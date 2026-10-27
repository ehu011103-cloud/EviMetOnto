import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from rdflib import Graph

from additional_baseline_common import (
    archive_task_views,
    call_json,
    is_done,
    merge_usage,
    output_empty,
    sanitize_predictions,
    write_task_views,
)
from preannotate_full_ontology_gold import (
    RELATION_TYPES,
    doc_text,
    load_dotenv_file,
    load_json,
    read_csv,
)
from preannotate_full_ontology_gold_split import aggregate_outputs
from run_ontogenia_deepseek_final50 import clean_evidence, create_requirements, turtle_report


DEFAULT_ROOT = Path("taxonomy_runs/gold_test_v2_high_signal_2026_06_25")
METHOD = "Memoryless-CQbyCQ-ESWC2025-adapted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memoryless CQbyCQ adapted ontology-generation baseline.")
    parser.add_argument("--manifest", default=str(DEFAULT_ROOT / "high_signal_final50_manifest.csv"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--requirements-dir", default="")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-input-chars", type=int, default=10000)
    parser.add_argument("--max-cqs", type=int, default=5)
    parser.add_argument("--max-hierarchy", type=int, default=18)
    parser.add_argument("--max-4d", type=int, default=28)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--timeout", type=float, default=420.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-empty", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def load_reused_requirements(directory: Path, sample_id: str) -> dict:
    path = directory / f"{sample_id}.hierarchy.json"
    if not path.exists():
        return {}
    payload = load_json(path)
    pipeline = payload.get("pipeline", {})
    value = pipeline.get("requirements", {})
    return value if isinstance(value, dict) else {}


def memoryless_module(
    client: OpenAI,
    args: argparse.Namespace,
    story: str,
    cq: dict,
) -> tuple[str, dict, dict]:
    prompt = f"""Create one self-contained ontology module for the current competency question.

Follow the official Memoryless CQbyCQ setup:
- The story contains no other competency questions.
- Work on only the current CQ.
- The previous RDF context is always empty; do not assume earlier modules exist.
- Decide whether the empty RDF answers the CQ. Normally it does not, so add the minimum reusable classes, strict subclass edges, properties, restrictions, and reification needed to answer it.
- Do not invent leaf concepts absent from the story or CQ. Do not model individuals, dates, coordinates, samples, or isolated measurements as reusable classes.
- Use reification only when a binary property cannot preserve all participants or qualifiers.

Return JSON with hierarchy_relations, nonhierarchical_relations, turtle, cq_answerability, and validation_notes.
Hierarchy fields: child_mention, parent_mention, child_canonical, parent_canonical, evidence_text, confidence.
Non-hierarchy fields: subject_mention, predicate, object_mention, subject_canonical, object_canonical, evidence_text, confidence.
Use only these non-hierarchical predicates: {', '.join(RELATION_TYPES)}.
A hierarchy edge is valid only if every child instance is a parent instance. Preserve relation direction and avoid cycles.
Evidence must be copied exactly from the current CQ evidence without quotation marks or ellipses.
Return at most 10 hierarchy edges and 10 non-hierarchical edges for this module.
Start Turtle with these prefixes:
@prefix ex: <http://www.example.org/mineral#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix dcterms: <http://purl.org/dc/terms/> .

Story:
{story}

Current CQ:
{json.dumps(cq, ensure_ascii=False)}

Previous RDF:
[EMPTY]"""
    return call_json(
        client,
        args,
        [
            {
                "role": "system",
                "content": "You are an ontology engineer applying the Memoryless CQbyCQ prompting technique.",
            },
            {"role": "user", "content": prompt},
        ],
        4200,
    )


def merge_modules(modules: list[dict], max_hierarchy: int, max_4d: int) -> tuple[dict, str, dict]:
    combined = {"hierarchy_relations": [], "nonhierarchical_relations": []}
    graph = Graph()
    valid_modules = 0
    for module in modules:
        parsed = module.get("parsed", {})
        combined["hierarchy_relations"].extend(parsed.get("hierarchy_relations", []))
        combined["nonhierarchical_relations"].extend(parsed.get("nonhierarchical_relations", []))
        turtle = str(parsed.get("turtle", "")).strip()
        if turtle:
            try:
                graph.parse(data=turtle, format="turtle")
                valid_modules += 1
            except Exception:
                pass
    predictions = sanitize_predictions(combined, max_hierarchy, max_4d)
    clean_evidence(predictions)
    merged_turtle = graph.serialize(format="turtle") if graph else ""
    report = turtle_report(merged_turtle)
    report["valid_modules"] = valid_modules
    report["total_modules"] = len(modules)
    return predictions, merged_turtle, report


def run_one(client: OpenAI, args: argparse.Namespace, raw_dir: Path, row: dict) -> dict:
    requirements = {}
    requirements_raw = ""
    requirements_usage = {}
    reused = False
    if args.requirements_dir:
        requirements = load_reused_requirements(Path(args.requirements_dir), row["sample_id"])
        reused = bool(requirements)
    if not requirements:
        text = doc_text(load_json(Path(row["file"])), args.max_input_chars)
        requirements_raw, requirements, requirements_usage = create_requirements(client, args, row, text)

    story = str(requirements.get("user_story", "")).strip()
    cqs = [item for item in requirements.get("competency_questions", []) if isinstance(item, dict)]
    cqs = cqs[: args.max_cqs]
    if not story or not cqs:
        raise ValueError("Requirement source returned an empty story or CQ list.")

    modules = []
    usages = [requirements_usage] if requirements_usage else []
    for index, cq in enumerate(cqs, start=1):
        raw, parsed, usage = memoryless_module(client, args, story, cq)
        usages.append(usage)
        modules.append(
            {
                "module": index,
                "cq": cq,
                "raw": raw,
                "parsed": parsed,
                "turtle_report": turtle_report(parsed.get("turtle")),
                "usage": usage,
            }
        )

    predictions, merged_turtle, merged_report = merge_modules(
        modules, args.max_hierarchy, args.max_4d
    )
    usage = merge_usage(*usages)
    write_task_views(
        raw_dir,
        row,
        args.model,
        args.base_url,
        METHOD,
        predictions,
        usage,
        {
            "source_paper": "Lippolis et al., Ontology Generation using Large Language Models, ESWC 2025",
            "official_method": "Memoryless CQbyCQ; each CQ receives empty previous RDF context",
            "requirements_reused": reused,
            "requirements_raw": requirements_raw,
            "requirements": requirements,
            "modules": modules,
            "merged_turtle": merged_turtle,
            "merged_turtle_report": merged_report,
        },
    )
    return {
        "sample_id": row["sample_id"],
        "cqs": len(cqs),
        "total_tokens": usage.get("total_tokens", 0),
        "hierarchy_items": len(predictions["hierarchy_relations"]),
        "relation_items": len(predictions["nonhierarchical_relations"]),
        "valid_turtle_modules": merged_report["valid_modules"],
    }


def main() -> int:
    load_dotenv_file()
    args = parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
    manifest = read_csv(Path(args.manifest))
    if args.limit:
        manifest = manifest[: args.limit]
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_memoryless_cqbycq_preannotations"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        pending = [
            row
            for row in manifest
            if not (
                args.resume
                and is_done(raw_dir, row["sample_id"])
                and not (args.retry_empty and output_empty(raw_dir, row["sample_id"]))
            )
        ]
        if args.retry_empty:
            for row in pending:
                if is_done(raw_dir, row["sample_id"]):
                    archive_task_views(raw_dir, row["sample_id"])
        client = OpenAI(
            api_key=api_key,
            base_url=args.base_url,
            timeout=args.timeout,
            max_retries=args.retries,
        )
        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, client, args, raw_dir, row): row for row in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                try:
                    result = future.result()
                    print(json.dumps({"completed": index, "pending_total": len(pending), **result}), flush=True)
                except Exception as exc:
                    failure = {
                        "sample_id": row["sample_id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    }
                    failures.append(failure)
                    print(json.dumps({"failed": index, "pending_total": len(pending), **failure}), flush=True)
                time.sleep(0.05)
        if failures:
            (out_dir / "run_failures.json").write_text(
                json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    summary = aggregate_outputs(out_dir, raw_dir, manifest)
    summary.update(
        {
            "baseline_method": METHOD,
            "model": args.model,
            "base_url": args.base_url,
            "calls_per_document": f"up to {args.max_cqs} independent CQ calls; requirement call omitted when reused",
            "adaptation_caveat": (
                "The official empty-RDF memoryless setup is retained. Paper text is converted to the user-story/CQ "
                "input expected by the method, independent CQ modules are deterministically unioned, and generated "
                "OWL content is mapped to the project's strict hierarchy and fixed 12-relation schema."
            ),
        }
    )
    (out_dir / "full_gold_draft_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

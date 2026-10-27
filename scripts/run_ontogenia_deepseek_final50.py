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


DEFAULT_ROOT = Path("taxonomy_runs/gold_test_v2_high_signal_2026_06_25")
METHOD = "Ontogenia-ESWC2025-adapted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ontogenia adapted ontology-generation baseline.")
    parser.add_argument("--manifest", default=str(DEFAULT_ROOT / "high_signal_final50_manifest.csv"))
    parser.add_argument("--out-dir", required=True)
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


def turtle_report(value: object) -> dict:
    turtle = str(value or "").strip()
    if not turtle:
        return {"syntax_valid": False, "triple_count": 0, "message": "No Turtle returned."}
    try:
        graph = Graph().parse(data=turtle, format="turtle")
        return {"syntax_valid": True, "triple_count": len(graph), "message": "RDFLib parse passed."}
    except Exception as exc:
        return {
            "syntax_valid": False,
            "triple_count": 0,
            "message": f"{type(exc).__name__}: {str(exc)[:500]}",
        }


def create_requirements(
    client: OpenAI, args: argparse.Namespace, row: dict, text: str
) -> tuple[str, dict, dict]:
    prompt = f"""Convert this mineral-geology paper excerpt into the requirements expected by Ontogenia.

Return JSON with:
- user_story: a concise ontology story defining purpose, scope, intended users, reusable domain concepts, and exclusions;
- competency_questions: exactly {args.max_cqs} focused objects with cq_id, question, expected_answer, evidence_text, and relation_family.

Each CQ must be answerable from an exact evidence span in the source and collectively cover strict hierarchy and the most important supported non-hierarchical relations. relation_family is HIERARCHY or one of {', '.join(RELATION_TYPES)}. Exclude bibliographic entities, coordinates, samples, isolated measurements, and unsupported background knowledge. Do not use or mention evaluation gold.

Document: {row['sample_id']} - {row['title']}
<source>{text}</source>"""
    return call_json(
        client,
        args,
        [
            {"role": "system", "content": "You prepare user stories and competency questions for evidence-grounded ontology engineering."},
            {"role": "user", "content": prompt},
        ],
        3200,
    )


def ontogenia_iteration(
    client: OpenAI,
    args: argparse.Namespace,
    story: str,
    cq: dict,
    previous: dict,
) -> tuple[str, dict, dict]:
    patterns = (
        "Explicit Typing; Tree for acyclic class hierarchies; Event and Participant Role for processes; "
        "Property Reification only for genuinely n-ary statements; Temporal Extent; Spatial Extent; "
        "Quantities and Units; Partonymy/Meronymy; Provenance."
    )
    procedure = """Apply the Ontogenia metacognitive procedure:
1. Analyze the competency question and identify its intended answer.
2. Identify domain context and scope from the user story.
3. Decompose the CQ into subject, predicate, object, and qualifiers.
4. Determine strict rdfs:subClassOf relations.
5. Add only necessary object properties, restrictions, and reification.
6. Consider equivalent/disjoint classes only when explicitly justified.
7. Integrate with and refine the previous ontology without deleting correct content.
8. Validate that the resulting ontology answers the CQ and remains logically coherent.
9. Test with minimal hypothetical instances, then report concise validation notes."""
    prompt = f"""Following the previous ontology, design and merge an ontology module that comprehensively answers the current competency question.

{procedure}

Ontology elements: classes, strict subclass relations, object properties with domain/range, labels, necessary OWL restrictions, and minimal test instances. Use these ontology design patterns when appropriate: {patterns}

Project adaptation constraints:
- Return JSON with hierarchy_relations, nonhierarchical_relations, turtle, cq_answerability, validation_notes, and test_instances.
- Hierarchy fields: child_mention, parent_mention, child_canonical, parent_canonical, evidence_text, confidence.
- Non-hierarchy fields: subject_mention, predicate, object_mention, subject_canonical, object_canonical, evidence_text, confidence.
- Only use non-hierarchical predicates: {', '.join(RELATION_TYPES)}.
- A hierarchy edge is valid only if every child instance is a parent instance.
- Preserve direction, avoid cycles, and do not turn paper-specific places, deposits, samples, dates, or quantities into reusable classes.
- Evidence must be copied exactly from the supplied CQ evidence without surrounding quotation marks or ellipses.
- Return the whole merged ontology at every iteration, as required by grouped-CQ Ontogenia.
- Keep no more than {args.max_hierarchy} hierarchy and {args.max_4d} non-hierarchical edges.
- Start every Turtle value with all six prefixes below, even when they appeared in the previous ontology:
  @prefix ex: <http://www.example.org/mineral#> .
  @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
  @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
  @prefix owl: <http://www.w3.org/2002/07/owl#> .
  @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
  @prefix dcterms: <http://purl.org/dc/terms/> .

User story:
{story}

Current CQ:
{json.dumps(cq, ensure_ascii=False)}

Previous ontology (empty on the first iteration):
{json.dumps(previous, ensure_ascii=False)}"""
    return call_json(
        client,
        args,
        [
            {"role": "system", "content": "You are an ontology engineer using the Ontogenia metacognitive prompting technique."},
            {"role": "user", "content": prompt},
        ],
        7000,
    )


def clean_evidence(predictions: dict) -> None:
    for group in ("hierarchy_relations", "nonhierarchical_relations"):
        for row in predictions[group]:
            row["evidence_text"] = str(row.get("evidence_text", "")).strip().strip(
                "\"'\u2018\u2019\u201c\u201d"
            )


def run_one(client: OpenAI, args: argparse.Namespace, raw_dir: Path, row: dict) -> dict:
    text = doc_text(load_json(Path(row["file"])), args.max_input_chars)
    requirements_raw, requirements, requirements_usage = create_requirements(client, args, row, text)
    story = str(requirements.get("user_story", "")).strip()
    cqs = [item for item in requirements.get("competency_questions", []) if isinstance(item, dict)]
    cqs = cqs[: args.max_cqs]
    if not story or not cqs:
        raise ValueError("Ontogenia requirement stage returned an empty story or CQ list.")

    previous = {}
    iterations = []
    usages = [requirements_usage]
    last_nonempty = {}
    for index, cq in enumerate(cqs, start=1):
        raw, parsed, usage = ontogenia_iteration(client, args, story, cq, previous)
        usages.append(usage)
        sanitized = sanitize_predictions(parsed, args.max_hierarchy, args.max_4d)
        nonempty = bool(sanitized["hierarchy_relations"] or sanitized["nonhierarchical_relations"])
        if nonempty:
            last_nonempty = parsed
            previous = parsed
        iterations.append(
            {
                "iteration": index,
                "cq": cq,
                "raw": raw,
                "parsed": parsed,
                "nonempty": nonempty,
                "turtle_report": turtle_report(parsed.get("turtle")),
                "usage": usage,
            }
        )

    final_source = previous or last_nonempty
    final_predictions = sanitize_predictions(final_source, args.max_hierarchy, args.max_4d)
    clean_evidence(final_predictions)
    usage = merge_usage(*usages)
    write_task_views(
        raw_dir,
        row,
        args.model,
        args.base_url,
        METHOD,
        final_predictions,
        usage,
        {
            "source_paper": "Lippolis et al., Ontology Generation using Large Language Models, ESWC 2025",
            "official_method": "Grouped-CQ Ontogenia with previous ontology in context",
            "requirements_raw": requirements_raw,
            "requirements": requirements,
            "iterations": iterations,
            "final_turtle_report": turtle_report(final_source.get("turtle")),
        },
    )
    return {
        "sample_id": row["sample_id"],
        "cqs": len(cqs),
        "total_tokens": usage.get("total_tokens", 0),
        "hierarchy_items": len(final_predictions["hierarchy_relations"]),
        "relation_items": len(final_predictions["nonhierarchical_relations"]),
        "turtle_valid": turtle_report(final_source.get("turtle"))["syntax_valid"],
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
    raw_dir = out_dir / "raw_ontogenia_preannotations"
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
            "calls_per_document": f"1 requirement call + up to {args.max_cqs} incremental CQ calls",
            "adaptation_caveat": (
                "The official grouped-CQ Ontogenia metacognitive procedure and previous-output memory are retained. "
                "Paper text is first converted to the user-story/CQ input expected by Ontogenia, and OWL output is "
                "mapped to the project's strict hierarchy and fixed 12-relation schema."
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

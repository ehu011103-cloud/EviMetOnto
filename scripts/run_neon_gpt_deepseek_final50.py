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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeOn-GPT adapted ontology-learning baseline.")
    parser.add_argument("--manifest", default=str(DEFAULT_ROOT / "high_signal_final50_manifest.csv"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://llm-api.net/v1"))
    parser.add_argument("--api-key-env", default=os.getenv("OPENAI_API_KEY_ENV", "DEEPSEEK_API_KEY"))
    parser.add_argument("--max-input-chars", type=int, default=10000)
    parser.add_argument("--max-hierarchy", type=int, default=18)
    parser.add_argument("--max-4d", type=int, default=28)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-empty", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def turtle_issue(value: object) -> str:
    turtle = str(value or "").strip()
    if not turtle:
        return "No Turtle draft was returned."
    try:
        Graph().parse(data=turtle, format="turtle")
        return "RDFLib syntax check passed."
    except Exception as exc:
        return f"RDFLib syntax error: {type(exc).__name__}: {str(exc)[:300]}"


def run_one(client: OpenAI, args: argparse.Namespace, raw_dir: Path, row: dict) -> dict:
    text = doc_text(load_json(Path(row["file"])), args.max_input_chars)
    relation_list = ", ".join(RELATION_TYPES)
    specification_prompt = f"""Act as an experienced knowledge engineer following the NeOn methodology.

From the mineral-geology paper below, specify the purpose, scope, ontology requirements, and up to six competency questions for a reusable concept-level ontology. Exclude samples, coordinates, one-off localities, bibliography items, and unsupported background knowledge.

Return JSON with: purpose, scope, requirements (array), competency_questions (array).

Document: {row['sample_id']} - {row['title']}
<source>{text}</source>"""
    spec_content, spec, spec_usage = call_json(
        client,
        args,
        [{"role": "system", "content": "You design evidence-grounded scientific ontologies."},
         {"role": "user", "content": specification_prompt}],
        1800,
    )

    draft_prompt = f"""Follow the NeOn conceptualization and implementation stages.

Use the source and ontology specification to extract reusable classes, strict subClassOf relations, and concept-level object-property relations. Use only these non-hierarchical predicates: {relation_list}.

Every relation must include a short verbatim evidence span from the source. A hierarchy edge is valid only when every instance of the child is an instance of the parent. Preserve direction and do not create sample, age, coordinate, mine, deposit-instance, or locality nodes.

Return JSON with hierarchy_relations, nonhierarchical_relations, and turtle. The two relation arrays use the same fields as below:
- hierarchy: child_mention, parent_mention, child_canonical, parent_canonical, evidence_text, confidence
- non-hierarchy: subject_mention, predicate, object_mention, subject_canonical, object_canonical, evidence_text, confidence
The Turtle must serialize the same classes and relations with valid RDF/OWL syntax.

Specification:
{json.dumps(spec, ensure_ascii=False)}

Document: {row['sample_id']} - {row['title']}
<source>{text}</source>"""
    draft_content, draft, draft_usage = call_json(
        client,
        args,
        [{"role": "system", "content": "You perform NeOn ontology conceptualization and Turtle implementation."},
         {"role": "user", "content": draft_prompt}],
        7000,
    )
    issue = turtle_issue(draft.get("turtle"))

    repair_prompt = f"""Perform the NeOn-GPT validation and repair stage on this ontology draft.

Validation requirements:
- Correct the reported Turtle syntax issue if present.
- Remove cycles, self-relations, non-is-a hierarchy edges, unsupported nodes, and relations without source evidence.
- Keep only the allowed predicates: {relation_list}.
- Do not add relations unless directly supported by the supplied source.
- Return final JSON with hierarchy_relations, nonhierarchical_relations, and corrected_turtle.
- Use the same relation fields as the draft and return at most {args.max_hierarchy} hierarchy and {args.max_4d} non-hierarchical relations.

Local validation report: {issue}
Specification: {json.dumps(spec, ensure_ascii=False)}
Draft: {json.dumps(draft, ensure_ascii=False)}
<source>{text}</source>"""
    repair_content, repaired, repair_usage = call_json(
        client,
        args,
        [{"role": "system", "content": "You repair ontology syntax, consistency, and modeling pitfalls."},
         {"role": "user", "content": repair_prompt}],
        7000,
    )
    final_predictions = sanitize_predictions(repaired, args.max_hierarchy, args.max_4d)
    fallback_used = False
    if not final_predictions["hierarchy_relations"] and not final_predictions["nonhierarchical_relations"]:
        final_predictions = sanitize_predictions(draft, args.max_hierarchy, args.max_4d)
        fallback_used = True
    usage = merge_usage(spec_usage, draft_usage, repair_usage)
    write_task_views(
        raw_dir,
        row,
        args.model,
        args.base_url,
        "NeOn-GPT-adapted",
        final_predictions,
        usage,
        {
            "specification_raw": spec_content,
            "draft_raw": draft_content,
            "repair_raw": repair_content,
            "local_turtle_check": issue,
            "draft_fallback_used": fallback_used,
        },
    )
    return {
        "sample_id": row["sample_id"],
        "total_tokens": usage["total_tokens"],
        "hierarchy_items": len(final_predictions["hierarchy_relations"]),
        "relation_items": len(final_predictions["nonhierarchical_relations"]),
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
    raw_dir = out_dir / "raw_neon_preannotations"
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
        client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout, max_retries=args.retries)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, client, args, raw_dir, row): row for row in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                print(json.dumps({"completed": index, "pending_total": len(pending), **result}), flush=True)
                time.sleep(0.05)

    summary = aggregate_outputs(out_dir, raw_dir, manifest)
    summary.update(
        {
            "baseline_method": "NeOn-GPT-adapted",
            "model": args.model,
            "base_url": args.base_url,
            "calls_per_document": 3,
            "stages": ["requirements_and_CQs", "conceptualization_and_Turtle", "validation_and_repair"],
            "adaptation_caveat": "RDFLib syntax and local cycle/schema checks replace the original HermiT and OOPS services; results are an adaptation, not an exact reproduction.",
        }
    )
    (out_dir / "full_gold_draft_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

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
    parser = argparse.ArgumentParser(description="POP plus LLM-Duo long-context adapted baseline.")
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


def run_one(client: OpenAI, args: argparse.Namespace, raw_dir: Path, row: dict) -> dict:
    text = doc_text(load_json(Path(row["file"])), args.max_input_chars)
    relation_list = ", ".join(RELATION_TYPES)
    explorer_prompt = f"""Act as the Explorer in a POP + LLM-Duo ontology annotation system.

Follow this prioritized ontology schedule, where each completed stage supplies context to the next:
1. Identify reusable geological concepts and their source mentions.
2. Annotate strict class-subclass relations between those concepts.
3. Annotate causal/logical predicates: CAUSES, CONTROLS, INDICATES, ASSOCIATED_WITH.
4. Annotate spatial/topological predicates: HOSTED_IN, INTRUDES_INTO, ZONES_WITH, DISTRIBUTED_ALONG.
5. Annotate temporal/evolution predicates: PRECEDES, OVERPRINTS, SYNCHRONOUS_WITH, FORMED_DURING.

Use only these predicates: {relation_list}. Preserve direction. Exclude samples, analytical spots, ages, coordinates, bibliography items, one-off localities, and relations lacking source evidence. Every candidate must include a short verbatim evidence span and a concise rationale.

Return JSON with hierarchy_relations and nonhierarchical_relations. Fields:
- hierarchy: child_mention, parent_mention, child_canonical, parent_canonical, evidence_text, rationale, confidence
- non-hierarchy: subject_mention, predicate, object_mention, subject_canonical, object_canonical, evidence_text, rationale, confidence
Return at most {args.max_hierarchy} hierarchy and {args.max_4d} non-hierarchical candidates.

Document: {row['sample_id']} - {row['title']}
<source>{text}</source>"""
    explorer_content, explorer, explorer_usage = call_json(
        client,
        args,
        [{"role": "system", "content": "You are a source-grounded ontology annotation explorer."},
         {"role": "user", "content": explorer_prompt}],
        7600,
    )

    evaluator_prompt = f"""Act as the independent Evaluator in LLM-Duo. You do not have the full paper. Adversarially review the Explorer's candidates using only their quoted evidence and rationale.

Reject hierarchy edges that are not necessary is-a relations. Reject non-hierarchical predicates outside this schema or with unsupported direction: {relation_list}. Flag invented nodes, instance leakage, vague co-occurrence, and evidence that does not entail the proposed relation.

Return concise JSON with rejected_hierarchy_indices, rejected_nonhierarchical_indices, hierarchy_revisions, and nonhierarchical_revisions. Each revision contains index, corrected_fields, and brief feedback. Any index not listed is KEEP. Do not repeat kept candidates and do not create unrelated new candidates.

Explorer output:
{json.dumps(explorer, ensure_ascii=False)}"""
    evaluator_content, evaluator, evaluator_usage = call_json(
        client,
        args,
        [{"role": "system", "content": "You are a skeptical ontology annotation evaluator."},
         {"role": "user", "content": evaluator_prompt}],
        5000,
    )

    revision_prompt = f"""Return the final Explorer revision after considering the Evaluator feedback.

Use the source as the authority. Keep well-supported candidates, apply valid corrections, reject unsupported candidates, and defend an original candidate only when the source clearly supports it. Use only these predicates: {relation_list}. Preserve strict is-a semantics and relation direction.

Return JSON with hierarchy_relations and nonhierarchical_relations. Keep the original relation fields except rationale may be omitted. Return at most {args.max_hierarchy} hierarchy and {args.max_4d} non-hierarchical relations.

Explorer draft: {json.dumps(explorer, ensure_ascii=False)}
Evaluator feedback: {json.dumps(evaluator, ensure_ascii=False)}
<source>{text}</source>"""
    revision_content, revision, revision_usage = call_json(
        client,
        args,
        [{"role": "system", "content": "You are the Explorer performing the final evidence-grounded revision."},
         {"role": "user", "content": revision_prompt}],
        7600,
    )
    final_predictions = sanitize_predictions(revision, args.max_hierarchy, args.max_4d)
    fallback_used = False
    if not final_predictions["hierarchy_relations"] and not final_predictions["nonhierarchical_relations"]:
        final_predictions = sanitize_predictions(explorer, args.max_hierarchy, args.max_4d)
        fallback_used = True
    usage = merge_usage(explorer_usage, evaluator_usage, revision_usage)
    write_task_views(
        raw_dir,
        row,
        args.model,
        args.base_url,
        "POP-LLM-Duo-LongContext-adapted",
        final_predictions,
        usage,
        {
            "explorer_raw": explorer_content,
            "evaluator_raw": evaluator_content,
            "revision_raw": revision_content,
            "schedule": ["concepts", "hierarchy", "causal_logical", "spatial", "temporal"],
            "explorer_fallback_used": fallback_used,
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
    raw_dir = out_dir / "raw_pop_duo_preannotations"
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
            "baseline_method": "POP-LLM-Duo-LongContext-adapted",
            "model": args.model,
            "base_url": args.base_url,
            "calls_per_document": 3,
            "stages": ["progressive_explorer", "independent_evaluator", "explorer_revision"],
            "adaptation_caveat": "Uses the paper's long-context LLM-Duo variant and a fixed mineral relation-schema traversal; it does not reuse GoldOntology learned nodes and is not the original RAG implementation.",
        }
    )
    (out_dir / "full_gold_draft_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

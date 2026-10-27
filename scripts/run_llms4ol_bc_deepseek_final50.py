"""Run a direct LLMs4OL Task B+C adaptation on the Final50 corpus.

This runner deliberately keeps taxonomy discovery and non-taxonomic relation
extraction independent.  It does not call the GoldOntology R-H-F cascade,
validation, confidence rubric, or endpoint-alignment modules.
"""

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
METHOD = "LLMs4OL-TaskB+C-adapted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct LLMs4OL Task B+C adapted full-graph baseline."
    )
    parser.add_argument(
        "--manifest", default=str(DEFAULT_ROOT / "high_signal_final50_manifest.csv")
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-input-chars", type=int, default=10000)
    parser.add_argument("--max-hierarchy", type=int, default=18)
    parser.add_argument("--max-4d", type=int, default=28)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--timeout", type=float, default=360.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-empty", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def taxonomy_prompt(row: dict, text: str, max_items: int) -> str:
    return f"""Perform the LLMs4OL Task B (taxonomy discovery) task on one mineral-geology paper.

Document: {row['sample_id']} - {row['title']}

Independently discover reusable domain concepts and strict is-a taxonomy edges from the source.
An edge child -> parent is valid only if every instance of the child is an instance of the parent.
Do not convert hosted-in, contains, part-of, deposit occurrence, commodity association, cause, time,
or locality relations into taxonomy. Do not create nodes for samples, mine/deposit instances, ages,
coordinates, or bibliography items. Use only source-supported evidence.

Return only a JSON object with at most {max_items} items:
{{
  "hierarchy_relations": [
    {{
      "child_mention": "source term",
      "parent_mention": "source term",
      "child_canonical": "PascalCaseClass",
      "parent_canonical": "PascalCaseClass",
      "evidence_text": "short verbatim evidence",
      "confidence": 0.0
    }}
  ]
}}

Source:
<source>{text}</source>"""


def relation_prompt(row: dict, text: str, max_items: int) -> str:
    predicates = ", ".join(RELATION_TYPES)
    return f"""Perform the LLMs4OL Task C (non-taxonomic relation extraction) task on one mineral-geology paper.

Document: {row['sample_id']} - {row['title']}

Independently extract evidence-supported, concept-level, non-hierarchical relations. This target-task
adaptation maps the output to only these fixed predicates: {predicates}.
Preserve direction. ASSOCIATED_WITH and SYNCHRONOUS_WITH are symmetric. Do not emit subClassOf here.
Do not invent predicates or use samples, isolated deposits/mines, coordinates, dates, or bibliography items
as reusable nodes. Evidence must be a short verbatim span from the source.

Return only a JSON object with at most {max_items} items:
{{
  "nonhierarchical_relations": [
    {{
      "subject_mention": "source term",
      "predicate": "one allowed predicate",
      "object_mention": "source term",
      "subject_canonical": "PascalCaseClass",
      "object_canonical": "PascalCaseClass",
      "evidence_text": "short verbatim evidence",
      "confidence": 0.0
    }}
  ]
}}

Source:
<source>{text}</source>"""


def run_one(client: OpenAI, args: argparse.Namespace, raw_dir: Path, row: dict) -> dict:
    text = doc_text(load_json(Path(row["file"])), args.max_input_chars)
    hierarchy_raw, hierarchy, hierarchy_usage = call_json(
        client,
        args,
        [
            {
                "role": "system",
                "content": "You are an ontology-learning system completing LLMs4OL taxonomy discovery.",
            },
            {"role": "user", "content": taxonomy_prompt(row, text, args.max_hierarchy)},
        ],
        4200,
    )
    relation_raw, relations, relation_usage = call_json(
        client,
        args,
        [
            {
                "role": "system",
                "content": "You are an ontology-learning system completing LLMs4OL non-taxonomic relation extraction.",
            },
            {"role": "user", "content": relation_prompt(row, text, args.max_4d)},
        ],
        5200,
    )
    final_predictions = sanitize_predictions(
        {
            "hierarchy_relations": hierarchy.get("hierarchy_relations", []),
            "nonhierarchical_relations": relations.get("nonhierarchical_relations", []),
        },
        args.max_hierarchy,
        args.max_4d,
    )
    usage = merge_usage(hierarchy_usage, relation_usage)
    write_task_views(
        raw_dir,
        row,
        args.model,
        args.base_url,
        METHOD,
        final_predictions,
        usage,
        {
            "source_paper": "Babaei Giglou et al., LLMs4OL: Large Language Models for Ontology Learning, 2023",
            "task_b_raw": hierarchy_raw,
            "task_c_raw": relation_raw,
            "tasks": ["taxonomy_discovery", "non_taxonomic_relation_extraction"],
            "postprocessing": ["JSON normalization", "canonical-key deduplication", "taxonomy cycle removal"],
            "excluded_goldontology_modules": [
                "R-H-F relation cascade",
                "counterexample validation",
                "bidirectional/evidence validation",
                "confidence rubric hard gate",
                "endpoint alignment",
            ],
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
    raw_dir = out_dir / "raw_llms4ol_bc"
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
            "calls_per_document": 2,
            "adaptation_caveat": (
                "Task B and Task C are run independently as in the LLMs4OL task split. "
                "Task C is constrained to the study's fixed 12-predicate target schema; this is a task adaptation, "
                "not an exact reproduction of every LLMs4OL benchmark setting."
            ),
            "excluded_goldontology_modules": [
                "R-H-F relation cascade",
                "counterexample validation",
                "bidirectional/evidence validation",
                "confidence rubric hard gate",
                "endpoint alignment",
            ],
        }
    )
    (out_dir / "full_gold_draft_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

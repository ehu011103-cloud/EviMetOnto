import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from preannotate_full_ontology_gold import (
    RELATION_TYPES,
    doc_text,
    load_dotenv_file,
    load_json,
    parse_json_object,
    read_csv,
)
from preannotate_full_ontology_gold_split import aggregate_outputs


def build_joint_prompt(sample_id: str, title: str, text: str, max_hierarchy: int, max_4d: int) -> str:
    relation_list = ", ".join(RELATION_TYPES)
    return f"""You are creating an LLM-assisted draft prediction set for evaluating a mineral-domain ontology extraction system.

Document ID: {sample_id}
Document title: {title}

Joint task: extract BOTH strict hierarchical relations and schema-constrained non-hierarchical relations in one response.

Hierarchy rules:
- Output only true class-subclass relations: every instance of Child must be an instance of Parent.
- Do not mark hosted-in, contains, part-of, caused-by, occurs-in, commodity association, or locality relations as subClassOf.
- Prefer reusable scientific classes, not individual deposits, mines, samples, ages, regions, or coordinates.
- Include concise evidence from the source text.
- Do not extract from bibliography/reference-list titles.
- Return at most {max_hierarchy} high-value hierarchy relations.

Non-hierarchical relation rules:
- Use ONLY these predicates: {relation_list}
- Extract concept-level mineral/geological relations only.
- Preserve direction. For symmetric relations, use ASSOCIATED_WITH or SYNCHRONOUS_WITH.
- Do not invent predicates outside the schema.
- Do not use individual sample IDs, analytical spots, coordinates, or one-off localities as canonical ontology nodes.
- Do not extract class-subclass relations in this list.
- Include concise evidence from the source text.
- Do not extract from bibliography/reference-list titles.
- Return at most {max_4d} high-value non-hierarchical relations.

Return only valid JSON:
{{
  "hierarchy_relations": [
    {{
      "child_mention": "string",
      "parent_mention": "string",
      "child_canonical": "PascalCaseClass",
      "parent_canonical": "PascalCaseClass",
      "evidence_text": "short evidence from source, 25 words or fewer",
      "confidence": 0.0
    }}
  ],
  "nonhierarchical_relations": [
    {{
      "subject_mention": "string",
      "predicate": "one of the 12 allowed predicates",
      "object_mention": "string",
      "subject_canonical": "PascalCaseClass",
      "object_canonical": "PascalCaseClass",
      "evidence_text": "short evidence from source, 25 words or fewer",
      "confidence": 0.0
    }}
  ]
}}

Source text:
<source>
{text}
</source>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint hierarchy and 4D ontology extraction ablation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key-env", default=os.getenv("OPENAI_API_KEY_ENV", "DEEPSEEK_API_KEY"))
    parser.add_argument("--max-input-chars", type=int, default=10000)
    parser.add_argument("--max-hierarchy", type=int, default=18)
    parser.add_argument("--max-4d", type=int, default=28)
    parser.add_argument("--max-tokens", type=int, default=5400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def task_paths(raw_dir: Path, sample_id: str) -> tuple[Path, Path]:
    return raw_dir / f"{sample_id}.hierarchy.json", raw_dir / f"{sample_id}.relations.json"


def is_done(raw_dir: Path, sample_id: str) -> bool:
    return all(path.exists() for path in task_paths(raw_dir, sample_id))


def run_one(client: OpenAI, args: argparse.Namespace, raw_dir: Path, row: dict) -> dict:
    sample_id = row["sample_id"]
    source_file = row["file"]
    title = row["title"]
    data = load_json(Path(source_file))
    prompt = build_joint_prompt(
        sample_id,
        title,
        doc_text(data, args.max_input_chars),
        args.max_hierarchy,
        args.max_4d,
    )
    request = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    is_deepseek = "deepseek" in args.base_url.lower() or "deepseek" in args.model.lower()
    if is_deepseek:
        request["extra_body"] = {"thinking": {"type": args.thinking}}
    elif args.reasoning_effort:
        request["reasoning_effort"] = args.reasoning_effort
    response = client.chat.completions.create(**request)
    content = response.choices[0].message.content or ""
    parsed = parse_json_object(content)
    usage = getattr(response, "usage", None)
    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else {}
    common = {
        "sample_id": sample_id,
        "source_file": source_file,
        "title": title,
        "model": args.model,
        "base_url": args.base_url,
    }
    hierarchy_path, relations_path = task_paths(raw_dir, sample_id)
    hierarchy_payload = {
        **common,
        "task": "joint_hierarchy_view",
        "raw_response": content,
        "parsed": {"hierarchy_relations": parsed.get("hierarchy_relations", [])},
        "usage": usage_dict,
    }
    relation_payload = {
        **common,
        "task": "joint_relations_view",
        "raw_response": content,
        "parsed": {
            "nonhierarchical_relations": parsed.get("nonhierarchical_relations", [])
            or parsed.get("relations", [])
        },
        "usage": {},
    }
    hierarchy_path.write_text(json.dumps(hierarchy_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    relations_path.write_text(json.dumps(relation_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "sample_id": sample_id,
        "total_tokens": int(usage_dict.get("total_tokens", 0) or 0),
        "hierarchy_items": len(hierarchy_payload["parsed"]["hierarchy_relations"]),
        "relation_items": len(relation_payload["parsed"]["nonhierarchical_relations"]),
    }


def main() -> int:
    load_dotenv_file()
    args = parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
    manifest_rows = read_csv(Path(args.manifest))
    if args.limit:
        manifest_rows = manifest_rows[: args.limit]
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_joint_preannotations"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        pending = [row for row in manifest_rows if not (args.resume and is_done(raw_dir, row["sample_id"]))]
        client = OpenAI(
            api_key=api_key,
            base_url=args.base_url,
            timeout=args.timeout,
            max_retries=args.retries,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, client, args, raw_dir, row): row for row in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                print(
                    json.dumps(
                        {"completed": index, "pending_total": len(pending), **result},
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
                time.sleep(0.05)

    summary = aggregate_outputs(out_dir, raw_dir, manifest_rows)
    summary.update(
        {
            "scope": f"Joint {args.model} hierarchy and non-hierarchical extraction prediction set.",
            "extraction_mode": "single_joint_call_per_document",
            "manifest": args.manifest,
            "model": args.model,
            "base_url": args.base_url,
            "api_calls_expected": len(manifest_rows),
            "max_input_chars": args.max_input_chars,
            "max_hierarchy": args.max_hierarchy,
            "max_4d": args.max_4d,
            "max_tokens": args.max_tokens,
            "thinking": args.thinking
            if "deepseek" in args.base_url.lower() or "deepseek" in args.model.lower()
            else "not_applicable",
            "reasoning_effort": args.reasoning_effort,
            "note": "Two raw task-view files are stored per document for reuse of the split-output aggregator, but only one API call was made per document.",
        }
    )
    (out_dir / "full_gold_draft_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import csv
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

from openai import OpenAI

from preannotate_full_ontology_gold import (
    RELATION_TYPES,
    as_float,
    canonical,
    doc_text,
    load_dotenv_file,
    load_json,
    norm_space,
    parse_json_object,
    read_csv,
    write_csv,
)


def build_hierarchy_prompt(sample_id: str, title: str, text: str, max_items: int) -> str:
    return f"""You are creating an LLM-assisted draft gold set for evaluating a mineral-domain ontology extraction system.

Document ID: {sample_id}
Document title: {title}

Task: extract strict hierarchical taxonomy relations from the source text.

Rules:
- Output only true class-subclass relations: every instance of Child must be an instance of Parent.
- Do not mark hosted-in, contains, part-of, caused-by, occurs-in, commodity association, or locality relations as subClassOf.
- Prefer reusable scientific classes, not individual deposits, mines, samples, ages, regions, or coordinates.
- Do not extract from bibliography/reference-list titles.
- Include concise evidence from the source text.
- Return at most {max_items} high-value hierarchy relations.

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
  ]
}}

Source text:
\"\"\"
{text}
\"\"\"
"""


def build_relation_prompt(sample_id: str, title: str, text: str, max_items: int) -> str:
    relation_list = ", ".join(RELATION_TYPES)
    return f"""You are creating an LLM-assisted draft gold set for evaluating a mineral-domain ontology extraction system.

Document ID: {sample_id}
Document title: {title}

Task: extract non-hierarchical ontology relations using ONLY these predicates:
{relation_list}

Rules:
- Extract concept-level mineral/geological relations only.
- Use only the 12 predicates listed above.
- Preserve direction. For symmetric relations, use ASSOCIATED_WITH or SYNCHRONOUS_WITH.
- Do not invent predicates outside the schema. If a fact is important but outside schema, omit it.
- Do not use individual sample IDs, analytical spots, coordinates, or one-off localities as canonical ontology nodes.
- Do not extract class-subclass relations here.
- Include concise evidence from the source text.
- Do not extract from bibliography/reference-list titles.
- Return at most {max_items} high-value non-hierarchical relations.

Return only valid JSON:
{{
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
\"\"\"
{text}
\"\"\"
"""


def api_json(
    client: OpenAI,
    model: str,
    base_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    thinking: str,
) -> tuple[str, dict, dict]:
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if "deepseek" in base_url.lower() or "deepseek" in model.lower():
        kwargs["extra_body"] = {"thinking": {"type": thinking}}
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else {}
    return content, parse_json_object(content), usage_dict


def raw_task_path(raw_dir: Path, sample_id: str, task: str) -> Path:
    return raw_dir / f"{sample_id}.{task}.json"


def completed_doc_ids(raw_dir: Path) -> set[str]:
    hierarchy = {path.name.split(".")[0] for path in raw_dir.glob("*.hierarchy.json")}
    relations = {path.name.split(".")[0] for path in raw_dir.glob("*.relations.json")}
    return hierarchy & relations


def load_raw_task(raw_dir: Path, sample_id: str, task: str) -> dict:
    path = raw_task_path(raw_dir, sample_id, task)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def relation_items(parsed: dict) -> list:
    rows = parsed.get("nonhierarchical_relations")
    if isinstance(rows, list):
        return rows
    rows = parsed.get("relations")
    if isinstance(rows, list):
        return rows
    return []


def hierarchy_items(parsed: dict) -> list:
    rows = parsed.get("hierarchy_relations")
    if isinstance(rows, list):
        return rows
    rows = parsed.get("relations")
    if isinstance(rows, list):
        return rows
    return []


def aggregate_outputs(out_dir: Path, raw_dir: Path, manifest_rows: list[dict]) -> dict:
    hierarchy_rows = []
    relation_rows = []
    alignment_rows_by_key: OrderedDict[tuple[str, str, str], dict] = OrderedDict()
    canonical_rows = []
    parse_empty_docs = []
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    task_counts = {"hierarchy": 0, "relations": 0}

    for row in manifest_rows:
        sample_id = row["sample_id"]
        source_file = row["file"]
        title = row["title"]
        hierarchy_task = load_raw_task(raw_dir, sample_id, "hierarchy")
        relation_task = load_raw_task(raw_dir, sample_id, "relations")
        if not hierarchy_task and not relation_task:
            continue

        model = hierarchy_task.get("model") or relation_task.get("model") or ""
        for task_name, task in [("hierarchy", hierarchy_task), ("relations", relation_task)]:
            if not task:
                continue
            task_counts[task_name] += 1
            usage = task.get("usage", {}) or {}
            for key in token_usage:
                token_usage[key] += int(usage.get(key, 0) or 0)

        h_items = hierarchy_items(hierarchy_task.get("parsed") or {})
        r_items = relation_items(relation_task.get("parsed") or {})
        if not h_items and not r_items:
            parse_empty_docs.append(sample_id)

        for rel_index, rel in enumerate(h_items if isinstance(h_items, list) else [], start=1):
            if not isinstance(rel, dict):
                continue
            child_m = norm_space(rel.get("child_mention"))
            parent_m = norm_space(rel.get("parent_mention"))
            child_c = norm_space(rel.get("child_canonical")) or canonical(child_m)
            parent_c = norm_space(rel.get("parent_canonical")) or canonical(parent_m)
            if not child_c or not parent_c:
                continue
            out_row = {
                "annotation_id": f"{sample_id}_H_{rel_index:03d}",
                "sample_id": sample_id,
                "source_file": source_file,
                "title": title,
                "child_mention": child_m,
                "parent_mention": parent_m,
                "child_canonical": child_c,
                "parent_canonical": parent_c,
                "predicate": "subClassOf",
                "evidence_text": norm_space(rel.get("evidence_text")),
                "preannotator_decision": "VALID",
                "confidence": as_float(rel.get("confidence")),
                "gold_decision": "",
                "gold_error_type": "",
                "gold_note": "",
                "model": model,
            }
            hierarchy_rows.append(out_row)
            canonical_rows.append(
                {
                    "annotation_id": out_row["annotation_id"],
                    "sample_id": sample_id,
                    "source_file": source_file,
                    "title": title,
                    "subject_canonical": child_c,
                    "predicate": "subClassOf",
                    "object_canonical": parent_c,
                    "evidence_text": out_row["evidence_text"],
                    "source_layer": "hierarchy",
                    "gold_decision": "",
                    "gold_note": "",
                }
            )
            for mention, node in [(child_m, child_c), (parent_m, parent_c)]:
                key = (sample_id, mention.lower(), node)
                alignment_rows_by_key.setdefault(
                    key,
                    {
                        "sample_id": sample_id,
                        "source_file": source_file,
                        "title": title,
                        "mention": mention,
                        "canonical_node": node,
                        "entity_type": "Other",
                        "evidence_text": out_row["evidence_text"],
                        "preannotator_decision": "VALID",
                        "confidence": out_row["confidence"],
                        "gold_decision": "",
                        "gold_note": "",
                    },
                )

        for rel_index, rel in enumerate(r_items if isinstance(r_items, list) else [], start=1):
            if not isinstance(rel, dict):
                continue
            predicate = norm_space(rel.get("predicate") or rel.get("relation")).upper()
            if predicate not in RELATION_TYPES:
                continue
            subject_m = norm_space(rel.get("subject_mention") or rel.get("subject"))
            object_m = norm_space(rel.get("object_mention") or rel.get("object"))
            subject_c = norm_space(rel.get("subject_canonical")) or canonical(subject_m)
            object_c = norm_space(rel.get("object_canonical")) or canonical(object_m)
            if not subject_c or not object_c:
                continue
            out_row = {
                "annotation_id": f"{sample_id}_R_{rel_index:03d}",
                "sample_id": sample_id,
                "source_file": source_file,
                "title": title,
                "subject_mention": subject_m,
                "predicate": predicate,
                "object_mention": object_m,
                "subject_canonical": subject_c,
                "object_canonical": object_c,
                "evidence_text": norm_space(rel.get("evidence_text") or rel.get("evidence")),
                "preannotator_decision": "VALID",
                "confidence": as_float(rel.get("confidence")),
                "gold_decision": "",
                "gold_error_type": "",
                "gold_note": "",
                "model": model,
            }
            relation_rows.append(out_row)
            canonical_rows.append(
                {
                    "annotation_id": out_row["annotation_id"],
                    "sample_id": sample_id,
                    "source_file": source_file,
                    "title": title,
                    "subject_canonical": subject_c,
                    "predicate": predicate,
                    "object_canonical": object_c,
                    "evidence_text": out_row["evidence_text"],
                    "source_layer": "nonhierarchical",
                    "gold_decision": "",
                    "gold_note": "",
                }
            )
            for mention, node in [(subject_m, subject_c), (object_m, object_c)]:
                key = (sample_id, mention.lower(), node)
                alignment_rows_by_key.setdefault(
                    key,
                    {
                        "sample_id": sample_id,
                        "source_file": source_file,
                        "title": title,
                        "mention": mention,
                        "canonical_node": node,
                        "entity_type": "Other",
                        "evidence_text": out_row["evidence_text"],
                        "preannotator_decision": "VALID",
                        "confidence": out_row["confidence"],
                        "gold_decision": "",
                        "gold_note": "",
                    },
                )

    hierarchy_fields = [
        "annotation_id",
        "sample_id",
        "source_file",
        "title",
        "child_mention",
        "parent_mention",
        "child_canonical",
        "parent_canonical",
        "predicate",
        "evidence_text",
        "preannotator_decision",
        "confidence",
        "gold_decision",
        "gold_error_type",
        "gold_note",
        "model",
    ]
    relation_fields = [
        "annotation_id",
        "sample_id",
        "source_file",
        "title",
        "subject_mention",
        "predicate",
        "object_mention",
        "subject_canonical",
        "object_canonical",
        "evidence_text",
        "preannotator_decision",
        "confidence",
        "gold_decision",
        "gold_error_type",
        "gold_note",
        "model",
    ]
    alignment_fields = [
        "sample_id",
        "source_file",
        "title",
        "mention",
        "canonical_node",
        "entity_type",
        "evidence_text",
        "preannotator_decision",
        "confidence",
        "gold_decision",
        "gold_note",
    ]
    canonical_fields = [
        "annotation_id",
        "sample_id",
        "source_file",
        "title",
        "subject_canonical",
        "predicate",
        "object_canonical",
        "evidence_text",
        "source_layer",
        "gold_decision",
        "gold_note",
    ]

    write_csv(out_dir / "full_gold_draft_hierarchy_review.csv", hierarchy_rows, hierarchy_fields)
    write_csv(out_dir / "full_gold_draft_4d_relations_review.csv", relation_rows, relation_fields)
    write_csv(out_dir / "full_gold_draft_alignment_review.csv", list(alignment_rows_by_key.values()), alignment_fields)
    write_csv(out_dir / "full_gold_draft_canonical_triples_review.csv", canonical_rows, canonical_fields)
    summary = {
        "scope": "Split GPT/LLM-assisted ontology draft; human verification required before final gold use.",
        "out_dir": str(out_dir),
        "documents_requested": len(manifest_rows),
        "documents_completed": len(completed_doc_ids(raw_dir)),
        "task_counts": task_counts,
        "hierarchy_rows": len(hierarchy_rows),
        "nonhierarchical_rows": len(relation_rows),
        "alignment_rows": len(alignment_rows_by_key),
        "canonical_triple_rows": len(canonical_rows),
        "parse_empty_docs": parse_empty_docs,
        "token_usage": token_usage,
        "outputs": {
            "hierarchy_review": str(out_dir / "full_gold_draft_hierarchy_review.csv"),
            "4d_relations_review": str(out_dir / "full_gold_draft_4d_relations_review.csv"),
            "alignment_review": str(out_dir / "full_gold_draft_alignment_review.csv"),
            "canonical_triples_review": str(out_dir / "full_gold_draft_canonical_triples_review.csv"),
            "raw_dir": str(raw_dir),
        },
    }
    (out_dir / "full_gold_draft_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split full ontology pre-annotation into shorter hierarchy/4D calls.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key-env", default=os.getenv("OPENAI_API_KEY_ENV", "DEEPSEEK_API_KEY"))
    parser.add_argument("--max-input-chars", type=int, default=10000)
    parser.add_argument("--max-hierarchy", type=int, default=18)
    parser.add_argument("--max-4d", type=int, default=24)
    parser.add_argument("--hierarchy-max-tokens", type=int, default=2200)
    parser.add_argument("--relations-max-tokens", type=int, default=3200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--api-timeout", type=float, default=240.0)
    parser.add_argument("--api-retries", type=int, default=1)
    parser.add_argument("--api-sleep", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_dotenv_file()
    args = parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_split_preannotations"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_csv(manifest_path)
    if args.limit:
        manifest_rows = manifest_rows[: args.limit]

    if not args.aggregate_only:
        client = OpenAI(
            api_key=api_key,
            base_url=args.base_url,
            timeout=args.api_timeout,
            max_retries=args.api_retries,
        )
        for index, row in enumerate(manifest_rows, start=1):
            sample_id = row["sample_id"]
            source_file = row["file"]
            title = row["title"]
            data = load_json(Path(source_file))
            text = doc_text(data, args.max_input_chars)
            tasks = [
                (
                    "hierarchy",
                    build_hierarchy_prompt(sample_id, title, text, args.max_hierarchy),
                    args.hierarchy_max_tokens,
                ),
                (
                    "relations",
                    build_relation_prompt(sample_id, title, text, args.max_4d),
                    args.relations_max_tokens,
                ),
            ]
            for task_name, prompt, max_tokens in tasks:
                path = raw_task_path(raw_dir, sample_id, task_name)
                if args.resume and path.exists():
                    continue
                content, parsed, usage = api_json(
                    client=client,
                    model=args.model,
                    base_url=args.base_url,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=args.temperature,
                    thinking=args.thinking,
                )
                payload = {
                    "sample_id": sample_id,
                    "source_file": source_file,
                    "title": title,
                    "task": task_name,
                    "model": args.model,
                    "base_url": args.base_url,
                    "raw_response": content,
                    "parsed": parsed,
                    "usage": usage,
                }
                path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                print(
                    json.dumps(
                        {
                            "completed_doc_index": index,
                            "total_docs": len(manifest_rows),
                            "sample_id": sample_id,
                            "task": task_name,
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
                if args.api_sleep:
                    time.sleep(args.api_sleep)

    summary = aggregate_outputs(out_dir, raw_dir, manifest_rows)
    summary.update(
        {
            "manifest": str(manifest_path),
            "model": args.model,
            "base_url": args.base_url,
            "max_input_chars": args.max_input_chars,
            "max_hierarchy": args.max_hierarchy,
            "max_4d": args.max_4d,
        }
    )
    (out_dir / "full_gold_draft_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

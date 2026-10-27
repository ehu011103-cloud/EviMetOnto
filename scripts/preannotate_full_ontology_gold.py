import argparse
import csv
import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path


DEFAULT_MANIFEST = (
    "taxonomy_runs/full_gold_ontology_gpt_2026_06_18/full_gold_gold_manifest_pure.csv"
)
DEFAULT_OUT_DIR = "taxonomy_runs/full_gold_ontology_gpt_2026_06_18"

RELATION_TYPES = [
    "ASSOCIATED_WITH",
    "HOSTED_IN",
    "CAUSES",
    "CONTROLS",
    "FORMED_DURING",
    "PRECEDES",
    "INDICATES",
    "OVERPRINTS",
    "DISTRIBUTED_ALONG",
    "INTRUDES_INTO",
    "SYNCHRONOUS_WITH",
    "ZONES_WITH",
]


def load_dotenv_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def doc_text(data: dict, max_chars: int) -> str:
    metadata = data.get("metadata", {}) if isinstance(data.get("metadata", {}), dict) else {}
    content = data.get("content", [])
    if isinstance(content, list):
        body = "\n".join(str(item) for item in content)
    else:
        body = str(content)

    body = trim_reference_section(body)
    prefix = [
        f"Title: {metadata.get('title', '')}",
        f"Abstract: {metadata.get('abstract', '')}",
        f"Keywords: {', '.join(str(x) for x in metadata.get('keywords', []) if x)}",
        "",
        "Full text:",
    ]
    text = "\n".join(prefix) + "\n" + body
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[TRUNCATED_FOR_PREANNOTATION]"
    return text


def trim_reference_section(text: str) -> str:
    patterns = [
        r"\n\s*references\s*\n",
        r"\n\s*bibliography\s*\n",
        r"\n\s*acknowledg(e)?ments\s*\n",
    ]
    lowered = text.lower()
    cut = len(text)
    for pattern in patterns:
        match = re.search(pattern, lowered, flags=re.I)
        if match:
            cut = min(cut, match.start())
    return text[:cut]


def parse_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {
        "hierarchy_relations": [],
        "nonhierarchical_relations": [],
        "alignment_mappings": [],
    }


def norm_space(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def canonical(value: object) -> str:
    text = norm_space(value)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    parts = [part for part in text.strip().split() if part]
    if not parts:
        return ""
    return "".join(part[:1].upper() + part[1:] for part in parts)


def completed_docs(raw_dir: Path) -> set[str]:
    return {path.stem for path in raw_dir.glob("*.json")}


def build_prompt(
    sample_id: str,
    title: str,
    text: str,
    max_hierarchy: int,
    max_4d: int,
    max_alignment: int,
) -> str:
    relation_list = ", ".join(RELATION_TYPES)
    return f"""You are creating a GPT/LLM-assisted draft gold set for evaluating a mineral-domain ontology extraction system.

Document ID: {sample_id}
Document title: {title}

Task: read the source text and extract a FULL ONTOLOGY GOLD DRAFT for this document.
This is pre-annotation only. A human will later verify every row.

Extract three things:
1. Strict hierarchical taxonomy relations: child subClassOf parent.
2. Non-hierarchical ontology relations using ONLY these predicates:
   {relation_list}
3. Mention-to-canonical-node mappings only for important aliases not already obvious from the extracted relations.

Hierarchy rules:
- Output only true class-subclass relations: every instance of Child must be an instance of Parent.
- Do not mark hosted-in, contains, part-of, caused-by, occurs-in, commodity association, or locality relations as subClassOf.
- Prefer reusable scientific classes, not individual deposits, mines, samples, ages, regions, or coordinates.
- Include explicit source evidence.
- Do not extract from bibliography/reference-list titles.

Non-hierarchical relation rules:
- Use only the 12 predicates listed above.
- Preserve direction. For symmetric relations, use ASSOCIATED_WITH or SYNCHRONOUS_WITH.
- Do not invent predicates outside the schema. If a fact is important but outside schema, omit it.
- Do not use individual sample IDs, analytical spots, coordinates, or one-off localities as canonical ontology nodes.
- Include concise source evidence.
- Do not extract from bibliography/reference-list titles.

Canonical node rules:
- canonical names should be ontology-style class labels, e.g. QuartzVein, GoldMineralization, HydrothermalAlteration.
- They do NOT need to already exist in the method ontology; this is gold draft.
- Keep canonical nodes scientifically reusable and stable.

Return only valid JSON with this exact schema:
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
  ],
  "alignment_mappings": [
    {{
      "mention": "string",
      "canonical_node": "PascalCaseClass",
      "entity_type": "Mineral|Rock|Alteration|DepositType|Structure|Process|GeologicalObject|Other",
      "evidence_text": "short evidence from source",
      "confidence": 0.0
    }}
  ]
}}

Limits:
- Return at most {max_hierarchy} hierarchy relations.
- Return at most {max_4d} non-hierarchical relations.
- Return at most {max_alignment} additional alignment mappings.
- Prioritize explicit, evidence-supported, reusable ontology facts.
- Obey the limits strictly; shorter complete JSON is better than long incomplete JSON.

Source text:
\"\"\"
{text}
\"\"\"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPT/LLM-assisted full ontology gold pre-annotation for selected documents."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key-env", default=os.getenv("OPENAI_API_KEY_ENV", "DEEPSEEK_API_KEY"))
    parser.add_argument("--max-input-chars", type=int, default=30000)
    parser.add_argument("--max-hierarchy", type=int, default=25)
    parser.add_argument("--max-4d", type=int, default=35)
    parser.add_argument("--max-alignment", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=7000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--api-timeout", type=float, default=240.0)
    parser.add_argument("--api-retries", type=int, default=1)
    parser.add_argument("--api-sleep", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def as_float(value: object) -> str:
    try:
        return str(round(float(value), 4))
    except Exception:
        return ""


def main() -> int:
    load_dotenv_file()
    args = parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")

    from openai import OpenAI

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_full_gold_preannotations"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv(manifest_path)
    if args.limit:
        manifest_rows = manifest_rows[: args.limit]

    done = completed_docs(raw_dir) if args.resume else set()
    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.api_timeout,
        max_retries=args.api_retries,
    )

    for index, row in enumerate(manifest_rows, start=1):
        sample_id = row["sample_id"]
        if sample_id in done:
            continue
        source_file = row["file"]
        title = row["title"]
        data = load_json(Path(source_file))
        prompt = build_prompt(
            sample_id=sample_id,
            title=title,
            text=doc_text(data, args.max_input_chars),
            max_hierarchy=args.max_hierarchy,
            max_4d=args.max_4d,
            max_alignment=args.max_alignment,
        )
        request_kwargs = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if "deepseek" in args.base_url.lower():
            request_kwargs["extra_body"] = {"thinking": {"type": args.thinking}}
        response = client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        payload = {
            "sample_id": sample_id,
            "source_file": source_file,
            "title": title,
            "model": args.model,
            "base_url": args.base_url,
            "raw_response": content,
            "parsed": parse_json_object(content),
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else {},
        }
        (raw_dir / f"{sample_id}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "completed": index,
                    "total": len(manifest_rows),
                    "sample_id": sample_id,
                    "title": title[:120],
                    "total_tokens": payload["usage"].get("total_tokens", 0),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        if args.api_sleep:
            time.sleep(args.api_sleep)

    hierarchy_rows = []
    relation_rows = []
    alignment_rows_by_key: OrderedDict[tuple[str, str], dict] = OrderedDict()
    canonical_rows = []
    parse_empty_docs = []
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    raw_files = sorted(raw_dir.glob("*.json"))
    for raw_file in raw_files:
        item = json.loads(raw_file.read_text(encoding="utf-8"))
        sample_id = item.get("sample_id", raw_file.stem)
        source_file = item.get("source_file", "")
        title = item.get("title", "")
        usage = item.get("usage", {}) or {}
        for key in token_usage:
            token_usage[key] += int(usage.get(key, 0) or 0)
        parsed = item.get("parsed") or {}
        h_items = parsed.get("hierarchy_relations", [])
        r_items = parsed.get("nonhierarchical_relations", [])
        a_items = parsed.get("alignment_mappings", [])
        if not h_items and not r_items and not a_items:
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
            row = {
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
                "model": item.get("model", ""),
            }
            hierarchy_rows.append(row)
            canonical_rows.append(
                {
                    "annotation_id": row["annotation_id"],
                    "sample_id": sample_id,
                    "source_file": source_file,
                    "title": title,
                    "subject_canonical": child_c,
                    "predicate": "subClassOf",
                    "object_canonical": parent_c,
                    "evidence_text": row["evidence_text"],
                    "source_layer": "hierarchy",
                    "gold_decision": "",
                    "gold_note": "",
                }
            )
            for mention, node, entity_type in [
                (child_m, child_c, "Other"),
                (parent_m, parent_c, "Other"),
            ]:
                key = (mention.lower(), node)
                alignment_rows_by_key.setdefault(
                    key,
                    {
                        "sample_id": sample_id,
                        "source_file": source_file,
                        "title": title,
                        "mention": mention,
                        "canonical_node": node,
                        "entity_type": entity_type,
                        "evidence_text": row["evidence_text"],
                        "preannotator_decision": "VALID",
                        "confidence": row["confidence"],
                        "gold_decision": "",
                        "gold_note": "",
                    },
                )

        for rel_index, rel in enumerate(r_items if isinstance(r_items, list) else [], start=1):
            if not isinstance(rel, dict):
                continue
            predicate = norm_space(rel.get("predicate")).upper()
            if predicate not in RELATION_TYPES:
                continue
            subject_m = norm_space(rel.get("subject_mention"))
            object_m = norm_space(rel.get("object_mention"))
            subject_c = norm_space(rel.get("subject_canonical")) or canonical(subject_m)
            object_c = norm_space(rel.get("object_canonical")) or canonical(object_m)
            if not subject_c or not object_c:
                continue
            row = {
                "annotation_id": f"{sample_id}_R_{rel_index:03d}",
                "sample_id": sample_id,
                "source_file": source_file,
                "title": title,
                "subject_mention": subject_m,
                "predicate": predicate,
                "object_mention": object_m,
                "subject_canonical": subject_c,
                "object_canonical": object_c,
                "evidence_text": norm_space(rel.get("evidence_text")),
                "preannotator_decision": "VALID",
                "confidence": as_float(rel.get("confidence")),
                "gold_decision": "",
                "gold_error_type": "",
                "gold_note": "",
                "model": item.get("model", ""),
            }
            relation_rows.append(row)
            canonical_rows.append(
                {
                    "annotation_id": row["annotation_id"],
                    "sample_id": sample_id,
                    "source_file": source_file,
                    "title": title,
                    "subject_canonical": subject_c,
                    "predicate": predicate,
                    "object_canonical": object_c,
                    "evidence_text": row["evidence_text"],
                    "source_layer": "nonhierarchical",
                    "gold_decision": "",
                    "gold_note": "",
                }
            )
            for mention, node in [(subject_m, subject_c), (object_m, object_c)]:
                key = (mention.lower(), node)
                alignment_rows_by_key.setdefault(
                    key,
                    {
                        "sample_id": sample_id,
                        "source_file": source_file,
                        "title": title,
                        "mention": mention,
                        "canonical_node": node,
                        "entity_type": "Other",
                        "evidence_text": row["evidence_text"],
                        "preannotator_decision": "VALID",
                        "confidence": row["confidence"],
                        "gold_decision": "",
                        "gold_note": "",
                    },
                )

        for align_index, align in enumerate(a_items if isinstance(a_items, list) else [], start=1):
            if not isinstance(align, dict):
                continue
            mention = norm_space(align.get("mention"))
            node = norm_space(align.get("canonical_node")) or canonical(mention)
            if not mention or not node:
                continue
            key = (mention.lower(), node)
            alignment_rows_by_key[key] = {
                "sample_id": sample_id,
                "source_file": source_file,
                "title": title,
                "mention": mention,
                "canonical_node": node,
                "entity_type": norm_space(align.get("entity_type")) or "Other",
                "evidence_text": norm_space(align.get("evidence_text")),
                "preannotator_decision": "VALID",
                "confidence": as_float(align.get("confidence")),
                "gold_decision": "",
                "gold_note": "",
            }

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
        "scope": "GPT/LLM-assisted full ontology gold draft; human verification required before final gold use.",
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "model": args.model,
        "base_url": args.base_url,
        "documents_requested": len(manifest_rows),
        "documents_completed": len(raw_files),
        "hierarchy_rows": len(hierarchy_rows),
        "nonhierarchical_rows": len(relation_rows),
        "alignment_rows": len(alignment_rows_by_key),
        "canonical_triple_rows": len(canonical_rows),
        "parse_empty_docs": parse_empty_docs,
        "allowed_nonhierarchical_relations": RELATION_TYPES,
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
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

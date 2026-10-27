import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


RELATION_TYPES = {
    "CAUSES",
    "CONTROLS",
    "INDICATES",
    "ASSOCIATED_WITH",
    "HOSTED_IN",
    "INTRUDES_INTO",
    "ZONES_WITH",
    "DISTRIBUTED_ALONG",
    "PRECEDES",
    "OVERPRINTS",
    "SYNCHRONOUS_WITH",
    "FORMED_DURING",
}

RELATION_DEFINITIONS = """
CORE CAUSAL & LOGICAL
1. CAUSES: one geological process or factor leads to another process or result.
2. CONTROLS: a structure, lithology, fluid, or factor constrains mineralization or ore-body distribution.
3. INDICATES: an observation, mineral, alteration, anomaly, or feature indicates a target, zone, or process.
4. ASSOCIATED_WITH: two concepts commonly co-occur, are paragenetically associated, or are geologically linked.

SPATIAL TOPOLOGY
5. HOSTED_IN: an ore body, mineralization, mineral, vein, or alteration is physically contained in a host.
6. INTRUDES_INTO: an intrusion, dike, vein, or magmatic body intrudes or cuts another geological unit.
7. ZONES_WITH: alteration, mineralization, or lithology shows spatial zoning with another zone or unit.
8. DISTRIBUTED_ALONG: veins, ore bodies, mineralization, or anomalies are distributed along a structure or trend.

TEMPORAL EVOLUTION
9. PRECEDES: one event, stage, assemblage, or mineralization occurs earlier than another.
10. OVERPRINTS: a later event, alteration, vein, or mineralization superimposes on or modifies an earlier one.
11. SYNCHRONOUS_WITH: two events occur during the same stage or time interval.
12. FORMED_DURING: an entity, mineralization, assemblage, or deposit formed during a specific age, stage, or event.
""".strip()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 12-schema 4D non-hierarchical metallogenic relations.")
    parser.add_argument("--input-dir", default="corpus/A_full_run_all")
    parser.add_argument("--file-glob", default="DOC_A_*.json")
    parser.add_argument("--out-jsonl", default="taxonomy_runs/relations_4d/raw_4d_relations_v4flash.jsonl")
    parser.add_argument("--out-csv", default="taxonomy_runs/relations_4d/raw_4d_relations_v4flash.csv")
    parser.add_argument("--summary", default="taxonomy_runs/relations_4d/raw_4d_relations_v4flash_summary.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--max-relations", type=int, default=15)
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key-env", default=os.getenv("OPENAI_API_KEY_ENV", "DEEPSEEK_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    return parser.parse_args()


def content_to_text(path: Path, max_chars: int) -> tuple[str, str]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    metadata = data.get("metadata") or {}
    source = data.get("source") or {}
    title = metadata.get("title") or source.get("display_name") or path.stem
    abstract = metadata.get("abstract") or ""
    paragraphs = []
    content = data.get("content")
    if isinstance(content, dict):
        for para in content.get("paragraphs") or []:
            if isinstance(para, dict):
                text = para.get("text") or para.get("clean_text") or para.get("raw_text") or ""
                if len(str(text)) > 20:
                    paragraphs.append(str(text))
    elif isinstance(content, list):
        paragraphs.extend(str(item) for item in content if len(str(item)) > 20)

    text = f"Title: {title}\nAbstract: {abstract}\nBody:\n" + "\n".join(paragraphs)
    return title, text[:max_chars]


def build_prompt(title: str, text: str, max_relations: int) -> list[dict]:
    system = f"""
You are an expert economic geologist building a concept-level metallogenic ontology.

Use the RHF extraction pattern:
- R: identify whether the evidence expresses one of the allowed relation types.
- H: identify the head concept / subject.
- F: complete the relation fact with object concept, direction, and evidence sentence.

Allowed relation types, and only these 12:
{RELATION_DEFINITIONS}

Hard constraints:
- Extract non-hierarchical relations only. Do not output is-a / subtype / subclass relations.
- Extract concept-level metallogenic relations only. If the text is not about ore deposits, mineralization,
  alteration, ore fluids, ore controls, or metallogenic timing, return an empty relations list.
- Do not extract facts about a specific named deposit, mine, author, sample, drill hole, locality, country,
  province, formation, group, district, or individual case name. Generalize only when the sentence supports a
  reusable concept-level relation; otherwise omit it.
- Use only relation labels from the 12 allowed labels.
- Keep relation direction scientifically correct.
- Evidence must be an exact English sentence or a short contiguous sentence span from the input text.
- Omit uncertain relations.
- Do not fill the quota. Return fewer relations when only a few strong concept-level relations are present.
- Return at most {max_relations} high-value relations, prioritizing CAUSES, CONTROLS, HOSTED_IN, PRECEDES, OVERPRINTS, and FORMED_DURING.
- Return strict JSON only, with the top-level key "relations".

JSON schema:
{{
  "relations": [
    {{
      "subject": "concept",
      "relation": "CAUSES",
      "object": "concept",
      "evidence": "exact evidence sentence from text"
    }}
  ]
}}
""".strip()
    user = f"Title: {title}\n\nExtract the 12-schema non-hierarchical 4D relations from this text:\n{text}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_json_response(content: str) -> list[dict]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    data = json.loads(content)
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("relations")
        if not isinstance(rows, list):
            rows = next((value for value in data.values() if isinstance(value, list)), [])
    else:
        rows = []
    return rows


def normalize_relation(row: dict, source_file: str, title: str, source_text: str) -> dict | None:
    subject = str(row.get("subject") or row.get("head") or "").strip()
    relation = str(row.get("relation") or row.get("predicate") or "").strip().upper()
    obj = str(row.get("object") or row.get("tail") or row.get("obj") or "").strip()
    evidence = str(row.get("evidence") or "").strip()
    if not subject or not relation or not obj or not evidence:
        return None
    relation = relation.replace("-", "_").replace(" ", "_")
    if relation not in RELATION_TYPES:
        return {
            "source_file": source_file,
            "title": title,
            "subject": subject,
            "relation": relation,
            "object": obj,
            "evidence": evidence,
            "schema_valid": "no",
            "evidence_in_text": "yes" if evidence in source_text else "no",
        }
    return {
        "source_file": source_file,
        "title": title,
        "subject": subject,
        "relation": relation,
        "object": obj,
        "evidence": evidence,
        "schema_valid": "yes",
        "evidence_in_text": "yes" if evidence in source_text else "no",
    }


def usage_dict(usage) -> dict:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        data = usage.model_dump()
    elif isinstance(usage, dict):
        data = usage
    else:
        data = {}
    return {k: int(v) for k, v in data.items() if isinstance(v, int)}


def estimate_deepseek_v4_flash_cost_rmb(usages: list[dict]) -> float:
    hit = sum(item.get("prompt_cache_hit_tokens", 0) for item in usages)
    miss = sum(item.get("prompt_cache_miss_tokens", 0) for item in usages)
    completion = sum(item.get("completion_tokens", 0) for item in usages)
    prompt_total = sum(item.get("prompt_tokens", 0) for item in usages)
    if hit == 0 and miss == 0:
        miss = prompt_total
    return hit / 1_000_000 * 0.02 + miss / 1_000_000 * 1.0 + completion / 1_000_000 * 2.0


def extract_one(path: Path, args: argparse.Namespace, api_key: str) -> dict:
    title, text = content_to_text(path, args.max_chars)
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout, max_retries=args.retries)
    messages = build_prompt(title, text, args.max_relations)
    kwargs = {}
    if args.thinking:
        kwargs["extra_body"] = {"thinking": {"type": args.thinking}}
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            response_format={"type": "json_object"},
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        parsed = parse_json_response(content)
        relations = []
        schema_drift = []
        for row in parsed:
            if not isinstance(row, dict):
                continue
            normalized = normalize_relation(row, path.name, title, text)
            if not normalized:
                continue
            if normalized["schema_valid"] == "yes":
                relations.append(normalized)
            else:
                schema_drift.append(normalized)
        return {
            "source_file": path.name,
            "title": title,
            "status": "OK",
            "relations": relations,
            "schema_drift": schema_drift,
            "relation_count": len(relations),
            "schema_drift_count": len(schema_drift),
            "usage": usage_dict(response.usage),
            "error": "",
        }
    except Exception as exc:
        return {
            "source_file": path.name,
            "title": title,
            "status": "ERROR",
            "relations": [],
            "schema_drift": [],
            "relation_count": 0,
            "schema_drift_count": 0,
            "usage": {},
            "error": str(exc),
        }
    finally:
        if args.sleep:
            time.sleep(args.sleep)


def load_done_sources(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "OK" and row.get("source_file"):
                done.add(row["source_file"])
    return done


def write_csv_from_jsonl(jsonl_path: Path, csv_path: Path) -> list[dict]:
    latest_by_source = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            source_file = item.get("source_file")
            if source_file:
                latest_by_source[source_file] = item
    rows = []
    for item in latest_by_source.values():
        if item.get("status") != "OK":
            continue
        rows.extend(item.get("relations", []))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_file", "title", "subject", "relation", "object", "evidence", "schema_valid", "evidence_in_text"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize(jsonl_path: Path, csv_rows: list[dict], summary_path: Path, args: argparse.Namespace) -> dict:
    latest_by_source = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            source_file = item.get("source_file")
            if source_file:
                latest_by_source[source_file] = item

    doc_status = Counter()
    relation_counts = Counter()
    evidence_counts = Counter()
    schema_drift_labels = Counter()
    usages = []
    doc_relation_counts = []
    for item in latest_by_source.values():
        doc_status[item.get("status", "UNKNOWN")] += 1
        usages.append(item.get("usage", {}))
        doc_relation_counts.append(int(item.get("relation_count", 0)))
        for drift in item.get("schema_drift", []):
            schema_drift_labels[drift.get("relation", "")] += 1
    for row in csv_rows:
        relation_counts[row["relation"]] += 1
        evidence_counts[row["evidence_in_text"]] += 1
    usage_totals = Counter()
    for usage in usages:
        usage_totals.update(usage)
    summary = {
        "model": args.model,
        "input_dir": args.input_dir,
        "max_chars": args.max_chars,
        "max_relations": args.max_relations,
        "documents": dict(doc_status),
        "total_relations": len(csv_rows),
        "relations_per_ok_doc_avg": round(sum(doc_relation_counts) / max(1, doc_status.get("OK", 0)), 3),
        "relation_distribution": dict(relation_counts),
        "evidence_in_text": dict(evidence_counts),
        "schema_drift_labels": dict(schema_drift_labels),
        "usage_totals": dict(usage_totals),
        "estimated_cost_rmb_deepseek_v4_flash": round(estimate_deepseek_v4_flash_cost_rmb(usages), 4),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    load_dotenv_file()
    args = parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key in environment variable {args.api_key_env}.")

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.file_glob))
    if args.limit:
        files = files[: args.limit]
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_sources(out_jsonl) if args.resume else set()
    pending = [path for path in files if path.name not in done]
    mode = "a" if args.resume else "w"

    print(f"[INFO] Files selected: {len(files)}")
    print(f"[INFO] Already done: {len(done)}")
    print(f"[INFO] Pending: {len(pending)}")
    print(f"[INFO] Model: {args.model}")

    with out_jsonl.open(mode, encoding="utf-8") as handle:
        if args.workers <= 1:
            iterator = (extract_one(path, args, api_key) for path in pending)
            for item in tqdm(iterator, total=len(pending), desc="4D extract"):
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(extract_one, path, args, api_key): path for path in pending}
                for future in tqdm(as_completed(futures), total=len(futures), desc="4D extract"):
                    item = future.result()
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    handle.flush()

    rows = write_csv_from_jsonl(out_jsonl, Path(args.out_csv))
    summary = summarize(out_jsonl, rows, Path(args.summary), args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

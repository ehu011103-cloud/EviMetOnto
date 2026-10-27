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


SYMMETRIC_RELATIONS = {"ASSOCIATED_WITH", "SYNCHRONOUS_WITH"}

RELATION_DEFINITIONS = {
    "CAUSES": "subject is a geological cause or driver of object.",
    "CONTROLS": "subject constrains or controls object distribution, occurrence, or localization.",
    "INDICATES": "subject is evidence, an indicator, or a proxy for object.",
    "ASSOCIATED_WITH": "subject and object are geologically linked, co-occurring, or paragenetically associated.",
    "HOSTED_IN": "subject is physically contained in or hosted by object.",
    "INTRUDES_INTO": "subject intrudes, cuts, or cross-cuts object.",
    "ZONES_WITH": "subject spatially zones with object.",
    "DISTRIBUTED_ALONG": "subject is distributed along object.",
    "PRECEDES": "subject occurs before object in geological time or paragenesis.",
    "OVERPRINTS": "subject overprints, replaces, or modifies object.",
    "SYNCHRONOUS_WITH": "subject and object formed or occurred during the same time interval.",
    "FORMED_DURING": "subject formed during object, where object is an age, stage, event, or process.",
}


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
    parser = argparse.ArgumentParser(
        description="Evidence-grounded bidirectional validation for full 4D non-hierarchical relations."
    )
    parser.add_argument("--input-csv", default="taxonomy_runs/relations_4d/full_raw_4d_relations_v4flash.csv")
    parser.add_argument("--out-jsonl", default="taxonomy_runs/relations_4d/full_4d_bidirectional_validation.jsonl")
    parser.add_argument("--out-csv", default="taxonomy_runs/relations_4d/full_4d_bidirectional_validation.csv")
    parser.add_argument("--summary", default="taxonomy_runs/relations_4d/full_4d_bidirectional_validation_summary.json")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    return parser.parse_args()


def relation_key(row: dict) -> str:
    payload = [
        row.get("source_file", ""),
        row.get("subject", ""),
        row.get("relation", ""),
        row.get("object", ""),
        row.get("evidence", ""),
    ]
    return "\u241f".join(str(x) for x in payload)


def read_rows(path: Path, limit: int = 0) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    for idx, row in enumerate(rows, start=1):
        row["_row_id"] = str(idx)
        row["_key"] = relation_key(row)
    return rows


def read_done(path: Path, include_errors: bool = False) -> dict[str, dict]:
    done = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = row.get("_row_id")
            if row_id and (include_errors or row.get("decision") != "ERROR"):
                done[row_id] = row
    return done


def build_prompt(row: dict) -> str:
    rel = (row.get("relation") or "").upper()
    definition = RELATION_DEFINITIONS.get(rel, "subject and object have the stated geological relation.")
    symmetric = "yes" if rel in SYMMETRIC_RELATIONS else "no"
    return f"""You are a strict economic-geology relation validator.
Validate this non-hierarchical relation using BOTH the evidence sentence and domain knowledge.

Relation label: {rel}
Definition: {definition}
Expected symmetric relation: {symmetric}

Forward claim:
Subject: {row.get('subject', '')}
Relation: {rel}
Object: {row.get('object', '')}

Reverse same-label claim:
Subject: {row.get('object', '')}
Relation: {rel}
Object: {row.get('subject', '')}

Evidence sentence/span:
{row.get('evidence', '')}

Decision rules:
- forward_supported=true only if the evidence supports the forward claim with the stated relation type.
- reverse_same_label_supported=true only if the same evidence supports the reverse claim with the same relation label.
- For ASSOCIATED_WITH and SYNCHRONOUS_WITH, reverse support is expected when the evidence genuinely supports a symmetric relation.
- For all other relations, a valid direction-sensitive extraction usually has forward_supported=true and reverse_same_label_supported=false.
- Mark wrong_relation_type if evidence supports a relation, but not the proposed label.
- Mark wrong_direction if the reverse same-label claim is better supported than the forward claim.
- Mark unsupported if the evidence does not support the proposed relation.

Return strict JSON only:
{{
  "forward_supported": true/false,
  "reverse_same_label_supported": true/false,
  "decision": "VALID" | "INVALID" | "UNCERTAIN",
  "direction_status": "correct" | "reversed" | "symmetric" | "bidirectional_non_symmetric" | "unsupported" | "unclear",
  "error_type": "none" | "unsupported" | "wrong_relation_type" | "wrong_direction" | "overstated" | "entity_boundary_error" | "unclear",
  "suggested_relation": "RELATION_OR_EMPTY",
  "reason": "one concise sentence"
}}"""


def parse_json(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_decision(row: dict, parsed: dict) -> dict:
    rel = (row.get("relation") or "").upper()
    fwd = bool(parsed.get("forward_supported"))
    rev = bool(parsed.get("reverse_same_label_supported"))
    decision = str(parsed.get("decision", "")).upper()
    direction_status = str(parsed.get("direction_status", "")).lower()
    error_type = str(parsed.get("error_type", "")).lower()

    if decision not in {"VALID", "INVALID", "UNCERTAIN"}:
        if rel in SYMMETRIC_RELATIONS:
            decision = "VALID" if fwd else "INVALID"
        else:
            decision = "VALID" if fwd and not rev else "INVALID"

    if not direction_status:
        if not fwd and rev:
            direction_status = "reversed"
        elif fwd and rev:
            direction_status = "symmetric" if rel in SYMMETRIC_RELATIONS else "bidirectional_non_symmetric"
        elif fwd:
            direction_status = "correct"
        else:
            direction_status = "unsupported"

    if not error_type:
        error_type = "none" if decision == "VALID" else "unclear"

    return {
        "forward_supported": "true" if fwd else "false",
        "reverse_same_label_supported": "true" if rev else "false",
        "decision": decision,
        "direction_status": direction_status,
        "error_type": error_type,
        "suggested_relation": str(parsed.get("suggested_relation", "") or ""),
        "reason": str(parsed.get("reason", "") or ""),
    }


def validate_one(client: OpenAI, args: argparse.Namespace, row: dict) -> dict:
    messages = [{"role": "user", "content": build_prompt(row)}]
    result = {
        "_row_id": row["_row_id"],
        "_key": row["_key"],
        "source_file": row.get("source_file", ""),
        "title": row.get("title", ""),
        "subject": row.get("subject", ""),
        "relation": row.get("relation", ""),
        "object": row.get("object", ""),
        "evidence": row.get("evidence", ""),
        "evidence_in_text": row.get("evidence_in_text", ""),
        "forward_supported": "",
        "reverse_same_label_supported": "",
        "decision": "ERROR",
        "direction_status": "",
        "error_type": "",
        "suggested_relation": "",
        "reason": "",
        "prompt_tokens": "",
        "completion_tokens": "",
        "total_tokens": "",
        "prompt_cache_hit_tokens": "",
        "prompt_cache_miss_tokens": "",
        "raw_response": "",
    }
    for attempt in range(args.retries + 1):
        try:
            extra_body = {}
            if "deepseek" in args.base_url.lower() and args.thinking == "disabled":
                extra_body["thinking"] = {"type": "disabled"}
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                extra_body=extra_body or None,
            )
            content = response.choices[0].message.content or ""
            parsed = parse_json(content)
            result.update(normalize_decision(row, parsed))
            result["raw_response"] = content
            usage = getattr(response, "usage", None)
            if usage:
                result["prompt_tokens"] = getattr(usage, "prompt_tokens", "") or ""
                result["completion_tokens"] = getattr(usage, "completion_tokens", "") or ""
                result["total_tokens"] = getattr(usage, "total_tokens", "") or ""
                result["prompt_cache_hit_tokens"] = getattr(usage, "prompt_cache_hit_tokens", "") or ""
                result["prompt_cache_miss_tokens"] = getattr(usage, "prompt_cache_miss_tokens", "") or ""
            return result
        except Exception as exc:
            result["reason"] = f"{type(exc).__name__}: {exc}"
            if attempt < args.retries:
                time.sleep(min(2**attempt, 8))
    return result


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields = [
        "_row_id",
        "source_file",
        "title",
        "subject",
        "relation",
        "object",
        "evidence",
        "evidence_in_text",
        "forward_supported",
        "reverse_same_label_supported",
        "decision",
        "direction_status",
        "error_type",
        "suggested_relation",
        "reason",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda x: int(x.get("_row_id") or 0)):
            writer.writerow({key: row.get(key, "") for key in fields})


def summarize(rows: list[dict], args: argparse.Namespace, input_rows: int) -> dict:
    token_fields = [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    ]
    tokens = {}
    for field in token_fields:
        total = 0
        for row in rows:
            try:
                total += int(row.get(field) or 0)
            except ValueError:
                pass
        tokens[field] = total
    return {
        "input_csv": args.input_csv,
        "model": args.model,
        "base_url": args.base_url,
        "rows_in_input_scope": input_rows,
        "rows_completed": len(rows),
        "api_errors": sum(1 for row in rows if row.get("decision") == "ERROR"),
        "decision_counts": dict(Counter(row.get("decision", "") for row in rows)),
        "direction_status_counts": dict(Counter(row.get("direction_status", "") for row in rows)),
        "error_type_counts": dict(Counter(row.get("error_type", "") for row in rows)),
        "relation_decision_counts": {
            rel: dict(Counter(row.get("decision", "") for row in rows if row.get("relation") == rel))
            for rel in sorted(set(row.get("relation", "") for row in rows))
        },
        "token_usage": tokens,
        "note": "ASSOCIATED_WITH and SYNCHRONOUS_WITH are treated as symmetric relation labels; other labels are direction-sensitive.",
    }


def main() -> None:
    load_dotenv_file()
    args = parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    input_rows = read_rows(Path(args.input_csv), args.limit)
    done = read_done(Path(args.out_jsonl)) if args.resume else {}
    rows_to_run = [row for row in input_rows if row["_row_id"] not in done]

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(validate_one, client, args, row): row for row in rows_to_run}
            for future in tqdm(as_completed(futures), total=len(futures), desc="4D bidirectional validation"):
                result = future.result()
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                if args.sleep:
                    time.sleep(args.sleep)

    all_done = read_done(out_jsonl)
    completed_rows = list(all_done.values())
    write_csv(completed_rows, Path(args.out_csv))
    summary = summarize(completed_rows, args, len(input_rows))
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

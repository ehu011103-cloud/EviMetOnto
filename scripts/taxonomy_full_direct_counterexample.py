import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


NOISE_NODES = {
    "ABC",
    "Subtypes",
    "Subtype",
    "Types",
    "Type",
    "Examples",
    "Example",
}

BROAD_PARENTS = {
    "GeologicalObject",
    "Object",
    "Thing",
    "Entity",
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


def split_camel(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
    return value.replace("_", " ").replace("-", " ")


def normalized_key(value: str) -> str:
    text = split_camel(value).lower()
    replacements = {
        "sulphide": "sulfide",
        "sulphur": "sulfur",
        "dyke": "dike",
        "k feldspar": "potassium feldspar",
        "kfeldspar": "potassium feldspar",
        "cu": "copper",
        "au": "gold",
    }
    for src, dst in replacements.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)
    return re.sub(r"[^a-z0-9]+", "", text)


def prompt_for(child: str, parent: str) -> str:
    return (
        "You are a strict mineralogy/geology ontology referee.\n"
        "Validate only this proposed taxonomy is-a edge:\n"
        f"Child: {split_camel(child)}\n"
        f"Parent: {split_camel(parent)}\n\n"
        "Use the necessary is-a rule: every instance of Child must necessarily be an instance of Parent. "
        "Reject the edge if Child merely can be Parent, is often associated with Parent, is hosted by Parent, "
        "contains Parent, forms from Parent, is caused by Parent, or belongs to a different classification axis.\n\n"
        "Return only JSON with keys: decision, error_type, reason. "
        "decision must be VALID, INVALID, or UNCERTAIN. "
        "error_type must be one of none, category_mismatch, not_is_a, too_broad, "
        "synonym_not_subtype, instance_as_class, commodity_vs_genetic_axis, unclear."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select and optionally run full direct counterexample validation.")
    parser.add_argument("--input", default="taxonomy_runs/validation/high_risk_isa_edges.csv")
    parser.add_argument("--out-candidates", default="taxonomy_runs/validation/full_direct_counterexample_candidates.csv")
    parser.add_argument("--out-results", default="taxonomy_runs/validation/full_direct_counterexample_results.csv")
    parser.add_argument("--summary", default="taxonomy_runs/validation/full_direct_counterexample_summary.json")
    parser.add_argument("--min-risk-score", type=int, default=0)
    parser.add_argument("--include-broad-parent", action="store_true")
    parser.add_argument("--api-validate", action="store_true")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--limit", type=int, default=0, help="Maximum rows to validate; 0 means all selected rows.")
    parser.add_argument("--resume", action="store_true", help="Skip already completed child-parent pairs in output.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument("--api-retries", type=int, default=1)
    parser.add_argument("--api-sleep", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent API workers for full validation.")
    return parser.parse_args()


def select_candidates(input_path: Path, min_risk_score: int, include_broad_parent: bool) -> tuple[list[dict], Counter]:
    rows = csv.DictReader(input_path.open(encoding="utf-8-sig"))
    selected = []
    excluded = Counter()
    for row in rows:
        child = row["child"]
        parent = row["parent"]
        reasons = set(filter(None, row["reasons"].split(";")))
        risk_score = int(float(row["risk_score"]))

        if risk_score < min_risk_score:
            excluded["below_min_risk_score"] += 1
            continue
        if child == parent:
            excluded["self_loop"] += 1
            continue
        if child in NOISE_NODES or parent in NOISE_NODES or "noise_node" in reasons:
            excluded["noise_or_meta_node"] += 1
            continue
        if normalized_key(child) == normalized_key(parent):
            excluded["synonym_or_spelling_variant"] += 1
            continue
        if not include_broad_parent and (parent in BROAD_PARENTS or "broad_parent" in reasons):
            excluded["broad_parent_policy"] += 1
            continue

        selected.append(
            {
                "child": child,
                "parent": parent,
                "risk_score": row["risk_score"],
                "risk_reasons": row["reasons"],
                "prompt": prompt_for(child, parent),
            }
        )
    return selected, excluded


def write_candidates(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["child", "parent", "risk_score", "risk_reasons", "prompt"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def completed_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done = set()
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        if row.get("decision") and row["decision"] != "ERROR":
            done.add((row["child"], row["parent"]))
    return done


def api_validate(rows: list[dict], args: argparse.Namespace) -> None:
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")

    from openai import OpenAI

    out_path = Path(args.out_results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_pairs(out_path) if args.resume else set()
    rows_to_run = [row for row in rows if (row["child"], row["parent"]) not in done]
    if args.limit:
        rows_to_run = rows_to_run[: args.limit]

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.api_timeout,
        max_retries=args.api_retries,
    )
    fields = [
        "child",
        "parent",
        "risk_score",
        "risk_reasons",
        "decision",
        "error_type",
        "reason",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "raw_response",
    ]
    mode = "a" if args.resume and out_path.exists() else "w"
    def validate_one(row: dict) -> dict:
        result = {
            "child": row["child"],
            "parent": row["parent"],
            "risk_score": row["risk_score"],
            "risk_reasons": row["risk_reasons"],
            "decision": "ERROR",
            "error_type": "",
            "reason": "",
            "prompt_tokens": "",
            "completion_tokens": "",
            "total_tokens": "",
            "prompt_cache_hit_tokens": "",
            "prompt_cache_miss_tokens": "",
            "raw_response": "",
        }
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": row["prompt"]}],
                temperature=0.0,
                max_tokens=args.max_tokens,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": args.thinking}},
            )
            content = response.choices[0].message.content or ""
            result["raw_response"] = content
            usage = getattr(response, "usage", None)
            if usage:
                result["prompt_tokens"] = getattr(usage, "prompt_tokens", "")
                result["completion_tokens"] = getattr(usage, "completion_tokens", "")
                result["total_tokens"] = getattr(usage, "total_tokens", "")
                result["prompt_cache_hit_tokens"] = getattr(usage, "prompt_cache_hit_tokens", "")
                result["prompt_cache_miss_tokens"] = getattr(usage, "prompt_cache_miss_tokens", "")
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", content, flags=re.S)
                parsed = json.loads(match.group(0)) if match else {}
            result["decision"] = str(parsed.get("decision", "")).upper() or "UNCERTAIN"
            result["error_type"] = str(parsed.get("error_type", ""))
            result["reason"] = str(parsed.get("reason", ""))
        except Exception as exc:
            result["reason"] = str(exc)
        if args.api_sleep:
            time.sleep(args.api_sleep)
        return result

    with out_path.open(mode, encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        if args.workers <= 1:
            for row in rows_to_run:
                writer.writerow(validate_one(row))
                handle.flush()
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(validate_one, row) for row in rows_to_run]
                for future in as_completed(futures):
                    writer.writerow(future.result())
                    handle.flush()


def main() -> int:
    load_dotenv_file()
    args = parse_args()
    candidates, excluded = select_candidates(Path(args.input), args.min_risk_score, args.include_broad_parent)
    write_candidates(candidates, Path(args.out_candidates))
    summary = {
        "input": args.input,
        "candidate_rows": len(candidates),
        "excluded_counts": dict(excluded),
        "min_risk_score": args.min_risk_score,
        "include_broad_parent": args.include_broad_parent,
        "outputs": {
            "candidates": args.out_candidates,
            "results": args.out_results if args.api_validate else None,
        },
        "api_validate": args.api_validate,
        "api_limit": args.limit,
        "resume": args.resume,
        "workers": args.workers,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.api_validate:
        api_validate(candidates, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

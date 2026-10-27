import argparse
import json
import os
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from preannotate_full_ontology_gold import (
    canonical,
    doc_text,
    load_dotenv_file,
    load_json,
    parse_json_object,
    read_csv,
    write_csv,
)


DEFAULT_ROOT = Path("taxonomy_runs/gold_test_v2_high_signal_2026_06_25")
DEFAULT_MANIFEST = DEFAULT_ROOT / "high_signal_final50_manifest.csv"
DEFAULT_OUT = DEFAULT_ROOT / "baseline_chisagent_deepseek_final50_2026_06_29"

DOMAINS = [
    "Mineral and Material",
    "Rock and Lithology",
    "Ore Deposit and Mineralization",
    "Alteration and Geochemistry",
    "Geological Structure",
    "Geological Process and Event",
    "Geological Time and Stratigraphy",
    "Exploration and Analytical Observation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CHisAgent-style DeepSeek hierarchy baseline on final50."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--raw-dir", default="")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-input-chars", type=int, default=10000)
    parser.add_argument("--max-terms", type=int, default=45)
    parser.add_argument("--max-local-edges", type=int, default=24)
    parser.add_argument("--max-induction-tokens", type=int, default=8000)
    parser.add_argument("--max-domain-edges", type=int, default=140)
    parser.add_argument("--max-expansion-edges", type=int, default=24)
    parser.add_argument("--max-enrichment-edges", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.set_defaults(resume=True)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def simple_norm(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.lower().replace("_", " ").replace("-", " ")
    text = text.replace("sulphide", "sulfide").replace("mineralisation", "mineralization")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def verified_in_text(term: str, text: str) -> bool:
    term_norm = simple_norm(term)
    return bool(term_norm and term_norm in simple_norm(text))


def clean_node(value: object) -> str:
    node = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n.,;:()[]{}\"")
    if not node or len(node) > 120 or len(node.split()) > 12:
        return ""
    return node


def normalize_domain(value: object) -> str:
    value_norm = simple_norm(value)
    for domain in DOMAINS:
        if value_norm == simple_norm(domain):
            return domain
    aliases = {
        "mineral": DOMAINS[0],
        "material": DOMAINS[0],
        "rock": DOMAINS[1],
        "lithology": DOMAINS[1],
        "deposit": DOMAINS[2],
        "mineralization": DOMAINS[2],
        "alteration": DOMAINS[3],
        "geochemistry": DOMAINS[3],
        "structure": DOMAINS[4],
        "process": DOMAINS[5],
        "event": DOMAINS[5],
        "time": DOMAINS[6],
        "stratigraphy": DOMAINS[6],
        "exploration": DOMAINS[7],
        "observation": DOMAINS[7],
    }
    for key, domain in aliases.items():
        if key in value_norm:
            return domain
    return ""


def api_json(
    client: OpenAI,
    args: argparse.Namespace,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> tuple[str, dict, dict]:
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=args.temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": args.thinking}},
    )
    content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else {}
    return content, parse_json_object(content), usage_dict


def save_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_payload(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def creates_cycle(edges: set[tuple[str, str]], child: str, parent: str) -> bool:
    child_norm = simple_norm(child)
    parent_norm = simple_norm(parent)
    if not child_norm or not parent_norm or child_norm == parent_norm:
        return True
    adjacency = defaultdict(set)
    for edge_child, edge_parent in edges:
        adjacency[simple_norm(edge_child)].add(simple_norm(edge_parent))
    stack = [parent_norm]
    seen = set()
    while stack:
        node = stack.pop()
        if node == child_norm:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency.get(node, ()))
    return False


def induction_prompt(row: dict, text: str, args: argparse.Namespace) -> tuple[str, str]:
    domain_list = "\n".join(f"- {domain}" for domain in DOMAINS)
    system = (
        "You are the Inducer agent in a CHisAgent-style taxonomy construction pipeline. "
        "Induce an explicit is-a taxonomy bottom-up from source-grounded economic-geology terms."
    )
    user = f"""Document ID: {row['sample_id']}
Title: {row['title']}

Allowed domains:
{domain_list}

Extract at most {args.max_terms} reusable domain terms occurring verbatim in the source and assign each to exactly one allowed domain.
Then propose at most {args.max_local_edges} explicit is-a edges. A leaf child must be a source term. A parent may be another source term or a concise intermediate class label needed to organize the terms.
Do not encode part-of, location, sequence, cause, association, or instance membership as is-a. Do not use document-specific sample IDs as classes.

Return JSON only:
{{"domains":[{{"domain":"one allowed domain","terms":[{{"term":"exact source span","evidence":"short source span"}}],"edges":[{{"child":"class","parent":"broader class","evidence":"source support when available"}}]}}]}}

<source>
{text}
</source>"""
    return system, user


def parse_induction(parsed: dict, text: str, args: argparse.Namespace) -> list[dict]:
    output = []
    total_terms = 0
    total_edges = 0
    seen_terms = set()
    seen_edges = set()
    raw_domains = parsed.get("domains", []) if isinstance(parsed, dict) else []
    if isinstance(raw_domains, dict):
        raw_domains = [
            {"domain": key, **(value if isinstance(value, dict) else {"terms": value})}
            for key, value in raw_domains.items()
        ]
    for raw_domain in raw_domains:
        if not isinstance(raw_domain, dict):
            continue
        domain = normalize_domain(raw_domain.get("domain"))
        if not domain:
            continue
        terms = []
        domain_term_norms = set()
        for raw_term in raw_domain.get("terms", []) or []:
            item = raw_term if isinstance(raw_term, dict) else {"term": raw_term}
            term = clean_node(item.get("term"))
            term_norm = simple_norm(term)
            if (
                total_terms >= args.max_terms
                or not term_norm
                or term_norm in seen_terms
                or not verified_in_text(term, text)
            ):
                continue
            evidence = clean_node(item.get("evidence"))
            if evidence and not verified_in_text(term, evidence):
                evidence = term
            terms.append({"term": term, "term_norm": term_norm, "evidence": evidence or term})
            domain_term_norms.add(term_norm)
            seen_terms.add(term_norm)
            total_terms += 1
        edges = []
        for raw_edge in raw_domain.get("edges", []) or []:
            if not isinstance(raw_edge, dict) or total_edges >= args.max_local_edges:
                continue
            child = clean_node(raw_edge.get("child"))
            parent = clean_node(raw_edge.get("parent"))
            key = (simple_norm(child), simple_norm(parent))
            if (
                not key[0]
                or not key[1]
                or key[0] == key[1]
                or key in seen_edges
                or (key[0] not in domain_term_norms and not verified_in_text(child, text))
            ):
                continue
            evidence = clean_node(raw_edge.get("evidence"))
            edges.append(
                {"child": child, "parent": parent, "evidence": evidence or child, "domain": domain}
            )
            seen_edges.add(key)
            total_edges += 1
        if terms or edges:
            output.append({"domain": domain, "terms": terms, "edges": edges})
    return output


def induce_document(
    client: OpenAI,
    args: argparse.Namespace,
    row: dict,
    out_path: Path,
) -> dict:
    data = load_json(Path(row["file"]))
    text = doc_text(data, args.max_input_chars)
    system, user = induction_prompt(row, text, args)
    content, parsed, usage = api_json(client, args, system, user, args.max_induction_tokens)
    domains = parse_induction(parsed, text, args)
    payload = {
        "stage": "inducer",
        "sample_id": row["sample_id"],
        "source_file": row["file"],
        "title": row["title"],
        "model": args.model,
        "raw_response": content,
        "domains": domains,
        "usage": usage,
    }
    save_payload(out_path, payload)
    return payload


def run_inducer(
    client: OpenAI,
    args: argparse.Namespace,
    manifest: list[dict],
    raw_dir: Path,
) -> None:
    pending = []
    for row in manifest:
        path = raw_dir / "inducer" / f"{row['sample_id']}.json"
        if args.resume and path.exists():
            payload = load_payload(path)
            if payload.get("domains") or not payload.get("raw_response"):
                continue
            archive_path = path.with_name(f"{path.stem}.truncated_attempt.json")
            if archive_path.exists():
                continue
            path.replace(archive_path)
        pending.append((row, path))
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(induce_document, client, args, row, path): row
            for row, path in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            payload = future.result()
            term_count = sum(len(item["terms"]) for item in payload["domains"])
            edge_count = sum(len(item["edges"]) for item in payload["domains"])
            print(
                json.dumps(
                    {
                        "stage": "inducer",
                        "completed": index,
                        "pending_total": len(pending),
                        "sample_id": payload["sample_id"],
                        "term_count": term_count,
                        "edge_count": edge_count,
                    }
                ),
                flush=True,
            )


def collect_induction(manifest: list[dict], raw_dir: Path) -> tuple[dict, dict]:
    pools = {domain: {"terms": [], "edges": []} for domain in DOMAINS}
    by_doc = {}
    for row in manifest:
        payload = load_payload(raw_dir / "inducer" / f"{row['sample_id']}.json")
        by_doc[row["sample_id"]] = payload.get("domains", []) or []
        for block in by_doc[row["sample_id"]]:
            domain = normalize_domain(block.get("domain"))
            if not domain:
                continue
            pools[domain]["terms"].extend(
                {**term, "sample_id": row["sample_id"]} for term in block.get("terms", []) or []
            )
            pools[domain]["edges"].extend(
                {**edge, "sample_id": row["sample_id"]} for edge in block.get("edges", []) or []
            )
    return pools, by_doc


def domain_consolidation_prompt(
    domain: str,
    terms: list[str],
    edges: list[dict],
    args: argparse.Namespace,
) -> tuple[str, str]:
    system = (
        "You are the ontology Merger used after the CHisAgent Inducer agents. "
        "Consolidate noisy bottom-up proposals into one acyclic explicit is-a hierarchy."
    )
    user = f"""Domain: {domain}

Observed terms:
{json.dumps(terms, ensure_ascii=False)}

Inducer edge proposals:
{json.dumps(edges, ensure_ascii=False)}

Return at most {args.max_domain_edges} high-quality direct is-a edges. Merge synonyms by choosing one label, remove duplicates, reject non-is-a relations, and add a small number of concise intermediate class labels only when needed to organize multiple observed terms.

Return JSON only:
{{"edges":[{{"child":"narrower class","parent":"broader class","definition":"brief taxonomic rationale"}}]}}"""
    return system, user


def parse_edges(parsed: dict, limit: int, domain: str, stage: str) -> list[dict]:
    rows = []
    seen = set()
    raw_edges = parsed.get("edges", []) if isinstance(parsed, dict) else []
    for raw in raw_edges or []:
        if not isinstance(raw, dict) or len(rows) >= limit:
            continue
        child = clean_node(raw.get("child"))
        parent = clean_node(raw.get("parent"))
        key = (simple_norm(child), simple_norm(parent))
        if not key[0] or not key[1] or key[0] == key[1] or key in seen:
            continue
        rows.append(
            {
                "child": child,
                "parent": parent,
                "evidence": clean_node(raw.get("definition") or raw.get("evidence")),
                "domain": domain,
                "stage": stage,
            }
        )
        seen.add(key)
    return rows


def consolidate_domain(
    client: OpenAI,
    args: argparse.Namespace,
    domain: str,
    pool: dict,
    out_path: Path,
) -> dict:
    term_counts = Counter(simple_norm(item.get("term")) for item in pool["terms"])
    term_labels = {}
    for item in pool["terms"]:
        norm = simple_norm(item.get("term"))
        if norm:
            term_labels.setdefault(norm, item["term"])
    terms = [
        term_labels[norm]
        for norm, _ in term_counts.most_common(180)
        if norm in term_labels
    ]
    edge_counts = Counter(
        (simple_norm(item.get("child")), simple_norm(item.get("parent")))
        for item in pool["edges"]
    )
    edge_labels = {}
    for item in pool["edges"]:
        key = (simple_norm(item.get("child")), simple_norm(item.get("parent")))
        if key[0] and key[1]:
            edge_labels.setdefault(key, {"child": item["child"], "parent": item["parent"]})
    edges = [edge_labels[key] for key, _ in edge_counts.most_common(180) if key in edge_labels]
    if not terms and not edges:
        payload = {
            "stage": "inducer_merge",
            "domain": domain,
            "model": args.model,
            "raw_response": "",
            "edges": [],
            "usage": {},
        }
        save_payload(out_path, payload)
        return payload
    system, user = domain_consolidation_prompt(domain, terms, edges, args)
    content, parsed, usage = api_json(client, args, system, user, 6000)
    consolidated = parse_edges(parsed, args.max_domain_edges, domain, "inducer_merge")
    payload = {
        "stage": "inducer_merge",
        "domain": domain,
        "model": args.model,
        "raw_response": content,
        "edges": consolidated,
        "usage": usage,
    }
    save_payload(out_path, payload)
    return payload


def run_domain_consolidation(
    client: OpenAI,
    args: argparse.Namespace,
    manifest: list[dict],
    raw_dir: Path,
) -> None:
    pools, _ = collect_induction(manifest, raw_dir)
    pending = []
    for index, domain in enumerate(DOMAINS, start=1):
        path = raw_dir / "inducer_merge" / f"domain_{index:02d}.json"
        if args.resume and path.exists():
            continue
        pending.append((domain, pools[domain], path))
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
        futures = {
            executor.submit(consolidate_domain, client, args, domain, pool, path): domain
            for domain, pool, path in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            payload = future.result()
            print(
                json.dumps(
                    {
                        "stage": "inducer_merge",
                        "completed": index,
                        "pending_total": len(pending),
                        "domain": payload["domain"],
                        "edge_count": len(payload["edges"]),
                    }
                ),
                flush=True,
            )


def load_global_edges(raw_dir: Path, include_expander: bool = True) -> list[dict]:
    rows = []
    for index, domain in enumerate(DOMAINS, start=1):
        merge = load_payload(raw_dir / "inducer_merge" / f"domain_{index:02d}.json")
        rows.extend(merge.get("edges", []) or [])
        if include_expander:
            expansion = load_payload(raw_dir / "expander" / f"domain_{index:02d}.json")
            rows.extend(expansion.get("edges", []) or [])
    return rows


def expansion_prompt(domain: str, edges: list[dict], args: argparse.Namespace) -> tuple[str, str]:
    system = (
        "You are the Expander agent in CHisAgent. Use geological domain knowledge to repair "
        "structural gaps in an induced taxonomy without adding facts unrelated to its observed concepts."
    )
    compact = [{"child": item["child"], "parent": item["parent"]} for item in edges]
    user = f"""Domain: {domain}
Current induced taxonomy edges:
{json.dumps(compact, ensure_ascii=False)}

Propose at most {args.max_expansion_edges} additional direct is-a edges that add useful missing intermediate classes or connect obvious orphan classes. Preserve the current node meanings, avoid synonyms and cycles, and output no part-of, causal, temporal, spatial, or associative relations. Return an empty list if no repair is needed.

Return JSON only:
{{"edges":[{{"child":"narrower class","parent":"broader class","definition":"brief domain-knowledge rationale"}}]}}"""
    return system, user


def expand_domain(
    client: OpenAI,
    args: argparse.Namespace,
    domain: str,
    edges: list[dict],
    out_path: Path,
) -> dict:
    if not edges:
        payload = {
            "stage": "expander",
            "domain": domain,
            "model": args.model,
            "raw_response": "",
            "edges": [],
            "usage": {},
        }
        save_payload(out_path, payload)
        return payload
    system, user = expansion_prompt(domain, edges, args)
    content, parsed, usage = api_json(client, args, system, user, 3600)
    proposed = parse_edges(parsed, args.max_expansion_edges, domain, "expander")
    existing = {(item["child"], item["parent"]) for item in edges}
    accepted = []
    for item in proposed:
        pair = (item["child"], item["parent"])
        if pair in existing or creates_cycle(existing, *pair):
            continue
        existing.add(pair)
        accepted.append(item)
    payload = {
        "stage": "expander",
        "domain": domain,
        "model": args.model,
        "raw_response": content,
        "edges": accepted,
        "usage": usage,
    }
    save_payload(out_path, payload)
    return payload


def run_expander(client: OpenAI, args: argparse.Namespace, raw_dir: Path) -> None:
    pending = []
    for index, domain in enumerate(DOMAINS, start=1):
        path = raw_dir / "expander" / f"domain_{index:02d}.json"
        if args.resume and path.exists():
            continue
        merge = load_payload(raw_dir / "inducer_merge" / f"domain_{index:02d}.json")
        pending.append((domain, merge.get("edges", []) or [], path))
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
        futures = {
            executor.submit(expand_domain, client, args, domain, edges, path): domain
            for domain, edges, path in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            payload = future.result()
            print(
                json.dumps(
                    {
                        "stage": "expander",
                        "completed": index,
                        "pending_total": len(pending),
                        "domain": payload["domain"],
                        "edge_count": len(payload["edges"]),
                    }
                ),
                flush=True,
            )


def build_graph_context(raw_dir: Path) -> tuple[set[tuple[str, str]], dict[str, str]]:
    edges = set()
    labels = {}
    for item in load_global_edges(raw_dir):
        child = clean_node(item.get("child"))
        parent = clean_node(item.get("parent"))
        pair = (child, parent)
        if not child or not parent or creates_cycle(edges, *pair):
            continue
        edges.add(pair)
        labels.setdefault(simple_norm(child), child)
        labels.setdefault(simple_norm(parent), parent)
    return edges, labels


def enrichment_prompt(
    row: dict,
    missing: list[dict],
    parent_candidates: list[str],
    args: argparse.Namespace,
) -> tuple[str, str]:
    system = (
        "You are the Enricher agent in CHisAgent. Place source-observed terms omitted by the "
        "current hierarchy into the most specific valid existing parent class."
    )
    user = f"""Document ID: {row['sample_id']}
Title: {row['title']}

Missing source terms with assigned domains:
{json.dumps(missing, ensure_ascii=False)}

Existing parent candidates:
{json.dumps(parent_candidates, ensure_ascii=False)}

Return at most {args.max_enrichment_edges} direct explicit is-a placements. The child must be one of the missing terms and the parent must be one of the existing parent candidates. Reject terms that are instances, measurements, or cannot be placed reliably.

Return JSON only:
{{"edges":[{{"child":"missing source term","parent":"existing broader class","evidence":"short rationale"}}]}}"""
    return system, user


def enrich_document(
    client: OpenAI,
    args: argparse.Namespace,
    row: dict,
    domains: list[dict],
    graph_labels: dict[str, str],
    out_path: Path,
) -> dict:
    missing = []
    for block in domains:
        domain = normalize_domain(block.get("domain"))
        for item in block.get("terms", []) or []:
            if item.get("term_norm") not in graph_labels:
                missing.append({"term": item.get("term"), "domain": domain, "evidence": item.get("evidence")})
    missing = missing[: args.max_enrichment_edges]
    if not missing or not graph_labels:
        payload = {
            "stage": "enricher",
            "sample_id": row["sample_id"],
            "source_file": row["file"],
            "title": row["title"],
            "model": args.model,
            "raw_response": "",
            "edges": [],
            "usage": {},
        }
        save_payload(out_path, payload)
        return payload
    parent_candidates = list(graph_labels.values())[:220]
    system, user = enrichment_prompt(row, missing, parent_candidates, args)
    content, parsed, usage = api_json(client, args, system, user, 3600)
    missing_labels = {simple_norm(item["term"]): item for item in missing}
    proposed = parse_edges(parsed, args.max_enrichment_edges, "", "enricher")
    accepted = []
    for item in proposed:
        child_norm = simple_norm(item["child"])
        parent_norm = simple_norm(item["parent"])
        if child_norm not in missing_labels or parent_norm not in graph_labels:
            continue
        source = missing_labels[child_norm]
        accepted.append(
            {
                **item,
                "child": source["term"],
                "parent": graph_labels[parent_norm],
                "domain": source["domain"],
                "evidence": source.get("evidence") or item.get("evidence", ""),
            }
        )
    payload = {
        "stage": "enricher",
        "sample_id": row["sample_id"],
        "source_file": row["file"],
        "title": row["title"],
        "model": args.model,
        "raw_response": content,
        "edges": accepted,
        "usage": usage,
    }
    save_payload(out_path, payload)
    return payload


def run_enricher(
    client: OpenAI,
    args: argparse.Namespace,
    manifest: list[dict],
    raw_dir: Path,
) -> None:
    _, by_doc = collect_induction(manifest, raw_dir)
    _, graph_labels = build_graph_context(raw_dir)
    pending = []
    for row in manifest:
        path = raw_dir / "enricher" / f"{row['sample_id']}.json"
        if args.resume and path.exists():
            continue
        pending.append((row, by_doc.get(row["sample_id"], []), path))
    if not pending:
        return
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(enrich_document, client, args, row, domains, graph_labels, path): row
            for row, domains, path in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            payload = future.result()
            print(
                json.dumps(
                    {
                        "stage": "enricher",
                        "completed": index,
                        "pending_total": len(pending),
                        "sample_id": payload["sample_id"],
                        "edge_count": len(payload["edges"]),
                    }
                ),
                flush=True,
            )


def aggregate_outputs(
    args: argparse.Namespace,
    manifest: list[dict],
    out_dir: Path,
    raw_dir: Path,
) -> dict:
    pools, by_doc = collect_induction(manifest, raw_dir)
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    term_docs: dict[str, list[dict]] = defaultdict(list)
    for sample_id, blocks in by_doc.items():
        for block in blocks:
            for term in block.get("terms", []) or []:
                term_docs[term["term_norm"]].append(
                    {
                        "sample_id": sample_id,
                        "term": term["term"],
                        "evidence": term.get("evidence", term["term"]),
                    }
                )

    attributed = []
    for sample_id, blocks in by_doc.items():
        for block in blocks:
            for edge in block.get("edges", []) or []:
                attributed.append({**edge, "sample_id": sample_id, "stage": "inducer"})
    for item in load_global_edges(raw_dir):
        for occurrence in term_docs.get(simple_norm(item.get("child")), []):
            attributed.append(
                {
                    **item,
                    "child": occurrence["term"],
                    "sample_id": occurrence["sample_id"],
                    "evidence": occurrence["evidence"],
                }
            )
    for row in manifest:
        payload = load_payload(raw_dir / "enricher" / f"{row['sample_id']}.json")
        for item in payload.get("edges", []) or []:
            attributed.append({**item, "sample_id": row["sample_id"], "stage": "enricher"})

    deduped: OrderedDict[tuple[str, str, str], dict] = OrderedDict()
    for item in attributed:
        key = (
            item.get("sample_id", ""),
            canonical(item.get("child")).lower(),
            canonical(item.get("parent")).lower(),
        )
        if not all(key) or key in deduped:
            continue
        deduped[key] = item

    hierarchy_rows = []
    canonical_rows = []
    for index, item in enumerate(deduped.values(), start=1):
        source = manifest_by_id[item["sample_id"]]
        row = {
            "annotation_id": f"CHIS_{item['sample_id']}_H_{index:05d}",
            "sample_id": item["sample_id"],
            "source_file": source["file"],
            "title": source["title"],
            "child_mention": item["child"],
            "parent_mention": item["parent"],
            "child_canonical": canonical(item["child"]),
            "parent_canonical": canonical(item["parent"]),
            "predicate": "subClassOf",
            "evidence_text": item.get("evidence", ""),
            "preannotator_decision": "VALID",
            "confidence": "",
            "gold_decision": "",
            "gold_error_type": "",
            "gold_note": "",
            "model": args.model,
            "baseline_method": "CHisAgent-DeepSeek-adapted",
            "stage": item.get("stage", ""),
            "domain": item.get("domain", ""),
        }
        hierarchy_rows.append(row)
        canonical_rows.append(
            {
                "annotation_id": row["annotation_id"],
                "sample_id": row["sample_id"],
                "source_file": row["source_file"],
                "title": row["title"],
                "subject_canonical": row["child_canonical"],
                "predicate": "subClassOf",
                "object_canonical": row["parent_canonical"],
                "evidence_text": row["evidence_text"],
                "source_layer": "hierarchy",
            }
        )

    hierarchy_fields = [
        "annotation_id", "sample_id", "source_file", "title", "child_mention",
        "parent_mention", "child_canonical", "parent_canonical", "predicate",
        "evidence_text", "preannotator_decision", "confidence", "gold_decision",
        "gold_error_type", "gold_note", "model", "baseline_method", "stage", "domain",
    ]
    relation_fields = [
        "annotation_id", "sample_id", "source_file", "title", "subject_mention",
        "predicate", "object_mention", "subject_canonical", "object_canonical",
        "evidence_text", "preannotator_decision", "confidence", "gold_decision",
        "gold_error_type", "gold_note", "model", "baseline_method",
    ]
    alignment_fields = [
        "sample_id", "source_file", "title", "mention", "canonical_node",
        "entity_type", "evidence_text", "preannotator_decision", "confidence",
        "gold_decision", "gold_note",
    ]
    canonical_fields = [
        "annotation_id", "sample_id", "source_file", "title", "subject_canonical",
        "predicate", "object_canonical", "evidence_text", "source_layer",
    ]
    write_csv(out_dir / "full_gold_draft_hierarchy_review.csv", hierarchy_rows, hierarchy_fields)
    write_csv(out_dir / "full_gold_draft_4d_relations_review.csv", [], relation_fields)
    write_csv(out_dir / "full_gold_draft_alignment_review.csv", [], alignment_fields)
    write_csv(out_dir / "full_gold_draft_canonical_triples_review.csv", canonical_rows, canonical_fields)

    token_usage = Counter()
    call_count = 0
    for path in raw_dir.rglob("*.json"):
        payload = load_payload(path)
        usage = payload.get("usage") or {}
        if not usage:
            continue
        call_count += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            token_usage[key] += int(usage.get(key, 0) or 0)

    stage_counts = Counter(item.get("stage", "unknown") for item in deduped.values())
    summary = {
        "scope": "CHisAgent-style DeepSeek adapted hierarchy baseline on final50; gold was not used during generation.",
        "method_label": "CHisAgent-DeepSeek-adapted",
        "original_method_source": "CHisAgent: Collaborative Hierarchical Multi-Agent Framework for Taxonomy Construction, arXiv:2601.05520v1.",
        "adaptations": [
            "All Inducer, Merger, Expander, and Enricher roles use DeepSeek V4 Flash instead of the original mixed-model agents.",
            "Eight economic-geology domains replace the source paper's historical-event domains.",
            "Document-level source terms replace the source paper's event-type inventory and retrieval store.",
            "The final global taxonomy is projected back to documents only when the child has a verified source occurrence, matching the document-grounded gold protocol.",
            "This baseline produces hierarchy only; no non-hierarchical or alignment metric is claimed.",
        ],
        "documents": len(manifest),
        "model": args.model,
        "domains": DOMAINS,
        "counts": {
            "hierarchy_rows": len(hierarchy_rows),
            "canonical_rows": len(canonical_rows),
            "stage_rows": dict(stage_counts),
            "api_calls_recorded": call_count,
            "inducer_terms": sum(len(pool["terms"]) for pool in pools.values()),
            "inducer_local_edges": sum(len(pool["edges"]) for pool in pools.values()),
        },
        "token_usage": dict(token_usage),
        "outputs": {
            "hierarchy": str(out_dir / "full_gold_draft_hierarchy_review.csv"),
            "canonical": str(out_dir / "full_gold_draft_canonical_triples_review.csv"),
            "raw_dir": str(raw_dir),
        },
    }
    (out_dir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


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
    raw_dir = Path(args.raw_dir) if args.raw_dir else out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.retries,
    )

    if not args.aggregate_only:
        run_inducer(client, args, manifest, raw_dir)
        run_domain_consolidation(client, args, manifest, raw_dir)
        run_expander(client, args, raw_dir)
        run_enricher(client, args, manifest, raw_dir)
    summary = aggregate_outputs(args, manifest, out_dir, raw_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

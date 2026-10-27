import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from preannotate_full_ontology_gold import (
    RELATION_TYPES,
    canonical,
    doc_text,
    load_dotenv_file,
    load_json,
    norm_space,
    parse_json_object,
    read_csv,
    write_csv,
)


DEFAULT_ROOT = Path("taxonomy_runs/gold_test_v2_high_signal_2026_06_25")
DEFAULT_MANIFEST = DEFAULT_ROOT / "high_signal_final50_manifest.csv"
DEFAULT_OUT = DEFAULT_ROOT / "baseline_ontogen_deepseek_final50_2026_06_29"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an OntoGen-style DeepSeek baseline on final50.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-input-chars", type=int, default=10000)
    parser.add_argument("--max-vocabulary", type=int, default=55)
    parser.add_argument("--max-hierarchy", type=int, default=24)
    parser.add_argument("--max-relations", type=int, default=28)
    parser.add_argument("--category-runs", type=int, default=3)
    parser.add_argument("--taxonomy-iterations", type=int, default=2)
    parser.add_argument("--self-consistency", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="disabled")
    parser.add_argument("--reasoning-effort", default="")
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
    text_norm = simple_norm(text)
    return bool(term_norm and term_norm in text_norm)


def api_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    thinking: str,
    reasoning_effort: str,
) -> tuple[str, dict, dict]:
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if "deepseek" in str(client.base_url).lower():
        request["extra_body"] = {"thinking": {"type": thinking}}
    elif reasoning_effort:
        request["reasoning_effort"] = reasoning_effort
    response = client.chat.completions.create(
        **request,
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


def vocabulary_prompt(sample_id: str, title: str, text: str, limit: int) -> tuple[str, str]:
    system = "You extract a scientific vocabulary for ontology induction from economic-geology literature."
    user = f"""Document ID: {sample_id}
Title: {title}

Extract at most {limit} reusable geological or mineral-domain terms that occur verbatim in the source.
Include deposit types, minerals, rocks, alteration types, structures, processes, geological events, and time concepts.
Prefer class-level concepts. Exclude coordinates, analytical spot IDs, bibliography entries, and generic words.
Do not paraphrase or invent terms. Every returned term must appear in the source text.

Return JSON only:
{{"terms":[{{"term":"exact source span","evidence":"short source evidence"}}]}}

<source>
{text}
</source>"""
    return system, user


def extract_vocabulary(
    client: OpenAI,
    args: argparse.Namespace,
    row: dict,
    out_path: Path,
) -> dict:
    data = load_json(Path(row["file"]))
    text = doc_text(data, args.max_input_chars)
    system, user = vocabulary_prompt(row["sample_id"], row["title"], text, args.max_vocabulary)
    content, parsed, usage = api_json(
        client, args.model, system, user, 3000, args.temperature, args.thinking,
        args.reasoning_effort,
    )
    raw_terms = parsed.get("terms", []) if isinstance(parsed, dict) else []
    terms = []
    seen = set()
    for item in raw_terms if isinstance(raw_terms, list) else []:
        if isinstance(item, str):
            term, evidence = norm_space(item), ""
        elif isinstance(item, dict):
            term = norm_space(item.get("term"))
            evidence = norm_space(item.get("evidence"))
        else:
            continue
        key = simple_norm(term)
        if not key or key in seen or not verified_in_text(term, text):
            continue
        seen.add(key)
        terms.append({"term": term, "term_norm": key, "evidence": evidence})
    payload = {
        "stage": "vocabulary",
        "sample_id": row["sample_id"],
        "source_file": row["file"],
        "title": row["title"],
        "model": args.model,
        "raw_response": content,
        "terms": terms[: args.max_vocabulary],
        "usage": usage,
    }
    save_payload(out_path, payload)
    return payload


def run_vocabulary_stage(
    client: OpenAI,
    args: argparse.Namespace,
    manifest: list[dict],
    raw_dir: Path,
) -> None:
    pending = []
    for row in manifest:
        path = raw_dir / "vocabulary" / f"{row['sample_id']}.json"
        if args.resume and path.exists():
            continue
        pending.append((row, path))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(extract_vocabulary, client, args, row, path): row
            for row, path in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            payload = future.result()
            print(
                json.dumps(
                    {
                        "stage": "vocabulary",
                        "completed": index,
                        "pending_total": len(pending),
                        "sample_id": payload["sample_id"],
                        "term_count": len(payload["terms"]),
                        "tokens": payload.get("usage", {}).get("total_tokens", 0),
                    }
                ),
                flush=True,
            )


def load_vocabularies(manifest: list[dict], raw_dir: Path) -> tuple[dict[str, list[dict]], Counter]:
    by_doc: dict[str, list[dict]] = {}
    frequency: Counter = Counter()
    for row in manifest:
        payload = load_payload(raw_dir / "vocabulary" / f"{row['sample_id']}.json")
        terms = payload.get("terms", []) or []
        by_doc[row["sample_id"]] = terms
        frequency.update(item.get("term_norm", "") for item in terms if item.get("term_norm"))
    return by_doc, frequency


def category_candidate_prompt(terms: list[str], run_index: int) -> tuple[str, str]:
    system = "You identify stable top-level categories for an economic-geology ontology."
    user = f"""Generate 8 to 12 broad, mutually distinguishable, reusable ontology categories that organize the supplied corpus vocabulary.
The categories must be scientific class labels, not named mines, deposits, countries, or paper topics.
Use PascalCase labels and concise definitions. This is category proposal run {run_index}.

Vocabulary sample:
{json.dumps(terms, ensure_ascii=False)}

Return JSON only:
{{"categories":[{{"name":"PascalCaseCategory","definition":"short definition"}}]}}"""
    return system, user


def run_category_stage(
    client: OpenAI,
    args: argparse.Namespace,
    manifest: list[dict],
    raw_dir: Path,
) -> list[dict]:
    final_path = raw_dir / "categories" / "final.json"
    if args.resume and final_path.exists():
        return load_payload(final_path).get("categories", [])
    _, frequency = load_vocabularies(manifest, raw_dir)
    top_terms = [term for term, _ in frequency.most_common(500)]
    candidate_payloads = []
    rng = random.Random(20260629)
    for run_index in range(1, args.category_runs + 1):
        terms = top_terms.copy()
        rng.shuffle(terms)
        system, user = category_candidate_prompt(terms, run_index)
        content, parsed, usage = api_json(
            client, args.model, system, user, 1800, args.temperature, args.thinking,
            args.reasoning_effort,
        )
        categories = parsed.get("categories", []) if isinstance(parsed, dict) else []
        payload = {
            "stage": "category_candidate",
            "run": run_index,
            "raw_response": content,
            "categories": categories,
            "usage": usage,
        }
        save_payload(raw_dir / "categories" / f"candidate_{run_index}.json", payload)
        candidate_payloads.append(payload)

    system = "You consolidate repeated ontology category proposals into one stable economic-geology schema."
    user = f"""Merge the candidate category lists below into 8 to 12 broad, non-overlapping categories.
Preserve important geological distinctions and remove synonyms or paper-specific categories.
Use PascalCase labels. Do not consult or reproduce any evaluation gold data.

Candidates:
{json.dumps([p['categories'] for p in candidate_payloads], ensure_ascii=False)}

Return JSON only:
{{"categories":[{{"name":"PascalCaseCategory","definition":"short definition"}}]}}"""
    content, parsed, usage = api_json(
        client, args.model, system, user, 1800, args.temperature, args.thinking,
        args.reasoning_effort,
    )
    categories = []
    seen = set()
    for item in parsed.get("categories", []) if isinstance(parsed, dict) else []:
        if not isinstance(item, dict):
            continue
        name = canonical(item.get("name"))
        if not name or simple_norm(name) in seen:
            continue
        seen.add(simple_norm(name))
        categories.append({"name": name, "definition": norm_space(item.get("definition"))})
    final = {
        "stage": "categories_final",
        "model": args.model,
        "categories": categories,
        "raw_response": content,
        "usage": usage,
    }
    save_payload(final_path, final)
    return categories


def taxonomy_prompt(
    row: dict,
    text: str,
    terms: list[dict],
    categories: list[dict],
    current_edges: set[tuple[str, str]],
    max_edges: int,
) -> tuple[str, str]:
    current_nodes = sorted(
        {node for edge in current_edges for node in edge}
        | {item["name"] for item in categories}
    )
    taxonomy_context = [f"{child} isA {parent}" for child, parent in sorted(current_edges)]
    system = "You incrementally place corpus vocabulary into a scientific isA taxonomy, following the OntoGen procedure."
    user = f"""Document ID: {row['sample_id']}
Title: {row['title']}

For terms from this document, propose at most {max_edges} strict child isA parent edges supported by the source.
Every instance of the child must be an instance of the parent.
Do not use hosted-in, part-of, association, causation, occurrence, locality, or commodity relations as isA.
The child must be one of the document vocabulary terms. The parent must be a corpus vocabulary term, a top category, or an existing taxonomy node.
Do not create other nodes. Do not output self-loops or duplicate edges.

Top categories:
{json.dumps(categories, ensure_ascii=False)}

Existing taxonomy edges:
{json.dumps(taxonomy_context[-1400:], ensure_ascii=False)}

Existing nodes:
{json.dumps(current_nodes[-1800:], ensure_ascii=False)}

Document vocabulary:
{json.dumps([item['term'] for item in terms], ensure_ascii=False)}

Return JSON only:
{{"hierarchy_relations":[{{"child":"exact vocabulary term","parent":"allowed parent node","evidence":"short source evidence","confidence":0.0}}]}}

<source>
{text}
</source>"""
    return system, user


def creates_cycle(edges: set[tuple[str, str]], child: str, parent: str) -> bool:
    adjacency: dict[str, set[str]] = {}
    for c, p in edges:
        adjacency.setdefault(c, set()).add(p)
    stack = [parent]
    seen = set()
    while stack:
        node = stack.pop()
        if node == child:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency.get(node, set()))
    return False


def parse_taxonomy_edges(
    parsed: dict,
    allowed_children: dict[str, str],
    allowed_parents: dict[str, str],
    limit: int,
) -> list[dict]:
    raw = parsed.get("hierarchy_relations", []) if isinstance(parsed, dict) else []
    rows = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        child_key = simple_norm(item.get("child"))
        parent_key = simple_norm(item.get("parent"))
        if not child_key or not parent_key or child_key == parent_key:
            continue
        if child_key not in allowed_children or parent_key not in allowed_parents:
            continue
        rows.append(
            {
                "child": allowed_children[child_key],
                "parent": allowed_parents[parent_key],
                "evidence": norm_space(item.get("evidence")),
                "confidence": item.get("confidence", ""),
            }
        )
    return rows[:limit]


def replay_taxonomy(raw_dir: Path, manifest: list[dict], iterations: int) -> tuple[set[tuple[str, str]], list[dict]]:
    edges: set[tuple[str, str]] = set()
    accepted_rows = []
    for iteration in range(1, iterations + 1):
        for row in manifest:
            payload = load_payload(raw_dir / "taxonomy" / f"iteration_{iteration}" / f"{row['sample_id']}.json")
            for edge in payload.get("accepted_edges", []) or []:
                child = edge.get("child", "")
                parent = edge.get("parent", "")
                if child and parent and (child, parent) not in edges and not creates_cycle(edges, child, parent):
                    edges.add((child, parent))
                    accepted_rows.append({**edge, "sample_id": row["sample_id"], "iteration": iteration})
    return edges, accepted_rows


def run_taxonomy_stage(
    client: OpenAI,
    args: argparse.Namespace,
    manifest: list[dict],
    raw_dir: Path,
    categories: list[dict],
) -> None:
    vocab_by_doc, _ = load_vocabularies(manifest, raw_dir)
    all_term_names = {
        item["term_norm"]: item["term"]
        for terms in vocab_by_doc.values()
        for item in terms
        if item.get("term_norm") and item.get("term")
    }
    category_names = {simple_norm(item["name"]): item["name"] for item in categories}
    edges, _ = replay_taxonomy(raw_dir, manifest, args.taxonomy_iterations)

    for iteration in range(1, args.taxonomy_iterations + 1):
        for doc_index, row in enumerate(manifest, start=1):
            out_path = raw_dir / "taxonomy" / f"iteration_{iteration}" / f"{row['sample_id']}.json"
            if args.resume and out_path.exists():
                continue
            terms = vocab_by_doc.get(row["sample_id"], [])
            data = load_json(Path(row["file"]))
            text = doc_text(data, args.max_input_chars)
            allowed_children = {item["term_norm"]: item["term"] for item in terms}
            existing_nodes = {simple_norm(node): node for edge in edges for node in edge}
            allowed_parents = {**all_term_names, **category_names, **existing_nodes}
            system, user = taxonomy_prompt(
                row, text, terms, categories, edges, args.max_hierarchy
            )
            def sample_taxonomy(sample_index: int) -> dict:
                content, parsed, usage = api_json(
                    client, args.model, system, user, 3200, args.temperature, args.thinking,
                    args.reasoning_effort,
                )
                sample_edges = parse_taxonomy_edges(
                    parsed, allowed_children, allowed_parents, args.max_hierarchy
                )
                return {
                    "sample_index": sample_index,
                    "raw_response": content,
                    "edges": sample_edges,
                    "usage": usage,
                }

            with ThreadPoolExecutor(
                max_workers=min(args.workers, args.self_consistency)
            ) as executor:
                samples = list(
                    executor.map(sample_taxonomy, range(1, args.self_consistency + 1))
                )

            counts: Counter = Counter()
            representative: dict[tuple[str, str], dict] = {}
            for sample in samples:
                seen_sample = set()
                for edge in sample["edges"]:
                    key = (simple_norm(edge["child"]), simple_norm(edge["parent"]))
                    if key in seen_sample:
                        continue
                    seen_sample.add(key)
                    counts[key] += 1
                    representative.setdefault(key, edge)
            majority = args.self_consistency // 2 + 1
            accepted = []
            for key, count in counts.most_common():
                if count < majority:
                    continue
                edge = representative[key]
                pair = (edge["child"], edge["parent"])
                if pair in edges or creates_cycle(edges, *pair):
                    continue
                edges.add(pair)
                accepted.append({**edge, "vote_count": count})
            payload = {
                "stage": "taxonomy",
                "iteration": iteration,
                "sample_id": row["sample_id"],
                "source_file": row["file"],
                "title": row["title"],
                "model": args.model,
                "self_consistency": args.self_consistency,
                "samples": samples,
                "accepted_edges": accepted,
            }
            save_payload(out_path, payload)
            print(
                json.dumps(
                    {
                        "stage": "taxonomy",
                        "iteration": iteration,
                        "doc": doc_index,
                        "total_docs": len(manifest),
                        "sample_id": row["sample_id"],
                        "accepted": len(accepted),
                        "graph_edges": len(edges),
                    }
                ),
                flush=True,
            )


def relation_prompt(row: dict, text: str, terms: list[dict], limit: int) -> tuple[str, str]:
    system = "You extract schema-constrained scientific relations after an OntoGen-style vocabulary stage."
    user = f"""Document ID: {row['sample_id']}
Title: {row['title']}

Extract at most {limit} non-hierarchical geological relations using only these predicates:
{', '.join(RELATION_TYPES)}

Both endpoints must be terms from the extracted document vocabulary below. Preserve direction. Do not output isA relations.
Every relation must be explicitly supported by the source. Do not invent predicates or endpoints.

Vocabulary:
{json.dumps([item['term'] for item in terms], ensure_ascii=False)}

Return JSON only:
{{"nonhierarchical_relations":[{{"subject":"exact vocabulary term","predicate":"allowed predicate","object":"exact vocabulary term","evidence":"short source evidence","confidence":0.0}}]}}

<source>
{text}
</source>"""
    return system, user


def extract_relations(
    client: OpenAI,
    args: argparse.Namespace,
    row: dict,
    terms: list[dict],
    out_path: Path,
) -> dict:
    data = load_json(Path(row["file"]))
    text = doc_text(data, args.max_input_chars)
    system, user = relation_prompt(row, text, terms, args.max_relations)
    content, parsed, usage = api_json(
        client, args.model, system, user, 3600, args.temperature, args.thinking,
        args.reasoning_effort,
    )
    allowed = {item["term_norm"]: item["term"] for item in terms}
    relations = []
    seen = set()
    raw = parsed.get("nonhierarchical_relations", []) if isinstance(parsed, dict) else []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        subject_key = simple_norm(item.get("subject"))
        object_key = simple_norm(item.get("object"))
        predicate = norm_space(item.get("predicate")).upper()
        key = (subject_key, predicate, object_key)
        if (
            subject_key not in allowed
            or object_key not in allowed
            or predicate not in RELATION_TYPES
            or subject_key == object_key
            or key in seen
        ):
            continue
        seen.add(key)
        relations.append(
            {
                "subject": allowed[subject_key],
                "predicate": predicate,
                "object": allowed[object_key],
                "evidence": norm_space(item.get("evidence")),
                "confidence": item.get("confidence", ""),
            }
        )
    payload = {
        "stage": "relations",
        "sample_id": row["sample_id"],
        "source_file": row["file"],
        "title": row["title"],
        "model": args.model,
        "raw_response": content,
        "relations": relations[: args.max_relations],
        "usage": usage,
    }
    save_payload(out_path, payload)
    return payload


def run_relation_stage(
    client: OpenAI,
    args: argparse.Namespace,
    manifest: list[dict],
    raw_dir: Path,
) -> None:
    vocab_by_doc, _ = load_vocabularies(manifest, raw_dir)
    pending = []
    for row in manifest:
        path = raw_dir / "relations" / f"{row['sample_id']}.json"
        if args.resume and path.exists():
            continue
        pending.append((row, vocab_by_doc.get(row["sample_id"], []), path))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(extract_relations, client, args, row, terms, path): row
            for row, terms, path in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            payload = future.result()
            print(
                json.dumps(
                    {
                        "stage": "relations",
                        "completed": index,
                        "pending_total": len(pending),
                        "sample_id": payload["sample_id"],
                        "relation_count": len(payload["relations"]),
                        "tokens": payload.get("usage", {}).get("total_tokens", 0),
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
    method_label = f"OntoGen-{args.model}-adapted"
    _, accepted_rows = replay_taxonomy(raw_dir, manifest, args.taxonomy_iterations)
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    hierarchy_rows = []
    relation_rows = []
    alignment_rows: OrderedDict[tuple[str, str, str], dict] = OrderedDict()
    canonical_rows = []

    for index, edge in enumerate(accepted_rows, start=1):
        sample_id = edge["sample_id"]
        source = manifest_by_id[sample_id]
        child = edge["child"]
        parent = edge["parent"]
        row = {
            "annotation_id": f"ONTOGEN_{sample_id}_H_{index:04d}",
            "sample_id": sample_id,
            "source_file": source["file"],
            "title": source["title"],
            "child_mention": child,
            "parent_mention": parent,
            "child_canonical": canonical(child),
            "parent_canonical": canonical(parent),
            "predicate": "subClassOf",
            "evidence_text": edge.get("evidence", ""),
            "preannotator_decision": "VALID",
            "confidence": edge.get("confidence", ""),
            "gold_decision": "",
            "gold_error_type": "",
            "gold_note": "",
            "model": args.model,
            "baseline_method": method_label,
            "iteration": edge.get("iteration", ""),
            "vote_count": edge.get("vote_count", ""),
        }
        hierarchy_rows.append(row)
        canonical_rows.append(
            {
                "annotation_id": row["annotation_id"],
                "sample_id": sample_id,
                "source_file": source["file"],
                "title": source["title"],
                "subject_canonical": row["child_canonical"],
                "predicate": "subClassOf",
                "object_canonical": row["parent_canonical"],
                "evidence_text": row["evidence_text"],
                "source_layer": "hierarchy",
            }
        )

    relation_index = 0
    for source in manifest:
        sample_id = source["sample_id"]
        payload = load_payload(raw_dir / "relations" / f"{sample_id}.json")
        for item in payload.get("relations", []) or []:
            relation_index += 1
            subject = item["subject"]
            obj = item["object"]
            row = {
                "annotation_id": f"ONTOGEN_{sample_id}_R_{relation_index:04d}",
                "sample_id": sample_id,
                "source_file": source["file"],
                "title": source["title"],
                "subject_mention": subject,
                "predicate": item["predicate"],
                "object_mention": obj,
                "subject_canonical": canonical(subject),
                "object_canonical": canonical(obj),
                "evidence_text": item.get("evidence", ""),
                "preannotator_decision": "VALID",
                "confidence": item.get("confidence", ""),
                "gold_decision": "",
                "gold_error_type": "",
                "gold_note": "",
                "model": args.model,
                "baseline_method": method_label,
            }
            relation_rows.append(row)
            canonical_rows.append(
                {
                    "annotation_id": row["annotation_id"],
                    "sample_id": sample_id,
                    "source_file": source["file"],
                    "title": source["title"],
                    "subject_canonical": row["subject_canonical"],
                    "predicate": row["predicate"],
                    "object_canonical": row["object_canonical"],
                    "evidence_text": row["evidence_text"],
                    "source_layer": "nonhierarchical",
                }
            )
            for mention, node in ((subject, row["subject_canonical"]), (obj, row["object_canonical"])):
                key = (sample_id, simple_norm(mention), node)
                alignment_rows.setdefault(
                    key,
                    {
                        "sample_id": sample_id,
                        "source_file": source["file"],
                        "title": source["title"],
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

    hierarchy_fields = [
        "annotation_id", "sample_id", "source_file", "title", "child_mention",
        "parent_mention", "child_canonical", "parent_canonical", "predicate",
        "evidence_text", "preannotator_decision", "confidence", "gold_decision",
        "gold_error_type", "gold_note", "model", "baseline_method", "iteration",
        "vote_count",
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
    write_csv(out_dir / "full_gold_draft_4d_relations_review.csv", relation_rows, relation_fields)
    write_csv(
        out_dir / "full_gold_draft_alignment_review.csv",
        list(alignment_rows.values()),
        alignment_fields,
    )
    write_csv(out_dir / "full_gold_draft_canonical_triples_review.csv", canonical_rows, canonical_fields)

    token_usage = Counter()
    call_count = 0
    for path in raw_dir.rglob("*.json"):
        payload = load_payload(path)
        usages = []
        if payload.get("usage"):
            usages.append(payload["usage"])
        usages.extend(sample.get("usage", {}) for sample in payload.get("samples", []) or [])
        for usage in usages:
            if not usage:
                continue
            call_count += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                token_usage[key] += int(usage.get(key, 0) or 0)

    summary = {
        "scope": "OntoGen-style adapted baseline; gold was not used during generation.",
        "method_label": method_label,
        "original_method_source": "Oarga et al., Scientific knowledge graph and ontology generation using open large language models, 2026.",
        "adaptations": [
            f"All LLM roles use {args.model} instead of the original open-model configuration.",
            "Vocabulary extraction is document-level with source-span verification rather than one API call per sentence.",
            "Top categories are generated by repeated corpus-vocabulary proposals and one consolidation call.",
            "Incremental taxonomy extraction uses the original explicit-isA, cycle-removal, and self-consistency principles.",
            "Non-hierarchical extraction is restricted to the GoldOntology 12-predicate schema for fair scoring.",
        ],
        "documents": len(manifest),
        "model": args.model,
        "params": {
            "max_input_chars": args.max_input_chars,
            "max_vocabulary": args.max_vocabulary,
            "max_hierarchy": args.max_hierarchy,
            "max_relations": args.max_relations,
            "category_runs": args.category_runs,
            "taxonomy_iterations": args.taxonomy_iterations,
            "self_consistency": args.self_consistency,
            "reasoning_effort": args.reasoning_effort,
        },
        "counts": {
            "hierarchy_rows": len(hierarchy_rows),
            "relation_rows": len(relation_rows),
            "alignment_rows": len(alignment_rows),
            "canonical_rows": len(canonical_rows),
            "api_calls_recorded": call_count,
        },
        "token_usage": dict(token_usage),
        "outputs": {
            "hierarchy": str(out_dir / "full_gold_draft_hierarchy_review.csv"),
            "relations": str(out_dir / "full_gold_draft_4d_relations_review.csv"),
            "alignment": str(out_dir / "full_gold_draft_alignment_review.csv"),
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
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.retries,
    )

    if not args.aggregate_only:
        run_vocabulary_stage(client, args, manifest, raw_dir)
        categories = run_category_stage(client, args, manifest, raw_dir)
        run_taxonomy_stage(client, args, manifest, raw_dir, categories)
        run_relation_stage(client, args, manifest, raw_dir)
    summary = aggregate_outputs(args, manifest, out_dir, raw_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

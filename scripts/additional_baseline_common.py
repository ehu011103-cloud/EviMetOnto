import json
import re
import shutil
from pathlib import Path

import networkx as nx

from preannotate_full_ontology_gold import RELATION_TYPES, parse_json_object


def call_json(client, args, messages: list[dict], max_tokens: int) -> tuple[str, dict, dict]:
    request = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if "deepseek" in args.base_url.lower() or "deepseek" in args.model.lower():
        request["extra_body"] = {"thinking": {"type": args.thinking}}
    response = client.chat.completions.create(**request)
    content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else {}
    return content, parse_json_object(content), usage_dict


def merge_usage(*items: dict) -> dict:
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {field: sum(int(item.get(field, 0) or 0) for item in items) for field in fields}


def confidence(row: dict) -> float:
    try:
        return float(row.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return {"high": 0.9, "medium": 0.7, "low": 0.5}.get(
            str(row.get("confidence", "")).strip().lower(), 0.0
        )


def canonical(value: object) -> str:
    words = re.findall(r"[A-Za-z0-9]+", str(value or ""))
    return "".join(word[:1].upper() + word[1:] for word in words)


def normalize_hierarchy_row(row: object) -> dict | None:
    if isinstance(row, list) and len(row) >= 3:
        child, relation, parent = row[:3]
        if str(relation).strip().lower() not in {
            "is_a", "isa", "subclass_of", "subclassof", "subclass", "rdfs:subclassof"
        }:
            return None
        row = {"child_mention": child, "parent_mention": parent}
    if not isinstance(row, dict):
        return None
    relation = str(row.get("relation", row.get("predicate", ""))).strip().lower()
    if relation and relation not in {
        "is_a", "isa", "subclass_of", "subclassof", "subclass", "rdfs:subclassof"
    }:
        return None
    child_mention = row.get("child_mention") or row.get("child") or row.get("source") or row.get("subject")
    parent_mention = row.get("parent_mention") or row.get("parent") or row.get("target") or row.get("object")
    child_canonical = row.get("child_canonical") or canonical(child_mention)
    parent_canonical = row.get("parent_canonical") or canonical(parent_mention)
    if not child_canonical or not parent_canonical:
        return None
    return {
        **row,
        "child_mention": str(child_mention or child_canonical),
        "parent_mention": str(parent_mention or parent_canonical),
        "child_canonical": str(child_canonical),
        "parent_canonical": str(parent_canonical),
        "evidence_text": str(row.get("evidence_text", row.get("evidence", ""))),
        "confidence": confidence(row),
    }


def normalize_relation_row(row: object) -> dict | None:
    if isinstance(row, list) and len(row) >= 3:
        subject, predicate, obj = row[:3]
        row = {"subject_mention": subject, "predicate": predicate, "object_mention": obj}
    if not isinstance(row, dict):
        return None
    predicate = str(row.get("predicate", row.get("relation", ""))).strip().upper()
    if predicate not in RELATION_TYPES:
        return None
    subject_mention = row.get("subject_mention") or row.get("subject") or row.get("source")
    object_mention = row.get("object_mention") or row.get("object") or row.get("target")
    subject_canonical = row.get("subject_canonical") or canonical(subject_mention)
    object_canonical = row.get("object_canonical") or canonical(object_mention)
    if not subject_canonical or not object_canonical:
        return None
    return {
        **row,
        "subject_mention": str(subject_mention or subject_canonical),
        "object_mention": str(object_mention or object_canonical),
        "subject_canonical": str(subject_canonical),
        "object_canonical": str(object_canonical),
        "predicate": predicate,
        "evidence_text": str(row.get("evidence_text", row.get("evidence", ""))),
        "confidence": confidence(row),
    }


def dedupe_rows(rows: list[dict], keys: tuple[str, ...], limit: int) -> list[dict]:
    result = []
    seen = set()
    for row in sorted((row for row in rows if isinstance(row, dict)), key=confidence, reverse=True):
        key = tuple(str(row.get(name, "")).strip().lower() for name in keys)
        if not all(key) or key in seen:
            continue
        seen.add(key)
        result.append(row)
        if len(result) >= limit:
            break
    return result


def sanitize_predictions(parsed: dict, max_hierarchy: int, max_4d: int) -> dict:
    hierarchy = parsed.get("hierarchy_relations", []) or []
    hierarchy = [normalized for row in hierarchy if (normalized := normalize_hierarchy_row(row))]
    hierarchy = dedupe_rows(
        hierarchy,
        ("child_canonical", "parent_canonical"),
        max_hierarchy * 2,
    )
    graph = nx.DiGraph()
    accepted_hierarchy = []
    for row in hierarchy:
        child = str(row.get("child_canonical", "")).strip()
        parent = str(row.get("parent_canonical", "")).strip()
        if not child or not parent or child.lower() == parent.lower():
            continue
        graph.add_edge(child.lower(), parent.lower())
        if not nx.is_directed_acyclic_graph(graph):
            graph.remove_edge(child.lower(), parent.lower())
            continue
        accepted_hierarchy.append(row)
        if len(accepted_hierarchy) >= max_hierarchy:
            break

    relations = parsed.get("nonhierarchical_relations", []) or parsed.get("relations", []) or []
    relations = [normalized for row in relations if (normalized := normalize_relation_row(row))]
    relations = [row for row in relations if row["subject_canonical"].lower() != row["object_canonical"].lower()]
    relations = dedupe_rows(
        relations,
        ("subject_canonical", "predicate", "object_canonical"),
        max_4d,
    )
    for row in relations:
        row["predicate"] = str(row.get("predicate", row.get("relation", ""))).strip().upper()
    return {
        "hierarchy_relations": accepted_hierarchy,
        "nonhierarchical_relations": relations,
    }


def task_paths(raw_dir: Path, sample_id: str) -> tuple[Path, Path]:
    return raw_dir / f"{sample_id}.hierarchy.json", raw_dir / f"{sample_id}.relations.json"


def is_done(raw_dir: Path, sample_id: str) -> bool:
    return all(path.exists() for path in task_paths(raw_dir, sample_id))


def output_empty(raw_dir: Path, sample_id: str) -> bool:
    hierarchy_path, relation_path = task_paths(raw_dir, sample_id)
    if not hierarchy_path.exists() or not relation_path.exists():
        return True
    try:
        hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))
        relations = json.loads(relation_path.read_text(encoding="utf-8"))
        return not hierarchy.get("parsed", {}).get("hierarchy_relations") and not relations.get(
            "parsed", {}
        ).get("nonhierarchical_relations")
    except Exception:
        return True


def archive_task_views(raw_dir: Path, sample_id: str) -> None:
    archive_dir = raw_dir.parent / "incomplete_raw"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in task_paths(raw_dir, sample_id):
        if not path.exists():
            continue
        target = archive_dir / path.name
        attempt = 1
        while target.exists():
            target = archive_dir / f"{path.stem}.attempt{attempt}{path.suffix}"
            attempt += 1
        shutil.copy2(path, target)


def write_task_views(
    raw_dir: Path,
    row: dict,
    model: str,
    base_url: str,
    method: str,
    final_predictions: dict,
    usage: dict,
    pipeline: dict,
) -> None:
    common = {
        "sample_id": row["sample_id"],
        "source_file": row["file"],
        "title": row["title"],
        "model": model,
        "base_url": base_url,
        "baseline_method": method,
        "pipeline": pipeline,
    }
    hierarchy_path, relation_path = task_paths(raw_dir, row["sample_id"])
    hierarchy_path.write_text(
        json.dumps(
            {
                **common,
                "task": f"{method}_hierarchy_view",
                "parsed": {"hierarchy_relations": final_predictions["hierarchy_relations"]},
                "usage": usage,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    relation_path.write_text(
        json.dumps(
            {
                **common,
                "task": f"{method}_relations_view",
                "parsed": {
                    "nonhierarchical_relations": final_predictions["nonhierarchical_relations"]
                },
                "usage": {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

"""Export non-text reproducibility artifacts for the public release package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "A_full_run_all"
GOLD = ROOT / "taxonomy_runs" / "gold_test_v2_high_signal_2026_06_25" / "final50_gold_v8_manual_disagreement_plus6_baseline_source_evidence_2026_07_11"
RELATIONS = ROOT / "taxonomy_runs" / "relations_4d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output: Path) -> int:
    rows = []
    for path in sorted(CORPUS.glob("DOC_A_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        source = record.get("source", {})
        metadata = record.get("metadata", {})
        rows.append({
            "document_id": source.get("document_id", ""),
            "title": metadata.get("title", ""),
            "journal": metadata.get("journal", ""),
            "date": metadata.get("date", ""),
            "doi": metadata.get("doi", ""),
            "language": metadata.get("language", ""),
            "page_count": source.get("page_count", ""),
            "pdf_sha256": source.get("pdf_sha256", ""),
            "record_sha256": sha256(path),
        })
    destination = output / "doc_a_manifest_5723.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_gold_labels(output: Path) -> dict[str, int]:
    sources = [
        ("hierarchy", GOLD / "full_gold_draft_hierarchy_review.csv"),
        ("non_hierarchical", GOLD / "full_gold_draft_4d_relations_review.csv"),
    ]
    rows = []
    counts: dict[str, int] = {}
    for layer, source in sources:
        with source.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if layer == "hierarchy":
                    subject = row.get("child_canonical", "")
                    obj = row.get("parent_canonical", "")
                else:
                    subject = row.get("subject_canonical", "")
                    obj = row.get("object_canonical", "")
                decision = row.get("gold_decision", "")
                key = f"{layer}:{decision}"
                counts[key] = counts.get(key, 0) + 1
                rows.append({
                    "annotation_id": row.get("annotation_id", ""),
                    "sample_id": row.get("sample_id", ""),
                    "source_file": Path(row.get("source_file", "")).name,
                    "layer": layer,
                    "subject_canonical": subject,
                    "predicate": row.get("predicate", ""),
                    "object_canonical": obj,
                    "gold_decision": decision,
                    "gold_error_type": row.get("gold_error_type", ""),
                    "gold_version": row.get("codex_gold_version", ""),
                })
    destination = output / "gold_v8_relation_labels.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), **counts}


def copy_json(source: Path, destination: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    document_count = write_manifest(args.output)
    gold_counts = write_gold_labels(args.output)
    copy_json(RELATIONS / "full_raw_4d_relations_v4flash_summary.json", args.output / "full_4d_extraction_summary.json")
    copy_json(RELATIONS / "full_4d_bidirectional_validation_deepseek_v4flash_summary.json", args.output / "full_4d_validation_summary.json")
    summary = {
        "document_manifest_count": document_count,
        "gold_label_export": gold_counts,
        "excluded_fields": ["full_text", "abstract", "body", "evidence_text", "gold_note", "raw_model_response"],
        "license_note": "The release contains metadata, hashes, derived labels, and aggregate results only; it does not redistribute third-party full text or verbatim evidence spans.",
    }
    (args.output / "release_artifact_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

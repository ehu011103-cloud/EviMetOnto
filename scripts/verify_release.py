"""Verify the public, non-restricted reproducibility artifacts.

This script intentionally checks only files that can be distributed without
redistributing licensed literature text or raw model outputs. It verifies the
frozen corpus manifest, Gold-label counts, and the arithmetic of the proposed
workflow rows in the published Table 3 artifact.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def read_csv(name: str) -> list[dict[str, str]]:
    with (ARTIFACTS / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def check(condition: bool, message: str) -> bool:
    print(("PASS" if condition else "FAIL") + f": {message}")
    return condition


def main() -> int:
    ok = True
    manifest = read_csv("doc_a_manifest_5723.csv")
    doc_ids = [row["document_id"] for row in manifest]
    ok &= check(len(manifest) == 5723, "DOC_A manifest contains 5,723 records")
    ok &= check(len(set(doc_ids)) == 5723, "DOC_A identifiers are unique")
    ok &= check(
        all(row.get("pdf_sha256") and row.get("record_sha256") for row in manifest),
        "every DOC_A record has source and structured-record hashes",
    )

    gold = read_csv("gold_v8_relation_labels.csv")
    decisions = Counter((row["layer"], row["gold_decision"]) for row in gold)
    ok &= check(len(gold) == 2180, "Gold file contains 2,180 reviewed candidate records")
    ok &= check(
        decisions[("hierarchy", "VALID")] == 680,
        "Gold file contains 680 positive hierarchy keys",
    )
    ok &= check(
        decisions[("non_hierarchical", "VALID")] == 1015,
        "Gold file contains 1,015 positive non-hierarchical keys",
    )

    results = read_csv("table_3_endpoint_compatibility_results_final.csv")
    proposed = [row for row in results if row["panel"] == "A"]
    ok &= check(len(proposed) == 3, "Table 3 includes three proposed-workflow rows")
    for row in proposed:
        tp, fp, fn = (int(row[key]) for key in ("tp", "fp", "fn"))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        ok &= check(
            all(
                math.isclose(value, float(row[key]), abs_tol=0.00005)
                for value, key in ((precision, "precision"), (recall, "recall"), (f1, "f1"))
            ),
            f"Table 3 arithmetic is consistent for {row['output_scope']}",
        )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

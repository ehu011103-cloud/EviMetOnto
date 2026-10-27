"""Check released artifacts locally; do not call models or reconstruct missing runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import rdflib
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS


def read_csv(root: Path, name: str) -> list[dict[str, str]]:
    with (root / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify(root: Path) -> dict:
    checks = []

    def check(name: str, passed: bool, details) -> None:
        checks.append({"check": name, "status": "pass" if passed else "fail", "details": details})

    inventory = read_csv(root, "SHA256SUMS.csv")
    hash_failures = []
    for row in inventory:
        file = (root / row["path"]).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            hash_failures.append(row["path"])
        elif hashlib.sha256(file.read_bytes()).hexdigest() != row["sha256"]:
            hash_failures.append(row["path"])
    check("file_integrity", not hash_failures, {"files_checked": len(inventory), "failures": hash_failures})

    view = json.loads((root / "ontology_release/release_view.json").read_text(encoding="utf-8"))
    graph_path = root / "ontology_release" / view["file"]
    graph = Graph().parse(data=graph_path.read_text(encoding="utf-8"), format="turtle")
    classes = {s for s in graph.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef)}
    properties = {s for s in graph.subjects(RDF.type, OWL.ObjectProperty) if isinstance(s, URIRef)}
    edges = {(s, o) for s, o in graph.subject_objects(RDFS.subClassOf) if s in classes and o in classes}
    relations = {(s, p, o) for s, p, o in graph if p in properties and s in classes and o in classes}
    hierarchy = nx.DiGraph()
    hierarchy.add_nodes_from(classes)
    hierarchy.add_edges_from(edges)
    isolated = len(list(nx.isolates(hierarchy)))
    components = list(nx.weakly_connected_components(hierarchy))
    largest = max(map(len, components), default=0)
    condensed = nx.condensation(hierarchy)
    measured = {
        "classes": len(classes), "object_properties": len(properties),
        "named_subclass_edges": len(edges), "nonhierarchical_edges": len(relations),
        "rdf_triples": len(graph), "hierarchy_coverage_percent": round(100 * (len(classes)-isolated) / len(classes), 2),
        "isolated_classes": isolated, "largest_component_coverage_percent": round(100 * largest / len(classes), 2),
        "max_depth": nx.dag_longest_path_length(condensed), "acyclic": nx.is_directed_acyclic_graph(hierarchy),
    }
    expected = view["counts"]
    check("ontology_release_counts", all(measured[k] == v for k, v in expected.items()), measured)
    own_row = next(row for row in read_csv(root, "current_audit/selected_structural_quality_comparison.csv") if row["ontology"] == "EviMetOnto-full-corpus-acyclic")
    fields = ["classes", "named_subclass_edges", "hierarchy_coverage_percent", "isolated_classes", "largest_component_coverage_percent", "max_depth"]
    check("ontology_descriptors_match_report", all(abs(float(own_row[k])-measured[k]) < 0.005 for k in fields) and measured["acyclic"], {k: measured[k] for k in fields})

    manifest = read_csv(root, "artifacts/doc_a_manifest_5723.csv")
    check("corpus_inventory", len(manifest) == 5723 and len({r["document_id"] for r in manifest}) == 5723, {"records": len(manifest)})
    gold = read_csv(root, "artifacts/gold_v8_relation_labels.csv")
    counts = Counter((row["layer"], row["gold_decision"]) for row in gold)
    positives = {"hierarchy": counts[("hierarchy", "VALID")], "non_hierarchical": counts[("non_hierarchical", "VALID")]}
    check("reference_inventory", len(gold) == 2180 and positives == {"hierarchy": 680, "non_hierarchical": 1015}, {"reviewed_rows": len(gold), "positive_rows": positives})

    split = read_csv(root, "reported_retrieval/query_source_split.csv")
    by_source = defaultdict(set)
    by_query = {}
    for row in split:
        by_source[row["source_file"]].add(row["split"])
        by_query[row["query_id"]] = row["split"]
    split_counts = Counter(row["split"] for row in split)
    check("query_source_group_split", len(by_query) == len(split) == 247 and split_counts == {"development": 89, "test": 158} and all(len(s) == 1 for s in by_source.values()), dict(split_counts))

    accuracy = read_csv(root, "reported_shared_model/reported_metrics.csv")
    scopes = Counter(row["scope"] for row in accuracy)
    arithmetic_ok = True
    for row in accuracy:
        p, r, f1 = (float(row[k]) for k in ("precision", "recall", "f1"))
        implied = 2*p*r/(p+r) if p+r else 0
        arithmetic_ok &= all(0 <= x <= 1 for x in (p, r, f1)) and abs(implied-f1) <= 0.0011
    check("reported_accuracy_file", len(accuracy) == 18 and len({r["method"] for r in accuracy}) == 9 and scopes == {"hierarchy": 9, "overall_graph": 9} and arithmetic_ok and all(r["reported_model"] == "deepseek-v4-flash" for r in accuracy), {"rows": len(accuracy), "verification": "summary structure and rounded P/R/F1 arithmetic only; not prediction-level reproduction"})

    rows = read_csv(root, "reported_retrieval/evidence_retrieval_query_results.csv")
    summaries = {r["system"]: r for r in read_csv(root, "reported_retrieval/reported_summary.csv")}
    systems = {
        "BM25_coarse_same_test_split": "BM25_coarse_rank",
        "BM25+schema_coarse_same_test_split": "BM25+schema_coarse_rank",
        "TFIDF_word_coarse_same_test_split": "TFIDF_word_coarse_rank",
        "TFIDF_char_coarse_same_test_split": "TFIDF_char_coarse_rank",
        "TFIDF_word_char_hybrid_coarse_same_test_split": "TFIDF_word_char_hybrid_coarse_rank",
        "BM25+EviMetOnto_coarse_same_test_split": "BM25+EviMetOnto_coarse_rank",
    }
    test_rows = [r for r in rows if by_query.get(r["query_id"]) == "test"]
    for system, column in systems.items():
        ranks = [int(float(r[column])) for r in test_rows]
        assert all(1 <= r <= 100 for r in ranks), (system, "invalid rank")
        values = {"MRR": sum(1/r for r in ranks)/len(ranks),
                  "nDCG@10": sum(1/math.log2(r+1) if r <= 10 else 0 for r in ranks)/len(ranks)}
        values.update({f"Recall@{k}": sum(r <= k for r in ranks)/len(ranks) for k in (5, 10, 20)})
        check("archived_rank_summary: " + system, len(ranks) == 158 and all(abs(v-float(summaries[system][key])) < 1e-12 for key, v in values.items()), values)
    hybrid = summaries["BM25+EviMetOnto_hybrid_coarse"]
    displayed = [float(hybrid[k]) for k in ("MRR", "nDCG@10", "Recall@5", "Recall@10", "Recall@20")]
    check("tuned_hybrid_aggregate", len(summaries) == 7 and int(hybrid["n_queries"]) == 158 and float(hybrid["tuned_alpha_on_dev"]) == 0.5 and [round(v, 4) for v in displayed] == [0.7418, 0.7745, 0.8671, 0.8924, 0.9557], {"verification": "recorded aggregate and display precision only; per-query hybrid ranks unavailable"})

    return {"status": "pass" if all(c["status"] == "pass" for c in checks) else "fail", "scope": "offline artifact consistency; not a rerun of unavailable model outputs", "network_or_model_calls": False,
            "verification_runtime": {"python": sys.version.split()[0], "platform": platform.system(), "rdflib": rdflib.__version__, "networkx": nx.__version__},
            "checks": checks, "not_verified": ["current shared-model prediction-level accuracy", "tuned ontology-hybrid per-query ranking", "external-baseline ontology descriptors without imported snapshots", "independent geological relevance judgments", "CRediT roles and code/data licensing"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.root.resolve())
    except (OSError, KeyError, ValueError, AssertionError, StopIteration) as exc:
        result = {"status": "fail", "error": type(exc).__name__, "message": str(exc), "network_or_model_calls": False}
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

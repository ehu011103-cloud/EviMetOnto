"""Export non-text prediction keys and re-score all workflows with one rule."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
RUNS = PROJECT / "taxonomy_runs/gold_test_v2_high_signal_2026_06_25"
METHODS = {
    "LLMs4OL Task B+C": "baseline_llms4ol_bc_deepseek_final50_2026_07_23",
    "POP-Duo": "baseline_pop_duo_deepseek_final50_2026_06_30",
    "Text2Onto--LLM": "baseline_text2onto_llm_deepseek_final50_2026_07_26",
    "OntoGen": "baseline_ontogen_deepseek_final50_2026_06_29",
    "NeOn-GPT": "baseline_neon_gpt_deepseek_final50_2026_06_30",
    "Ontogenia": "baseline_ontogenia_deepseek_final50_2026_07_02",
    "Memoryless-CQ": "baseline_memoryless_cqbycq_deepseek_final50_2026_07_03",
    "Planning-First": "baseline_planning_first_deepseek_v4flash_final50_2026_07_03",
    "EviMetOnto": "hard_gate_validation_final50_v4",
}
FIELDS = ("method", "sample_id", "layer", "subject", "predicate", "object", "decision")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def norm(value):
    # Identical to evaluate_gold_test_v2_pilot.norm, not fuzzy matching.
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value or "")
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = text.lower().replace("_", " ").replace("-", " ")
    text = text.replace("sulphide", "sulfide").replace("sulphur", "sulfur").replace("dyke", "dike")
    return re.sub(r"[^a-z0-9]+", "", text)


def key(row):
    return (row["sample_id"], norm(row["subject"]), row["predicate"], norm(row["object"]))


def keys(rows):
    return {k for row in rows for k in [key(row)] if all(k) and k[1] != k[3]}


def score(gold, pred):
    tp, fp, fn = len(gold & pred), len(pred - gold), len(gold - pred)
    p = tp / len(pred) if pred else 0
    r = tp / len(gold) if gold else 0
    return dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0)


def export(out):
    out.mkdir(exist_ok=True)
    gold = read(PROJECT / "release/goldontology-reproducibility/artifacts/gold_v8_relation_labels.csv")
    gold_rows = [dict(method="Gold", sample_id=r["sample_id"], layer=r["layer"],
                      subject=r["subject_canonical"], predicate=r["predicate"],
                      object=r["object_canonical"], decision=r["gold_decision"]) for r in gold]
    write(out / "gold_keys.csv", gold_rows, FIELDS)
    samples = {r["sample_id"] for r in gold_rows}
    assert len(samples) == 50
    predictions, provenance = [], {}
    for method, dirname in METHODS.items():
        folder = RUNS / dirname
        ours = method == "EviMetOnto"
        paths = [folder / ("hard_gate_hierarchy_predictions.csv" if ours else "full_gold_draft_hierarchy_review.csv"),
                 folder / ("hard_gate_4d_predictions.csv" if ours else "full_gold_draft_4d_relations_review.csv")]
        models, input_hashes = set(), {}
        for layer, path in zip(("hierarchy", "nonhierarchical"), paths):
            records = read(path)
            input_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for row in records:
                if row.get("sample_id") not in samples:
                    continue
                if row.get("model"):
                    models.add(row["model"])
                decision = row.get("validation_decision", "") if ours else row.get("preannotator_decision", "VALID")
                if decision not in ({"VALID", "UNCERTAIN"} if ours else {"VALID"}):
                    continue
                subject = (row.get("child_canonical") or row.get("child_mention")) if layer == "hierarchy" else (row.get("subject_canonical") or row.get("subject_mention"))
                obj = (row.get("parent_canonical") or row.get("parent_mention")) if layer == "hierarchy" else (row.get("object_canonical") or row.get("object_mention"))
                predictions.append(dict(method=method, sample_id=row["sample_id"], layer=layer,
                                        subject=subject or "", predicate="subClassOf" if layer == "hierarchy" else (row.get("predicate") or "").upper().strip(),
                                        object=obj or "", decision=decision))
        provenance[method] = dict(source_directory=dirname, input_sha256=input_hashes, recorded_model_fields=sorted(models))
    write(out / "prediction_keys.csv", predictions, FIELDS)
    (out / "input_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def evaluate(out):
    gold = [r for r in read(out / "gold_keys.csv") if r["decision"] == "VALID"]
    pred = read(out / "prediction_keys.csv")
    gh = keys([r for r in gold if r["layer"] == "hierarchy"])
    gr = keys([r for r in gold if r["layer"] != "hierarchy"])
    assert (len(gh), len(gr)) == (680, 1015)
    results, decisions = [], []
    for method in METHODS:
        rows = [r for r in pred if r["method"] == method]
        ph = keys([r for r in rows if r["layer"] == "hierarchy"])
        pr = keys([r for r in rows if r["layer"] != "hierarchy"])
        for scope, g, p in (("hierarchy", gh, ph), ("nonhierarchical", gr, pr), ("overall_graph", gh | gr, ph | pr)):
            results.append(dict(method=method, scope=scope, gold_count=len(g), pred_count=len(p), **score(g, p)))
        for k in sorted(gh | gr | ph | pr):
            decisions.append(dict(method=method, sample_id=k[0], subject=k[1], predicate=k[2], object=k[3],
                                  in_gold=int(k in gh | gr), in_prediction=int(k in ph | pr),
                                  outcome="TP" if k in (gh | gr) & (ph | pr) else "FN" if k in gh | gr else "FP"))
    write(out / "unified_metrics.csv", results, results[0].keys())
    write(out / "key_decisions.csv", decisions, decisions[0].keys())
    tests = [norm("Sulphide-Mineral") == norm("sulfide_mineral"),
             score({("d", "a", "CAUSES", "b")}, {("d", "b", "CAUSES", "a")})["tp"] == 0,
             score({("d1", "a", "r", "b")}, {("d2", "a", "r", "b")})["tp"] == 0,
             len(keys([pred[0], pred[0]])) == 1]
    assert all(tests)
    ours = next(r for r in results if r["method"] == "EviMetOnto" and r["scope"] == "overall_graph")
    assert (ours["tp"], ours["fp"], ours["fn"]) == (1483, 304, 212)
    protocol = dict(version="unified_exact_20261010", status="retrospective_re_scoring_no_generation_rerun",
                    matching="Exact sample identifier, predicate, and normalized canonical endpoints; no alias ontology, semantic similarity, threshold, or symmetric swapping.",
                    normalization="CamelCase splitting, lower case, underscore/hyphen removal, sulphide/sulphur/dyke spelling substitutions, non-alphanumeric removal.",
                    counting="Unique set intersection; each normalized key contributes at most once, with no many-to-many matching.",
                    direction="Endpoint order preserved for every predicate, including declared symmetric relations.",
                    tests_passed=len(tests), input_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/"gold_keys.csv", out/"prediction_keys.csv")},
                    note="Re-scoring unifies evaluation only, not generation models, prompts, or candidate-conditioned annotation coverage.")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for r in results:
        if r["scope"] in {"hierarchy", "overall_graph"}:
            print(r["method"], r["scope"], *(f"{r[k]:.4f}" for k in ("precision", "recall", "f1")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="Author workspace only: extract non-text keys from cached predictions.")
    parser.add_argument("--data-dir", type=Path, default=ROOT/"unified_accuracy_20261010")
    args = parser.parse_args()
    if args.export:
        export(args.data_dir)
    evaluate(args.data_dir)

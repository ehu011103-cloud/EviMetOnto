# Input-matched retrieval audit: local candidate 2026-11-04

The new manuscript Table 3 and Supplementary S7 use `retrieval_controls_20261023/`. Files in `reported_retrieval/` remain the older seven-system report, not an updated table silently disguised under an old name. The author reconfirmed the same-model main diagnostic experiment; its original predictions/logs are still unavailable. Earlier mixed-model caches and the ten-document V4-Pro pilot are not substitutes.

## Offline verification

Install the supplied requirements in an isolated Python environment. From this software directory run:

```sh
python scripts/test_matched_retrieval.py
python scripts/verify_matched_retrieval.py --output audit_check.json
python scripts/verify_manuscript_artifacts.py --output base_artifact_check.json
```

The new verifier reconstructs all twelve systems' ranks, metrics, development tuning, fusion scores and paired cluster intervals from complete stored score vectors. The older verifier checks its documented artifact subset and ontology. Neither performs model inference or independently scores literature texts.

## Full text-scoring rerun with authorized inputs

```sh
python scripts/run_matched_retrieval.py --corpus-dir /authorized/corpus_json --ontology /authorized/final_4D_root_dedup_validated_latest.ttl --out /new/nonexistent/output_directory
```

Use the exact legacy retrieval ontology and corpus identified by the original audit input hashes. The large final strict ontology is a different graph and must not replace the retrieval graph. The output path must not exist. The runner freezes archived queries, rejects mismatched expansions/ranks, reproduces the legacy aggregate, and tunes fusion only on development queries. Input paths have been made configurable; scoring, ranking and statistical functions are unchanged from the 10-23 runner. This portable version is not falsely labelled as the original byte-identical script.

## Artifact meanings and limits

- `manifest.json`: original audit environment, document order, hashes and selected weights.
- `full_scores.jsonl`: all query/system/document scores; ties preserve document order.
- `query_tokens.json`: actual concatenated token inputs with multiplicities.
- `positive_ranks.csv`, `metrics.csv`, `dev_tuning.csv`, `paired_deltas.csv`: twelve systems, both splits, tuning, metrics and descriptive intervals.
- `output_sha256.csv`: original audit output inventory, retained unchanged.

There are 247 queries, 89 development and 158 test; test queries come from 19 source documents. Positives are inherited, not an expert-judged relevance pool. Coarse queries themselves use ontology coarsening. Two thousand paired document-cluster bootstrap draws use seed 20261023; intervals condition on fixed split/weights, exclude tuning uncertainty and are unadjusted for multiplicity. No confirmatory test or preregistration is claimed. Do not tune further on this test set and call that independent validation.

Matched-base nDCG@10 is 0.8646 without expansion and 0.7951 with hierarchy expansion; corresponding fusion gives 0.8363 and 0.8013. These results do not show a hierarchy-expansion benefit and do not measure the resource's semantic correctness or general utility. Text-dependent reruns require lawful source access. The synthetic example is a usage demonstration, not a new experiment.

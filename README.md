# EviMetOnto: Evidence-Constrained Metallogenic Ontology Construction

Version: 2026-10-27.

Repository: https://github.com/ehu011103-cloud/EviMetOnto

This repository supplies available code, derived records and manuscript-table summaries. Original project code and documentation are provided under the MIT license; see LICENSE and NOTICE.md for scope. This is not a claim of complete prediction-level reproduction of every reported experiment, and no archival DOI has been assigned.

Start with `docs/user_guide.md` for installation, input/output descriptions, examples and troubleshooting. `docs/software_metadata.json` records the actual artifact-verification environment, contact and package size. It is not the environment record for the original model experiments.

Run `python scripts/synthetic_example.py` for a small, invented concept-graph walkthrough requiring no model API, source text or downloaded data. Its outputs are not additional experimental results.

## Current Manuscript Map

| Manuscript item | Files | Reproduction scope |
| --- | --- | --- |
| Table 2, shared-model diagnostic accuracy | reported_shared_model/ | Aggregate results; original predictions and logs for this run were not supplied |
| Table 3, source-record-assisted retrieval | reported_retrieval/ | Reported aggregate results, query definitions, split and rank sets for six systems other than the tuned ontology hybrid; tuned-hybrid per-query ranks are not archived |
| Table 4, construction and validation counts | artifacts/ | Derived summaries and corpus manifest |
| Table 5, ontology structure | current_audit/, ontology_release/ | Final exported graph, structural summaries, comparator-file manifests and scoring code; external ontology snapshots must be obtained from their stated repositories |
| Diagnostic Gold | artifacts/gold_v8_relation_labels.csv | Candidate-conditioned final relation labels and canonical keys, without restricted evidence spans |

docs/current_protocol.md defines the current evaluation interpretation. Files under historical_base_release/docs/ retain earlier protocol descriptions, numbering and method-to-script mappings for provenance, not as specifications of current manuscript tables. Historical CHisAgent artifacts do not establish a run-script mapping for the currently reported Text2Onto--LLM row.

historical_cached_accuracy/ contains an executable exact-key audit of earlier cached predictions, not the current shared-model run. These files preserve their original model metadata. current_audit/retrieval_corrected_* belongs to a different historical retrieval audit and is not the source of current Table 3.

## Offline Artifact Verification

Run python scripts/verify_manuscript_artifacts.py --output verification_report.json from this package's root. This makes no model API calls and downloads no external resources. It checks the file inventory, recomputes the released ontology descriptors and six archived retrieval-rank summaries, and checks corpus/Gold inventory and source-group separation.

The runtime in this report belongs to artifact verification, not the original model experiment. Current Table 2 predictions/logs and tuned ontology-hybrid per-query ranks remain unavailable. A passing report is not complete experiment reproduction or evidence of independently judged geological relevance.

## Environment and Use

Python 3 is required. Install requirements.txt in a suitable environment. The API-based construction scripts require an authorized model account and environment configuration. No credentials are distributed. No API call is made by this packaging work.

- python scripts/verify_release.py checks the base release.
- python historical_cached_accuracy/rescore_unified_exact.py --data-dir historical_cached_accuracy re-scores the historical cached prediction keys only.
- current_audit/compare_ontology_extended_metrics.py and current_audit/remove_taxonomy_cycles.py document structural processing; check their recorded input paths and obtain the comparator snapshots before running.
- Construction and adapted-baseline scripts in scripts/ are retained as supplied. They are not a substitute for the unavailable current-run prediction files. Some historical figure-rendering helpers depend on local modules not included here; the six final manuscript figures are supplied separately without regeneration.

This is a separate, curated software release, not the complete development archive. Its MIT license does not grant rights in third-party literature or comparator resources. The two offline examples require no model credentials or restricted source texts.

No claim is made that installing dependencies alone reproduces all manuscript metrics. docs/current_protocol.md and the table above specify what can be checked from the released material.

## Data and Rights

The strict final concept-level ontology contains 23,725 named classes, 35,040 named subclass edges, 12 object properties, 16,371 unique non-hierarchical edges and 98,887 RDF triples. Cycle-processing records and file hashes accompany it.

Third-party PDFs, full structured source texts and verbatim evidence are excluded. Text-dependent construction and retrieval require authorized access to source literature and corresponding structured records. The source-document retrieval label is inherited from a relation record, not independently judged relevance.

Users must obtain any required rights to third-party source material for text-dependent reruns. Repository access does not override those restrictions; see NOTICE.md.

SHA256SUMS.csv and source_snapshot.json record package integrity and the manuscript/figure snapshot used for this delivery.

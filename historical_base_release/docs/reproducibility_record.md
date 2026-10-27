# Reproducibility Record

## Scope

This record distinguishes the full-corpus construction runs from the separate 50-document diagnostic and module-ablation runs. Results from the two settings must not be pooled.

## Corpus and Selection

- Input population: all 5,723 `DOC_A_*.json` records in `corpus/A_full_run_all`.
- Structured fields: title, abstract, body paragraphs, document identifier, and source-file metadata.
- Document-level selection: no additional inclusion, exclusion, or post hoc corpus filter is applied after PDF-to-JSON conversion.
- Source PDFs: third-party literature; they are not redistributed with the derived dataset.
- Release inventory: document identifier, bibliographic metadata, source-file checksum, canonical relation keys, final annotation decisions, aggregate result summaries, and hashes for the released artifacts. Full text and evidence spans are excluded.

## Frozen Run Labels

| Label | Purpose | Script | Model and decoding controls |
| --- | --- | --- | --- |
| `TAX-v1` | Code-style taxonomy parent completion | `step1_json_batch.py` | `deepseek-v4-flash`; temperature 0.0; 12,000-character input cap |
| `RHF-v1` | Evidence-anchored 12-schema relation extraction | `tools/run_4d_relation_extract.py` | `deepseek-v4-flash`; temperature 0.0; thinking disabled; 12,000-character input cap; 15 relations/document; 4,096 output tokens |
| `HVAL-v1` | Direct taxonomy counterexample screening | `tools/taxonomy_full_direct_counterexample.py` | `deepseek-v4-flash`; temperature 0.0; thinking disabled; 256 output tokens |
| `RVAL-v1` | Bidirectional non-hierarchical validation | `tools/validate_4d_bidirectional.py` | `deepseek-v4-flash`; temperature 0.0; thinking disabled; 220 output tokens |
| `DIAG-v1` | Joint extraction for the 50-document diagnostic setting | `tools/preannotate_full_ontology_gold_joint.py` | `deepseek-v4-flash`; temperature 0.0; 10,000-character input cap; 18 hierarchy edges; 28 non-hierarchical relations; 5,400 output tokens |
| `BASELINE-v1` | Eight adapted external baseline workflows | `scripts/run_*_deepseek_final50.py` and `docs/baseline_adaptation_manifest.md` | `deepseek-v4-flash`; temperature 0.0; the same 50 documents; no Gold labels during generation |

No run script explicitly sets `top_p`, random seed, frequency penalty, or presence penalty. JSON-only response formats are requested for structured LLM outputs. The diagnostic run is candidate-conditioned and is reported separately from full-corpus construction statistics.

## Source Snapshot Checksums

The following SHA-256 checksums freeze the current public source snapshot. Paths are relative to `scripts/`.

- `step1_json_batch.py`: `4E98397CF6E3E420314FA4DA1C3FABA4EDCA4D8DA2A53DD13DDA0E0E52157FFE`
- `tools/run_4d_relation_extract.py`: `D14BC50877CD63C52BF1203C7AA09690683F6F597159C10307DA501A68370A14`
- `tools/taxonomy_full_direct_counterexample.py`: `4DE9A67CFE094ECDC27379D29B03D4B0B9FBC7C34B33516D09271B522FAC067B`
- `validate_4d_bidirectional.py`: `CC6F9A21488DD41F88BA9FDB76F77164E40CF320E4D519596E27CE79F22A1605`
- `preannotate_full_ontology_gold_joint.py`: `B1BA4BC92E6B32CF3C69B8116A8955CDBE407794B16920F0707FD78445595999`
- `run_neon_gpt_deepseek_final50.py`: `77C4837B522FD35F46603C5AD9118C06A566216CC36C0A027F5EF854414E7AD0`
- `run_llms4ol_bc_deepseek_final50.py`: `5CE64D423FF3EE06440CDCA91BC7BF4754843846E1A2A8CCF6A178BA9399A6D8`
- `run_pop_duo_deepseek_final50.py`: `1A63E9CC391275E048FA6AC5DC1349A4D45675D75DA610B9949732B5213460E9`

## Artifact Mapping

- Taxonomy construction: `taxonomy_runs/raw_taxonomy_5723_deepseek_v4_flash.py` and `taxonomy_runs/stats/structure_summary.json`.
- Full 4D extraction: `taxonomy_runs/relations_4d/full_raw_4d_relations_v4flash.csv` and `taxonomy_runs/relations_4d/full_raw_4d_relations_v4flash_detailed_stats.json`.
- Full 4D validation: `taxonomy_runs/relations_4d/full_4d_bidirectional_validation_deepseek_v4flash_summary.json`.
- Full taxonomy validation: `taxonomy_runs/validation/full_direct_counterexample_combined_summary_5723.json`.
- Table 3 diagnostic results: `artifacts/table_3_endpoint_compatibility_results_final.csv` and `docs/table_3_endpoint_compatibility_manifest.md`.
- Diagnostic ablations: `docs/unified_v4_three_module_ablation_2026-07-06.md`.
- Baseline adaptation mapping: `docs/baseline_adaptation_manifest.md` and `artifacts/table_3_endpoint_compatibility_results_final.csv`.

## Repository and Archival Actions

- The non-restricted scripts and derived records accompany the revised manuscript in Supplementary Software 1.
- Before submission, deposit any legally redistributable derived data, document manifest, ontology exports, and final result summaries in an archival repository and add its DOI or accession identifier to the manuscript.
- The current release makes no formal retrieval-coverage claim because the pre-conversion collection log is unavailable. See `docs/corpus_provenance.md` for the frozen input boundary and the fields that cannot be reconstructed.

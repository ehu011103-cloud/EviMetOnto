# EviMetOnto: Installation and User Guide

Version: 2026-10-27. Original project code and documentation use the MIT license; third-party source material remains subject to its own rights and access conditions.

## Scope

EviMetOnto separates concept-level taxonomic and relation extraction, evidence screening, canonical integration and evaluation. This package includes construction scripts, derived records, aggregate results and executable checks for the available artifacts. It does not contain third-party source PDFs, full DOC_A texts or verbatim evidence spans.

## Install

Use a Python environment matching the recorded verification environment in `docs/software_metadata.json`. The standalone checks use Python 3, RDFLib and NetworkX. No GPU, model API account or source-literature access is needed for the two commands below. Other Python or operating-system versions have not been validated by this packaging task.

```sh
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or activate it on a POSIX shell:

```sh
. .venv/bin/activate
```

Install the small dependency subset needed by the offline examples:

```sh
python -m pip install rdflib networkx
```

The supplied `requirements.txt` records the broader construction/evaluation dependencies. Installing those dependencies does not supply restricted source texts, model access or unavailable predictions. Do not put API credentials into source files or public configuration files.

## Example 1: Synthetic Input/Output Walkthrough

```sh
python scripts/synthetic_example.py --output synthetic_example_report.json
```

Input is an invented six-class Turtle graph embedded in the script. The hierarchy contains two three-class branches, with child-to-parent subclass edges. A toy non-hierarchical association remains separate from the subclass graph. Outputs are structural descriptors and descendant expansion of the toy `HydrothermalAlteration` concept.

Expected structural output: six classes, four subclass edges, 100% hierarchy participation, zero isolated classes, 50% largest-component coverage and maximum depth two. The expanded concept set is `HydrothermalAlteration` and `Silicification`. The script checks these values internally and exits unsuccessfully if an assertion fails.

This demonstration is synthetic. It is not an additional geological experiment, a semantic-validity test, or evidence for any manuscript performance result. It never calls a language model or downloads data.

## Example 2: Verify Available Manuscript Artifacts

From the package root, run:

```sh
python scripts/verify_manuscript_artifacts.py --output artifact_verification_report.json
```

Required inputs are the packaged `SHA256SUMS.csv`, `ontology_release/`, `current_audit/`, `artifacts/`, `reported_shared_model/` and `reported_retrieval/` records. The JSON output contains individual pass/fail checks and the verification runtime. An overall failure returns a nonzero exit code.

The checker recomputes EviMetOnto graph descriptors and summary retrieval metrics for six systems with archived ranks. It also checks label/corpus inventory, source-group splitting, file hashes and aggregate-score consistency. It makes no model calls and does not retrieve external ontology snapshots.

Do not store output JSON over an input file. Newly generated output files are not original experiment records and are not added automatically to the integrity manifest.

## Full Construction and Evaluation

Refer to `docs/current_protocol.md` before using historical scripts. The current Table 2 shared-model experiment has aggregate scores but no supplied original predictions or run logs. The tuned ontology hybrid in Table 3 has aggregate scores but no archived per-query ranks. Historical model caches and the corrected retrieval audit are separate records, not substitutes for either missing output.

Full text-dependent reruns require authorized literature and structured inputs, an appropriately configured model account, and checking historical script paths before execution. External ontology comparisons require the exact public-resource snapshots and import scope documented in the manifests. The offline checker and synthetic example do not reconstruct these missing inputs or outputs.

## Troubleshooting

- `ModuleNotFoundError`: use the intended environment and install RDFLib/NetworkX for the standalone checks.
- Missing packaged input: run the checker from the complete package root, not from the `scripts/` subdirectory or a source-only ZIP.
- Integrity failure: compare the reported path and digest with the packaged file; do not rewrite the expected digest merely to force a pass.
- Changed ranks or descriptors: preserve the original artifact and report the discrepancy instead of modifying manuscript values automatically.
- API or local-path errors in historical construction scripts: consult their configuration and the current protocol. Those scripts are not invoked by the offline examples.

## Publication and Contact

Software contact: Lijun Dong, `ljdong@cug.edu.cn`.

Public repository: https://github.com/ehu011103-cloud/EviMetOnto . Original third-party full texts, personal files, model credentials and unrelated projects are excluded. See LICENSE and NOTICE.md for license scope; obtain authorized source access before text-dependent reruns.

# Corpus Provenance and Access

## Frozen structured input

The reported full-corpus workflow takes as input all 5,723 `DOC_A_*.json` records in the locally assembled `A_full_run_all` working collection. No additional document-level inclusion, exclusion, or post-conversion corpus filter is applied after PDF-to-JSON conversion. `artifacts/doc_a_manifest_5723.csv` freezes this input population with a document identifier, bibliographic metadata when locally available, page count, source-PDF SHA-256, and structured-record SHA-256.

The manifest contains no abstract, body text, evidence span, local source path, or PDF. It therefore permits input identity checks without redistributing third-party full text.

## Interpretation boundary

This is a local geoscience-literature working collection assembled for ontology construction. It is not released or described as a systematic-review corpus, nor are its document counts used as field-wide prevalence estimates. The manuscript reports construction coverage over this frozen structured input only.

## Access boundary

The source publications must be obtained from their original publishers, archives, or other lawful access routes. The repository does not redistribute PDFs, full-text DOC_A records, or generated artifacts that reproduce verbatim source text.

## Historical collection record

The current conversion record does not preserve the bibliographic databases searched, search strings, search dates, language/date limits, pre-conversion deduplication rules, or PDF acquisition route. The release makes no claim about these unavailable fields and does not infer them retrospectively. Should a later paper make a formal retrieval-coverage claim, the original collection log must be archived and cited separately.

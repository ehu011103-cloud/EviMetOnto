# Gold Reference Protocol

## Scope

The submitted Gold package supports the 50-document diagnostic evaluation. It is a candidate-conditioned, relation-level expert-annotated reference, as confirmed by the authors and described in the revised manuscript. It is intended for controlled workflow diagnosis and ablation, not as a population-representative estimate for all 5,723 source documents.

## Released labels

`artifacts/gold_v8_relation_labels.csv` contains 2,180 reviewed candidate records: 880 hierarchy candidates and 1,300 non-hierarchical candidates. The final positive reference contains 680 hierarchy and 1,015 non-hierarchical relation keys. Each released row retains the document-level sample identifier, canonical endpoints, predicate, final decision, error category, and Gold version; it excludes evidence spans, raw model text, and reviewer notes.

## Matching protocol

Table 2 in the revised manuscript reports the diagnostic and external-baseline comparison using the fixed endpoint-compatibility protocol recorded in the historically named `table_3_endpoint_compatibility_manifest.md`. Predictions and reference keys must share a document identifier and predicate and satisfy the same frozen endpoint-compatibility configuration; non-hierarchical direction is retained except for predefined symmetric predicates. Figure 4 is the separate within-method ablation evaluated with Exact canonical endpoint matching. Historical base-release records may retain the former Table 3 and Figure 6 numbering.

## Annotation provenance

Gold decisions are relation-level expert annotations, as confirmed by the authors. The archive provides canonical relation keys, final decisions, error categories, and version identifiers. The reference remains candidate-conditioned and is not presented as an independently sampled expert-blind benchmark. Detailed annotation personnel records and adjudication logs are not included in this archive. No inter-annotator agreement statistic is claimed. These documentation limits do not change the authors' statement that the reference was expert annotated.

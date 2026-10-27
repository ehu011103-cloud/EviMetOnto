# Current C&G Manuscript Protocol

## Diagnostic Accuracy, Table 2

All nine reported workflows use DeepSeek-V4 Flash according to the authors' confirmed experiment description. The supplied current-run record contains aggregate scores, not original predictions or logs.

All workflows are described under an exact-key protocol: document identifier, predicate and ordered canonical endpoints; normalized mention fallback when canonical fields are absent. Normalization splits CamelCase, lowercases strings, removes underscores, hyphens and non-alphanumeric characters, and standardizes sulphide/sulfide, sulphur/sulfur and dyke/dike. No semantic threshold or alias inventory is used. Endpoint order is retained even for symmetric predicates. Duplicate keys collapse before set intersection.

The expert reference is candidate-conditioned and contains 680 positive hierarchy keys and 1,015 positive non-hierarchical keys. No unprovided expert count, professional profile, blinding procedure or agreement score is asserted.

The earlier endpoint-compatibility manifests under historical_base_release/docs/ are historical and do not override this current exact-key description. Historical prediction caches are not relabeled as the current model run.

## Retrieval, Table 3

There are 100 documents and 247 source-derived queries, split by source-document group into 89 development queries and 158 test queries within a fixed retrieval collection. Each query inherits one source-document positive. The five lexical systems use coarse queries; the two ontology systems additionally receive original fine-grained endpoints for hierarchy expansion. Source identifiers are not ranking terms.

The original untruncated reciprocal-rank field is correctly named MRR, not MRR@10. nDCG@10 and Recall@5/10/20 retain their definitions. The ontology hybrid uses min-max-normalized BM25 score fusion with development-selected alpha 0.5. Aggregate hybrid scores are retained; per-query hybrid ranks were not supplied. Reported results describe a source-record-assisted configuration, not a matched-information isolation of ontology effects.

## Ontology Structure, Table 5

Use the final concept-only export in ontology_release/, not the provisional diagnostic graph or auxiliary V4 inventory. It retains strict VALID assertions and the recorded structural processing. Count named classes, incident named-subclass hierarchy participation, isolated classes, largest weakly connected component coverage and maximum root-to-class depth after component collapse. Apply identical definitions to the loaded comparator snapshots.

These are structural descriptors rather than a composite semantic-quality ranking. Resource-source manifests define the imported scope of each comparator. Supplementary Information S4 and S5 provide provenance and count reconciliation.

## Artifact Checker

scripts/verify_manuscript_artifacts.py recomputes hierarchy descriptors from the released EviMetOnto graph. It recomputes MRR, nDCG@10 and Recall@5/10/20 from archived ranks for six systems other than the tuned ontology hybrid. It checks shared-model score-file shape and rounded arithmetic, but cannot re-evaluate that run without predictions. The checker tests the documented release subset rather than inventing missing records.

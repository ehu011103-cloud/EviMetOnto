# Table 3 Endpoint-Compatibility Manifest

## Scope

This manifest freezes the paper-facing Table 3 comparison: the proposed workflow, eight external hierarchy workflows, and the two external workflows that produce both hierarchy and non-hierarchical relations.

## Common input and generation conditions

- Reference set: the 50-document, candidate-conditioned, relation-level reviewed Gold set, version V8.
- Positive reference keys: 680 hierarchy edges and 1,015 non-hierarchical relations; 1,695 relation keys in the pooled graph.
- Generation model: `deepseek-v4-flash` for the proposed diagnostic workflow and every external baseline adaptation.
- Decoding: temperature 0.0; reference labels are not read during generation.
- Relation schema: 12 non-hierarchical predicates; predicates remain exact and directed except for declared symmetric predicates.

## Fixed endpoint-compatibility protocol

For every Table 3 panel, a predicted relation and a reference relation must share the document identifier and predicate and satisfy the same frozen endpoint-compatibility configuration. Hierarchy comparisons apply the configuration to child and parent endpoints; non-hierarchical comparisons apply it to subject and object endpoints while preserving directed predicates. Pooled graph scores are micro-aggregates over the two relation groups.

This protocol is the paper-facing common comparison rule. It is intentionally distinct from the Exact canonical endpoint rule used only for the Figure 6 internal module ablation.

## Released result artifact

`artifacts/table_3_endpoint_compatibility_results_final.csv` contains every Table 3 row. `docs/baseline_adaptation_manifest.md` maps every external workflow to its released run script. These files replace the earlier Exact-only Table 3 records.

## Boundaries

The package does not redistribute source full text, raw model responses, or verbatim evidence spans. The 50-document reference supports controlled diagnostic comparison; it is not a population-representative, independently sampled benchmark for all 5,723 DOC_A records.

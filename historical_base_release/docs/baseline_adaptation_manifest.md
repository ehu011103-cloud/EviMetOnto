# External Baseline Adaptation Manifest

## Shared execution boundary

All eight external workflows are adapted to the same 50 DOC_A records and run with `deepseek-v4-flash` at temperature 0.0. Gold labels are unavailable during generation. Table 3 compares the generated outputs under the shared endpoint-compatibility protocol in `table_3_endpoint_compatibility_manifest.md`.

| Workflow | Method family | Released run script |
| --- | --- | --- |
| NeOn-GPT | Generic ontology learning | `scripts/run_neon_gpt_deepseek_final50.py` |
| OntoGen | Generic ontology learning | `scripts/run_ontogen_deepseek_final50.py` |
| Ontogenia | CQ-driven ontology generation | `scripts/run_ontogenia_deepseek_final50.py` |
| Memoryless-CQ | CQ-driven ontology generation | `scripts/run_memoryless_cqbycq_deepseek_final50.py` |
| LLMs4OL Task B+C | Task-specific ontology learning | `scripts/run_llms4ol_bc_deepseek_final50.py` |
| POP-Duo | Progressive dual-LLM prompting | `scripts/run_pop_duo_deepseek_final50.py` |
| Planning-First | Planning-first multi-agent generation | `scripts/run_planning_first_multiagent_upperbound.py` |
| CHisAgent | Taxonomy-specific multi-agent construction | `scripts/run_chisagent_deepseek_final50.py` |

## Interpretation boundary

The scripts implement task adaptations, not claims of bit-identical reproduction of the original systems. The paper reports a controlled same-input, same-model, same-evaluation-rule workflow comparison. Prompt/version identifiers and execution controls are recorded in `docs/reproducibility_record.md`.

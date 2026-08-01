# Result Card: uav_polarity_prototype_blind_smoke_v1

- Evidence level: `smoke`
- Seeds: `7`
- Evaluation episodes per seed/tier: `1`
- Primary objective: Evaluate 15-way frozen-embedding polarity prototypes on v7 wording frozen before observing outputs.

## Variants

- `polarity_prototype_cbf`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `nli_prototype_gated_cbf`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `objective_profile_oracle_cbf`: representation=`structured_rule_context`, algorithm=`rule_planner`

## Intent generalization protocol

- Suite: `uav_intent_generalization_v7_polarity_prototype_blind`
- Training intents: `19`
- Queries: seen=`2`, paraphrase=`4`, unseen=`6`
- Query texts are averaged within each seed before cross-seed uncertainty is computed.
- Random-dense and one-hot paraphrase queries receive canonical-label identity as an oracle control.

## Interpretation guardrails

- `legacy_hash` and `random_dense` are representation controls and must not be described as semantic embeddings.
- Paired confidence intervals that include zero do not support a stable directional advantage.
- Safety improvements must be reported together with task completion and resource costs.
- Representation retrieval metrics diagnose geometry and are not behavioral performance evidence.
- Paired variants use the same deterministic environment-reset seed schedule.
- This automatically generated card records protocol facts; paper claims require researcher review.

## Artifact status

- Variant summaries: `3`
- Raw per-seed results: retained under each variant directory
- Checksums: `checksums.sha256`

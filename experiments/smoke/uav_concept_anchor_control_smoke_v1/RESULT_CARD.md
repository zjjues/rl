# Result Card: uav_concept_anchor_control_smoke_v1

- Evidence level: `smoke`
- Seeds: `7`
- Evaluation episodes per seed/tier: `1`
- Primary objective: Test a monotonic objective-concept bottleneck against direct ridge decoding, posture retrieval, and an oracle rule controller.

## Variants

- `concept_anchor_residual`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `dual_ridge_residual`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `posture_retrieval_residual`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `rule_planner_oracle`: representation=`structured_rule_context`, algorithm=`rule_planner`

## Intent generalization protocol

- Suite: `uav_intent_generalization_v1`
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

- Variant summaries: `4`
- Raw per-seed results: retained under each variant directory
- Checksums: `checksums.sha256`

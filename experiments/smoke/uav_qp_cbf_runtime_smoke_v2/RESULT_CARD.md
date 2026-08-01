# Result Card: uav_qp_cbf_runtime_smoke_v2

- Evidence level: `smoke`
- Seeds: `7`
- Evaluation episodes per seed/tier: `5`
- Primary objective: Measure optimized QP-CBF latency against 4- and 32-cycle projections with identical online instrumentation.

## Variants

- `rule_qp_cbf`: representation=`structured_rule_context`, algorithm=`rule_planner`
- `rule_cyclic_4`: representation=`structured_rule_context`, algorithm=`rule_planner`
- `rule_cyclic_32`: representation=`structured_rule_context`, algorithm=`rule_planner`

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

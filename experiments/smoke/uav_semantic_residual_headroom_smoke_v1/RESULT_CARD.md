# Result Card: uav_semantic_residual_headroom_smoke_v1

- Evidence level: `smoke`
- Seeds: `7`
- Evaluation episodes per seed/tier: `3`
- Primary objective: Verify the latent headroom residual parameterization and PPO probability-ratio path.

## Variants

- `objective_semantic_residual`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `nonsemantic_residual`: representation=`none`, algorithm=`mappo`
- `rule_planner`: representation=`structured_rule_context`, algorithm=`rule_planner`

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

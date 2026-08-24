# Result Card: uav_imappo_ablation_calibration_v1

- Evidence level: `pilot`
- Seeds: `7`
- Evaluation episodes per seed/tier: `20`
- Primary objective: Single-seed active-time and pipeline calibration for the registered ten-variant UAV chained ablation; not an effect study.
- Valid scope: `uav`
- Registered primary metrics: `collision_rate, task_completion`

## Variants

- `imappo_full`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `no_mask`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `no_attention`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `no_intent_reward`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `no_cbf`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `no_nli_gate`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `prior_only`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `no_profile_prior`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `identity_oracle`: representation=`onehot`, algorithm=`imappo`
- `no_intent`: representation=`none`, algorithm=`imappo`

## Interpretation guardrails

- `legacy_hash` and `random_dense` are representation controls and must not be described as semantic embeddings.
- Paired confidence intervals that include zero do not support a stable directional advantage.
- Representation retrieval metrics diagnose geometry and are not behavioral performance evidence.
- Paired variants use the same deterministic environment-reset seed schedule.
- This automatically generated card records protocol facts; paper claims require researcher review.
- Safety improvements must be reported together with task completion and resource costs.

## Artifact status

- Variant summaries: `10`
- Raw per-seed results: retained under each variant directory
- Checksums: `checksums.sha256`

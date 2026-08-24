# Result Card: uav_marl_architecture_v3_calibration

- Evidence level: `pilot`
- Seeds: `7`
- Evaluation episodes per seed/tier: `20`
- Primary objective: Single-seed active-time and stability calibration for the corrected six-algorithm UAV architecture protocol; not an effect study.
- Valid scope: `uav`
- Registered primary metrics: `collision_rate, task_completion`

## Variants

- `imappo`: representation=`onehot`, algorithm=`imappo`
- `imappo_no_mask`: representation=`onehot`, algorithm=`imappo`
- `mappo`: representation=`none`, algorithm=`mappo`
- `ippo`: representation=`none`, algorithm=`ippo`
- `happo`: representation=`none`, algorithm=`happo`
- `matd3`: representation=`none`, algorithm=`matd3`

## Interpretation guardrails

- `legacy_hash` and `random_dense` are representation controls and must not be described as semantic embeddings.
- Paired confidence intervals that include zero do not support a stable directional advantage.
- Representation retrieval metrics diagnose geometry and are not behavioral performance evidence.
- Paired variants use the same deterministic environment-reset seed schedule.
- This automatically generated card records protocol facts; paper claims require researcher review.
- Safety improvements must be reported together with task completion and resource costs.

## Artifact status

- Variant summaries: `6`
- Raw per-seed results: retained under each variant directory
- Checksums: `checksums.sha256`

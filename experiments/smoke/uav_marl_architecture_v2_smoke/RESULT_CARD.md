# Result Card: uav_marl_architecture_v2_smoke

- Evidence level: `smoke`
- Seeds: `7`
- Evaluation episodes per seed/tier: `3`
- Primary objective: Pipeline validation for corrected UAV MARL architecture baselines.

## Variants

- `imappo`: representation=`onehot`, algorithm=`imappo`
- `imappo_no_mask`: representation=`onehot`, algorithm=`imappo`
- `mappo`: representation=`none`, algorithm=`mappo`
- `ippo`: representation=`none`, algorithm=`ippo`
- `matd3`: representation=`none`, algorithm=`matd3`

## Interpretation guardrails

- `legacy_hash` and `random_dense` are representation controls and must not be described as semantic embeddings.
- Paired confidence intervals that include zero do not support a stable directional advantage.
- Safety improvements must be reported together with task completion and resource costs.
- Representation retrieval metrics diagnose geometry and are not behavioral performance evidence.
- Paired variants use the same deterministic environment-reset seed schedule.
- This automatically generated card records protocol facts; paper claims require researcher review.

## Artifact status

- Variant summaries: `5`
- Raw per-seed results: retained under each variant directory
- Checksums: `checksums.sha256`

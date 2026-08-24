# Result Card: vmas_navigation_architecture_v1_calibration

- Evidence level: `pilot`
- Seeds: `7`
- Evaluation episodes per seed/tier: `20`
- Primary objective: 100-training-episode runtime and variance calibration for the architecture-only VMAS navigation paper protocol; not an effect study.
- Valid scope: `architecture_only`
- Registered primary metrics: `episode_return`

## Variants

- `imappo_attention`: representation=`none`, algorithm=`imappo`
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
- This study aggregates only the environment-native episode return.
- It cannot support language, preference, UAV safety, or UAV task-completion claims.

## Artifact status

- Variant summaries: `5`
- Raw per-seed results: retained under each variant directory
- Checksums: `checksums.sha256`

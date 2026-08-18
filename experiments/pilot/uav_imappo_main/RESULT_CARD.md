# Result Card: uav_imappo_main

- Evidence level: `pilot`
- Seeds: `7, 11, 23, 42, 100, 256, 512, 1024, 2048, 4096`
- Evaluation episodes per seed/tier: `50`
- Primary objective: Composite resumed study; see objectives for registered run scopes.

## Variants

- `ippo`: representation=`none`, algorithm=`ippo`
- `imappo`: representation=`onehot`, algorithm=`imappo`
- `mappo`: representation=`onehot`, algorithm=`imappo`
- `matd3`: representation=`none`, algorithm=`matd3`

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

# Result Card: uav_imappo_paper_smoke

- Evidence level: `smoke`
- Seeds: `7`
- Evaluation episodes per seed/tier: `3`
- Primary objective: Smoke test: verify I-MAPPO vs MAPPO pipeline before full paper run.

## Variants

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

- Variant summaries: `3`
- Raw per-seed results: retained under each variant directory
- Checksums: `checksums.sha256`

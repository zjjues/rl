# Result Card: vmas_navigation_pilot_v1

- Evidence level: `pilot`
- Seeds: `7, 11, 23, 42, 100`
- Evaluation episodes per seed/tier: `10`
- Primary objective: Estimate five-seed optimizer stability on the public VMAS navigation continuous-control scenario; no language-generalization claim.

## Variants

- `imappo_onehot`: representation=`onehot`, algorithm=`imappo`
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

- Variant summaries: `4`
- Raw per-seed results: retained under each variant directory
- Checksums: `checksums.sha256`

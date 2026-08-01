# Result Card: uav_intent_generalization_feasibility_v1

- Evidence level: `pilot`
- Seeds: `7, 11, 23, 42, 100`
- Evaluation episodes per seed/tier: `5`
- Primary objective: Measure runtime, variance, and early generalization signals before committing to the full 300-episode pilot.

## Variants

- `objective_grounded_semantic`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `pretrained_semantic`: representation=`pretrained_semantic`, algorithm=`imappo`
- `ippo`: representation=`none`, algorithm=`ippo`

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

- Variant summaries: `3`
- Raw per-seed results: retained under each variant directory
- Checksums: `checksums.sha256`

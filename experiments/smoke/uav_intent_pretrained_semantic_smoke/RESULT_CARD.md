# Result Card: uav_intent_pretrained_semantic_smoke

- Evidence level: `smoke`
- Seeds: `7`
- Evaluation episodes per seed/tier: `1`
- Primary objective: Verify that the pinned pretrained semantic encoder can be loaded and used end-to-end in the UAV research pipeline.

## Variants

- `pretrained_semantic`: representation=`pretrained_semantic`, algorithm=`imappo`

## Interpretation guardrails

- `legacy_hash` and `random_dense` are representation controls and must not be described as semantic embeddings.
- Paired confidence intervals that include zero do not support a stable directional advantage.
- Safety improvements must be reported together with task completion and resource costs.
- This automatically generated card records protocol facts; paper claims require researcher review.

## Artifact status

- Variant summaries: `1`
- Raw per-seed results: retained under each variant directory
- Checksums: `checksums.sha256`

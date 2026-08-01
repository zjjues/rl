# Result Card: uav_intent_generalization_smoke

- Evidence level: `smoke`
- Seeds: `7`
- Evaluation episodes per seed/tier: `1`
- Primary objective: Verify leakage-resistant seen, paraphrase, and held-out intent evaluation for semantic and identity-code representations.

## Variants

- `pretrained_semantic`: representation=`pretrained_semantic`, algorithm=`imappo`
- `random_dense`: representation=`random_dense`, algorithm=`imappo`
- `legacy_hash`: representation=`legacy_hash`, algorithm=`imappo`
- `onehot`: representation=`onehot`, algorithm=`imappo`

## Interpretation guardrails

- `legacy_hash` and `random_dense` are representation controls and must not be described as semantic embeddings.
- Paired confidence intervals that include zero do not support a stable directional advantage.
- Safety improvements must be reported together with task completion and resource costs.
- This automatically generated card records protocol facts; paper claims require researcher review.

## Artifact status

- Variant summaries: `4`
- Raw per-seed results: retained under each variant directory
- Checksums: `checksums.sha256`

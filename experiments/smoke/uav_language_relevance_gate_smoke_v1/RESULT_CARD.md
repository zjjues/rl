# Result Card: uav_language_relevance_gate_smoke_v1

- Evidence level: `smoke`
- Seeds: `7`
- Evaluation episodes per seed/tier: `3`
- Primary objective: Pipeline and identity smoke for a frozen-embedding preference relevance abstention gate; not final language evidence.

## Variants

- `imappo_relevance_gated`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `imappo_relevance_ungated`: representation=`objective_grounded_semantic`, algorithm=`imappo`

## Interpretation guardrails

- `legacy_hash` and `random_dense` are representation controls and must not be described as semantic embeddings.
- Paired confidence intervals that include zero do not support a stable directional advantage.
- Safety improvements must be reported together with task completion and resource costs.
- Representation retrieval metrics diagnose geometry and are not behavioral performance evidence.
- Paired variants use the same deterministic environment-reset seed schedule.
- This automatically generated card records protocol facts; paper claims require researcher review.

## Artifact status

- Variant summaries: `2`
- Raw per-seed results: retained under each variant directory
- Checksums: `checksums.sha256`

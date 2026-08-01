# Result Card: uav_nli_semantic_cbf_pilot_v1

- Evidence level: `pilot`
- Seeds: `7, 11, 23, 42, 100`
- Evaluation episodes per seed/tier: `5`
- Primary objective: Confirm multi-seed blind-wording objective grounding and isolate NLI relevance gating and pairwise barrier projection.

## Variants

- `nli_gated_cbf`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `nli_gated_no_filter`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `nli_ungated_no_filter`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `prototype_prior`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `neutral_prior`: representation=`none`, algorithm=`mappo`
- `objective_profile_oracle_cbf`: representation=`structured_rule_context`, algorithm=`rule_planner`

## Intent generalization protocol

- Suite: `uav_intent_generalization_v4_blind_counterfactual`
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

- Variant summaries: `6`
- Raw per-seed results: retained under each variant directory
- Checksums: `checksums.sha256`

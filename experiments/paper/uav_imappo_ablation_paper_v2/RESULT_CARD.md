# Result Card: uav_imappo_ablation_paper_v2

- Evidence level: `paper`
- Seeds: `7, 11, 23, 42, 100, 256, 512, 1024, 2048, 4096`
- Evaluation episodes per seed/tier: `100`
- Primary objective: Pre-registered ten-seed chained causal ablation of semantic representation, NLI prototype gating, objective-profile rule prior, learned residual, attention critic, action masking, intent shaping, and pairwise CBF filtering.
- Valid scope: `uav`
- Registered primary metrics: `collision_rate, task_completion`

## Variants

- `imappo_full`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `no_mask`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `no_attention`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `no_intent_reward`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `no_cbf`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `no_nli_gate`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `prior_only`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `no_profile_prior`: representation=`objective_grounded_semantic`, algorithm=`imappo`
- `identity_oracle`: representation=`onehot`, algorithm=`imappo`
- `no_intent`: representation=`none`, algorithm=`imappo`

## Interpretation guardrails

- `legacy_hash` and `random_dense` are representation controls and must not be described as semantic embeddings.
- Paired confidence intervals that include zero do not support a stable directional advantage.
- Representation retrieval metrics diagnose geometry and are not behavioral performance evidence.
- Paired variants use the same deterministic environment-reset seed schedule.
- This automatically generated card records protocol facts; paper claims require researcher review.
- Safety improvements must be reported together with task completion and resource costs.

## Artifact status

- Variant summaries: `10`
- Raw per-seed results: retained under each variant directory
- Checksums: `checksums.sha256`

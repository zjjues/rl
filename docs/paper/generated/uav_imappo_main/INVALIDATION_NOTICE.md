# Artifact interpretation invalidated

Date: 2026-08-20

The numeric files remain checksummed historical outputs, but the method comparison is invalid. The variant named `mappo` used `algorithm="imappo"`; the requested `critic_mode="concat"` had no implementation branch and executed the attention path. MATD3 ignored that field and IPPO silently replaced it with a local critic.

Do not cite this directory as evidence for MAPPO, attention-versus-concat, or a strong-baseline comparison. The authoritative audit is `docs/paper/audits/uav_imappo_main_semantic_protocol_audit.json`. Corrected protocols are `configs/research/uav_marl_architecture_v2.smoke.json` and `configs/research/uav_marl_architecture_v2.paper.json`.

# Stage 7 Closed-Loop MARL Validation Report

- Status: **not_met**
- Tuning rounds executed: **1**
- Episodes per training run: **3**
- Seeds: **7**
- Output directory: `experiments/stage7_semantic_library`

## Success Criteria

| Metric | Threshold | Final I-MAPPO (Semantic Library) |
| --- | ---: | ---: |
| Medium collision rate | < 0.30 | 0.0000 |
| Hard collision rate | < 0.30 | 1.0000 |
| Medium task completion | >= 0.75 | 0.5814 |
| Hard task completion | >= 0.75 | 0.3903 |
| Step-30 re-planning latency | <= 3.5 | 5.0000 |

## Optimal Hyperparameters

```json
{
  "lambda_3": 1.0,
  "lambda_1": 1.2,
  "eta": 0.5,
  "eta_end": 0.1,
  "eps_clip": 0.1,
  "critic_lr": 0.0003,
  "attention_dim": 128,
  "hard_train_interval": 6,
  "hard_train_spawn_scale": 0.31,
  "hard_train_separation_scale": 0.86,
  "collision_probe_spawn_scale": 0.29,
  "collision_probe_separation_scale": 0.82
}
```

## Notes

- All Stage 7 paths use `experiments/stage7_semantic_library/`.
- The continuous UAV environment cannot directly run the repo's discrete QMIX/VDN learner; the matrix records the compatible shared-value continuous baseline and its limitation in JSON metadata.

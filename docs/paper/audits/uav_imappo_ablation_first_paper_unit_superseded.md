# Superseded first UAV ablation paper unit

The preserved directory `experiments/paper/uav_imappo_ablation_paper_v2_superseded_pre_result_fingerprint_20260824/` contains the first `imappo_full × seed 7` formal run from clean commit `0447ffd604c64d3c9b8d8c2f2140095d6440b092`.

The numerical result and its original checksums completed successfully, but the post-run audit found that completed `result.json` files did not persist the checkpoint implementation SHA-256. A later resume could therefore mix source implementations while retaining only Git commit strings. The fragment is preserved unchanged as failure-analysis evidence and must not be merged into paper statistics. The registered unit is rerun after manifest/result provenance enforcement is committed.

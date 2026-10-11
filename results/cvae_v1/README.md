# CVAE v1: first cost-aware CVAE and its CVAE-FOP closed-loop evaluation (2026-10-11)

Reference point for later CVAE versions and ablations. Git tag `cvae-v1` in this repository and in
the submodule `CVAE_efficient_sampling` (branch `cost-aware`).

## Code

| Part | Version |
|---|---|
| Planners, evaluation scripts, configs | tag `cvae-v1` (branch `Experiment`) |
| CVAE training code | submodule `363389f` (later submodule commits change only Slurm and evaluation) |
| CVAE inference / evaluation code | submodule tag `cvae-v1` (`9d5906d`) |
| C++ build | `ENABLE_DATA_COLLECTION=OFF`, Release |

## Model

- **Location:** frozen copy in `fiss_plus_planner/data/models/cvae_v1/` (not in git; 44 MB).
  `SHA256SUMS` there was read directly from disk.
  - `model_best.pth`: sha256 `2b64ddb93cf2965405159d53e2b95a70fff702e25d59fb637575c9e2f9dfbd04`.
  - Source: `data/output/cvae_local/runs/attn_tau0.25_beta0.1`, trained locally on the RTX 5070 Ti
    on 2026-10-11 between 01:26 and 03:44.
  - `config.json`, `normalizer/normalization_stats.txt`, `log.tsv` and the offline evaluations
    `eval.txt` / `eval_verification.txt` are also in this folder.
- **Training:**
  - Model: attnCVAE (ResNet18, latent 64, 11.3 M parameters). Condition: 3 BEV frames 256x256.
    Target: z = [d, v, T].
  - Cost-aware importance resampling: weights exp(-J~/tau) / q(z) with tau 0.25 and density
    correction, one draw per cycle.
  - KL weight beta: sigmoid annealing up to 0.1. 20 epochs, batch 64, lr 1e-4 (halved after
    epoch 5), seed 0.
- **Data:**
  - CEM R8x250 collection v2 on data/10k/Train: 6808 scenarios, 519,001 cycles, 826 M feasible
    candidates.
  - Training uses 488,838 cycles. Validation: 5 % of the scenarios, 20,000 cycles. Failed
    scenarios: the last 2.5 s before a non-rear-end failure are dropped.
- **Offline result:** Verification set, 20,000 cycles, K 16.

  | Sampler | best_cost@16 | mean_dist | spread |
  |---|---|---|---|
  | CVAE v1 | 0.137 | 0.378 | 0.389 |
  | Perfect CVAE (target) | 0.117 | 0.184 | 0.475 |
  | Marginal | 0.091 | 1.105 | 0.952 |

## Closed-loop evaluation (data/10k/Test)

- **How it ran:** `fiss_plus_planner/scripts/run_test_comparison.sh` on 2026-10-11 between 12:45 and
  13:25. The planners ran one after the other, 22 processes each.
- **Configs:** `cfgs/cvae_fop.yaml`, `cfgs/cem_test.yaml` and `cfgs/fop_test.yaml`.
  - SAFETY: tau 2 s, d0 3 m, margin 0.2 m, grace 1 s, recovery 3 s, clearance_fallback on.
  - Low speed: 4 m/s, 5 m.
  - CEM: R8x250, elite 0.1, smoothing 0.7, init_std 0.3, seed 0. FOP: 10x10x10.
  - CVAE-FOP: K 16, seed 0. Fallback order: samples -> FOP grid -> FOP grid without clearance.
- **Files in this folder:**
  - `report.txt`, `summary.csv`, `paired_costs.csv`: output of `scripts/compare_planners.py`;
  - `measurements/<planner>/`: per-scenario measurement CSVs (input of `compare_planners.py`).
- **Archived only** (`fiss_plus_planner/data/results/cvae_v1/`, not in git, `SHA256SUMS` inside):
  - the same measurements;
  - per-cycle logs `cvae_cycles/<scenario>.csv` of CVAE-FOP (stage, feasible samples, rejection
    reasons, J of samples / grid / plan, chosen z, stage times).

| Planner | Success | J_total (822 solved by all) | vs CEM | Runtime mean / median* |
|---|---|---|---|---|
| CVAE-FOP v1 | 87.6 % | 13.86 | +2.02 ± 0.31 | 22.8 / 11.9 ms |
| CEM R8x250 | 86.1 % | 11.84 | | 123 / 98 ms |
| FOP 10x10x10 | 88.2 % | 14.57 | +2.73 ± 1.03 | 60 / 49 ms |

\* Measured with 22 parallel processes, so the times only show relative size. The time excludes BEV
rendering.

### CVAE-FOP v1

**Cycles:**
- 77.5 % are solved by the samples, 13.2 % by the grid, 9.2 % by the grid without clearance, and
  0.15 % fail.
- 72 % of the 16 samples are feasible.
- In fallback cycles, 92 % of the sample rejections are clearance.

**Cost:**
- The gap to CEM is mostly J_V: 7.92 vs 6.43 (FOP 8.14).
- J_J is 0.82, lower than FOP (1.30).
- Against FOP, the median difference is +0.03 (CVAE-FOP lower in 40 % of the scenarios).
- Even the scenarios without any fallback are +1.17 above CEM.

**Hypotheses for v2:**
- The BEV frames carry no ego-speed cue.
- tau could be smaller.
- A warm start (the previous best z as a sample) is missing.

## Reproduce

```bash
git checkout cvae-v1 && git submodule update --init   # submodule at cvae-v1
cd C_Planner/build && cmake .. -DENABLE_DATA_COLLECTION=OFF && make -j16 && cd ../..
cd fiss_plus_planner/data/models/cvae_v1 && sha256sum -c SHA256SUMS && cd -
bash fiss_plus_planner/scripts/run_test_comparison.sh    # overwrites data/measurements/*_test
```

The planners are deterministic: a rerun gives the same J and success. The runtimes will differ.
Note: the collection PC has a RAM fault (see CLAUDE.md).

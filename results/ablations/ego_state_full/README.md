# Ablation: full ego-state condition (2026-10-11), a negative result

**Question:** Does the CVAE plan speeds better if it also gets the ego state and the task as input,
besides the 3 BEV frames? v1 sees neither the ego speed (the BEV has no speed cue) nor v_des.
Reference: Ichter et al. 2018 condition the sampling distribution on start state, goal and obstacles.

**Answer:** Offline yes, in closed loop no. The model predicts speed better, but in closed loop it
gets stuck at low speed in a minority of scenarios. This is the inertia problem (Codevilla et al.
2019) / causal confusion (de Haan et al. 2019).

Git tag `ablation-ego-full` (this repository and the submodule).

## Design

- **Input:** 8 values per cycle, `cem_dataset.EGO_COLUMNS` (cache columns, from contexts.parquet):

  | Column | Meaning |
  |---|---|
  | `ego_v`, `ego_a` | Cartesian speed and acceleration of the start state |
  | `ego_s_d`, `ego_s_dd` | Frenet longitudinal speed and acceleration |
  | `ego_d`, `ego_d_d`, `ego_d_dd` | Frenet lateral offset and its derivatives |
  | `ego_v_des` | v_des of J_V |

- **Scaling:** `EgoScaler` clips each value to the 0.1-99.9 % range of the training cycles (one
  scenario starts at 258 m/s), then standardises it. The scaler is saved as
  `normalizer/ego_scaler.json` (copy in this folder).
- **Model:** attnCVAE with `ego_dim=8`. A `VectorTokenizer` makes 8 tokens of the vector and appends
  them to the 64 image tokens, so the encoder, prior and decoder attend to both. This adds 2,560
  parameters.
- **Training:** `train_cost_cvae.py --ego --tau 0.25 --beta_end 0.1`, 20 epochs, seed 0, v2 cache
  with the ego columns. Otherwise identical to CVAE v1. Run time 2 h 24 min locally; no batch was
  skipped by the non-finite-loss guard.
- **Planner:** CVAE-FOP passes `[v, a, s_d, s_dd, d, d_d, d_dd, v_des]` of the cycle, taken from the
  same sources as `planning.py cem_cycle_record`. K = 16, warm start on.
- **Frozen model:** `fiss_plus_planner/data/models/ablation_ego_full/` (not in git; SHA256SUMS;
  `model_best.pth` 292d9fc8...925a4).
- **Archive:** raw measurements, per-cycle logs and per-cycle diagnosis in
  `fiss_plus_planner/data/results/ablation_ego_full/`.

## Results

**Offline, Verification set, 20,000 cycles, K 16** (`eval_verification.txt`): best_cost@16 0.155
(v1 0.137), mean_dist 0.359 (v1 0.378), spread 0.401 (target 0.498).

**Closed loop, data/10k/Test, 972 scenarios** (`report.txt`, `measurements/`, `config.yaml`):

| | Success | J_total (817 solved by all) | vs CEM | FOP fallback | J_V |
|---|---|---|---|---|---|
| CVAE v1 | 87.6 % | 13.90 | +2.02 | 22.4 % | 7.96 |
| v1 + warm start | 87.6 % | **13.25** | **+1.38** | 11.6 % | 7.43 |
| full ego + warm start | 87.4 % | 13.85 | +1.98 | 11.5 % | 7.97 |
| CEM R8x250 | 86.1 % | 11.87 | | | 6.46 |

Paired against v1 + warm start (844 scenarios): **+0.58 ± 0.17** (lower in only 31 %); +0.52 of it
comes from J_V.

## Diagnosis

The diagnosis used 3000 Verification cycles; its data is `diagnosis_cycles.csv` in the archive.

1. **The model uses the inputs as the data does.** Linear fits of the mean sample speed:

   | Source | Fit |
   |---|---|
   | CEM target | -0.85 + 0.77 v0 + 0.08 v_des |
   | ego model | -0.67 + 0.73 v0 + 0.08 v_des |
   | v1 | +0.49 + 0.72 v0 + 0.02 v_des (ignores v_des) |

   Changing the input speed by ±3 m/s moves the samples by -1.10 / +1.39 m/s. v_des 8.3 / 13.9 /
   27.8 gives 4.85 / 5.38 / 6.11 m/s.
2. **Offline it is more accurate than v1 in every group** (accelerating, braking, keeping speed,
   v_des ≠ 14, low speed):
   - mean-speed error 0.77 vs 1.12 m/s;
   - closest sample to CEM's best speed 0.65 vs 0.80 m/s;
   - its only bias is -0.14 m/s (slower).
3. **In closed loop the speed gap to the warm-start run grows:**
   - -0.03 m/s at cycle 10, -0.22 m/s at cycle 30 and -0.24 m/s at the end;
   - the ego run is slower in 52 % of the scenarios and faster in 10 %;
   - 85 % of the J gap comes from the worst 10 % of the scenarios. There the vehicle creeps
     slowly to the goal without stopping (e.g. POL_Krakow-13_1: 200 cycles vs 85, J 44.4 vs
     16.2). The other 760 scenarios differ by only +0.10.

**Interpretation:** the speed the model sees is produced by its own previous plans. A slightly slow
plan lowers the next input speed, which lowers the next plan, and so on (feedback loop). v1 cannot
see its speed and does not enter this loop.

**Remedies tested next:**
- B: task-only input (v_des and the lateral state, no ego speed or acceleration);
- A (if needed): ego-speed dropout during training (ChauffeurNet's past-motion dropout,
  Bansal et al. 2019).

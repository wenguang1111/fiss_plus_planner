# Ablation: DAgger round 1 (2026-10-14), a positive result

**Question:** Does the CVAE creep less at low speed when it is also trained on the states it reaches
itself? The diagnosis of v3_img (see "Background") found that the gap to CEM comes mostly from
low-speed phases the CVAE drives into itself and CEM rarely visits (compounding error /
distribution shift). The literature remedy is DAgger (Ross, Gordon & Bagnell, "A Reduction of
Imitation Learning and Structured Prediction to No-Regret Online Learning", AISTATS 2011).

**Answer:** Yes, partly. Against v3_img (same data, settings and seed, plus the DAgger data) the
scenario cost drops by **-0.214 ± 0.111** (Wilcoxon p 4e-4), mostly in J_V. The extra low-speed
cycles over CEM shrink by about 30 %, the gap to CEM from +1.54 to +1.32. Success is unchanged.

## Background

v3_img = CVAE v1 setting retrained on the v3 CEM collection (`runs/v3_attn_tau0.25_beta0.1`). On
Test it is +1.54 J per scenario behind CEM, 73 % of it J_V. Below 1 m/s it spends 10.2 % of its
cycles (CEM 5.7 %); the J_V accumulated there is 47 % of the gap. Offline at CEM's own states the
CVAE is not slower at low speed, so the creeping comes from states the CVAE reaches itself. More
samples (K = 32) barely changed it.

## Design

- **Collection** (`fiss_plus_planner/cfgs/collect_dagger1.yaml`, `scripts/run_collection.sh`, C_Planner
  built with `-DENABLE_DATA_COLLECTION=ON`):
  - the learner CVAE-FOP with v3_img (K 16, warm start) drives every data/10k/Train scenario;
  - the expert, a separate CEM R8x250 with the collection settings, plans in every state the learner
    reaches; its plan is recorded but never executed. It warm-starts from its own previous best;
  - the CEM data (candidates, proposals, contexts best_*, best_trajectory, sampled_vars) are the
    expert's; the BEV images, scenario.json and the trajectory are the learner's run.
  - The correction happens in training: the targets in these states are drawn from the expert's
    feasible candidates as usual, with probability exp(-J~/tau) / q(z).
- **Failed scenarios (rule B):** in the DAgger data the cycles before a learner failure are kept
  (no 2.5 s tail drop, failed scenarios kept; `CEMCache.dagger`, from `dagger_learner` in
  dataset_info). The CEM data keep the 2.5 s rule.
- **Training:** `train_cost_cvae.py --cache cem_train_cache_v3 cem_dagger1_cache --val_cache
  cem_verif_cache_v3 --tau 0.25 --beta_end 0.1` (`CEMCycleConcat`; draws of cache k > 0 use seed +
  1000 k). From scratch, otherwise identical to v3_img: attnCVAE, 20 epochs, batch 64, lr 1e-4, seed
  0, draws_per_cycle 1; validation set Verification v3.
- **Planner:** CVAE-FOP K 16, warm start, dense FOP 10x10x10 fallback; only `CVAE_FOP.run` changed
  (`config.yaml`).

## Data and model

- **Collection** (2026-10-14, 12 processes; 3 of them died silently, the RAM fault of this PC, and
  were rerun with 16 processes; `check_cem_data.py` and a PNG / scenario.json / sampled_vars scan
  are clean): 6808 scenarios, 474,877 cycles, 759.9 M feasible expert candidates, 8.03 M world-model
  trajectories. Settings in `collection_info.json` (`git_dirty: true`: the DAgger code was committed
  after the collection, unchanged).
- **The learner on Train vs CEM on the same scenarios (v3):**

  | | CEM drives (v3) | CVAE drives (DAgger 1) |
  |---|---|---|
  | Cycles | 439,918 | 474,877 (+8 %) |
  | Reached the goal | 4930 | 4501 |
  | Time limit | 1048 | 1557 |
  | Failed / rear-end | 824 / 385 | 746 / 354 |

- **Training:** 909,499 training cycles (435,270 v3 + 474,229 DAgger), 20,000 validation cycles, ESS
  median 126.5; 4 h 22 min on the RTX 5070 Ti; no batch skipped; model_best = epoch 20 (`log.tsv`).
- **Frozen model:** `fiss_plus_planner/data/models/dagger1/` (not in git; SHA256SUMS read with
  O_DIRECT; `model_best.pth` 86e3c9aa...daad3a, equal to the hash written at training time).
- **Archive:** raw measurements, per-cycle logs (`cvae_cycles`) and the low-speed analyses in
  `fiss_plus_planner/data/results/dagger1/` (not in git, SHA256SUMS). Collection in
  `data/output/dagger1_train`, cache `data/output/cvae_local/cem_dagger1_cache`.

## Results

**Offline, Verification v3, 20,000 cycles, K 16** (`eval_verification.txt`):

| | in_bounds | best_cost@16 | mean_dist | spread |
|---|---|---|---|---|
| v3_img | 0.935 | 0.182 | 0.376 | 0.407 |
| dagger1 | 0.933 | 0.176 | 0.381 | 0.405 |
| target | 1.000 | 0.122 | 0.190 | 0.488 |

Equal, as expected: the validation states are CEM's, and DAgger adds the learner's. It shows that
the DAgger data do not hurt at CEM's states.

**Closed loop, data/10k/Test, 972 scenarios** (2026-10-14, 12 processes; `report.txt`,
`measurements/`, `config.yaml`). J_total on the 846 scenarios solved by all planners:

| | Success | Reached | J_total | vs CEM | J_V | FOP fallback |
|---|---|---|---|---|---|---|
| v3_img | 89.5 % | 68.3 % | 11.16 | +1.54 ± 0.23 | 6.17 | 11.8 % |
| **dagger1** | 89.6 % | **70.6 %** | **10.94** | **+1.32 ± 0.19** | 5.99 | 11.6 % |
| CVAE_warm (v1 + warm start) | 89.6 % | 71.0 % | 10.86 | +1.24 ± 0.21 | 5.95 | 12.0 % |
| CEM R8x250 | 88.6 % | 74.0 % | 9.62 | | 5.05 | |
| FISS+ | 89.0 % | 73.9 % | 9.84 | +0.23 ± 0.10 | 5.20 | |
| FOP 10x10x10 | 89.7 % | 66.5 % | 12.27 | +2.65 ± 1.02 | 6.75 | |

**Paired against v3_img** (`report_vs_v3_img.txt`, `paired_costs_vs_v3_img.csv`):

| Subset | n | ΔJ (95 % CI) | Wilcoxon p | lower in |
|---|---|---|---|---|
| solved by both | 869 | **-0.214 ± 0.111** | 4e-4 | 48 % (higher 40 %) |
| both reached the goal | 660 | -0.140 ± 0.126 | 0.11 | 44 % |
| both ended the same way | 841 | -0.139 ± 0.104 | 0.01 | 47 % |

- J_V gives -0.17 of the -0.21.
- Endings v3_img -> dagger1: 24 scenarios time limit -> goal, 4 goal -> time limit, 2 failed -> goal,
  1 time limit -> failed.
- Against CVAE_warm: +0.08 ± 0.12 (equal). DAgger recovers what v3_img lost against v1 + warm start
  (+0.29); v3 Train has a third fewer cycles below 1 m/s than v2, which the DAgger data add back.

**Low speed** (`scripts/lowspeed_gap.py`; `lowspeed_gap_dagger1.txt`, `lowspeed_gap_v3_img.txt`;
scenarios solved by the CVAE and CEM):

| | v3_img | dagger1 | CEM |
|---|---|---|---|
| Cycles below 1 m/s | 10.2 % | 8.9 % | 5.7 % |
| Low-speed J_V above CEM | 620 | 446 (-28 %) | |
| Share of the J gap | 47 % | 40 % | |
| Low-speed cycles not held back by clearance | 80 % | 75 % | |

- Fixed: HRV_Centar-51 (goal right behind a standing car; 180 -> 121 cycles, J 38.8 -> 24.1),
  GRC_NeaSmyrni-76 / -133, DEU_Herzogenaurach-120, DEU_Bremen-7.
- New regression: USA_Phoenix-75_10 (+20.5; 50 -> 75 cycles, now creeps; CEM never drives below
  1 m/s there).
- Still on top: POL_Krakow-62, GRC_NeaSmyrni-73, DEU_Bonn-11, where the CVAE is held back by clearance
  and CEM drives in its clearance fallback (no clearance). DAgger cannot change that.

## Caveats

- One training seed per model; the seed noise of the closed loop is not measured.
- dagger1 trains on twice the items of v3_img (twice the updates). Part of the gain could come from
  more data / longer training rather than from the learner's states. Control: v3 alone with
  `--draws_per_cycle 2` (same number of updates, no new states).
- Runtimes come from 12 parallel processes and are not timing results.

## Next

- the control above and a second seed;
- DAgger round 2 (learner = dagger1; train on v3 + dagger1 + dagger2) for the remaining ~70 % of the
  low-speed excess;
- the clearance-fallback part of the gap (16 %) needs a rule decision, not more data.

Settings of all Test numbers: SAFETY tau 2 s, d0 3 m, lateral margin 0.2 m, grace 1 s, T_rec 3 s,
clearance_fallback on; CEM R8x250; CVAE-FOP K 16 + warm start; FOP 10x10x10.

# Experiments: Sparse Planner paper re-measurement and FOP-based learning (2026-11-01 .. 2026-11-06)

This branch (`sp-paper-experiments`, from `develop` @ `a585cac`) records the code, the measured results and the notes of
the experiments for the revised Sparse Planner paper (IROS'27 draft, "Sparse Planner: A Hybrid Planner for Efficient
Sampling via a Conditional Variational Autoencoder"). All numbers below were measured on this branch's code; every
number is backed by a folder under `fiss_plus_planner/data/measurements/` that is committed here.

**What is committed:** code changes, run / summary scripts, paper drafts, all result folders of these experiments (CSV,
logs, configs, `.complete` markers), training logs / configs of the Gaussian / MoG samplers.
**What stays local only** (too large, see [Local-only data](#local-only-data)): scenario sets, the SP training data and
its image cache, model weights / TensorRT engines, the Gaussian / MoG checkpoints.

Paths below are relative to `fiss_plus_planner/` (the scripts are run from there or from the repo root as shown).

---

## 1. Setup

| Item | Value |
|---|---|
| Machine | local PC `dsc-cuda-01`, Intel Core Ultra 9 285K (24 cores), RTX 5070 Ti |
| Python | conda env `world-model` (`~/miniconda3/envs/world-model/bin/python`), default `PY` of all scripts |
| C++ build | CMakeLists hardcode a dead `pybind11_DIR`: configure with the `world-model` prefix; rebuild after branch switches |
| Scenario sets | `data/demo/UsedForSelfEvaluation` (26 scenarios, Table I/II), `data/demo/Test` (250 scenarios, clean test set) |
| Cost | paper J = per-step mean cost + `w_T * t_f`; `PLANNING_OBSTACLE_COST: False` (no J_D in the planning cost, as the paper) |
| Clearance | off unless stated (`SAFETY` in `cfgs/demo_config.yaml`: tau 2 s, d0 3 m, margin 0.2 m, grace 1 s, T_rec 3 s, fallback) |
| CEM / MPPI | elite 0.1 / smoothing 0.7, temperature 0.1 / lr 0.7, init_std 0.3, seed 0, warm start; Table I: 8 it x N/8; Table II: clamp(N/8, 1, 8) iterations |
| SP | `SP_CVAE: paper` (paper CVAE weights + paper statistics), N CVAE samples, 5x5x5 FOP fallback; SP* = SP without fallback |

**Definitions**

- *Failure* (26-scenario tables): planning failure or standstill (v < 0.01, `planning.py` default). Step-0 failures write
  no CSV row; `table_ii.py` counts missing rows as failures.
- *Failure* (Test 250, user's definition): only "no feasible trajectory"; reaching the scenario end (time / map end) is OK.
  Classify from `log.txt` ("Planning failed at time step k"). `STOP_ON_STANDSTILL=0` keeps simulating at v < 0.01.
- *t_warm*: mean per-cycle planning time without each scenario's first cycle. Only valid for sequential runs (one
  process, pinned CPU); runs marked *parallel* have invalid runtimes.
- *p_B / P_fb*: share of cycles in which SP needed the FOP fallback.
- *J_common*: J averaged over the scenarios all compared planners complete (per column).

**Main run script:** `scripts/run_paper_table.sh` (header lists all env switches: `PLANNERS NS SP_NS CLEARANCE PARALLEL
THREADS OUT CPUS SP_IMPL SP_CVAE_BACKEND SP_FALLBACK SP_SAMPLER SP_SHADOW ITER_SPLIT STOP_ON_STANDSTILL SP_GAUSSIAN_WEIGHTS`).
Runs are resumable (`.complete` marker per run folder; delete the folder to re-measure). Each result folder has
`<planner>_<N>/{measurement_<planner>.csv, config.yaml, log.txt}` and `machine.txt` (CPU, GPU, commit).

---

## 2. Code changes in this branch (relative to a585cac)

- **C++ / Python alignment** of FOP, FISS+, CEM, MPPI (spline transpose bug, `arange` grid, stale grid speed, J_D switch);
  C++ numpy binding `best_traj_generation_array` for evaluating externally given samples.
- **Clearance** constraint ported to all planners (`SAFETY` block, clearance fallback).
- `planners/sparse_planning/cvae_paper.py`: paper CVAE (encode once, paper statistics), backends `eager`, `graph_fp32`,
  `graph_fp16` (CUDA graph, `CVAEGraphSampler`), `trt_fp16` (TensorRT, `CVAETRTSampler`, engines cached in
  `data/models/trt_cache`).
- `planners/sparse_planner_cpp.py`: `Sparse_CPP` = CVAE samples evaluated by the C++ FOP + C++ FOP 5x5x5 fallback;
  `SP_SAMPLER` cvae / marginal / uniform / mean; `SP_FALLBACK`; `SP_SHADOW` same-state evaluation; `torch.set_num_threads(1)`.
- `planners/sparse_planning/cond_gaussian.py` + `planners/gaussian_planner_cpp.py`: conditional Gaussian (K=1) / MoG (K=5)
  sampler (`Gaussian_CPP`), backends eager / graph / `trt_fp16` (`GaussianTRTSampler`).
- `planners/benchmark/planning.py`: `STOP_ON_STANDSTILL` switch; traceback printed before "<file> not feasible!".
- Scripts: `run_paper_table.sh`, `paper_table.py`, `run_table_ii.sh`, `table_ii.py`, `plot_table_ii.py`,
  `compare_cvae_backend.py`, `sp_ablation.py`, `run_sp_ablation_fallback.sh`, `sp_ablation_fallback.py`,
  `build_sp_image_cache.py`, `train_sp_sampler.py`, `run_sp_sampler_training.sh`, `select_sp_checkpoint.py`,
  `run_gaussian_test250.sh`, `summarize_test250_sp.py`.
- Paper drafts: `paper/evaluation.tex`, `paper/ablation_sampling.tex`, `paper/related_work.tex`,
  `paper/sparse_planner_zh.md` (Chinese full draft), `paper/images/fail_vs_runtime.pdf`.

---

## 3. Experiment index

| # | Experiment | Folder (`data/measurements/`) | Script | Runtime valid |
|---|---|---|---|---|
| E1 | Table I re-measurement, no clearance | `paper_table/` | `run_paper_table.sh` | yes |
| E2 | Table I with clearance | `paper_table_clearance/` | `CLEARANCE=1 run_paper_table.sh` | **no** (parallel, THREADS=8) |
| E3 | SP without fallback (Python draws) | `paper_table_sp_nofallback/` | `SP_FALLBACK=0 ...` | no (4 parallel) |
| E4 | Runtime vs threads, SP_CPP (Leipzig) | `runtime_threads_leipzig/` | `rt_table.py` inside | yes |
| E5 | FOP large grids / lane centre | `fop_cpp_large_fop3/`, `fop_cpp_lane_centre/`, `fop_cpp_14000/` | `run_paper_table.sh` GRID | yes |
| E6 | Learned-distribution ablation, no fallback + same-state shadow eval | `sp_ablation/` | `sp_ablation.py` | no |
| E7 | Table II: performance under N (all C++ + SP_CPP graph_fp16) | `table_ii/` | `run_table_ii.sh`, `table_ii.py` | yes |
| E8 | Table II with TensorRT fp16 CVAE | `table_ii_trt/` | same, `SP_CVAE_BACKEND=trt_fp16` | yes |
| E9 | Sampler ablation with fallback | `sp_ablation_fallback/` | `run_sp_ablation_fallback.sh` | yes |
| E10 | Test 250 FOP / CEM N-sweep | `test250/`, `test250_nostop/{grid,iter,fop14000_chunks}` | `run_paper_table.sh` | no |
| E11 | Test 250 SP / SP* (paper CVAE) | `test250_nostop/{sp,cvae}` | `run_paper_table.sh` | no |
| E12 | Gaussian / MoG training | `data/output/sp_sampler/runs/` (logs only) | `run_sp_sampler_training.sh` | - |
| E13 | Gaussian / MoG, 26 scenarios, no fallback | `selfeval_gauss/` | `run_gaussian_test250.sh INPUT=...` | yes |
| E14 | Gaussian / MoG, Test 250, no fallback (clean) | `test250_gauss_nofb/` | `run_gaussian_test250.sh` (4 parallel) | no |
| E15 | Gaussian / MoG TRT runtime, SP with fallback | `selfeval_gauss_trt/` | `BACKEND=trt_fp16 ...` | yes |
| - | superseded / invalid | `paper_table_sp_stats_b382f22/`, `paper_table_sp_expand/`, `test250_nostop/gauss/` | | |

---

## 4. Results

### E1 Table I, no clearance (`paper_table/`)

Python planners, 26 scenarios, 16 numba threads, sequential. Table: `scripts/paper_table.py` -> `table_paper_layout.csv`,
`table_long.csv`.

| N | 125 | 216 | 512 | 1000 |
|---|---|---|---|---|
| Success /26: FOP / FISS+ / CEM / MPPI | 23/17/15/15 | 20/17/15/16 | 19/17/16/15 | 17/16/15/15 |
| J_all: FOP | 72.38 | 72.93 | 72.27 | 72.20 |
| J_all: FISS+ | 72.18 | 71.81 | 71.66 | 71.18 |
| J_all: CEM | 69.82 | 71.25 | 70.88 | 70.66 |
| J_all: MPPI | 71.41 | 71.16 | 70.65 | 70.87 |
| t_warm ms: FOP / FISS+ / CEM / MPPI | 44.7/19.4/48.8/47.7 | 78.1/34.3/81.7/82.6 | 184.3/58.1/192.1/189.7 | 357.6/113.0/384.2/377.9 |

- J_common over the 15 scenarios all four complete: ~41.0 for all. FOP / FISS+ J reproduce the old paper CSVs exactly.
- Failures are nested in N. FOP 125's 3 failures: AUT_Haag-15 (no plan), DEU_Weimar-84, ESP_Barcelona-48 (standstill).
- **SP** (N 1/8/16/64/128/256/512/1000): 26/26 at every N; J_all 72.23/71.54/71.40/71.13/70.98/70.82/70.59/70.45
  (paper 72.50 .. 70.33); t_warm 12.8/17.7/20.4/39.3/59.2/102.5/190.4/354.4 ms; p_B 3.43/2.63/2.69/2.76/2.93/3.04/3.21/3.23 %
  (the paper's 17.4..9.3 % came from a never-reset counter bug).

**Reading:** without clearance, failures grow with N for every baseline (more per-cycle optimisation of J -> closer to
queues -> dead end; recursive feasibility / inevitable collision states).

### E2 Table I with clearance (`paper_table_clearance/`, runtimes invalid)

N_fail % (125/216/512/1000): FOP 15.38/0/3.85/3.85 (all standstills), CEM 3.85/3.85/3.85/0, MPPI 0. clear_fb <= 3.7 % of
cycles. J_common (22 sc.): FOP 63.57/64.02/62.01/62.87, CEM 61.95/61.80/61.74/61.67, MPPI 61.80/61.78/61.72/61.70.
Comparison with E1: `compare_clearance.csv`. With clearance the rise with N disappears and FOP no longer beats CEM / MPPI.
SP with clearance falls back in ~62-100 % of cycles (3-scenario check; CVAE samples do not keep the gap) -> not run.

### E3 SP without fallback (`paper_table_sp_nofallback/`)

Success N 1/8/16/64/128/256/512/1000 = 19/21/21/20/20/20/20/19 of 26; all failures = no feasible CVAE sample. A scenario
fails exactly when the Table I SP run needed >= 1 fallback. 5 fail at every N (DEU_Nuremberg-24_6, DEU_Heilbronn-287_1,
BEL_Antwerp-10_4, DEU_Heilbronn-140_1, DEU_Kiel-2_58), all solved by FOP 125.

### E4 Runtime vs threads (`runtime_threads_leipzig/`, see `summary.txt`)

SP_CPP (CUDA graph fp16 CVAE + C++ FOP evaluation + C++ fallback), Leipzig, t_warm ms N 1/8/16/64/128/256/512/1000:
1 core 0.78/0.83/0.90/1.39/1.97/3.18/5.67/10.62; 16 cores 0.88/1.04/1.19/1.47/1.81/2.42/3.68/5.51. eager SP_CPP ==
Python Sparse_FOP to 1e-14. `torch.set_num_threads(1)` is required (OpenMP spinning stole the C++ threads' cores).

### E5 FOP grids (`fop_cpp_large_fop3/`, `fop_cpp_lane_centre/`, `fop_cpp_14000/`)

- FOP's 3 failures persist at N = 10648 (22^3), failing even earlier. FOP_CPP 1000 == Python FOP 1000 bit-identical.
- Lane centre d = 0 is in `linspace(-0.828, 0.828, N_W)` only for odd N_W: 6x6x6 20/26 vs 7x6x5 (210) 23/26; 8x8x8 19 vs
  9x8x7 (504) 19; 10x10x10 17 vs 11x10x9 (990) 17. Table II therefore uses odd-N_W grids.

### E6 Learned-distribution ablation, no fallback (`sp_ablation/`)

`SP_SAMPLER` cvae / marginal (rows of the CVAE training targets) / uniform (FOP sampling box) / mean.
Closed loop success /26, N 1/8/64: cvae 19/21/20, marginal 4/10/19, uniform 5/12/15, mean (N 1) 8.
Same states (`SP_SHADOW`, `shadow_eval*/`), N = 1 sample feasible: cvae 96.6 %, marginal 73.8, uniform 77.8, mean 71.6
(grid 77 %). Regret of the best sample vs the 10^3 grid (mean/median), N 1/8/64: cvae 15.6/2.9, 14.6/2.3, 13.9/1.8 (does
not fall with N); marginal 88/29, 10.2/2.3, 2.1/0.27; uniform 60/29, 8.8/3.3, 1.7/0.64.

Completed 2026-11-06 for all N (`<sampler>_nofallback/`), SP* success /26, N 1/8/16/64/128/256/512/1024:
cvae 18/21/21/20/20/20/20/19; uniform 5/12/14/15/16/16/17/15; marginal 4/10/13/19/18/22/22/23 -> **marginal beats the
CVAE without fallback at N >= 256** (CVAE nearly deterministic: more samples do not help when its region is infeasible).

Train/test overlap: 6 of the 26 scenarios are in the CVAE training set, 11 have an unseen map. Unseen maps: N=1
feasibility cvae 97.9 % vs marginal 67.7 / uniform 69.8 / mean 60.1; closed loop SP* cvae 9/10/10 (N 1/8/64) vs marginal
1/3/6, uniform 2/3/5, FOP 125/216/1000 8/7/6.

### E7 / E8 Table II: performance under N (`table_ii/`, `table_ii_trt/`)

All C++ planners + SP_CPP, sequential, CPU 2, 1 thread, no clearance, J_D off. Grids (odd N_W) 12/18/60/125/252/504/990
for columns 8..1024. Table: `scripts/table_ii.py` -> `table_ii.tex`, `table_ii_long.csv`; figure `scripts/plot_table_ii.py`.

| N | 1 | 8 | 16 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| N_fail %: FOP | - | 26.9 | 11.5 | 11.5 | 11.5 | 19.2 | 26.9 | 34.6 |
| FISS+ | - | 11.5 | 3.8 | 19.2 | 34.6 | 34.6 | 38.5 | 38.5 |
| CEM | 61.5 | 26.9 | 30.8 | 34.6 | 42.3 | 42.3 | 42.3 | 42.3 |
| MPPI | 61.5 | 26.9 | 26.9 | 42.3 | 42.3 | 42.3 | 42.3 | 42.3 |
| CVAE (SP*, no fallback) | 30.8 | 19.2 | 19.2 | 23.1 | 23.1 | 23.1 | 23.1 | 23.1 |
| SP | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| t ms SP graph_fp16 (E7) | 0.84 | 0.97 | 1.11 | 1.97 | 3.07 | 5.29 | 9.79 | 19.16 |
| t ms SP trt_fp16 (E8) | 0.48 | 0.58 | 0.76 | 1.58 | 2.69 | 4.86 | 9.35 | 18.58 |
| p_B % SP | 3.1 | 2.6 | 2.4 | 2.7 | 2.7 | 2.9 | 3.0 | 3.0 |

- J over per-column common scenarios: CVAE = SP (common scenarios never need the fallback); small N CVAE competitive
  (N=1 43.68 vs CEM 44.12), N >= 64 CVAE ~3 % worse than the baselines (~43.0-43.6 vs ~41.7-42.3).
- Baseline t ms: FOP (12..990) 0.23..16.16, FISS+ 0.16..2.74, CEM 0.04..20.34.
- TRT vs graph (`table_ii_trt/compare_backend.csv`): J_all diff -0.02..+0.17, same p_B, CVAE failure sets identical except
  N=1024. TRT fp16 forward GPU ms N 1/64/1024: 0.175/0.204/0.551 (torch graph fp16 0.65/0.70/1.25).

### E9 Sampler ablation with fallback (`sp_ablation_fallback/`, table `sp_ablation_fallback.csv`)

Completed /26, N 1/8/16/64/128/256/512/1024: CVAE 26 everywhere; uniform 26/26/26/26/25/23/19/17; marginal
24/26/26/25/25/25/23/24; mean 26.
p_B %: CVAE 3.2/2.7/2.5/2.4/2.4/2.7/3.0/3.1; uniform 28.0/9.3/5.4/4.5/3.3/3.1/2.3/1.6; marginal 24.9/6.5/3.1/0.7/1.1/0.3/0.1/0.05.
J_all: CVAE 72.55..70.46 (lowest except uniform N=8), uniform 89.5/71.5/71.5/71.9/71.5/73.7/73.3/72.3, marginal
112.1/74.1/73.1/74.0/74.7/73.7/75.1/73.4.
**Reading:** the CVAE is the only sampler that needs little fallback at small N and never dead-ends at large N.

### E10 Test 250 FOP / CEM N-sweep (`test250/`, `test250_nostop/`)

C++ planners, Table I settings, parallel (runtime invalid). `test250/` excludes the 5 v0 = 0 scenarios (base 245);
`test250_nostop/` uses `STOP_ON_STANDSTILL=0` and all 250.
No-feasible rate (of 250, nostop): FOP 12/18/60/125/252/504/990/14000 5.2/1.2/0.8/6.8/10.8/12.0/18.4/29.2 %;
CEM 1/8/16/64/128/256/512/1024 55.2/19.2/18.8/22.4/30.4/30.8/31.6/31.2 %. Paired: FOP 60 vs 990 38 vs 3 (p 1e-8),
CEM 16 vs 1024 34 vs 3 (p 1e-7). Large-N failures happen late (median 90 % of the scenario). Same trend as Table I.
Summaries: `summary_no_feasible.csv`, `summary.csv`, `outcomes_<planner>_<N>.csv`, `cost_summary.csv`.

Test 250 is a clean test set: the 622 SP training scenarios all come from the old `selected_scenarios/Verification`.

### E11 Test 250 SP / SP* (`test250_nostop/{sp,cvae}`, `sp_summary.csv`)

Sparse_CPP trt_fp16, fallback 5x5x5, 16 parallel. No-feasible of 250, N 1..1024: SP 2/2/3/3/3/3/2/2 (FRA_Dijon-24_2 and
USA_Austin-57_27 at every N, also failed by FOP 125/990 and CEM 16); SP* 68/57/54/53/56/54/55/54 (27 % -> ~22 %, flat, no
rise with N). P_fb 3.89/3.35/3.35/3.55/3.60/3.51/3.17/3.13 %.
Cost, paired on scenarios SP, CEM and FOP all complete: SP vs CEM +10.7..+18.7 %, SP vs FOP +12..+18 % (only -3.9 % vs the
12-point grid at N=8) -> SP is **not** cheaper on Test 250; its small-N advantage is the pass rate.

### E12 Gaussian / MoG sampler training (`data/output/sp_sampler/runs/<model>/`)

`scripts/train_sp_sampler.py --model gaussian|mog`, data `data/sp_training_2026-02-15` (622 scenarios, 58,427 cycles,
image cache `cache_256/`), AdamW 1e-4, 20 epochs, batch 64, scenario split val 10 % (split_seed 0), seeds 0/1/2.
Committed per run: `config.json`, `split.json`, `training_log.tsv` (complete), `train.log` (truncated for gaussian_s0/s1,
mog_s0), `selected.json`. Checkpoint choice (user): epoch with min `best_dist@16`
(`scripts/select_sp_checkpoint.py`): gaussian ep 7/7/5 (bd16 0.45-0.46, spread 0.26-0.35, out-of-range 35-39 %), mog ep
20/14/15 (bd16 0.41-0.42, spread 0.50-0.58). Paper CVAE on the same split: bd16 0.785, spread 0.07, out-of-range 18.7 %.
The single Gaussian's val NLL diverges after epoch 1 (std collapses on the main mode); MoG stays stable.

### E13 Gaussian / MoG, 26 scenarios, no fallback (`selfeval_gauss/`, `compare_cvae_per_seed.csv`)

Failed /26, N 1..1024: CVAE 8/5/5/6/6/6/6/7; Gaussian (mean of 3 seeds) 18/12.7/11/9/8/7.7/6.7/6.3; MoG
17.7/10/8/7.3/6.7/5.7/3.7/2.7. **User verdict: not usable for the paper** (too few scenarios, cost difference not
significant).

### E14 Gaussian / MoG, Test 250, no fallback (`test250_gauss_nofb/`)

Clean run (4 parallel, graph_fp16; all tracebacks are step-0 "No ego vehicle trajectory" = valid failures).
Summary: `scripts/summarize_test250_sp.py --root <dir with symlinks cvae, sp -> test250_nostop/{cvae,sp}, gauss -> test250_gauss_nofb>`.

| Fail %, N | 1 | 8 | 16 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| CVAE | 27.2 | 22.8 | 21.6 | 21.2 | 22.4 | 21.6 | 22.0 | 21.6 |
| Gaussian (3 seeds) | 62.9 | 37.1 | 32.4 | 25.7 | 24.3 | 22.8 | 22.1 | 20.4 |
| MoG (3 seeds) | 62.1 | 34.8 | 30.1 | 23.7 | 20.1 | 15.3 | 12.1 | 10.4 |

Paired dJ (model - CVAE, CVAE J ~57-68): Gaussian +7.4 at N=1, -3.1..-4.0 at N >= 8; MoG +1.0 at N=1, then
-3.9/-4.2/-5.9/-6.8/-6.4/-6.0/-6.0 (win rate ~0.70). **Reading:** the CVAE is better only at small N (pass rate N <= 64,
cost N = 1); the MoG is better in pass rate and cost at N >= 128.

### E15 Gaussian / MoG TRT runtime, SP with fallback (`selfeval_gauss_trt/`, seed 0 only)

Sequential, CPU 2, 1 thread (= table_ii_trt). t_warm ms N 1..1024: CVAE 0.48/0.58/0.76/1.58/2.69/4.86/9.35/18.58,
Gaussian 0.77/0.75/0.85/1.58/2.62/4.71/8.97/17.71, MoG 0.70/0.65/0.78/1.52/2.55/4.69/9.03/17.60. P_fb %: CVAE 3.2..2.4..3.0,
Gaussian 16.6/10.9/9.1/7.8/7.6/7.2/5.0/7.7, MoG 14.1/6.8/6.0/3.4/2.0/1.2/0.7/1.4. Forward GPU ms N 1/64/1024: CVAE trt
0.20/0.24/0.60, Gaussian 0.20/0.21/0.21, MoG 0.21/0.21/0.22.

### Superseded / invalid runs (kept for the record)

- `paper_table_sp_stats_b382f22/`: SP with the submodule CVAE statistics (wrong statistics, p_B ~30 %).
- `paper_table_sp_expand/`: SP encoding N image copies: J valid, runtime too high at large N.
- `test250_nostop/gauss/`: first Gaussian / MoG Test 250 run with 16 GPU processes: only 21-215 of 250 rows per run
  (scenarios swallowed by GPU errors in the parallel run) -> **invalid**, replaced by E14.

---

## 5. Paper decisions taken (2026-11-06)

- No multimodality claim for the CVAE (spread 0.07, top-1 labels, no basin-coverage evaluation).
- Supported claims: scene-conditioned feasible-region prior at small N (E6, E9, unseen maps), one forward pass for any N /
  runtime (E7, E8), no failure rise with N (E11), fallback keeps the safety checks (P_fb ~3 %).
- "Why CVAE" rests on the Gaussian / MoG ablation (E13-E15); current evidence favours the CVAE only at small N.
- Mean sampler dropped from the paper's V-B. `paper/ablation_sampling.tex` is not yet synced with
  `paper/sparse_planner_zh.md`.

## 6. Open items

- Gaussian / MoG SP mode (with fallback) on Test 250 not run cleanly yet; E15 is single-seed.
- Paper Sec. IV-A text (3500 scenarios / 2450 training) does not match the actual 622 training scenarios; case-study
  Fig. 3 scenarios are not in the 26 and come from old Python runs.
- Still missing for the paper: seeds for the SP runs, equal-budget FOP N=8, chosen (d, v, T) + gap logs for the 3
  greedy-failure scenarios, sequential 26-scenario runtime on another PC.
- SP / Sparse_CPP with clearance not evaluated (only a 3-scenario check, see E2).

## Local-only data

Not committed (sizes on `dsc-cuda-01`, 2026-11-06):

| Path | Size | Content |
|---|---|---|
| `data/demo/` | 0.5 GB | scenario sets (UsedForSelfEvaluation, Test) |
| `data/10k/` | 57 GB | 10k Train / Verification / Test scenario splits |
| `data/sp_training_2026-02-15/` | 7.5 GB | SP CVAE training data (sampled_vars.15, imgs.5, cache_256); only its README is committed |
| `data/models/` | 0.7 GB | CVAE weights, TensorRT engine cache (`trt_cache`, rebuilt ~16 s per N on first use) |
| `data/output/sp_sampler/runs/*/*.pt`, `epochs/` | 5.6 GB | Gaussian / MoG checkpoints (`selected.pt` symlink) |
| `data/output/` (rest) | ~550 GB | planner outputs, images, collection data |
| `Verification.zip`, `*.pdf` (repo root) | | archive / reference papers |

Older measurement folders of the dropped cost-aware CVAE direction (`cvae_*`, `compare_*`, `10k*`, `weight_study*`, ...)
also remain local only.

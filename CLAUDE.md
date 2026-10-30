# Project context for Claude Code (fiss_plus_planner)

Research code for a paper on sampling-based trajectory planning in Frenet space (CommonRoad):
FOP, FISS+, CEM and MPPI share one C++ backend, and a cost-aware CVAE is trained on CEM data to
propose the sampling parameters (CVAE-FOP with a dense-FOP fallback).
This file holds the decisions and state from the work so far (until 2026-10-09). Read it before
changing planner, cost or training code.
A self-contained summary of all CVAE findings for external readers is `results/cvae_findings.md` (keep it updated).

## Working with the user

- Reply in Chinese. The user knows FOP / C++ / optimisation well but has no deep-learning
  background: explain ML parts in plain terms, with what each step does.
- Do not invent new algorithms; base methods on literature and cite them.
- Every planner has its own class/file and inherits the FOP base; code should be concise.
- After any parameter or cost change, **re-run the closed loop**; never re-score old trajectories.
- Logging inside a planner's `plan()` must sit behind a CMake option, so timing runs don't pay
  for it.
- Remind the user of the quality-critical settings (below) when presenting or comparing results.
- Commit only when asked, with
  `git -c user.name=wenguang -c user.email=xuwenguang1111@gmail.com commit`, ending the message
  with `Co-Authored-By: Claude <noreply@anthropic.com>`. Never push. Do not commit `.vscode/`,
  the FISS PDF, `Verification.zip` or `__pycache__/`.
- The user runs long jobs in tmux / on the HPC; give them the command and analyse afterwards.

## Planner backend (C_Planner, Python in fiss_plus_planner/)

- **One evaluator for all planners** (`common/evaluator/trajectory_evaluator.*`): generation,
  Frenet->Cartesian, dynamic/road checks, clearance, polygon collision, cost.
- **Ranking is lexicographic (feasible, V, J):** feasible before infeasible; feasible ones by J,
  infeasible ones by V.
  - V of a check = share of the horizon from its first violation on, (N - k) / N. V_total sums
    speed (incl. reversing), acceleration, road, clearance, collision and transform. There is no
    curvature check.
  - All planners use `EvalMode::kEarlyExit`.
- **Cost** (FISS+ paper Eq. 4 + J_D, no normalisation): w_T=0.1, w_V=0.01, w_A=0.1, w_J=0.1,
  w_D=1.0, w_dist=0.1, w_LC=10.
  - FISS+ plans without J_D.
  - Scenario cost J_total = J_run + J_ter with J_ter = w_T * t_f (the user treats Eq. 8 as
    misprinted).
- **v_des:** the goal velocity if there is one, else the route speed-limit sign, else 14 m/s.
  The sampling bound is max(v_des, current speed).
- **Safety (clearance) defaults:**
  - tau = 2.0 s, d0 = 3 m, lateral margin 0.2 m, grace 1 s;
  - recovery T_rec = 3 s with an r^2 ramp from the constant-speed gap;
  - fallback without clearance if nothing is feasible.
  - The scan suggested tau 1.5 / T_rec 5; **the user has not decided — ask before changing.**
- **Low-speed model** (Werling thesis Sec. 3.5.1, App. A.1; CommonRoad reactive planner):
  - below 4 m/s the lateral motion is d(s), with a minimum lateral length of 5 m;
  - heading and curvature are computed analytically;
  - standstill is allowed.
- **Map handling:**
  - the reference line is extended straight past the route end;
  - lanes merging into the route widen the road profile.
- **Goal reached (2026-10-12):** the ego position lies in the goal region (CommonRoad `contains_point`,
  without the time window, velocity and orientation of `GoalRegion.is_reached`), or the route end / map end
  is reached. The measurement CSVs and scenario.json have `termination` = goal / map_end / time_limit /
  failed; compare_planners.py reports the costs also on the scenarios where all planners reached the goal.
- **Failure label:** `rear_end_failure` = a recorded follower drives into the standing ego
  (nuPlan at-fault idea). The recorded traffic is not reactive.
- **CEM** (data source for the CVAE): R8 x 250 (8 iterations, 250 samples each), elite 0.1,
  smoothing 0.7, init_std 0.3, seed 0, warm start from the previous best.
  - Tuned on data/demo/Test (250 scenarios, 3 seeds).
  - Larger N saturates: R8x2000 is only -0.1 % J.
  - The greedier settings fail DEU_Bonn-16_18 (dead end, recursive-feasibility issue).
  - CEM ≈ MPPI < FISS+ < FOP in cost at every budget 200-27000.

## CVAE training data (collected 2026-10-08/09)

- **Collection:** `Collect_Data_For_ML` + `PLANNER: CEM_CPP` on data/10k/Train, run with
  `demo_cr.py --shard k --num_shards n`.
  - Needs C_Planner built with `-DENABLE_DATA_COLLECTION=ON`; rebuild OFF before timing runs
    (the build is ON since 2026-10-14 23:30, for the gap diagnosis E1).
  - Recording evaluates every check (kFullViolation) but ranks the early-exit view, so its
    decisions are bitwise identical to a normal run.
  - Guide: README.md "Collecting CVAE training data with CEM".
- **Output** in `OUTPUT_DIR` (v2: `data/output/cem_train_R8x250_v2/`):
  - `cem_data/<scenario>/{contexts,candidates,proposals,conditions,best_trajectory,reference_line}.parquet`,
    `scenario.json`;
  - `imgs/<scenario>/<t>.png`;
  - `dataset_info.json`.
- **Reference path** (`conditions.parquet`, added for v2, for the planned route condition of the
  CVAE): per cycle, in the ego frame. It runs from the nearest reference point up to 43.75 m ahead
  (VIEW_SIZE_DEFAULT / 2), with points every 0.1 m (`ref_x`, `ref_y`, float32).
  - This is the same format as the FOP path data and as Giovanni's world-model CVAE
    (github.com/giovannilucente/CVAE_trajectory_planning). That CVAE resamples every 2 m and uses
    20 points (40 m) as deltas.
  - The user chose 43.75 m (not longer).
- **Candidate trajectories (option A, user's choice):** not stored; storing all of them would be
  320 GB to 1 TB.
  - `scripts/cem_trajectories.py` (`CEMTrajectories`) rebuilds any candidate exactly with
    `FrenetPlanner.generate_trajectory`. It uses the start state in contexts (including `d_s`,
    `d_ss`), `reference_line.parquet` and the candidate's (d, v, T).
  - The executed plan per cycle is stored in `best_trajectory.parquet` (ego frame).
  - `--verify` checks the best plans against the stored ones (float32 storage) and the speed /
    acceleration labels of all candidates.
  - Tested on ARG_Carcarana-1_3 (34 of 36 cycles in low-speed mode) and ZAM_Tjunction-1_374:
    - 325k feasible trajectories the planner evaluated, and all 390k candidates generated in the run,
      rebuilt **bit for bit** from the files.
    - This requires d, v, T stored as float64. With float32 the error was about 2e-6 m, and one
      trajectory had a different number of points.
- **Collections:**
  - **v3** (2026-10-12, collected by the user on another PC): recollection with the new goal logic.
    Configs `cfgs/collect_{train,verification}_v3.yaml`, run with `scripts/run_collection.sh CFG`.
    - It also writes `cem_data/<s>/sampled_vars.parquet` for Giovanni's world-model CVAE
      (`planning.wm_samples`, `ML_DATA`): per cycle the executed plan + 16 feasible candidates drawn
      with p ∝ exp(-J~/0.25)/q(z), with x, y, yaw in the ego frame.
    - `scripts/merge_wm_data.py` merges these into the WorldModel-branch layout
      (OUT/sampled_vars.parquet, conditions.parquet).
    - The CEM data are identical to v2 up to the changed episode ends.
    - **State 2026-10-13 (collected on the local PC despite the RAM fault):** Train 6808 scenarios
      (439,094 cycles with a plan, 7.44 M trajectories), Verification 1947 (127,508 cycles). Checked:
      no empty / unreadable files, CEM files byte-identical to v2 for equal episode lengths and row-identical
      on the common cycles otherwise; 316 / 79 scenarios fail at cycle 0 and have no sampled_vars.
    - Crashes of that run left completed scenarios with 0-byte scenario.json / truncated PNGs (kernel crash
      right after they finished); scan every file, not only the `completed` count, before packaging.
    - Fixed in `wm_samples`: draw at most as many candidates as have a non-zero weight (exp underflow made
      `rng.choice` fail in near-failure cycles; those cycles now have fewer than 17 rows).
  - **v1** (2026-10-08/09) had no reference paths. It was moved off this PC (Train.zip).
  - **v2** is a full recollection with reference paths, best trajectories, reference line and
    d_ss, planned with the same code. The CEM data
    should be byte-identical to v1, which also checks the suspected hardware fault.
  - v2 uses OUTPUT_DIR `cem_train_R8x250_v2` and MEASUREMENTS_DIR `data/measurements/10k_v2`.
  - A full run takes about 2 h with 22 shards (v1: 1 h 55 min; the rerun of 150 scenarios: 5 min).
- **v1 state (complete, 2026-10-09):** 6808 of 6832 scenarios. The other 24 fail before planning
  (18 without a global route, 6 without an ego trajectory) and have no data.
  - The missing ones were recollected with `cfgs/rerun_missing.yaml` (not committed).
  - An integrity scan of all PNG and parquet files was clean.
  - **RAM fault on the local PC (confirmed 2026-10-11; non-ECC DDR5, no MCE/EDAC reports):** pages in
    the page cache get corrupted, so a process can read wrong bytes from a correct file.
    - It broke 10 files inside Train.zip (the zip's CRCs match the bad data; fixed on alex).
    - It showed 6 wrong d values in one Verification file until the cache was evicted.
    - Reruns are bit-identical (deterministic planner), so rerun + sha256 is the reliable check.
      `scripts/check_cem_data.py` finds only gross corruption.
    - Compare sha256 of every copy, also against direct-from-disk reads (O_DIRECT).
    - Advise memtest86+ / JEDEC memory speed before trusting large data jobs on this PC.
    - The kernel log (journalctl, needs the user; `/proc/vmstat` oom_kill is readable) shows it too:
      kernel crash 2026-10-11 14:32, `BUG: Bad page state` and random python segfaults / general
      protection faults since 2026-10-05. On 2026-10-12, 5 of 16 shards of the Train v3 collection died
      this way (not OOM, oom_kill 0), and one input XML was read as corrupt.
- **v1 statistics:** 86.1 % success; 945 failures (419 rear-end, 316 failing at cycle 0);
  519,001 cycles; 826 M feasible candidates; 54 GB cem_data plus 5.2 GB images. The cache
  (cem_cache.py) is 48 GB, built in 73 s with nothing skipped.
- **Splits:**
  - data/10k Train / Verification / Test are disjoint.
  - data/demo/Test (used for all tuning) overlaps 10k/Train by 57 scenarios. Evaluate the CVAE
    planner on 10k/Test, or on demo/Test without the overlaps.
  - **Verification** (2026-10-11, `cfgs/collect_verification.yaml`, same settings and build as v2):
    - `data/output/cem_verif_R8x250_v2`: 1947 of 1952 scenarios (5 fail before planning),
      149,326 cycles;
    - checksums (read directly from disk) in `cem_verif_R8x250_v2.sha256`;
    - cache `data/output/cvae_local/cem_verif_cache`. Not used for training or model selection.
  - **From v3 on (user's choice, 2026-10-13):** Verification v3 is the validation set of the training
    (`--val_cache`: val loss, model_best, per-epoch generation check) and of the offline evaluation; the
    model trains on all Train scenarios. The independent test is the closed loop on data/10k/Test.
    Local v3 baseline: `data/output/cvae_local/train_v3.sh` (caches `cem_{train,verif}_cache_v3`, run
    `runs/v3_attn_tau0.25_beta0.1`).

## Cost-aware CVAE (CVAE_efficient_sampling, branch `cost-aware`)

- **Target:** z = [d, v, T] in this order everywhere (the old code used [t, d, v]).
- **Condition:** 3 consecutive 256x256 grayscale BEV frames, inverted and scaled to [-1, 1].
- **Cost awareness by importance resampling:** per planning cycle, one feasible candidate is
  drawn with probability exp(-J~/tau) / q(z), and the model is trained with the usual ELBO.
  - J~ = (J - J_min) / (J_90 - J_min) within the cycle.
  - q = CEM's mixture sampling density over the cycle's iterations (balance heuristic, Veach &
    Guibas 1995). It removes CEM's concentration near the optimum.
  - Nothing is weighted at collection time; tau etc. are training options.
- **Failed scenarios:** keep rear-end failures (minus the failing cycle). Drop the last 2.5 s
  before other failures (`--drop_tail_s`, `--drop_failed`).
- **Files:**
  - `cem_cache.py`: one-time memory-mapped cache, about 47 GB.
  - `cem_dataset.py`.
  - `train_cost_cvae.py`: checkpoints, SIGUSR1/SIGTERM and `--max_hours` stop, resumes with the
    same command. Exit code 3 = stopped and must be resumed; 0 = complete.
  - `test_cost_cvae.py`: held-out check against uniform and marginal samplers, plus a `target`
    row (K draws from the target itself: the ceiling of a perfect CVAE). `--cache` evaluates on
    another data set (e.g. Verification).
  - `CVAE.py: CostAwareCVAE`: inference, frames -> [d, v, T].
  - `slurm_train.sh`: one sbatch is enough; it resubmits itself on exit code 3 (at most
    MAX_RESUBMIT = 5).
  - Steps: CVAE_efficient_sampling/README.md.
- **Models:**
  - attnCVAE (default; ResNet18, one latent level, 11.3 M params; the model the sparse planners
    use);
  - HCVAE (`--model hcvae`; four hierarchical latent levels, about 2 M params).
- **Measured locally:** about 1200 items/s on an RTX 5070 Ti, about 7 min/epoch, 2 GB GPU at
  batch 64.
- **First full trainings (2026-10-10/11):**
  - Setup: attnCVAE, 20 epochs, draws_per_cycle 1, seed 0, v2 cache.
  - Local runs are in `data/output/cvae_local/runs/<name>`; the same runs on alex agree within
    noise.
  - Verification, 20000 cycles, K 16. best_cost@16 for cvae | target ceiling | marginal:
    - tau 0.25, beta_end 0.1: 0.137 | 0.117 | 0.091 (best so far; CVAE-FOP uses it);
    - tau 0.5, beta_end 0.1: 0.236 | 0.180 | 0.116;
    - tau 0.5, beta_end 1.0: collapsed (spread 0.085 vs 0.571), 0.390.
  - The held-out split gave the same numbers, so the model generalises.
  - best_cost uses the nearest CEM candidate's J~ and so favours broad samplers (marginal beats even
    the target).
  - Room left: mean_dist 0.378 vs target 0.184, spread 0.389 vs target 0.498.
  - Next options: tau 0.1, free bits / smaller beta_end, closed-loop evaluation.
- **Posterior collapse** (2-epoch smoke test, spread 0.03 vs 0.61 with `--beta_end 1.0`): fixed by
  `--beta_end 0.1`.
  - `generation_check.py` runs after every epoch (`--gen_check_cycles` 300, K 16) and writes
    gen_in_bounds, gen_best_cost, gen_mean_dist, gen_spread and target_spread to `log.tsv`.
  - First remedy: `--beta_end 0.1`. Then free bits (Kingma 2016), cyclical annealing (Fu 2019),
    then MMD / InfoVAE.
  - The user decided against SIGReg.
  - Both models already clamp logvar to [-10, 10].
- **CVAE v1 = reference for later versions and ablations** (git tag `cvae-v1` here and in the
  submodule; record `results/cvae_v1/README.md`):
  - frozen model `fiss_plus_planner/data/models/cvae_v1` (tau 0.25, beta_end 0.1; sha256 in
    SHA256SUMS);
  - Test result: success 87.6 % (CEM 86.1 %, FOP 88.2 %). J_total on the 822 scenarios solved by
    all: CVAE-FOP 13.86, CEM 11.84, FOP 14.57. The gap to CEM is mostly J_V.
  - Raw measurements and per-cycle logs are archived in `fiss_plus_planner/data/results/cvae_v1`.
  - New CVAE versions get a new run folder and `data/models/cvae_vN`; never overwrite v1.
- **CVAE-FOP planner** (2026-10-11): `fiss_plus_planner/planners/cvae_fop_planner.py`,
  `PLANNER: CVAE_FOP`, config `cfgs/cvae_fop.yaml` (data/10k/Test).
  - Per cycle:
    1. render the BEV frame (same drawer as the training images; frame 0 is pixel-identical);
    2. draw K = 16 CVAE samples, clipped to the search space (C++ `search_space`);
    3. evaluate them with the clearance requirement;
    4. if none is feasible: the dense FOP grid 10x10x10 (num_FOP_intervention) with the clearance
       requirement, then without it if `SAFETY: clearance_fallback`;
    5. otherwise the cycle fails.
  - The user chose this order on 2026-10-11, so the rules match the other planners and the CEM data.
  - Without the clearance fallback, FOP 10x10x10 alone fails ESP_Barcelona-70_11 and
    POL_Krakow-93_1, as CVAE-FOP did.
  - Runtime excludes the BEV rendering (user's choice; the raster would come from perception) and
    the untimed diagnostics.
  - Target hardware: this PC (RTX 5070 Ti, PyTorch). Time with one process; parallel shards share
    the GPU and CPU. Use about 12 shards for quality runs (each loads CUDA and the model).
  - Config `CVAE_FOP`:
    - `run`, `num_samples`, `seed` (cycle t uses seed + t);
    - `sampler` cvae | uniform | marginal (ablation baselines with the same K);
    - `log_cycles`: per-cycle CSV in `OUTPUT_DIR/cvae_cycles`: stage
      samples/grid/grid_no_clearance/failed, feasible count, rejection reasons, J of the samples /
      grid / plan, chosen z, stage times;
    - `regret`: also plan the grid in cycles solved by the samples (untimed);
    - `warm_start` (default true since 2026-10-11, CVAE v1: false): the previous cycle's executed z
      is sample K + 1.
    - `fallback` fop | cem (2026-10-15; cfgs/cvae_fop.yaml and gap_diagnosis.yaml: cem, code default fop for the
      older configs): when no sample keeps the clearance requirement, the teacher CEM of the CEM section plans (with
      clearance, then without if `clearance_fallback`, as the teacher), warm-started from the previous executed z
      (C++ `CEMPlanner.set_warm_start`). Stages cem / cem_no_clearance. Test: with no samples at all, CVAE+CEM
      reproduces the CEM baseline bit for bit (3 scenarios, J to 12 digits). User's evaluation route (2026-10-15):
      CVAE + CEM fallback vs CEM, reporting (compare_planners.py) **fallback** = share of the cycles without a
      feasible CVAE sample (CEM planned, whatever it did inside about clearance), **interrupted** = share of the
      scenarios that ended because no plan was found (termination failed), and the paired J_total of the whole run.
  - **Warm start result** (`results/ablations/warm_start/README.md`, Test):
    - success unchanged at 87.6 %; the FOP fallback share drops from 22.4 % to 11.6 %;
    - J_total 13.21 vs 13.86 (v1) and 11.84 (CEM); paired vs v1 -0.63 ± 0.20, lower in 66 %;
    - the warm sample is chosen in 78 % of the cycles the samples solve.
  - **Full ego-state condition (negative result, tag `ablation-ego-full`,
    `results/ablations/ego_state_full/README.md`):**
    - Setup: `train_cost_cvae.py --ego` gives the attnCVAE 8 extra tokens from
      `cem_dataset.EGO_COLUMNS` (v, a, s_d, s_dd, d, d_d, d_dd, v_des), clipped to 0.1-99.9 % and
      standardised (`normalizer/ego_scaler.json`).
    - Offline it is more accurate in speed (error 0.77 vs 1.12 m/s; it uses v_des like the data),
      but in closed loop it is worse than v1 + warm start (J 13.85 vs 13.25; +0.58 ± 0.17 paired).
    - Cause: inertia / causal confusion (Codevilla 2019, de Haan 2019). The input speed comes from
      its own plans, so the vehicle gets stuck creeping slowly in about 10 % of the scenarios
      (85 % of the gap).
    - Next: B = task-only input (no ego speed / acceleration), then A = ego-speed dropout
      (ChauffeurNet, Bansal 2019).
  - **Variant runs:** `bash fiss_plus_planner/scripts/run_cvae_variant.sh NAME KEY=VALUE ...`
    (e.g. `k32 CVAE_FOP.num_samples=32`):
    - runs `cfgs/cvae_fop.yaml` with the overrides on Test into `data/{measurements,output}/cvae_NAME`;
    - compares against the stored CVAE v1, CEM and FOP results (`results/cvae_v1`) into
      `data/measurements/compare_NAME/report.txt`;
    - keep each ablation's report, config and measurements in `results/ablations/<name>/`.
  - Smoke test (5 Verification scenarios):
    - all succeed;
    - samples-only cycles take 2.5 ms (CVAE 1.9 ms, C++ evaluation 0.5 ms); grid cycles 9.5 ms;
      rendering 19 ms (untimed);
    - 95 % of the samples are feasible in samples-stage cycles;
    - the 16 samples beat the 1000-point grid in 58 % of the cycles;
    - ESP_Barcelona-70_11 needs the grid in 63 % of its cycles (worth a look).
  - Ablation ideas not built yet: regret vs the CEM teacher on the same state, inference time vs
    K / bf16 / TensorRT, storing the K raw samples.
  - **Test comparison:** `bash fiss_plus_planner/scripts/run_test_comparison.sh`
    (`PLANNERS="CVAE_FOP CEM"` for a subset, `SHARDS=n`, default 22).
    - Runs `cfgs/cvae_fop.yaml`, `cfgs/cem_test.yaml` (CEM R8x250) and `cfgs/fop_test.yaml`
      (FOP 10x10x10) one after the other. The configs differ only in PLANNER and paths.
    - Then `scripts/compare_planners.py --ref CEM` writes `data/measurements/compare_test/`
      (report.txt, summary.csv, paired_costs.csv):
      - success (rear-end / other failures), FOP fallback share, runtime per cycle;
      - `trajs_per_cycle` (trajectories evaluated per cycle, fallback grids included: the real budget) and, for
        CVAE-FOP runs with cycle logs, the stage shares samples / grid / grid_no_clearance / failed and the
        feasible share of the K samples (since 2026-10-15; `grid` is the coverage signal of the CVAE);
      - J_total paired on the scenarios all planners solve: diff with 95 % CI, Wilcoxon p, share
        lower, cost terms.
    - Logs are in `logs_test/`. Every planner run starts from scratch: the script deletes its
      measurement and output folders and logs. It refuses to start while demo_cr.py processes run,
      and reports scenarios done vs 976. Planners running in parallel (load 370) were much slower
      than one after the other.
    - Ctrl+C in the script's terminal can leave the shard processes running. Check with
      `pgrep -fa demo_cr.py` and stop them with `pkill -f scripts/demo_cr.py`.
    - Parallel runs only give J and success. Measure runtime separately with one process.
    - torch is limited to `Num_Threads_For_CollisionChecker` CPU threads. Its default pool made 12
      parallel CVAE processes reach load 200.
  - Next: the comparison above, then single-process timing and FISS+ with the same SAFETY settings.
- **Route condition (`--route`, implemented 2026-10-13; closed-loop result 2026-10-14: no effect)**, taken from
  Giovanni's world-model CVAE.
  - Test (972, vs goalpos baselines, `data/measurements/compare_v3_{img,route}`): v3_route vs v3_img paired
    J +0.008 ± 0.128 (p 0.55), success 89.4 vs 89.5 %, reached 69.3 vs 68.3 %; offline also equal.
  - Reason: z = [d, v, T] is in the Frenet frame of the reference line, so the branch at intersections is
    already chosen by the reference line; the route can only inform the speed (curvature ahead), and J_V did
    not change (6.23 vs 6.25). It matters for Giovanni's Cartesian (x, y) outputs, not for ours.
  - v3 image-only baseline (`runs/v3_attn_tau0.25_beta0.1`, CVAE v1 setting on v3 data) vs CVAE v1 + warm
    start: J +0.29 ± 0.13 paired (lower in 36 %), success 89.5 vs 89.6 %, reached 68.3 vs 71.0 %, J_V 6.25 vs
    6.03. One training seed each, so this may be seed noise; not checked yet.
  - **Where the CVAE loses to CEM (v3_img, Test, 851 scenarios solved by both, 2026-10-14):** J gap 1.54 per
    scenario, 73 % J_V. Below 1 m/s the CVAE spends 10.2 % of its cycles (CEM 5.7 %); the J_V accumulated there
    is 47 % of the J gap (50 % with the time term); in 80 % of these cycles no sample was rejected by clearance
    (not held back); the top 10 % of scenarios hold 94 % of it (e.g. HRV_Centar-51: goal rectangle right behind
    a standing car, the CVAE stops ~6 m back and creeps). Another 16 % of the gap is CEM in the clearance
    fallback (no clearance, drives close: POL_Krakow-62, GRC_NeaSmyrni-73, DEU_Bonn-11).
  - Offline at CEM's own states (Verification, per ego-speed bin) the CVAE is not slower at low speed (v bias
    +0.26 m/s below 1 m/s, -0.37 m/s above 8 m/s); its v spread is about half the target's in every bin.
    So the creeping comes from states the CVAE reaches itself and CEM rarely visits (compounding error /
    distribution shift, Ross et al. 2011): DAgger with CEM as the expert is the literature remedy.
  - K = 32 (v3_img, `compare_v3_k32`): vs K = 16 J -0.077 ± 0.071 (small), vs CEM +1.46 (K16 +1.54, CVAE_warm
    +1.24); low-speed cycles 10.0 vs 10.2 %, so more samples barely touch the creeping -> not mainly sample
    concentration. v3 Train has a third fewer cycles below 1 m/s than v2 (25.7k vs 38.5k: the episodes end at
    the goal), a candidate cause of v3 < v1 (untested).
  - `data/output/cvae_warm/cvae_cycles` belongs to an older run (before goalpos), not to goalpos/CVAE_warm.
  - GIF tools: `scripts/compare_gifs.sh SCEN...` (CEM, CVAE_warm, v3_img; goal region drawn, cycle in the title).
  - The BEV images do not show the route (`ScenarioDrawer._render_frame` draws all lanes in one gray), so at
    intersections the image-only CVAE cannot see which branch is planned.
  - Input: the reference path ahead (conditions.parquet; planner: `planners.common.utils.reference_ahead_local`,
    43.75 m) -> `cem_dataset.route_features`: resampled every 2 m by arc length, 20 points (0-38 m), straight
    extension past the end, deltas (first from the ego), divided by 2 m.
  - Model: `attnCVAE(route_points=20)`, `PointTokenizer` (shared MLP + position embedding, Giovanni's
    cond_n); the 20 route tokens are appended to the condition tokens (as `--ego`), so encoder cross-attention,
    prior and decoder see them. Giovanni uses FiLM for them instead.
  - Cache: raw paths in `route_xy.bin` + `route_offsets.bin` (about 1.5 GB for Train), converted when read.
    Storing the resampled points is not equivalent (resampling again moves points up to 0.15 m on curves).
    `cem_cache.py --add_route` adds them to an existing cache.
  - Tests (30 scenarios): build == add_route, cache == conditions.parquet, the planner's path at cycle 0
    equals the data (1e-6 m), smoke trainings with / without --route, cvae_v1 still loads.
  - Trained by `data/output/cvae_local/train_v3.sh` after the baseline (run `v3_attn_route_tau0.25_beta0.1`);
    compare image-only vs image + route in closed loop on Test.
- **DAgger (Ross, Gordon & Bagnell, AISTATS 2011; round 1 done 2026-10-14, positive; user's choice: on the v3
  data):**
  - Collection: `PLANNER: CVAE_FOP` + `Collect_Data_For_ML: True` (planning.py `expert`): CVAE-FOP drives, the
    CEM of the collection (CEM section, record_candidates) plans in every state it reaches; the CEM data
    (candidates, contexts best_*, best_trajectory, sampled_vars) are the expert's, scenario.json and the images
    are the learner's run. Config `cfgs/collect_dagger1.yaml` (learner = v3_img, K 16, warm start;
    Runtime_Measurement True), run with `scripts/run_collection.sh`; needs the ON build (rebuilt OFF after the
    round-1 collection, 2026-10-14). The script does not notice dead shards: check that every shard log ends at
    its last scenario (3 of 12 died silently in round 1, RAM fault); a rerun skips completed scenarios, use a
    different SHARDS so the rest spreads over all processes (scenarios are assigned by index mod SHARDS).
  - Training on the aggregate: `train_cost_cvae.py --cache <v3 cache> <dagger cache> --val_cache <verif v3>`
    (`CEMCycleConcat`; cache k > 0 draws with seed + 1000 k; one cache trains as before).
  - Tests (3 Train scenarios): the learner's run is identical with and without the expert; the expert's cycle-0
    candidates equal the v3 CEM data byte for byte; images = cycles; cache build + 2-cache training work.
  - Rule B (user's choice): DAgger caches (`CEMCache.dagger`, from dataset_info `dagger_learner`) keep the cycles
    before a learner failure (no drop_tail, failed scenarios kept); the CEM data keep the 2.5 s rule.
  - Analysis tools: `scripts/lowspeed_gap.py` (share of the gap below 1 m/s), `scripts/speed_bins_offline.py`.
- **Gap diagnosis E1 (implemented 2026-10-14, defines the paper's problem: P1 target / P2 shift / P3 cost):**
  - `scripts/run_gap_diagnosis.sh NAME RUN [KEY=VALUE]` on data/10k/Verification (`cfgs/gap_diagnosis.yaml`):
    learner mode = DAgger collection with `log_cycles` (ON build); expert mode = `SHADOW_CVAE: True` with CEM_CPP
    (CEM drives, CVAE-FOP plans every state, never executed, warm-started from the CEM's executed z).
  - Both write the CVAE-FOP cycle_log with the CEM's plan in the same state (`J_cem`, `cem_d/v_end/T`,
    `cem_no_clearance`; planning.py `cem_log_columns`).
  - `scripts/gap_decomposition.py NAME`: per cycle r = J_plan - J_cem and rT = exact E[best of K target draws] -
    J_cem (with / without /q), per speed bin and mode; low-speed check (cem_v_end vs v_end); closed-loop Spearman.
    Expert states are checked against cem_verif_R8x250_v3 (smoke: 100 % identical).
  - Offline result on all 127,508 Verification v3 states (perfect CVAE = target, tau 0.25): best-of-K regret per
    cycle K 1/4/16/64 with /q 0.86/0.41/0.23/0.14, without /q 0.18/0.055/0.021/0.009 (J_cem ~5). The density
    correction makes the target ~10x worse for best-of-16 (P1 candidate; closed-loop test: `--no_density_correction`).
  - **Round 1 result (2026-10-14, `results/ablations/dagger1/README.md`; frozen model `data/models/dagger1`,
    archive `data/results/dagger1`):**
    - Data: 6808 scenarios, 474,877 learner cycles (learner on Train: reached 4501 vs CEM 4930, time limit 1557
      vs 1048); training on 909,499 cycles (v3 + DAgger), 4 h 22 min.
    - Offline on Verification v3 equal to v3_img (best_cost@16 0.176 vs 0.182): expected, those are CEM's states.
    - Test vs v3_img paired: J **-0.214 ± 0.111** (p 4e-4), mostly J_V; reached 68.3 -> 70.6 %, success 89.6 %;
      gap to CEM +1.54 -> +1.32 (CVAE_warm +1.24: equal to dagger1, +0.08 ± 0.12). Cycles below 1 m/s 10.2 ->
      8.9 % (CEM 5.7 %), low-speed J_V excess -28 % (47 -> 40 % of the gap).
    - Caveats: one seed; twice the training items. Next: control v3 + `--draws_per_cycle 2`, 2nd seed, round 2
      (learner = dagger1), as the user decides.
    - The paired comparison against v3_img is not in run_cvae_variant's report: `compare_planners.py --run
      dagger1=... --run v3_img=data/measurements/cvae_v3_img --ref v3_img`.
  - **Control draws2 (2026-10-15, run `runs/v3_draws2_tau0.25_beta0.1`: v3 only, `--draws_per_cycle 2`):** Test vs
    v3_img -0.149 ± 0.092 (p 0.006), vs dagger1 +0.066 ± 0.093 (p 0.29). ~70 % of DAgger's gain is the doubled
    training items; the on-policy part is within one-seed noise (memory cvae-ablations entry 9).
  - **E4 A no /q (2026-10-15, run `runs/v3_draws2_noq_tau0.25_beta0.1`, = draws2 + `--no_density_correction`):** Test
    vs draws2 -0.281 ± 0.152 (p 4e-11), vs CVAE_warm -0.134 ± 0.110: new best CVAE (gap to CEM 1.10); low-speed J_V
    excess halved; FOP fallback 11.7 -> 13.0 % (coverage cost). P1 confirmed (memory cvae-ablations entry 10).
  - **Success-only data (2026-10-29/30, `results/ablations/success_only/README.md`):** scenario folders
    `data/10k/{Train,Verification,Test}_cem_success` (5984 / 1711 / 861 scenarios CEM completes; outcome lists in
    `data/10k/cem_outcomes/`). `train_cost_cvae.py --success_only` restricts the v3 caches to them (no recollection).
    Run `runs/succ_draws2_tau0.05_beta0.1` (B settings), frozen `data/models/succ_draws2_tau005`. Test_cem_success with
    `scripts/run_test_set.sh` (CEM / FOP / FISS+ / B_v3 / B_succ, each from its earlier config; results in
    `data/measurements/cem_success/`): J vs CEM FISS+ +0.23, B_succ +0.68, B_v3 +0.75, FOP +2.66; interrupted
    2 / 5 / 7 / 8; B_succ vs B_v3 -0.075 ± 0.058 (p 0.001, one seed). FISS+ stays ahead of every CVAE.
  - **Python env (2026-10-29/30):** the user works in conda `world-model`; all scripts default to it. That env has
    another checkout (`~/workspace/CVPR/fiss_plus_planner`) installed as `fiss_plus_planner`; `scripts/demo_cr.py` now
    puts this repository first on sys.path. Other scripts importing the package need
    `PYTHONPATH=~/workspace/fiss_plus_planner` there.
  - Design, results and commands of all ablations: memory cvae-ablations (overview table + entries 4-14) and
    `results/ablations/CLAUDE.md`.
- **Later ideas** (user wants them after V1): use feasible / V labels for the CVAE. Independent
  CEM restarts and a curvature constraint are not planned.

## Running on the HPC (runbook)

The CVAE is trained on an HPC with A40 / A100 GPUs and Slurm, with jobs of at most 24 h. The data
are collected on the local PC. Paths below are examples.

**0. On the local PC: package the data.** Do this after the rerun and a clean integrity scan.
The raw data are needed for later V / feasibility training; the cache keeps only feasible
candidates.

```bash
cd fiss_plus_planner/data/output
tar --exclude='*.gif' -cf - cem_train_R8x250_v2/cem_data cem_train_R8x250_v2/imgs cem_train_R8x250_v2/completed \
  | zstd -T0 -3 -o cem_train_R8x250_v2.tar.zst
sha256sum cem_train_R8x250_v2.tar.zst > cem_train_R8x250_v2.tar.zst.sha256
```

The submodule branch `cost-aware` must be pushed before the main repo (`Experiment`), which
points to it. The user pushes.

**1. Code and environment** (once):

```bash
git clone --recurse-submodules <fiss_plus_planner repo> && cd fiss_plus_planner && git checkout Experiment
git submodule update --init && git -C CVAE_efficient_sampling checkout cost-aware
conda create -n cvae python=3.10 && conda activate cvae
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.x build
pip install numpy pandas pyarrow scikit-learn pillow matplotlib tqdm
python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"
```

The last line downloads the attnCVAE backbone on a login node; compute nodes may have no internet.

**2. Data and cache** (once; the cache build needs about 1 min with 16 CPU workers, and can
run interactively or as a small CPU job):

```bash
cd $SCRATCH && sha256sum -c cem_train_R8x250_v2.tar.zst.sha256 && zstd -dc cem_train_R8x250_v2.tar.zst | tar -xf -
python <repo>/CVAE_efficient_sampling/cem_cache.py --data $SCRATCH/cem_train_R8x250_v2 --out $SCRATCH/cem_train_cache --workers 16
```

Check `meta.json` of the cache: `skipped` should be empty, about 6800 scenarios.

**3. Configure `CVAE_efficient_sampling/slurm_train.sh`** (once): in the block "adjust to the
cluster", set the partition / account lines if required, `CACHE=$SCRATCH/cem_train_cache`,
`RUN`, `CONDA_ENV=cvae`, and `module load` if needed.

**4. Smoke test** (about 5 min), then check the result:

```bash
cd <repo>/CVAE_efficient_sampling
RUN=$HOME/runs/smoke sbatch slurm_train.sh --limit_scenarios 300 --epochs 2 --draws_per_cycle 4
python test_cost_cvae.py --run $HOME/runs/smoke --cycles 300 --k 16     # on a GPU node / interactive job
```

Expected: the cvae row has a clearly smaller `mean_dist` than uniform / marginal, and
`in_bounds` around 0.9.

**5. Full trainings**, one run directory per setting:

```bash
RUN=$HOME/runs/attn_tau0.5_beta1.0  sbatch slurm_train.sh
RUN=$HOME/runs/attn_tau0.5_beta0.1  sbatch slurm_train.sh --beta_end 0.1
RUN=$HOME/runs/attn_tau0.25_beta0.1 sbatch slurm_train.sh --tau 0.25 --beta_end 0.1
squeue -u $USER;  tail -f cost_cvae_<jobid>.log;  column -t $RUN/log.tsv
```

About 7 min/epoch, so 20 epochs is about 2.5 h; a single job usually finishes. If not, the job
resubmits itself.

**6. Evaluate each run:** `python test_cost_cvae.py --run $RUN --cycles 2000 --k 16`.
- Compare `best_cost@K` and `mean_dist`.
- Compare `spread` with the target spread; a much smaller spread means posterior collapse.
- Report the settings (tau, beta_end, model, draws_per_cycle) with the numbers.
- Record the chosen setting in this file.

**After the CVAE training:** evaluate the CVAE-FOP planner (see "CVAE-FOP planner" above) in
closed loop on data/10k/Test.

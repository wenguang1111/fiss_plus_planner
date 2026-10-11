# Project context for Claude Code (fiss_plus_planner)

Research code for a paper on sampling-based trajectory planning in Frenet space (CommonRoad):
FOP, FISS+, CEM and MPPI share one C++ backend, and a cost-aware CVAE is trained on CEM data to
propose the sampling parameters (CVAE-FOP with a dense-FOP fallback).
This file holds the decisions and state from the work so far (until 2026-10-09). Read it before
changing planner, cost or training code.

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
    (the build is OFF since 2026-10-11).
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
  - **RAM fault on the local PC (confirmed 2026-10-11; non-ECC DDR5, no kernel reports):** pages in
    the page cache get corrupted, so a process can read wrong bytes from a correct file.
    - It broke 10 files inside Train.zip (the zip's CRCs match the bad data; fixed on alex).
    - It showed 6 wrong d values in one Verification file until the cache was evicted.
    - Reruns are bit-identical (deterministic planner), so rerun + sha256 is the reliable check.
      `scripts/check_cem_data.py` finds only gross corruption.
    - Compare sha256 of every copy, also against direct-from-disk reads (O_DIRECT).
    - Advise memtest86+ / JEDEC memory speed before trusting large data jobs on this PC.
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
    - `regret`: also plan the grid in cycles solved by the samples (untimed).
  - Smoke test (5 Verification scenarios):
    - all succeed;
    - samples-only cycles take 2.5 ms (CVAE 1.9 ms, C++ evaluation 0.5 ms); grid cycles 9.5 ms;
      rendering 19 ms (untimed);
    - 95 % of the samples are feasible in samples-stage cycles;
    - the 16 samples beat the 1000-point grid in 58 % of the cycles;
    - ESP_Barcelona-70_11 needs the grid in 63 % of its cycles (worth a look).
  - Ablation ideas not built yet: regret vs the CEM teacher on the same state, the previous best z
    as an extra sample (warm start), inference time vs K / bf16 / TensorRT, storing the K raw
    samples.
  - **Test comparison:** `bash fiss_plus_planner/scripts/run_test_comparison.sh`
    (`PLANNERS="CVAE_FOP CEM"` for a subset, `SHARDS=n`, default 22).
    - Runs `cfgs/cvae_fop.yaml`, `cfgs/cem_test.yaml` (CEM R8x250) and `cfgs/fop_test.yaml`
      (FOP 10x10x10) one after the other. The configs differ only in PLANNER and paths.
    - Then `scripts/compare_planners.py --ref CEM` writes `data/measurements/compare_test/`
      (report.txt, summary.csv, paired_costs.csv):
      - success (rear-end / other failures), FOP fallback share, runtime per cycle;
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
- **Planned: route condition (`--nav`)**, taken from Giovanni's world-model CVAE.
  - The BEV images do not show the route, so at intersections the CVAE cannot see which branch is
    planned.
  - Add a navigation tokenizer (MLP + position embedding over the resampled reference path:
    2 m, 20 points, deltas). Add cross-attention to these tokens in the encoder and decoder, and
    condition the prior on them.
  - Compare image-only vs image + route.
  - Needs the v2 data (`conditions.parquet`) in the cache (`ref_path.bin`). The inference interface
    gets `generate_samples(frames, k, ref_path=...)`, computed with the same `reference_ahead_local`.
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

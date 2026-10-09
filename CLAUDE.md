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
  - Needs C_Planner built with `-DENABLE_DATA_COLLECTION=ON`; rebuild OFF before timing runs.
  - Recording evaluates every check (kFullViolation) but ranks the early-exit view, so its
    decisions are bitwise identical to a normal run.
  - Guide: README.md "Collecting CVAE training data with CEM".
- **Output** in `data/output/cem_train_R8x250/`:
  - `cem_data/<scenario>/{contexts,candidates,proposals}.parquet`, `scenario.json`;
  - `imgs/<scenario>/<t>.png`;
  - `dataset_info.json`.
- **State (complete, 2026-10-09):** 6808 of 6832 scenarios. The other 24 fail before planning
  (18 without a global route, 6 without an ego trajectory) and have no data.
  - The missing ones were recollected with `cfgs/rerun_missing.yaml` (not committed).
  - An integrity scan of all PNG and parquet files was clean.
  - **Possible SSD/RAM fault on the collection PC:** 3 files had changed on disk without being
    rewritten; they were recollected. Verify with sha256 after every copy.
- **Statistics:** 86.1 % success; 945 failures (419 rear-end, 316 failing at cycle 0);
  519,001 cycles; 826 M feasible candidates; 54 GB cem_data plus 5.2 GB images. The cache
  (cem_cache.py) is 48 GB, built in 73 s with nothing skipped.
- **Splits:**
  - data/10k Train / Verification / Test are disjoint.
  - data/demo/Test (used for all tuning) overlaps 10k/Train by 57 scenarios. Evaluate the CVAE
    planner on 10k/Test, or on demo/Test without the overlaps.
  - Verification has not been collected yet (optional validation set).

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
  - `test_cost_cvae.py`: held-out check against uniform and marginal samplers.
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
- **Open issue:** the 2-epoch smoke test collapsed to almost one point per scene (sample spread
  0.03 vs target 0.61, posterior collapse with `--beta_end 1.0`).
  - The pipeline itself works: the sample mean follows the scene.
  - Watch `spread` in `test_cost_cvae.py`; try `--beta_end 0.1` or `--model hcvae`.
- **Not done yet:** the CVAE-FOP planner (sampling with CostAwareCVAE, dense-FOP fallback when
  no sample is feasible). It needs:
  - clipping of the samples to the search space;
  - the 3-frame history padded as [0, 0, 1] at t = 1 (the old sparse planners use [0, 1, 1]).

  Planned evaluation metrics: best cost@K, feasible ratio, fallback rate, regret vs dense FOP.
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
tar --exclude='*.gif' -cf - cem_train_R8x250/cem_data cem_train_R8x250/imgs cem_train_R8x250/completed \
  | zstd -T0 -3 -o cem_train_R8x250.tar.zst
sha256sum cem_train_R8x250.tar.zst > cem_train_R8x250.tar.zst.sha256
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
cd $SCRATCH && sha256sum -c cem_train_R8x250.tar.zst.sha256 && zstd -dc cem_train_R8x250.tar.zst | tar -xf -
python <repo>/CVAE_efficient_sampling/cem_cache.py --data $SCRATCH/cem_train_R8x250 --out $SCRATCH/cem_train_cache --workers 16
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

**After the CVAE training:** build the CVAE-FOP planner (see "Not done yet" above) and evaluate
it in closed loop on data/10k/Test.

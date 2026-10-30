# Cost-aware CVAE proposals for sampling-based Frenet planning: state of the findings

State: 2026-10-30. Self-contained summary for an external reviewer of the research direction. The project-internal
log with every path and command is `CLAUDE.md` (repository root); per-ablation records are in `results/ablations/`.

## 1. Goal of the paper

A sampling-based trajectory planner in Frenet space evaluates candidate trajectories, each defined by its sampling
parameters z = [d, v, T] (lateral offset, end speed and duration of a quintic/quartic polynomial, as in FOP,
Werling et al. 2010). A learned sampler (a conditional VAE, CVAE) proposes K = 16 values of z per planning cycle from
the scene, and the planner executes the best feasible one. **The research question is how to train the CVAE so that
the cost of its own samples gets as close as possible to that of a strong optimiser (CEM).** Target venue: IROS.
Methods must come from the literature (no invented algorithms); the novelty is in the problem analysis and in what
the sampler is trained on.

## 2. System

**Planners** (one C++ backend for all; same trajectory generator, checks and cost):
- FOP: dense grid of z (10 x 10 x 10).
- FISS+ (re-implementation of the FISS+ paper): coarse-to-fine grid search, ~250 trajectories per cycle.
- CEM: cross-entropy method over z, 8 iterations x 250 samples, elite 0.1, smoothing 0.7, init std 0.3, warm start
  from the previous cycle's executed solution. **CEM is the teacher of the CVAE and the main baseline.**
- CVAE-FOP: the CVAE's K = 16 samples plus the previous cycle's executed z ("warm start", sample 17).

**Ranking and cost.** Candidates are ranked lexicographically: feasible before infeasible, feasible ones by cost J.
J follows the FISS+ paper (Eq. 4) plus a lane-centre term: w_T = 0.1 (time), w_V = 0.01 (squared speed deviation
from v_des), w_A = 0.1, w_J = 0.1 (jerk), w_D = 1.0, w_dist = 0.1, w_LC = 10. Scenario cost J_total = running cost of
the executed trajectory + w_T * t_final.

**Safety (clearance) constraint** for all planners: keep a gap of d0 + tau_gap * v (d0 = 3 m, tau_gap = 2 s, lateral
margin 0.2 m, grace 1 s, recovery within 3 s) to obstacles ahead in the ego path. If no candidate keeps it, a planner
re-plans without it (clearance fallback, also in the teacher CEM). In ~10 % of the teacher's cycles this happens.

**Scenarios.** CommonRoad, 10k split: Train 6832, Verification 1952, Test 976 scenarios (disjoint); recorded traffic,
not reactive. A run ends at the goal region / map end, at the time limit, or when no plan is found.

**Training data.** The teacher CEM drives every Train scenario; every candidate of every cycle is stored with its J,
feasibility and CEM's per-iteration sampling distributions (v3: 439k cycles, 6808 scenarios). Condition = three
consecutive 256x256 bird's-eye-view images.

**Model.** attnCVAE: ResNet18 image encoder, one latent level, 11.3 M parameters; trained with the ELBO (beta
annealed to 0.1; beta 1.0 collapsed the posterior).

**Training target (the core of the method).** Per cycle, one feasible CEM candidate is drawn as the regression
target with probability w ~ exp(-J~ / tau) / q(z) and the model is trained with the usual ELBO (importance
resampling; in expectation the cost-weighted ELBO):
- J~ = (J - J_min) / (J_90 - J_min) within the cycle (normalised cost);
- q(z) = CEM's mixture sampling density over its iterations (balance heuristic, Veach & Guibas 1995). Dividing by q
  undoes CEM's concentration of samples near the optimum, so the target is the Boltzmann distribution
  exp(-J~/tau) over the feasible search space, independent of how CEM searched.

**Planner used for evaluation (reference setup since 2026-10-15).** 17 samples evaluated with the clearance
constraint; if none is feasible, **fallback = the teacher CEM** (same rules, warm-started from the last executed
solution). A test with zero CVAE samples reproduces the CEM baseline bit for bit.

**Metrics** (Test, closed loop):
- fallback = share of cycles in which no CVAE sample was feasible (CEM planned);
- interrupted = share of scenarios that ended because no plan was found;
- J_total paired on the scenarios both planners complete (mean difference, 95 % CI, Wilcoxon p), plus cost terms.
The ± in the tables is the 95 % CI over scenarios; it does **not** include training randomness (one training seed).

## 3. Results

### 3.1 Baselines (Test, all planners complete 846-850 scenarios)

| Planner | J_total vs CEM | trajectories / cycle | interrupted |
|---|---|---|---|
| CEM (teacher) | 0 (J ~ 9.6) | 2180 | 11.3 % |
| FISS+ | +0.23 | 250 | 10.9 % |
| FOP 10x10x10 | +2.65 | 1094 | 10.2 % |
| **CVAE + CEM fallback, model B (current best)** | **+0.75** | 17 + fallback | 11.8 % |

CEM and MPPI tie and are below FISS+ and FOP at every budget from 200 to 27000 trajectories per cycle.

### 3.2 Ablations (closed loop on Test, paired difference in J_total; negative = better)

Until entry 10 the fallback was the FOP grid; entries 11-12 use the CEM fallback.

| # | Change | Result | Verdict |
|---|---|---|---|
| 1 | Warm start: previous executed z as sample 17 | -0.63 ± 0.20 | kept |
| 2 | Ego state (speed, acceleration, Frenet state, v_des) as extra input | +0.58 ± 0.17 | worse: causal confusion (de Haan 2019, Codevilla 2019): the model copies its own slow speed and creeps |
| 3 | Only task-related ego inputs | worse than #1 | negative |
| 4 | New data collection v3 (episodes end at the goal) | +0.29 ± 0.13 | unclear (one seed); v3 has 1/3 fewer low-speed cycles |
| 5 | Planned route (reference path ahead) as input | +0.008 ± 0.128 | no effect: z lives in the Frenet frame of the route already |
| 6 | K = 32 instead of 16 | -0.077 ± 0.071 | small: sample count is not the bottleneck |
| 7 | DAgger round 1 (Ross et al. 2011): CEM labels the states the CVAE drives into | -0.214 ± 0.111 | see #8 |
| 8 | Control: two target draws per cycle, no new states | -0.149 ± 0.092; DAgger vs control -0.066 ± 0.093 (n.s.) | most of DAgger's gain was the doubled training items |
| 9 | A: target without /q (tau 0.25) | -0.281 ± 0.152 vs #8 | better |
| 10 | B: target with /q, tau 0.05 | -0.259 ± 0.150 vs #8; equal to A (+0.022 ± 0.079) | better; chosen (clean target) |
| 11 | CEM fallback instead of the FOP grid (model B) | gap to CEM +1.13 -> +0.75 | reference setup |
| 12 | tau 0.25 vs 0.05, both with CEM fallback | B **-0.483 ± 0.122** (p 3e-17) | target effect in the final setup |

| 13 | Train only on scenarios the teacher completes (`--success_only`, 5984 of 6808 Train scenarios), evaluated on the Test scenarios CEM completes (861) | -0.075 ± 0.058 vs B (p 0.001); gap to CEM +0.75 -> +0.68; interrupted 7 -> 5 (4 of them rear-end) | small gain, one seed |

On the 861 Test scenarios CEM completes (2026-10-30): J_total vs CEM FISS+ +0.23, CVAE (#13) +0.68, FOP +2.66;
interrupted FISS+ 2, CVAE 5, FOP 8. FISS+ remains ahead of the CVAE in cost, reliability and trajectories per cycle.

Entry 12 in detail (only tau differs):

| | tau 0.25 (draws2_cem) | tau 0.05 (B_cem) |
|---|---|---|
| J_total vs CEM | +1.23 | +0.75 |
| J_V (speed term) | 6.06 | 5.66 (CEM 5.18) |
| fallback | 16.7 % | 21.9 % |
| interrupted | 12.0 % | 11.8 % |
| reached the goal | 69.3 % | 71.0 % (CEM 74.0 %) |

### 3.3 Offline analysis of the training target

On all 127,508 Verification states, the expected cost of the best of K draws from the training target itself (what
a perfect CVAE would reach), minus CEM's best (mean CEM plan cost J ~ 4.1):

| tau | with /q, K = 16 | without /q, K = 16 |
|---|---|---|
| 0.05 | 0.059 | 0.009 |
| 0.10 | 0.118 | 0.015 |
| 0.25 | 0.231 | 0.021 |
| 0.50 | 0.362 | 0.026 |

With /q and tau 0.25: K = 1/4/16/64 gives 0.86/0.41/0.23/0.14 (without /q 0.18/0.055/0.021/0.009). The original
target (tau 0.25 with /q) puts only 6 % of its draws into the best region (J~ < 0.1); even a perfect model would pay
~5.6 % per cycle. Offline best_cost@16 of the trained models (Verification, normalised J~): tau 0.25 0.166, B 0.076,
A 0.042; the models are narrower than their targets (spread 60-76 % of the target's).

### 3.4 Diagnostics of the remaining gap

- **Low-speed creeping** (tau 0.25 model, FOP fallback): 47 % of the gap to CEM was speed cost accumulated below
  1 m/s (CVAE 10.2 % of cycles, CEM 5.7 %), concentrated in 10 % of the scenarios (e.g. stopping 6 m behind a
  standing car instead of entering the goal region). The narrower target (A) halved this excess; DAgger reduced it
  by a quarter. So creeping came mostly from the broad target, not mainly from distribution shift.
- **Clearance-boundary states** (model B, CEM fallback): the CVAE falls into fallback in 3 % of the cycles after a
  CVAE cycle, but once CEM plans, the next cycle is again fallback in 90 % (episodes 8.9 cycles on average). CEM
  drives the car into states close to the clearance limit (close following); there 94 % of the CVAE's 17 samples
  violate the clearance constraint. The narrower target raises this (fallback 16.7 % -> 21.9 %).
- Candidate cause: ~10 % of the training targets come from CEM's re-plan without clearance (the teacher dropped the
  constraint in those cycles), so the CVAE learns clearance-violating proposals in such states. Not tested yet.
- Offline metrics misled twice (ego input better offline, worse in closed loop); conclusions are drawn from closed
  loop only.

## 4. Current conclusions

1. **The training target is the main lever.** Making the cost-weighted target narrower (tau 0.25 -> 0.05) lowers
   J_total by 0.48 in the final setup, closing ~39 % of the gap to CEM; the offline target analysis predicted this.
   The target was calibrated for coverage of all good solutions, while the planner uses the best of K.
2. More inputs (ego state, route) do not help or hurt; more samples (K 32) help little; DAgger's on-policy effect is
   small. The image condition is sufficient for z.
3. **Trade-off:** narrower targets fail more often at the clearance boundary (more fallback), without more
   interrupted scenarios.
4. Remaining gap: +0.75 to CEM, +0.52 to FISS+, mostly the speed term J_V.

## 5. Open problems and candidate next steps

- Sample quality at constraint boundaries: e.g. train only on clearance-keeping targets (feasibility labels exist
  per candidate), or a target that accounts for the best of K (e.g. "best of many" objective, Bhattacharyya et al.,
  CVPR 2018). Measure by fallback share and paired J_total.
- Isolate the sampler from the fallback: the "shadow" evaluation (CEM drives, the CVAE proposes in the same states,
  regret of its best sample) is implemented but not run yet.
- FISS+ reaches +0.23 with ~250 trajectories and no learning: the paper must position the CVAE against it.

## 6. Threats to validity

- One training seed per configuration: differences around 0.1-0.3 may be seed noise (the ± does not cover it).
- Several design choices were made by looking at Test results; final numbers should come from a selection on
  Verification and a single final Test run.
- Runtimes were measured with parallel processes and a debug-capable build; not valid for timing claims.
- The recorded traffic is not reactive (rear-end failures by followers are labelled separately).

## 7. Where things are

- Project log and decisions: `CLAUDE.md`. Ablation records: `results/ablations/<name>/README.md`, CVAE v1:
  `results/cvae_v1/`.
- Planner: `fiss_plus_planner/planners/cvae_fop_planner.py`; evaluation: `fiss_plus_planner/scripts/run_cvae_variant.sh`,
  `compare_planners.py`, `gap_decomposition.py`, `lowspeed_gap.py`.
- Training: `CVAE_efficient_sampling/` (`cem_dataset.py` target weights, `train_cost_cvae.py`, `test_cost_cvae.py`).
- Reports: `fiss_plus_planner/data/measurements/compare_*/report.txt` (not in git).

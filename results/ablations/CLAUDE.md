# CVAE ablations: what was tested, how, and what came out

State 2026-10-30. Written for an agent that designs the next ablations. Background of the whole project:
`../cvae_findings.md` (method, metrics, numbers) and the repository's `CLAUDE.md` (decisions, paths, commands).

## The question the ablations serve

The paper studies how to train a CVAE that proposes sampling parameters z = [d, v, T] (lateral offset, end speed,
duration of a Frenet polynomial) so that **the cost of its own samples** gets close to the teacher, a CEM optimiser
(8 x 250 samples per cycle). The planner evaluates K = 16 CVAE samples plus the previous cycle's executed z ("warm
start", sample 17) with a safety-gap (clearance) constraint and executes the cheapest feasible one; if none is
feasible, the teacher CEM plans that cycle ("fallback"). The fallback is only a safety net and is not a subject of
study (its size or budget does not matter).

## Rules for a new ablation

- Change **one** thing against a stored reference run; everything else (data, model, seed, K, warm start, fallback,
  safety settings, scenarios) stays equal. Name the reference run in the design.
- Judge in **closed loop** on data/10k/Test (976 scenarios; 972 run), never by offline metrics alone (they misled
  twice, see #2). Offline numbers (Verification v3) may motivate a run, not decide it.
- Report the three metrics (scripts/compare_planners.py prints them):
  - **J_total** paired with the reference on the scenarios both complete: mean difference, 95 % CI, Wilcoxon p,
    cost terms (J_V speed, J_A, J_J jerk, J_LC, J_D, J_ter time);
  - **fallback** = share of cycles without a feasible CVAE sample (CEM planned);
  - **interrupted** = share of scenarios that ended because no plan was found.
- One training seed so far; the ± is the CI over scenarios, not over training randomness. A second seed of the
  reference is still missing; differences below ~0.2 are not safe to claim yet.
- Methods must come from the literature (cite them); no invented algorithms. A change of the planner's cost or
  parameters requires re-running every compared planner in closed loop.
- Cost per training run: ~4.5 h on one RTX 5070 Ti (870k items, 20 epochs); closed loop on Test: ~1-1.5 h.

## Reference setup (current best, "B_cem")

- Data: v3 CEM collection on data/10k/Train (6808 scenarios, 439k cycles); validation = Verification v3.
- Model: attnCVAE (ResNet18 on 3 BEV frames, one latent level, 11.3 M params), ELBO with beta annealed to 0.1,
  20 epochs, lr 1e-4, batch 64, seed 0.
- Target: per cycle one feasible CEM candidate drawn with w ~ exp(-J~/tau) / q(z) (J~ = cost normalised in the
  cycle, q = CEM's sampling density; importance resampling), **tau 0.05**, `--draws_per_cycle 2`.
- Run `fiss_plus_planner/data/output/cvae_local/runs/v3_draws2_tau0.05_beta0.1`; planner config
  `fiss_plus_planner/cfgs/cvae_fop.yaml` (K 16, warm start, `fallback: cem`).
- Test: J_total +0.75 vs CEM (FISS+ +0.23), fallback 21.9 %, interrupted 11.8 % (CEM 11.3 %).

How to run an ablation (from the repository root):

```bash
# training (in CVAE_efficient_sampling/): the reference plus ONE changed flag
python train_cost_cvae.py --cache ../fiss_plus_planner/data/output/cvae_local/cem_train_cache_v3 \
  --val_cache ../fiss_plus_planner/data/output/cvae_local/cem_verif_cache_v3 --out <run dir> \
  --tau 0.05 --beta_end 0.1 --draws_per_cycle 2 [CHANGED FLAG]
# closed loop on Test (any key of cfgs/cvae_fop.yaml can be overridden)
EXTRA="B_cem=data/measurements/cvae_B_cem" bash fiss_plus_planner/scripts/run_cvae_variant.sh NAME CVAE_FOP.run=<run dir> [KEY=VALUE]
# paired comparison against the reference
python fiss_plus_planner/scripts/compare_planners.py --run B_cem=... --run NAME=... --ref B_cem
```

## Ablations done (Test, paired difference in J_total; negative = better)

#1-#10 used the old fallback (FOP 10x10x10 grid); #11-#12 the CEM fallback.

| # | Hypothesis / change | Reference | Result | Verdict and reason |
|---|---|---|---|---|
| 1 | Warm start: previous executed z as sample 17 | CVAE v1 | -0.63 ± 0.20; fallback 22 -> 12 % | kept |
| 2 | Ego state (v, a, Frenet state, v_des) as 8 extra tokens | v1 + warm | +0.58 ± 0.17 | negative: causal confusion / inertia (de Haan 2019, Codevilla 2019); offline speed error was better (0.77 vs 1.12 m/s) |
| 3 | Only task inputs (v_des, d, d_d, d_dd), no speed feedback | v1 + warm | worse than reference | negative |
| 4 | New collection v3 (episodes end in the goal region) | v1 + warm | +0.29 ± 0.13 | unclear: one seed; v3 has 1/3 fewer low-speed cycles |
| 5 | Planned route (20 points ahead) as extra tokens | v3 image-only | +0.008 ± 0.128 | no effect: z is in the route's Frenet frame already |
| 6 | K = 32 samples | v3, K 16 | -0.077 ± 0.071 | small: sample count is not the bottleneck |
| 7 | DAgger round 1 (Ross et al. 2011): CEM labels the states the CVAE drives into, train on the aggregate | v3 | -0.214 ± 0.111 | confounded by 2x training items, see #8 |
| 8 | Control of #7: `--draws_per_cycle 2`, no new states | v3 | -0.149 ± 0.092; #7 vs #8 -0.066 ± 0.093 (n.s.) | DAgger's own effect is small |
| 9 | A: target without the density correction (`--no_density_correction`) | #8 | -0.281 ± 0.152 | positive: narrower target |
| 10 | B: keep /q, tau 0.05 | #8 | -0.259 ± 0.150; = A (+0.022 ± 0.079) | positive; chosen (target independent of CEM's sampler) |
| 11 | Fallback CEM instead of the FOP grid (model B) | B with FOP fallback | gap to CEM +1.13 -> +0.75 | new evaluation setup |
| 12 | tau 0.25 vs 0.05, both with CEM fallback | B_cem | tau 0.25 +0.483 ± 0.122 (p 3e-17); fallback 16.7 vs 21.9 %; interrupted 12.0 vs 11.8 % | target effect holds in the final setup; narrower = cheaper but more fallback |
| 13 | Train only on scenarios the teacher completes (`--success_only`), evaluated on Test_cem_success (861) | B_cem (= B_v3 there) | -0.075 ± 0.058 (p 0.001); gap to CEM +0.75 -> +0.68; interrupted 7 -> 5 (4 rear-end); fallback 20.5 -> 21.7 % | small gain, one seed |

Earlier trainings (v2 data, offline only): tau 0.5 / beta_end 1.0 collapsed the posterior (sample spread 0.085 vs
0.571 of the target); beta_end 0.1 fixed it.

Since #13 the user evaluates on `data/10k/Test_cem_success` (the 861 Test scenarios CEM completes; CEM success 100 % by
construction) with `fiss_plus_planner/scripts/run_test_set.sh` (CEM, FOP, FISS+, B_v3, B_succ from their earlier
configs; they reproduce the full-Test results exactly). There: FISS+ +0.23 vs CEM with 2 interruptions, ahead of every
CVAE in cost, interruptions and trajectories per cycle.

## What the ablations established

1. **The training target is the lever.** Narrowing the cost-weighted target (tau 0.25 -> 0.05 with /q, or dropping
   /q) is the largest model-side gain (-0.26 with FOP fallback, -0.48 with CEM fallback). Offline, a perfect model of
   the old target (tau 0.25, /q) would still be 0.23 per cycle above CEM with the best of 16 (CEM plan cost ~4.1);
   tau 0.05 /q: 0.059; no /q: 0.021. The old target spread its mass over all good regions while the planner needs
   one good sample among 16.
2. Extra inputs do not help (#2, #3, #5); more samples barely help (#6); on-policy data (DAgger) adds little beyond
   more training items (#7, #8).
3. Low-speed creeping (the main J_V gap of the tau 0.25 models) was mostly caused by the broad target: the narrower
   target halved it, DAgger cut a quarter.
4. **Trade-off at the safety constraint:** narrower targets fail more often at the clearance boundary. With model B,
   after a CEM fallback cycle the next one is again a fallback in 90 % of the cases (episodes 8.9 cycles on
   average); CEM drives close to the clearance limit and 94 % of the CVAE's samples then violate it.
5. The trained models are narrower than their targets (spread 60-76 % of the target's) and farther from them the
   narrower the target is (offline best_cost@16: B 0.076 vs its target 0.029; A 0.042 vs 0.006).

## Open design questions for the next ablations

- **Clearance boundary:** ~10 % of the training targets come from the teacher's re-plan without the clearance
  constraint (candidates table column `pass` = 1). Does training only on clearance-keeping targets (`pass` 0)
  lower the fallback share at equal cost? Needs a cache option in `CVAE_efficient_sampling/cem_cache.py`.
- **Best-of-K objective:** the planner uses the minimum cost of K samples; a target or loss that optimises the best of
  K (e.g. "best of many" sample objective, Bhattacharyya et al., CVPR 2018) instead of a temperature choice.
- **Tau sweep** between 0.25 and 0.05 (and below 0.05) with the CEM fallback: the J_total vs fallback trade-off curve.
- **Model fidelity:** the narrower the target, the larger the model's distance to it; capacity / longer training /
  more draws per cycle (`--draws_per_cycle 4`) are candidates.
- **Seeds:** a second seed of B_cem (and of the tau 0.25 model) to size the training noise before any new claim.
- **Sampler-only metric:** the "shadow" mode (CEM drives, the CVAE proposes in the same states, regret of its best
  sample, warm start off; `fiss_plus_planner/scripts/run_gap_diagnosis.sh`, `MODES=expert`) separates sample quality
  from how often the fallback helps; implemented, not yet run.

## Records

- `results/ablations/{warm_start,ego_state_full,dagger1,success_only}/README.md` (configs, reports, measurements).
- Training runs: `fiss_plus_planner/data/output/cvae_local/runs/` (v3_attn_tau0.25_beta0.1 = v3 image-only,
  v3_attn_route_..., v3_dagger1_..., v3_draws2_tau0.25_beta0.1 = #8, v3_draws2_noq_tau0.25_beta0.1 = A,
  v3_draws2_tau0.05_beta0.1 = B, succ_draws2_tau0.05_beta0.1 = #13); frozen models:
  `fiss_plus_planner/data/models/{cvae_v1,ablation_ego_full,dagger1,succ_draws2_tau005}`.
- Closed-loop reports: `fiss_plus_planner/data/measurements/compare_<name>/report.txt` (not in git).

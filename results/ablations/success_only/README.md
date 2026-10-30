# Ablation: train only on scenarios the teacher completes (2026-10-29/30)

**Question:** Does the CVAE improve if the training data contain only the scenarios in which the CEM teacher
succeeded (no rear-end failures, no other failures)? Before, failed scenarios contributed 3 % of the training cycles
(rear-end failures minus the failing cycle, other failures minus their last 2.5 s).

**Answer:** Slightly. On the Test scenarios CEM completes, the success-only model is **-0.075 ± 0.058** lower in
J_total than the same model trained on all scenarios (p 0.001), mostly in J_V. One training seed only.

## Design

- **Scenario sets** (`fiss_plus_planner/data/10k/*_cem_success`, not in git; outcome lists with reasons in
  `data/10k/cem_outcomes/`): the scenarios CEM R8x250 completes (goal, time limit or map end).
  Train 5984 of 6832, Verification 1711 of 1952, Test 861 of 976. Source: the scenario.json of the v3 CEM collections;
  8 Verification/Test scenarios were rerun with the current code and gave identical outcomes.
- **Training (B_succ):** model B settings (attnCVAE, target exp(-J~/tau)/q with tau 0.05, `--draws_per_cycle 2`,
  beta_end 0.1, 20 epochs, seed 0) plus `--success_only`: the v3 caches restricted to the successful scenarios
  (422,413 training cycles instead of 435,270; validation on Verification_cem_success, 121,751 cycles). No new data
  collection: the CEM data of these scenarios are the same.
- **Reference (B_v3):** model B trained on all v3 Train scenarios (`runs/v3_draws2_tau0.05_beta0.1`).
- **Test:** all planners on `data/10k/Test_cem_success` with `fiss_plus_planner/scripts/run_test_set.sh`, each from the
  config of its earlier Test run (`configs/`). CEM, FOP, FISS+ and B_v3 reproduce their earlier Test results on these
  scenarios exactly (same endings, J difference 0).
- **Frozen model:** `fiss_plus_planner/data/models/succ_draws2_tau005/` (not in git; SHA256SUMS read with O_DIRECT;
  `model_best.pth` 97517872...3a77f4, equal to the hash written at training time). The training was interrupted by a
  system crash in epoch 20 and resumed from the epoch-19 checkpoint (data and checkpoint verified intact).
- **Archive:** all measurement CSVs and the CVAE cycle logs in `fiss_plus_planner/data/results/success_only/`
  (SHA256SUMS).

## Results

**Offline** (`eval_verification.txt`, Verification_cem_success, 20,000 cycles, K 16): best_cost@16 0.078 (target 0.029),
mean_dist 0.254, spread 0.218 (target 0.329), in_bounds 0.922. Not comparable with B's offline numbers, which were
computed on all Verification scenarios.

**Closed loop, Test_cem_success, 861 scenarios** (`report.txt`):

| | J vs CEM (845 solved by all) | all reached the goal (618) | interrupted | fallback | trajectories / cycle |
|---|---|---|---|---|---|
| CEM | 0 (J 9.68) | 0 | 0 (by construction) | | 2160 |
| FISS+ | +0.23 ± 0.10 | +0.06 | 2 | | 231 |
| **B_succ** | **+0.68 ± 0.10** | **+0.51** | 5 (4 rear-end) | 21.7 % | 608 |
| B_v3 | +0.75 ± 0.10 | +0.63 | 7 (1 rear-end) | 20.5 % | 580 |
| FOP 10x10x10 | +2.66 ± 1.02 | +1.46 | 8 | | 1073 |

**Paired B_succ vs B_v3:** -0.075 ± 0.058 on 852 scenarios solved by both (p 0.001, lower in 45 %), -0.126 ± 0.051 on
the 685 where both reached the goal; J_V 5.59 vs 5.66.

- Interrupted scenarios: B_v3 fails the 6 "other" failures analysed before (CVAE drives into a state CEM cannot rescue,
  e.g. DEU_Bremen-44: a faster recorded follower) plus 1 rear-end. B_succ fails HRV_Centar-59 and 4 rear-ends
  (DEU_Nuremberg-16 turned into a rear-end; ESP_Vigo-93 and DEU_Lengede-19 are new).
- FISS+ stays ahead of both CVAE models in cost, interruptions and trajectories per cycle.

## Caveats

- One training seed; a difference of 0.075 may be within training noise.
- The Test set is selected by the CEM outcome, so CEM's success is 100 % by construction; compare costs and
  interruptions, not success rates against CEM.
- Runtimes come from parallel processes (22 / 12) and are not timing results.

Settings of all numbers: SAFETY tau 2 s, d0 3 m, lateral margin 0.2 m, grace 1 s, T_rec 3 s, clearance_fallback on;
CEM R8x250; CVAE-FOP K 16 + warm start + teacher-CEM fallback; FOP 10x10x10.

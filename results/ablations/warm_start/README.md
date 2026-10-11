# Ablation: warm start of CVAE-FOP (2026-10-11)

**What changed against CVAE v1:** the previous cycle's executed z = (d, v, T) is added as sample
K + 1 (`CVAE_FOP.warm_start: true`). It is clipped and evaluated together with the K = 16 CVAE
samples. CEM uses the same idea: its proposal starts at the previous best.

The model, data, settings and the CEM / FOP baselines are the same as in `results/cvae_v1`. CEM and
FOP were not rerun; they are deterministic, and their code and settings did not change.

**Run:** `bash fiss_plus_planner/scripts/run_cvae_variant.sh warm CVAE_FOP.warm_start=true`, on
2026-10-11 from 14:17 to 14:31 (22 processes).
- `config.yaml`: the exact config of the run.
- `measurements/`: the per-scenario CSVs.
- `report.txt`: the comparison with CVAE v1, CEM and FOP.
- Per-cycle logs: `fiss_plus_planner/data/output/cvae_warm/cvae_cycles` (not in git).

## Result on data/10k/Test (972 scenarios)

| | CVAE v1 | + warm start | CEM R8x250 | FOP 10x10x10 |
|---|---|---|---|---|
| Success | 87.6 % | 87.6 % | 86.1 % | 88.2 % |
| FOP fallback share of cycles | 22.4 % | **11.6 %** | | |
| J_total, 820 scenarios solved by all | 13.86 | **13.21** | 11.84 | 14.57 |
| Difference to CEM | +2.02 ± 0.32 | **+1.37 ± 0.20** | | +2.73 ± 1.03 |
| J_V | 7.93 | 7.41 | 6.44 | 8.15 |
| Runtime mean / median* | 22.8 / 11.9 ms | 18.6 / 10.2 ms | 123 / 98 ms | 60 / 49 ms |

Paired against CVAE v1 (845 scenarios solved by both), the warm start changes J by
**-0.63 ± 0.20** (median -0.04; lower in 66 % of the scenarios; Wilcoxon p 5e-21).

The warm-start sample is chosen in 78 % of the cycles that the samples solve. So the 16 CVAE samples
are rarely better than keeping the previous plan, which points at the sample quality (see
`results/cvae_v1/README.md`, hypotheses for v2).

\* With 22 parallel processes, the runtimes only show relative size.

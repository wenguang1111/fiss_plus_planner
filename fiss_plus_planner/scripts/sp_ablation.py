"""Ablation of the Sparse Planner's learned sampling distribution (runs of run_paper_table.sh in data/measurements/sp_ablation):
closed loop without fallback (cvae from paper_table_sp_nofallback, marginal / uniform / mean) and the same-state evaluation
(output/shadow_eval.csv: along SP's closed loop, every cycle the N samples of each sampler and the 10^3 grid evaluated by the
same C++ FOP pipeline, J_D off)."""
import argparse, glob, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="data/measurements/sp_ablation")
ap.add_argument("--cvae", default="data/measurements/paper_table_sp_nofallback")
args = ap.parse_args()
DROP = {"AUT_Haag-15_1_T-9", "ESP_Barcelona-48_3", "DEU_Weimar-84_1"}  # FOP 125's failures


def load(path):
    d = pd.read_csv(path, skipinitialspace=True); d.columns = [c.strip() for c in d.columns]
    d["scenario"] = d["scenario"].str.replace(r"(_T-1)?\.xml", "", regex=True)
    d["ok"] = d.success.astype(str).str.strip() == "True"
    return d.set_index("scenario")


runs = {}
for n in [1, 8, 64]:
    runs[("cvae", n)] = load(glob.glob(f"{args.cvae}/SP_{n}/measurement_*.csv")[0])
for path in glob.glob(f"{args.dir}/*_nofallback/SP_*/measurement_*.csv"):
    sampler = os.path.basename(os.path.dirname(os.path.dirname(path))).split("_")[0]
    runs[(sampler, int(os.path.basename(os.path.dirname(path)).split("_")[1]))] = load(path)

print("== Closed loop without fallback: scenarios completed (26 / without FOP 125's 3 failures 23)")
rows = []
for (sampler, n), d in sorted(runs.items()):
    ref = runs[("cvae", n)]
    c = d.index[d.ok & ref.reindex(d.index).ok]
    rows.append(dict(sampler=sampler, N=n, ok26=int(d.ok.sum()), ok23=int(d.ok[~d.index.isin(DROP)].sum()),
                     common_with_cvae=len(c), J=d.loc[c, "final_trajector_cost"].mean(), J_cvae=ref.loc[c, "final_trajector_cost"].mean(),
                     lower_than_cvae=int((d.loc[c, "final_trajector_cost"] < ref.loc[c, "final_trajector_cost"]).sum())))
print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\n== Same-state evaluation along SP's closed loop (cvae + fallback, Table I trajectories)")
for path in sorted(glob.glob(f"{args.dir}/shadow_eval*/SP_*/output/shadow_eval.csv")):
    n = int(path.split("/SP_")[1].split("/")[0])
    s = pd.read_csv(path, header=None, names=["scenario", "t", "sampler", "n", "n_feasible", "best"])
    p = s.pivot_table(index=["scenario", "t"], columns="sampler", values=["n_feasible", "best"], aggfunc="first")
    best, nf = p["best"], p["n_feasible"]
    grid = best["grid"]
    out = []
    for smp in ["cvae", "marginal", "uniform", "mean", "grid"]:
        feas = np.isfinite(best[smp])
        both = feas & np.isfinite(grid)
        reg = (best[smp] - grid)[both]
        out.append(dict(sampler=smp, cycles=len(best), cycle_feasible=100 * feas.mean(),
                        sample_feasible=100 * (nf[smp] / (1000 if smp == "grid" else n)).mean(),
                        regret_mean=reg.mean(), regret_median=reg.median(), regret_p90=reg.quantile(0.9),
                        within_1=100 * (reg <= 1.0).mean(), beats_grid=100 * (reg < -1e-9).mean()))
    print(f"-- N = {n}  ({len(best)} cycles, {best.index.get_level_values(0).nunique()} scenarios; regret = best cost - best of the 10^3 grid, J_D off)")
    print(pd.DataFrame(out).round(2).to_string(index=False))
    # all four N-samplers feasible in the same cycle: their best costs side by side
    allf = np.isfinite(best[["cvae", "marginal", "uniform", "mean"]]).all(axis=1)
    print(f"   cycles where cvae, marginal, uniform and mean all have a feasible sample: {allf.sum()}; mean best cost there:",
          best.loc[allf, ["cvae", "marginal", "uniform", "mean", "grid"]].mean().round(2).to_dict())
    w = {smp: int((best.loc[allf, "cvae"] < best.loc[allf, smp] - 1e-9).sum()) for smp in ["marginal", "uniform", "mean"]}
    print(f"   cvae strictly better than: {w} of {allf.sum()} cycles")

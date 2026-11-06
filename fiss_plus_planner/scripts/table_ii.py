"""Table "performance under sample number N" from the runs of scripts/run_table_ii.sh.

J_all  mean final trajectory cost over all scenarios with a row (the old paper's J; a failed scenario counts with the cost
       up to the failure)
J      mean final trajectory cost; per N column over the scenarios that every method of that column completes
       (n_common), so the methods of a column are compared on the same scenarios
t      mean planning time per cycle [ms] without each scenario's first cycle (SP / CVAE: CVAE inference + C++ evaluation +
       fallback, without the BEV image and its preprocessing)
p_B    SP: share of all cycles that fall back to the FOP grid [%]
N_fail share of the 26 scenarios not completed [%], in the LaTeX table followed by their number; a scenario that fails at time step 0 has no row in the measurement
       CSV and counts as failed
Grid planners (FOP, FISS+) have odd N_W (the lane centre d = 0 is a sample) and N_S, N_T >= 2: 12 (3x2x2) -> column 8, 18 -> 16, 60 -> 64, 125 -> 128,
252 -> 256, 504 -> 512, 990 -> 1024.
"""
import argparse, ast, glob, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="data/measurements/table_ii")
ap.add_argument("--sp_dir", default=None, help="read sp/ and cvae/ from here (e.g. data/measurements/table_ii_trt)")
ap.add_argument("--out", default=None, help="output folder (default --dir)")
args = ap.parse_args()
COLS = [1, 8, 16, 64, 128, 256, 512, 1024]
GRID_COL = {12: 8, 18: 16, 60: 64, 125: 128, 252: 256, 504: 512, 990: 1024}
METHODS = [("FOP", "grid", "FOP_CPP"), ("FISS+", "grid", "FISS+_CPP"), ("CEM", "iter", "CEM_CPP"), ("MPPI", "iter", "MPPI_CPP"),
           ("CVAE", "cvae", "SP"), ("SP", "sp", "SP")]

runs = {}
for name, sub, prefix in METHODS:
    root = args.sp_dir if args.sp_dir and sub in ("sp", "cvae") else args.dir
    for path in glob.glob(os.path.join(root, sub, f"{prefix}_*", "measurement_*.csv")):
        n = int(os.path.basename(os.path.dirname(path)).rsplit("_", 1)[1])
        d = pd.read_csv(path, skipinitialspace=True); d.columns = [c.strip() for c in d.columns]
        d["scenario"] = d["scenario"].str.replace(".xml", "")
        d["ok"] = d.success.astype(str).str.strip() == "True"
        runs[(name, GRID_COL.get(n, n) if sub == "grid" else n)] = (n, d.set_index("scenario"))
scenarios = sorted(set().union(*[d.index for _, d in runs.values()]))
for key, (n, d) in runs.items():  # no row: failed at time step 0
    d = d.reindex(scenarios)
    d["ok"] = d.ok.astype("boolean").fillna(False).astype(bool)
    d["steps"] = d.steps.fillna(1)
    d["num_FOP_intervence_for_SP"] = d.num_FOP_intervence_for_SP.fillna(0)
    runs[key] = (n, d)

out = args.out or args.dir
os.makedirs(out, exist_ok=True)
rows = []
for col in COLS:
    present = [(m, runs[(m, col)]) for m, _, _ in METHODS if (m, col) in runs]
    if not present:
        continue
    common = set.intersection(*[set(d.index[d.ok]) for _, (_, d) in present])
    for m, (n_real, d) in present:
        h = np.concatenate([np.array(ast.literal_eval(x))[1:] for x in d["runtime history [s]"].dropna()])
        rows.append(dict(method=m, N=col, N_real=n_real, n_common=len(common),
                         J=d.loc[sorted(common), "final_trajector_cost"].mean() if common else np.nan,
                         J_all=d.final_trajector_cost.mean(),
                         t_ms=1e3 * h.mean(), t_p95_ms=1e3 * np.percentile(h, 95),
                         p_B=100 * d.num_FOP_intervence_for_SP.sum() / d.steps.sum() if m == "SP" else np.nan,
                         N_fail=100 * (~d.ok).mean(), n_ok=int(d.ok.sum()), trajs=d.num_trajs_generated.mean(), n_rows=int(d["runtime history [s]"].notna().sum())))
t = pd.DataFrame(rows)
t.to_csv(os.path.join(out, "table_ii_long.csv"), index=False)
pd.set_option("display.width", 250)
for metric, nd in [("J_all", 2), ("J", 2), ("t_ms", 2), ("t_p95_ms", 2), ("p_B", 2), ("N_fail", 1), ("n_ok", 0), ("n_rows", 0), ("N_real", 0), ("trajs", 1), ("n_common", 0)]:
    p = t.pivot_table(index="method", columns="N", values=metric).reindex([m for m, _, _ in METHODS]).round(nd)
    print(metric); print(p.to_string()); print()

# LaTeX in the layout of the paper draft
order = [m for m, _, _ in METHODS]
def cell(m, col, metric, fmt):
    r = t[(t.method == m) & (t.N == col)]
    if r.empty or pd.isna(r[metric].iloc[0]):
        return "--"
    if metric == "N_fail":  # share and number of the failed scenarios
        return fmt.format(r[metric].iloc[0]) + " (%d)" % (len(scenarios) - r.n_ok.iloc[0])
    return fmt.format(r[metric].iloc[0])
lines = [r"\begin{tabular}{ll" + "c" * len(COLS) + "}", r"\toprule",
         r"& & \multicolumn{%d}{c}{$N$} \\" % len(COLS), r"\cmidrule(lr){3-%d}" % (2 + len(COLS)),
         "Metric & Method & " + " & ".join(map(str, COLS)) + r" \\", r"\midrule"]
for metric, label, methods, fmt in [("J", r"$\bar{J}$", order, "{:.2f}"), ("t_ms", r"$\bar{t}$ [ms]", order, "{:.2f}"),
                                    ("p_B", r"$p_{B}$ (\%)", ["SP"], "{:.1f}"), ("N_fail", r"$N_{fail}$ (\%)", order, "{:.1f}")]:
    for i, m in enumerate(methods):
        head = r"\multirow{%d}{*}{%s}" % (len(methods), label) if i == 0 else ""
        lines.append(f"{head} & {m} & " + " & ".join(cell(m, c, metric, fmt) for c in COLS) + r" \\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines.append(r"\end{tabular}")
common_row = " & ".join(str(int(t[t.N == c].n_common.iloc[0])) if (t.N == c).any() else "--" for c in COLS)
tex = "\n".join(lines) + f"\n% n_common (scenarios of the J average per column): {common_row}\n"
open(os.path.join(out, "table_ii.tex"), "w").write(tex)
print(tex)

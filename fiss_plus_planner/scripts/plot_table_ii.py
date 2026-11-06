"""Figure for the paper: failed scenarios (of 26) over the mean planning time per cycle, one line per planner, points =
the N columns of table_ii (from table_ii_long.csv). Replaces the old cost-runtime figure: J over the common scenarios is
only comparable within a column, the failure count is comparable everywhere."""
import argparse, os
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/measurements/table_ii_trt/table/table_ii_long.csv")
ap.add_argument("--out", default="data/measurements/table_ii_trt/table/fail_vs_runtime.pdf")
args = ap.parse_args()
t = pd.read_csv(args.csv)
# fixed categorical order (dataviz reference palette slots 1-6) + distinct markers as secondary encoding for print
STYLE = {"FOP": ("#2a78d6", "s"), "FISS+": ("#eb6834", "D"), "CEM": ("#1baf7a", "^"), "MPPI": ("#eda100", "v"),
         "CVAE": ("#e87ba4", "o"), "SP": ("#008300", "o")}
LABEL = {"CVAE": r"SP$^{*}$ (no fallback)"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, ax = plt.subplots(figsize=(3.45, 2.5))
for m, (c, mk) in STYLE.items():
    d = t[t.method == m].sort_values("N")
    ax.plot(d.t_ms, 26 - d.n_ok, color=c, marker=mk, markersize=4, linewidth=1.2, label=LABEL.get(m, m),
            markeredgecolor="#fcfcfb", markeredgewidth=0.6, zorder=3 if m == "SP" else 2)
sp = t[t.method == "SP"].sort_values("N")
for _, r in sp.iterrows():
    ax.annotate(f"{r.N}", (r.t_ms, 0), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6, color="#52514e")
ax.set_xscale("log")
ax.set_xlabel(r"mean runtime per cycle $\bar{t}$ [ms]")
ax.set_ylabel("failed scenarios (of 26)")
ax.set_ylim(-0.8, 17)
ax.grid(True, which="major", color="#e4e3df", linewidth=0.5)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(ncol=3, fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.0)
fig.tight_layout(pad=0.3)
fig.savefig(args.out)
fig.savefig(args.out.replace(".pdf", ".png"), dpi=200)
print(args.out)

"""Table I of the Sparse Planner paper from the runs of scripts/run_paper_table.sh.

Per planner and N:
    J_all    mean final_trajector_cost over every scenario with a row, as the summary rows of the paper's CSVs
    J_common mean over the scenarios that every new run completes (success), the comparison the paper text describes
    t_ms     mean of 'average runtime_plan [s]' over the scenarios (the paper's t)
    t_warm   the same without the first cycle of each scenario (numba compilation, CUDA initialisation and CVAE
             loading fall into it), from 'runtime history [s]'; for SP without the image generation only in runs
             made since 2026-11-02 (the paper's SP histories include it)
    N_fail   share of the input scenarios without success (a scenario that fails in its first cycle has no row),
             split into fail_plan (no feasible trajectory: step_number_for_break > 0, or no row) and fail_stop
             (develop ends a run as unsuccessful when the speed drops below 0.01 m/s)
    p_B      SP only: mean share of the cycles in which SP falls back to FOP ('Percent_FOP_Intervence')
--paper DIR adds the paper's own FOP / FISS+ / SP CSVs as rows with source 'paper' (not used for J_common).
Writes table_long.csv and table_paper_layout.csv (metric x method rows, N columns) into --dir.
"""
import argparse
import ast
import glob
import os
import re

import numpy as np
import pandas as pd


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    d = d.dropna(subset=["scenario"]).drop_duplicates("scenario", keep="last").set_index("scenario")
    d["ok"] = d.success.astype(str).str.strip().eq("True")
    return d


def warm(history: str) -> float:
    h = ast.literal_eval(history)
    return float(np.mean(h[1:])) if len(h) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/measurements/paper_table")
    ap.add_argument("--input", default="data/demo/UsedForSelfEvaluation")
    ap.add_argument("--paper", default=None, help="folder of the paper's CSVs (rows marked 'paper')")
    args = ap.parse_args()
    num_inputs = sum(f.endswith(".xml") for f in os.listdir(args.input))

    runs = {}
    for path in glob.glob(os.path.join(args.dir, "*_*", "measurement_*.csv")):
        planner, n = os.path.basename(os.path.dirname(path)).rsplit("_", 1)
        runs[(planner, int(n), "now")] = load(path)
    if args.paper and os.path.isdir(args.paper):
        for path in glob.glob(os.path.join(args.paper, "measurement_*.csv")):
            m = re.fullmatch(r"measurement_(FOP|FISS\+|Sparse_FOP)[-_]?(\d+)\.csv", os.path.basename(path))
            if m and len(load(path)) > 5:
                planner = "SP" if m.group(1) == "Sparse_FOP" else m.group(1)
                runs[(planner, int(m.group(2)), "paper")] = load(path)

    new = [d for (p, _, src), d in runs.items() if src == "now" and p != "SP"]  # SP never fails, see p_B
    common = sorted(set.intersection(*[set(d.index[d.ok]) for d in new])) if new else []
    rows = [dict(planner=p, N=n, source=src, rows=len(d), success=int(d.ok.sum()),
                 N_fail=100.0 * (1 - d.ok.sum() / num_inputs),
                 fail_plan=int((~d.ok & (d.step_number_for_break > 0)).sum()) + num_inputs - len(d),
                 fail_stop=int((~d.ok & (d.step_number_for_break == 0)).sum()),
                 J_all=d.final_trajector_cost.mean(), J_common=d.reindex(common).final_trajector_cost.mean(),
                 t_ms=1e3 * d["average runtime_plan [s]"].mean(), t_warm=1e3 * d["runtime history [s]"].map(warm).mean(),
                 trajs_per_cycle=d.num_trajs_generated.mean(),
                 p_B=d.Percent_FOP_Intervence.mean() if p == "SP" else float("nan"))
            for (p, n, src), d in runs.items()]
    table = pd.DataFrame(rows).sort_values(["source", "planner", "N"])
    table.to_csv(os.path.join(args.dir, "table_long.csv"), index=False)

    order = [p for p in ["FOP", "FISS+", "CEM", "MPPI", "SP"] if p in set(table.planner)]
    now = table[table.source == "now"]
    layout = pd.concat({metric: now.pivot(index="planner", columns="N", values=metric).reindex(order).dropna(how="all")
                        for metric in ["J_common", "J_all", "t_ms", "t_warm", "p_B", "N_fail", "success", "fail_plan", "fail_stop",
                                       "trajs_per_cycle"]})
    layout.to_csv(os.path.join(args.dir, "table_paper_layout.csv"))

    pd.set_option("display.width", 200)
    print(f"{num_inputs} input scenarios; J_common over the {len(common)} scenarios every new run completes\n")
    print(layout.to_string(float_format=lambda x: f"{x:.2f}"))
    print("\nAll runs (source 'paper' = the paper's CSVs):")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()

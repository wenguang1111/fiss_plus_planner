"""Summary of the J_D-on N-sweep (scripts/run_jd_sweep.sh) and the comparison with the J_D-off runs.

Per (planner, N column): fail_pct, no_feasible, J_common, J_all, t_ms.
  fail       --def success (26 scenarios, as table_ii.py): CSV success False or no CSV row (step-0 failure);
             includes standstill (STOP_ON_STANDSTILL=1).
             --def nofeasible (Test 250, user's definition): "Planning failed at time step k" in log.txt only;
             reaching the scenario end is fine (STOP_ON_STANDSTILL=0).
  no_feasible  number of scenarios with "Planning failed at time step k" (both definitions).
  J_common   mean final_trajector_cost over the scenarios that every planner of this root and column completes
             (n_common); J_all = mean over all scenarios with a row (failed ones count with the cost up to the failure).
  t_ms       mean planning time per cycle without each scenario's first cycle (valid only for sequential runs).
With --ref (J_D-off root with the same grid/ iter/ layout): the same metrics of the reference, and J_paired_on /
J_paired_off = mean J of a planner on the scenarios it completes both with J_D on and off (n_paired).
Grid planners (odd N_W): 12 -> column 8, 18 -> 16, 60 -> 64, 125 -> 128, 252 -> 256, 504 -> 512, 990 -> 1024.

    python scripts/jd_summary.py --dir data/measurements/table_ii_jd --ref data/measurements/table_ii --def success
    python scripts/jd_summary.py --dir data/measurements/test250_jd --ref data/measurements/test250_nostop --def nofeasible
"""
import argparse
import ast
import glob
import os
import re

import numpy as np
import pandas as pd

GRID_COL = {12: 8, 18: 16, 60: 64, 125: 128, 252: 256, 504: 512, 990: 1024}
PLANNERS = [("FOP", "grid", "FOP_CPP"), ("FISS+", "grid", "FISS+_CPP"), ("CEM", "iter", "CEM_CPP"), ("MPPI", "iter", "MPPI_CPP")]


def read_run(run_dir, definition):
    if not os.path.exists(os.path.join(run_dir, ".complete")):
        return None
    processed, nofeas, current = [], set(), None
    for line in open(os.path.join(run_dir, "log.txt"), errors="replace"):
        m = re.match(r"Processing (\S+?)(\.xml)?\.\.\.", line)
        if m:
            current = m.group(1)
            processed.append(current)
        elif line.startswith("Planning failed at time step") and current:
            nofeas.add(current)
    csv = glob.glob(os.path.join(run_dir, "measurement_*.csv"))[0]
    d = pd.read_csv(csv, skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    d["scenario"] = d["scenario"].str.replace(r"\.xml$", "", regex=True)
    d = d.set_index("scenario").reindex(processed)
    t = pd.DataFrame(index=pd.Index(processed, name="scenario"))
    t["nofeas"] = t.index.isin(nofeas)
    success = d["success"].astype(str).str.strip() == "True"
    t["failed"] = t.nofeas if definition == "nofeasible" else ~success
    t["J"] = d["final_trajector_cost"]
    t["rt"] = d["runtime history [s]"]
    return t


def load(root, definition):
    runs = {}
    for name, sub, prefix in PLANNERS:
        for f in glob.glob(os.path.join(root, sub, f"{prefix}_*")):
            n = int(f.rsplit("_", 1)[1])
            col = GRID_COL.get(n, n) if sub == "grid" else n
            t = read_run(f, definition)
            if t is not None:
                runs[(name, col)] = (n, t)
    return runs


def summarize(runs):
    rows = []
    for col in sorted({c for _, c in runs}):
        present = [(p, runs[(p, col)]) for p, _, _ in PLANNERS if (p, col) in runs]
        common = set.intersection(*[set(t.index[~t.failed & t.J.notna()]) for _, (_, t) in present])
        for p, (n, t) in present:
            h = [np.array(ast.literal_eval(x))[1:] for x in t.rt.dropna()]
            h = np.concatenate(h) if h else np.array([np.nan])
            rows.append(dict(planner=p, N=col, N_real=n, scenarios=len(t), fail_pct=round(100 * t.failed.mean(), 2),
                             n_fail=int(t.failed.sum()), no_feasible=int(t.nofeas.sum()), n_common=len(common),
                             J_common=t.J.loc[sorted(common)].mean() if common else np.nan, J_all=t.J.mean(),
                             t_ms=1e3 * np.nanmean(h)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", default=None, help="J_D-off root for the comparison")
    ap.add_argument("--def", dest="definition", choices=["success", "nofeasible"], required=True)
    args = ap.parse_args()
    on = load(args.dir, args.definition)
    s = summarize(on)
    s.to_csv(os.path.join(args.dir, "jd_summary.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(f"== J_D on: {args.dir} (failure definition: {args.definition})")
    print(s.round(3).to_string(index=False))
    if args.ref:
        off = load(args.ref, args.definition)
        r = summarize(off).set_index(["planner", "N"])
        rows = []
        for (p, col), (n, t) in sorted(on.items(), key=lambda kv: (kv[0][1], [x[0] for x in PLANNERS].index(kv[0][0]))):
            if (p, col) not in off:
                continue
            t0 = off[(p, col)][1]
            both = sorted(set(t.index[~t.failed & t.J.notna()]) & set(t0.index[~t0.failed & t0.J.notna()]))
            rows.append(dict(planner=p, N=col, fail_on=s.set_index(["planner", "N"]).loc[(p, col), "fail_pct"],
                             fail_off=r.loc[(p, col), "fail_pct"], n_paired=len(both), J_paired_on=t.J.loc[both].mean(),
                             J_paired_off=t0.J.loc[both].mean(),
                             t_on=s.set_index(["planner", "N"]).loc[(p, col), "t_ms"], t_off=r.loc[(p, col), "t_ms"]))
        c = pd.DataFrame(rows)
        c.to_csv(os.path.join(args.dir, "jd_vs_off.csv"), index=False)
        print(f"\n== J_D on vs off ({args.ref})")
        print(c.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

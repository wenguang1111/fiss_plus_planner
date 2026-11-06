"""Summary of the Sparse-Planner-type closed loops on data/demo/Test (250): the paper CVAE (test250_nostop/{sp,cvae})
and the Gaussian Planner models (test250_nostop/gauss/<model>/{sp,cvae}, scripts/run_gaussian_test250.sh).

Failure = no feasible trajectory ("Planning failed at time step k" in log.txt; step-0 failures write no CSV row);
reaching the scenario end is fine (STOP_ON_STANDSTILL=0). Per (method, mode, N):
  no_feasible, fail_pct; with fallback (mode sp): fallback cycles, P_fb = fallback cycles / cycles, scen_with_fb,
  top5_fb_share (as sp_summary.csv of 2026-11-05);
  J = mean final_trajector_cost of the completed scenarios; paired with the CVAE of the same mode and N on the
  scenarios both complete: n_paired, J_paired, J_cvae_paired, dJ = J - J_cvae (mean per scenario), win = share of
  paired scenarios with a lower J than the CVAE.
Writes <root>/gauss_summary.csv (one row per run) and gauss_summary_models.csv (gaussian / mog: mean and std over
the seeds), and prints both.

    python fiss_plus_planner/scripts/summarize_test250_sp.py
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/measurements/test250_nostop")


def read_run(run_dir: str):
    """(per-scenario table, number of scenarios) of one SP_<N> folder; None if incomplete."""
    if not os.path.exists(os.path.join(run_dir, ".complete")):
        return None
    failed, processed, current = set(), [], None
    for line in open(os.path.join(run_dir, "log.txt"), errors="replace"):
        m = re.match(r"Processing (\S+)\.\.\.", line)
        if m:
            current = m.group(1)
            processed.append(current)
        elif line.startswith("Planning failed at time step") and current:
            failed.add(current)
    csv = glob.glob(os.path.join(run_dir, "measurement_*.csv"))
    d = pd.read_csv(csv[0]) if csv else pd.DataFrame(columns=["scenario"])
    d.columns = [c.strip() for c in d.columns]
    d = d.set_index("scenario")
    table = pd.DataFrame(index=pd.Index(processed, name="scenario"))
    table["failed"] = table.index.isin(failed)
    table["J"] = d["final_trajector_cost"].reindex(table.index)
    table["fb"] = d["num_FOP_intervence_for_SP"].reindex(table.index).fillna(0)
    table["steps"] = d["steps"].reindex(table.index).fillna(0)
    return table, len(processed)


def runs(root: str):
    """(method, mode, N, folder) of every run."""
    for mode in ("sp", "cvae"):
        for f in glob.glob(os.path.join(root, mode, "SP_*")):
            yield "cvae_paper", mode, int(f.rsplit("_", 1)[1]), f
    for f in glob.glob(os.path.join(root, "gauss", "*", "*", "SP_*")):
        parts = f.split(os.sep)
        yield parts[-3], parts[-2], int(f.rsplit("_", 1)[1]), f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT)
    args = parser.parse_args()
    tables = {}
    for method, mode, n, f in runs(args.root):
        r = read_run(f)
        if r is not None:
            tables[(method, mode, n)] = r
    rows = []
    for (method, mode, n), (t, total) in sorted(tables.items()):
        done = t[~t.failed & t.J.notna()]
        row = {"method": method, "mode": mode, "N": n, "scenarios": total, "no_feasible": int(t.failed.sum()),
               "fail_pct": round(100 * t.failed.mean(), 2), "J": round(done.J.mean(), 3)}
        if mode == "sp":
            fb = t.fb.sum()
            row.update(fb_cycles=int(fb), cycles=int(t.steps.sum()), P_fb=round(100 * fb / max(t.steps.sum(), 1), 2),
                       scen_with_fb=int((t.fb > 0).sum()), top5_fb_share=round(100 * t.fb.nlargest(5).sum() / max(fb, 1), 1))
        ref = tables.get(("cvae_paper", mode, n))
        if ref is not None and method != "cvae_paper":
            rt = ref[0]
            common = done.index.intersection(rt[~rt.failed & rt.J.notna()].index)
            diff = done.J[common] - rt.J[common]
            row.update(n_paired=len(common), J_paired=round(done.J[common].mean(), 3),
                       J_cvae_paired=round(rt.J[common].mean(), 3), dJ=round(diff.mean(), 3),
                       win=round(float((diff < 0).mean()), 3))
        rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(args.root, "gauss_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(s.to_string(index=False))

    g = s[s.method != "cvae_paper"].copy()
    if len(g):
        g["model"] = g.method.str.replace(r"_s\d+$", "", regex=True)
        cols = [c for c in ["fail_pct", "P_fb", "J", "J_paired", "J_cvae_paired", "dJ", "win"] if c in g]
        agg = g.groupby(["model", "mode", "N"])[cols].agg(["mean", "std"]).round(3)
        agg.insert(0, "seeds", g.groupby(["model", "mode", "N"]).size())
        agg.to_csv(os.path.join(args.root, "gauss_summary_models.csv"))
        print()
        print(agg.to_string())


if __name__ == "__main__":
    main()

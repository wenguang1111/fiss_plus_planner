"""How often a CVAE-FOP run falls back to the dense FOP grid, and whether that goes with a larger cost gap.

Per scenario: share of cycles planned by the grid (measurement column num_FOP_intervence_for_SP). On the
scenarios solved by the CVAE-FOP run, CEM and FOP: the gap J_CVAE - J_CEM per fallback-share group, next
to J_FOP - J_CEM; with --cycles also the stages and the sample rejection reasons of the fallback cycles.

python scripts/fallback_report.py --run data/measurements/cvae_warm --cem ../results/cvae_v1/measurements/cem_test \\
    --fop ../results/cvae_v1/measurements/fop_test [--cycles data/output/cvae_warm/cvae_cycles] [--top 10]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from compare_planners import load

GROUPS = ((0, 0.5), (0.5, 10), (10, 30), (30, 101))  # fallback share of a scenario's cycles [%]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="measurement folder of the CVAE-FOP run")
    p.add_argument("--cem", required=True, help="measurement folder of CEM")
    p.add_argument("--fop", required=True, help="measurement folder of FOP")
    p.add_argument("--cycles", help="cvae_cycles folder of the run (per-cycle logs)")
    p.add_argument("--top", type=int, default=10, help="scenarios listed with the largest gap to CEM")
    args = p.parse_args()

    run, cem, fop = load(args.run), load(args.cem), load(args.fop)
    share = 100 * run.num_FOP_intervence_for_SP / run.steps
    print(f"{len(run)} scenarios; with any fallback: {int((share > 0).sum())} ({(share > 0).mean():.0%}); "
          f"fallback cycles {run.num_FOP_intervence_for_SP.sum() / run.steps.sum():.1%} of all cycles")

    ok = run.index[run.success & cem.success.reindex(run.index).fillna(False) & fop.success.reindex(run.index).fillna(False)]
    d = pd.DataFrame({"J_run": run.loc[ok, "J_total"], "J_CEM": cem.loc[ok, "J_total"], "J_FOP": fop.loc[ok, "J_total"],
                      "fallback_%": share[ok]})
    d["gap"], d["fop_gap"] = d.J_run - d.J_CEM, d.J_FOP - d.J_CEM
    print(f"\n{len(d)} scenarios solved by the run, CEM and FOP: gap J_run - J_CEM by fallback share")
    for lo, hi in GROUPS:
        m = (d["fallback_%"] >= lo) & (d["fallback_%"] < hi)
        print(f"  fallback {lo:>4}-{hi:<3}%: {int(m.sum()):4d} scenarios | gap mean {d.gap[m].mean():+.2f} "
              f"median {d.gap[m].median():+.2f} | FOP - CEM mean {d.fop_gap[m].mean():+.2f}")
    print(f"  correlation of fallback share and gap: {np.corrcoef(d['fallback_%'], d.gap)[0, 1]:+.2f}")
    print(f"\n{args.top} largest gaps to CEM:")
    print(d.nlargest(args.top, "gap").round(2).to_string())

    if args.cycles:
        c = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(args.cycles, "*.csv"))])
        print(f"\n{len(c)} cycles; stages: {c.stage.value_counts(normalize=True).round(4).to_dict()}")
        fb = c[c.stage != "samples"]
        rej = fb[["rejected_dynamic", "rejected_offroad", "rejected_clearance", "rejected_collision"]].sum()
        print(f"why the samples failed in the fallback cycles: {(rej / max(rej.sum(), 1)).round(3).to_dict()}")


if __name__ == "__main__":
    main()

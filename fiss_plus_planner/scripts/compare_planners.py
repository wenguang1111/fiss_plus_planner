"""Closed-loop comparison of planners from their measurement CSVs (demo_cr.py, all shards of a run).

Per planner: success rate (failures split into rear-end and others), how the runs ended (goal region /
map end / time limit / failed, from the termination column of newer runs; goal region and map end count as
reached), FOP fallback share (CVAE_FOP), trajectories evaluated per cycle (the real sample budget, a fallback's
grid included) and runtime per cycle (from the runtime history). For CVAE-FOP runs with cycle logs (the
measurement folder's twin under data/output/.../cvae_cycles): the fallback share (cycles in which no CVAE sample
was feasible and the fallback planner, CEM or the FOP grid, planned, with or without the clearance requirement),
the scenarios with a fallback, the failed cycles, and the feasible share of the K samples. `interrupted` is the
share of the scenarios that ended because no plan was found (termination failed). The scenario cost J_total = J_run + J_ter is compared
pairwise on the scenarios every planner solved and, with the termination column, on the scenarios where
every planner reached the goal (as CommonRoad compares the costs of valid solutions only) and where every
planner ended the same way: mean, difference to the reference planner with a 95 % confidence interval and the Wilcoxon
signed-rank p-value, share of scenarios with a lower J, and the mean cost terms.

python scripts/compare_planners.py --run CVAE_FOP=data/measurements/cvae_fop_test \\
    --run CEM=data/measurements/cem_test [--run FOP=data/measurements/fop_test] [--ref CEM] [--out <dir>]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

TERMS = ["J_ter", "J_V", "J_A", "J_J", "J_LC", "J_D"]
REACHED = ("goal", "map_end")  # terminations that count as the goal reached


def load(measurement_dir: str) -> pd.DataFrame:
    """All shard CSVs of a run, one row per scenario (the last one if a scenario was rerun)."""
    files = sorted(glob.glob(os.path.join(measurement_dir, "measurement_*.csv")))
    if not files:
        raise SystemExit(f"no measurement_*.csv in {measurement_dir}")
    parts = [pd.read_csv(f, skipinitialspace=True) for f in files]
    parts = [d for d in parts if len(d)]
    if not parts:
        raise SystemExit(f"{measurement_dir}: no scenario in the measurement CSVs (did the run fail? see its logs)")
    df = pd.concat(parts, ignore_index=True)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates("scenario", keep="last").set_index("scenario")
    for col in ("success", "rear_end_failure"):
        df[col] = df[col].astype(str).str.strip() == "True"
    df["J_total"] = pd.to_numeric(df["final_trajector_cost"], errors="coerce")
    if "termination" in df:
        df["termination"] = df["termination"].fillna("").astype(str).str.strip()
    return df


def runtime_ms(df: pd.DataFrame) -> np.ndarray:
    return 1e3 * np.concatenate([json.loads(h) for h in df["runtime history [s]"] if isinstance(h, str)] or [[]])


def cvae_stages(measurement_dir: str, scenarios: list) -> pd.Series:
    """Stage shares of a CVAE-FOP run's cycles on the given scenarios, from its cycle logs (log_cycles) in the
    output folder of the run (data/measurements/X -> data/output/X); empty if there are none."""
    out_dir = os.path.join(measurement_dir.replace("measurements", "output", 1), "cvae_cycles")
    files = [os.path.join(out_dir, s.removesuffix(".xml") + ".csv") for s in scenarios]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        return pd.Series(dtype=float)
    c = pd.concat([pd.read_csv(f, usecols=["stage", "num_samples", "num_feasible"]).assign(scenario=f) for f in files],
                  ignore_index=True)
    fallback = ~c.stage.isin(["samples", "failed"])  # grid / cem, with or without the clearance requirement
    return pd.Series({"scenarios": len(files), "fallback": fallback.mean(),
                      "scenarios_with_fallback": fallback.groupby(c.scenario).any().mean(),
                      "failed_cycles": (c.stage == "failed").mean(),
                      "feasible": (c.num_feasible / c.num_samples.clip(lower=1)).mean()})


def paired_report(data: dict, names: list, ref: str, scenarios: list, label: str) -> pd.DataFrame:
    paired = pd.DataFrame({n: data[n].loc[scenarios, "J_total"] for n in names})
    print(f"J_total on the {len(scenarios)} scenarios {label} (difference to {ref}):")
    for n in names:
        line = f"  {n:12s} mean {paired[n].mean():8.3f}"
        if n != ref and len(scenarios) > 1:
            diff = paired[n] - paired[ref]
            half = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
            p_w = stats.wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
            line += (f" | diff {diff.mean():+.3f} ± {half:.3f} (95 % CI), median {diff.median():+.3f}, "
                     f"Wilcoxon p {p_w:.2g}, lower than {ref} in {(diff < 0).mean():.1%}")
        print(line)
    terms = pd.DataFrame({n: data[n].loc[scenarios, TERMS].astype(float).mean() for n in names}).T
    print("mean cost terms on these scenarios:\n" + terms.round(3).to_string() + "\n")
    return paired


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="NAME=MEASUREMENTS_DIR (repeat)")
    p.add_argument("--ref", help="reference planner of the paired differences (default: the second --run)")
    p.add_argument("--out", help="directory for summary.csv and paired_costs.csv")
    args = p.parse_args()

    runs = dict(r.split("=", 1) for r in args.run)
    data = {name: load(path) for name, path in runs.items()}
    names = list(data)
    ref = args.ref or names[min(1, len(names) - 1)]
    common = sorted(set.intersection(*(set(d.index) for d in data.values())))
    if not common:
        raise SystemExit("no scenario was run by every planner")
    print(f"{len(common)} scenarios in every run ({', '.join(f'{n}: {len(d)}' for n, d in data.items())})\n")

    rows = []
    for name, d in data.items():
        d = d.loc[common]
        rt = runtime_ms(d)
        fails = ~d.success
        rows.append({
            "planner": name, "scenarios": len(d), "success": int(d.success.sum()),
            "success_rate": d.success.mean(), "rear_end_failures": int((fails & d.rear_end_failure).sum()),
            "other_failures": int((fails & ~d.rear_end_failure).sum()),
            "fop_fallback_share": (d["num_FOP_intervence_for_SP"].sum() / d["steps"].sum()) if len(d) else np.nan,
            "trajs_per_cycle": (d.num_trajs_generated * d.steps).sum() / d.steps.sum() if len(d) else np.nan,
            "runtime_mean_ms": rt.mean() if rt.size else np.nan, "runtime_median_ms": np.median(rt) if rt.size else np.nan,
            "runtime_p95_ms": np.percentile(rt, 95) if rt.size else np.nan,
            **({"interrupted": (d.termination == "failed").mean(), "reached": d.termination.isin(REACHED).mean(),
                **{f"end_{k}": (d.termination == k).mean() for k in ("goal", "map_end", "time_limit", "failed")}}
               if "termination" in d else {}),
        })
    summary = pd.DataFrame(rows).set_index("planner")
    print(summary.round(4).to_string(), "\n")
    stages = pd.DataFrame({n: cvae_stages(runs[n], common) for n in names}).T.dropna(how="all")
    if len(stages):
        print("CVAE-FOP cycles: fallback = share of the cycles without a feasible CVAE sample (planned by the fallback "
              "planner); feasible = share of the K samples that passed every check:\n" + stages.round(4).to_string() + "\n")

    solved = [s for s in common if all(data[n].at[s, "success"] for n in names)]
    paired = paired_report(data, names, ref, solved, "solved by all planners")
    if all("termination" in d for d in data.values()):
        at_goal = [s for s in common if all(data[n].at[s, "termination"] in REACHED for n in names)]
        paired_report(data, names, ref, at_goal, "where every planner reached the goal")
        # the same end for all: all reached the goal, or all ran to the time limit (same duration, e.g. the
        # scenarios without a goal position)
        same_end = at_goal + [s for s in common if all(data[n].at[s, "termination"] == "time_limit" for n in names)]
        paired_report(data, names, ref, same_end, "where every planner ended the same way (all reached / all timed out)")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        summary.to_csv(os.path.join(args.out, "summary.csv"))
        paired.to_csv(os.path.join(args.out, "paired_costs.csv"))
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()

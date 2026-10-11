"""Integrity check of a CEM data set (Collect_Data_For_ML with CEM_CPP), e.g. after a collection or a copy.

Per scenario every file must be readable, and the values must be consistent:
    contexts         finite state and search space (best_* may be NaN in a cycle without a feasible candidate)
    candidates       finite d, v, T inside the cycle's search space; finite J for feasible candidates;
                     finite, non-negative V; rank and index unique within each CEM iteration
    proposals        finite mean, positive std
    conditions       finite, smooth reference path: consecutive points at most MAX_STEP apart, every point
                     within MAX_PERP of the chord of its neighbours (a corrupted point is a spike)
    best_trajectory  finite points; reference_line finite
    time steps       candidates, conditions and best trajectories belong to cycles in contexts
A flipped bit that moves a value out of its range, breaks a unique / smooth sequence is found; one that only
moves a value slightly inside its range is not. For that, compare with a second copy (checksums), or with a
second collection run (the planner is deterministic: a rerun gives bit-identical files).

python scripts/check_cem_data.py --data <OUTPUT_DIR> [--input <scenario dir>] [--workers 16]
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

MAX_STEP = 0.5     # [m] reference path: points of the planner's reference line, 0.1-0.19 m apart in Verification
MAX_PERP = 0.15    # [m] deviation of a point from its neighbours' chord: up to 0.074 m at a map kink (BEL_Putte)
BOUND_TOL = 1e-6
FILES = ("contexts", "candidates", "proposals", "conditions", "best_trajectory", "reference_line")


def finite(a) -> bool:
    return bool(np.all(np.isfinite(np.asarray(a, dtype=np.float64))))


def check_scenario(args) -> list:
    """Problems found in one scenario (empty list = ok)."""
    data_dir, scenario = args
    folder = os.path.join(data_dir, "cem_data", scenario)
    try:
        info = json.load(open(os.path.join(folder, "scenario.json")))
        f = {name: pd.read_parquet(os.path.join(folder, f"{name}.parquet")) for name in FILES}
    except Exception as e:
        return [f"unreadable: {type(e).__name__}: {e}"]
    bad = []
    ctx, cand, prop, cond, best = f["contexts"], f["candidates"], f["proposals"], f["conditions"], f["best_trajectory"]
    steps = set(ctx.time_step)

    if info.get("cycles") is not None and info["cycles"] != len(ctx):
        bad.append(f"scenario.json cycles {info['cycles']} != {len(ctx)} contexts")
    if not finite(ctx.drop(columns=[c for c in ctx.columns if c.startswith("best_")])):
        bad.append("contexts: non-finite value")
    if (ctx.d_min > ctx.d_max).any() or (ctx.v_min > ctx.v_max).any() or (ctx.T_min > ctx.T_max).any():
        bad.append("contexts: search space with min > max")

    z = cand[["d", "v", "T"]].to_numpy(np.float64)
    if not finite(z):
        bad.append(f"candidates: {int((~np.isfinite(z)).any(1).sum())} non-finite d/v/T")
    if not set(cand.time_step).issubset(steps):
        bad.append("candidates: time step not in contexts")
    else:
        b = ctx.set_index("time_step").loc[cand.time_step]
        lo = b[["d_min", "v_min", "T_min"]].to_numpy() - BOUND_TOL
        hi = b[["d_max", "v_max", "T_max"]].to_numpy() + BOUND_TOL
        out = ~((z >= lo) & (z <= hi)).all(1)
        if out.any():
            bad.append(f"candidates: {int(out.sum())} outside the search space, e.g. {z[out][0]}")
    feas = cand.feasible == 1
    if not finite(cand.loc[feas, [c for c in cand.columns if c.startswith("J_")]]):
        bad.append("candidates: non-finite J of a feasible candidate")
    v = cand[[c for c in cand.columns if c.startswith("V_")]].to_numpy(np.float64)
    if not finite(v) or (v < 0).any():
        bad.append("candidates: non-finite or negative V")
    keys = ["time_step", "pass", "iteration"]
    for col in ("rank", "index"):
        if cand.duplicated(keys + [col]).any():
            bad.append(f"candidates: duplicate {col} within an iteration")

    if not finite(prop.filter(like="mean_")) or not finite(prop.filter(like="std_")) \
            or (prop.filter(like="std_").to_numpy() <= 0).any():
        bad.append("proposals: non-finite mean or non-positive std")

    for name, df, xc, yc in (("conditions", cond, "ref_x", "ref_y"), ("best_trajectory", best, "x", "y")):
        if not set(df.time_step).issubset(steps):
            bad.append(f"{name}: time step not in contexts")
        for t, x, y in zip(df.time_step, df[xc], df[yc]):
            x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
            if len(x) != len(y) or not finite(x) or not finite(y):
                bad.append(f"{name} t={t}: non-finite or unequal point lists")
                break
            if name == "conditions" and len(x) > 2:
                step = np.hypot(np.diff(x), np.diff(y)).max()
                cx, cy, mx, my = x[2:] - x[:-2], y[2:] - y[:-2], x[1:-1] - x[:-2], y[1:-1] - y[:-2]
                perp = np.abs(cx * my - cy * mx) / np.maximum(np.hypot(cx, cy), 1e-9)
                if step > MAX_STEP or perp.max() > MAX_PERP:
                    bad.append(f"conditions t={t}: point step {step:.3g} m, off the path by {perp.max():.3g} m")
                    break
    if not finite(f["reference_line"]):
        bad.append("reference_line: non-finite point")
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="OUTPUT_DIR of the collection (contains cem_data/, completed/)")
    p.add_argument("--input", help="scenario directory of the collection (INPUT_DIR), to list scenarios without data")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    root = os.path.join(args.data, "cem_data")
    scenarios = sorted(s for s in os.listdir(root) if os.path.isdir(os.path.join(root, s)))
    done = {f[:-5] for f in os.listdir(os.path.join(args.data, "completed")) if f.endswith(".done")}
    print(f"{len(scenarios)} scenarios with data, {len(done)} completed markers")
    if args.input:
        inputs = {os.path.splitext(f)[0] for f in os.listdir(args.input) if f.endswith(".xml")}
        missing = sorted(inputs - set(scenarios))
        print(f"{len(inputs)} input scenarios, {len(missing)} without data" + (f": {missing[:10]}" if missing else ""))
    if set(scenarios) - done:
        print(f"data without completed marker (unfinished?): {sorted(set(scenarios) - done)[:10]}")

    with Pool(args.workers) as pool:
        results = pool.map(check_scenario, [(args.data, s) for s in scenarios], chunksize=8)
    failed = [(s, b) for s, b in zip(scenarios, results) if b]
    for s, b in failed:
        print(f"{s}: " + "; ".join(b))
    print(f"{len(failed)} of {len(scenarios)} scenarios with problems")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

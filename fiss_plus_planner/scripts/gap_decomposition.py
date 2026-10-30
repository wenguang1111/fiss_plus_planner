"""Gap diagnosis (E1): how much worse the CVAE-FOP's plan is than the CEM's in the same state, on the CEM's own
states (expert: CEM drove, SHADOW_CVAE) and on the CVAE-FOP's states (learner: DAgger collection), from the runs of
scripts/run_gap_diagnosis.sh NAME.

Per planning cycle (state s), with the costs of the shared evaluator:
    r(s)    = J_cvae(s) - J_cem(s)        regret of the CVAE-FOP's plan (J_plan of its cycle_log, samples or grid)
    rS(s)   = J_samples(s) - J_cem(s)     regret of the best of the K samples alone, in the cycles where one was
                                          feasible (the share of the others is `grid`): the sampler's own quality
    rT(s)   = E[min J of K draws] - J_cem(s), the K draws from the training target p ~ exp(-J~/tau) / q(z) over the
              CEM's feasible candidates in s (exact expectation): what a CVAE that learned the target perfectly
              would reach with K samples (without the warm start and the grid fallback)
    rT_noq  the same without the density correction (p ~ exp(-J~/tau))
r~ = r / (J_90 - J_min) of the cycle is the regret in the units of J~ (the training's normalised cost).
Cycles where either planner dropped the clearance requirement (cem_no_clearance, stage grid_no_clearance) plan
under another rule and are reported apart. On the expert states, the CEM candidates come from the CEM collection of
the same scenarios (CEM_DATA, default the Verification v3 collection); the CEM is deterministic, so its states are
the same, which is checked (J_cem of the shadow log vs best_J of the collection).

Answers (see the decision table in memory cvae-ablations, E1):
    rT large on the expert states             -> the training target is poor for best-of-K (P1)
    rT small, r large on the expert states    -> the model does not reach its target
    r small on expert, large on learner states-> distribution shift (P2); at low speed: does the CEM drive faster
                                                 where the CVAE-FOP creeps (cem_v_end vs v_end)?
    r small on both, closed-loop gap large    -> per-cycle cost and scenario cost disagree (P3)

    python scripts/gap_decomposition.py v3_img [--cem_data data/output/cem_verif_R8x250_v3] [--k 16] [--tau 0.25]
Run from fiss_plus_planner/. Writes data/measurements/gap_cycles_NAME.parquet (one row per cycle and mode).
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from CVAE_efficient_sampling.cem_cache import EPS, J_SCALE_QUANTILE, log_mixture_density, unit  # noqa: E402

SPEED_BINS = [0.0, 1.0, 4.0, 8.0, np.inf]  # [m/s]: creeping, low-speed model, urban, faster


def expected_min(j: np.ndarray, logw: np.ndarray, k: int) -> float:
    """E[min of k draws with replacement] of the values j drawn with weights exp(logw)."""
    p = np.exp(logw - logw.max())
    order = np.argsort(j)
    j, p = j[order], p[order] / p.sum()
    tail = np.clip(np.cumsum(p[::-1])[::-1], 0.0, 1.0)  # P(draw rank >= i)
    tail_next = np.append(tail[1:], 0.0)
    return float(np.sum(j * (tail ** k - tail_next ** k)))


def target_costs(args) -> pd.DataFrame:
    """Per cycle of one scenario of a CEM data folder: J_min, J_90 of the feasible candidates and the expected best
    of k target draws (with / without the density correction), as in cem_cache.read_scenario."""
    folder, k, tau = args
    try:
        ctx = pd.read_parquet(os.path.join(folder, "contexts.parquet")).set_index("time_step")
        cand = pd.read_parquet(os.path.join(folder, "candidates.parquet"),
                               columns=["time_step", "d", "v", "T", "feasible", "J_total"])
        prop = pd.read_parquet(os.path.join(folder, "proposals.parquet"))
    except Exception as e:  # e.g. a scenario that failed before planning
        print(f"  skipped {os.path.basename(folder)}: {type(e).__name__}: {e}", file=sys.stderr)
        return pd.DataFrame()
    by_step = dict(tuple(cand[cand.feasible == 1].groupby("time_step")))
    props = dict(tuple(prop.groupby("time_step")))
    rows = []
    for t in ctx.index:
        row = {"time_step": int(t), "best_J": ctx.at[t, "best_J"], "cem_s_d": ctx.at[t, "s_d"]}
        f = by_step.get(t)
        if f is not None and len(f):
            j = f.J_total.to_numpy(np.float64)
            j_min, j_90 = j.min(), np.percentile(j, J_SCALE_QUANTILE)
            jn = (j - j_min) / (j_90 - j_min + EPS)
            c = ctx.loc[t]
            lo, hi = np.array([c.d_min, c.v_min, c.T_min]), np.array([c.d_max, c.v_max, c.T_max])
            p = props[t]
            logq = log_mixture_density(unit(f[["d", "v", "T"]].to_numpy(np.float64), lo, hi),
                                       p[["mean_d", "mean_v", "mean_T"]].to_numpy(),
                                       np.maximum(p[["std_d", "std_v", "std_T"]].to_numpy(), 1e-6)).astype(np.float64)
            row.update(J_min=j_min, J_90=j_90, n_feasible=len(j), J_T=expected_min(j, -jn / tau - logq, k),
                       J_T_noq=expected_min(j, -jn / tau, k))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.insert(0, "scenario", os.path.basename(folder))
    return out


def load_mode(name: str, mode: str, cem_data: str, k: int, tau: float, workers: int) -> pd.DataFrame:
    odir = f"data/output/gap_{mode}_{name}"
    logs = sorted(glob.glob(f"{odir}/cvae_cycles/*.csv"))
    if not logs:
        raise SystemExit(f"no cycle logs in {odir}/cvae_cycles")
    cyc = pd.concat([pd.read_csv(f).assign(scenario=os.path.basename(f)[:-4]) for f in logs], ignore_index=True)
    root = f"{odir}/cem_data" if mode == "learner" else f"{cem_data}/cem_data"
    with Pool(workers) as pool:
        tgt = pd.concat(pool.map(target_costs, [(f"{root}/{s}", k, tau) for s in cyc.scenario.unique()]),
                        ignore_index=True)
    df = cyc.merge(tgt, on=["scenario", "time_step"], how="left")
    if mode == "expert":  # the shadow's states must be those of the CEM collection
        same = np.isclose(df.J_cem, df.best_J, rtol=1e-6, atol=1e-9) | (df.J_cem.isna() & df.best_J.isna())
        bad = df.loc[~same, "scenario"].unique()
        print(f"expert states vs {cem_data}: {same.mean():.4%} of the cycles have the same CEM plan; "
              f"{len(bad)} scenarios differ and are dropped{(': ' + ', '.join(bad[:5])) if len(bad) else ''}")
        df = df[~df.scenario.isin(bad)]
    df["mode"] = mode
    df["other_rule"] = (df.cem_no_clearance == 1) | df.stage.str.endswith("_no_clearance")
    df["r"] = df.J_plan - df.J_cem
    df["rS"] = df.J_samples - df.J_cem  # the K samples alone (NaN where none was feasible, the FOP grid planned)
    df["rT"] = df.J_T - df.J_cem
    df["rT_noq"] = df.J_T_noq - df.J_cem
    scale = df.J_90 - df.J_min + EPS
    for c in ("r", "rT", "rT_noq"):
        df[c + "~"] = df[c] / scale
    df["dv_end"] = df.cem_v_end - df.v_end  # > 0: the CEM plans to be faster at the horizon end
    df["speed_bin"] = pd.cut(df.v, SPEED_BINS, right=False)
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per mode (and speed bin): share of the cycles, mean regrets of the CVAE-FOP and of the K-draw target, with
    the clearance requirement in both plans."""
    ok = df[~df.other_rule & df.J_plan.notna() & df.J_cem.notna() & df.J_T.notna()]
    g = ok.groupby(["mode", "speed_bin"], observed=True)
    out = pd.DataFrame({
        "cycles": g.size(),
        "share": g.size() / ok.groupby("mode").size().reindex(g.size().index.get_level_values(0)).to_numpy(),
        "r": g.r.mean(), "r_median": g.r.median(), "r>0": g.r.apply(lambda x: (x > 1e-6).mean()),
        "rS": g.rS.mean(), "rS_median": g.rS.median(), "rT": g.rT.mean(), "rT_noq": g.rT_noq.mean(),
        "r~": g["r~"].mean(), "rT~": g["rT~"].mean(),
        "grid": g.stage.apply(lambda s: (s != "samples").mean()),
        "dv_end": g.dv_end.mean(),
    })
    tot = ok.groupby("mode")
    all_rows = pd.DataFrame({"cycles": tot.size(), "share": 1.0, "r": tot.r.mean(), "r_median": tot.r.median(),
                             "r>0": tot.r.apply(lambda x: (x > 1e-6).mean()), "rS": tot.rS.mean(),
                             "rS_median": tot.rS.median(), "rT": tot.rT.mean(),
                             "rT_noq": tot.rT_noq.mean(), "r~": tot["r~"].mean(), "rT~": tot["rT~"].mean(),
                             "grid": tot.stage.apply(lambda s: (s != "samples").mean()), "dv_end": tot.dv_end.mean()})
    all_rows.index = pd.MultiIndex.from_tuples([(m, "all") for m in all_rows.index], names=["mode", "speed_bin"])
    return pd.concat([out, all_rows]).sort_index(level=0, sort_remaining=False)


def closed_loop(name: str, df: pd.DataFrame) -> None:
    """Scenario level: the closed-loop gap (CVAE-FOP run of the learner mode vs the CEM run of the expert mode) and
    the summed per-cycle regret on the learner's states. Plan costs are not scenario costs (a plan covers its whole
    horizon), so only the rank correlation is meaningful, not the ratio."""
    def load(d):
        files = glob.glob(f"{d}/measurement_*.csv")
        if not files:
            return None
        m = pd.concat([pd.read_csv(f, skipinitialspace=True) for f in files])
        m.columns = m.columns.str.strip()
        m["scenario"] = m.scenario.str.replace(".xml", "", regex=False)
        m = m.drop_duplicates("scenario", keep="last").set_index("scenario")
        m["J"] = pd.to_numeric(m.final_trajector_cost, errors="coerce")
        return m
    cv, cem = load(f"data/measurements/gap_learner_{name}"), load(f"data/measurements/gap_expert_{name}")
    if cv is None or cem is None:
        print("closed loop: measurements of both modes needed, skipped")
        return
    both = [s for s in cv.index.intersection(cem.index)
            if cv.at[s, "termination"] != "failed" and cem.at[s, "termination"] != "failed"]
    gap = cv.loc[both, "J"] - cem.loc[both, "J"]
    learner = df[(df["mode"] == "learner") & ~df.other_rule]
    r_sum = learner.groupby("scenario").r.sum().reindex(both)
    low = learner[learner.v < 1.0].groupby("scenario").size().reindex(both).fillna(0)
    ok = r_sum.notna() & gap.notna()
    rho = stats.spearmanr(gap[ok], r_sum[ok]).statistic
    rho_low = stats.spearmanr(gap[ok], low[ok]).statistic
    print(f"closed loop on {ok.sum()} scenarios solved by both: gap J {gap[ok].mean():+.3f} per scenario "
          f"(CVAE-FOP {cv.loc[both, 'J'].mean():.3f}, CEM {cem.loc[both, 'J'].mean():.3f}); Spearman of the gap with "
          f"the summed regret on the learner's states {rho:.2f}, with its number of cycles below 1 m/s {rho_low:.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name")
    p.add_argument("--cem_data", default="data/output/cem_verif_R8x250_v3", help="CEM collection of the scenarios")
    p.add_argument("--k", type=int, default=16, help="samples of the CVAE-FOP (K)")
    p.add_argument("--tau", type=float, default=0.25, help="temperature of the training target")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    modes = [m for m in ("expert", "learner") if os.path.isdir(f"data/output/gap_{m}_{args.name}/cvae_cycles")]
    df = pd.concat([load_mode(args.name, m, args.cem_data, args.k, args.tau, args.workers) for m in modes],
                   ignore_index=True)
    df.assign(speed_bin=df.speed_bin.astype(str)).to_parquet(f"data/measurements/gap_cycles_{args.name}.parquet")
    print(f"{args.name}: K {args.k}, tau {args.tau}; cycles per mode: " + ", ".join(
        f"{m} {n} ({o:.1%} under another clearance rule, {f} failed)" for m, n, o, f in zip(
            modes, df.groupby("mode").size()[modes], df.groupby("mode").other_rule.mean()[modes],
            df.groupby("mode").J_plan.apply(lambda x: x.isna().sum())[modes])))
    print("\nRegret per cycle (both plans with the clearance requirement). r = J_cvae - J_cem, rT = best of K target "
          "draws - J_cem\n(rT_noq without /q), ~ = in J~ units, grid = share planned by the FOP fallback, dv_end = "
          "cem_v_end - v_end [m/s]:")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary(df).round(4).to_string())
    for m in modes:  # where the CVAE-FOP is slow: does the CEM plan to be faster from the same state?
        low = df[(df["mode"] == m) & (df.v < 1.0) & ~df.other_rule & df.J_plan.notna()]
        if len(low):
            print(f"\n{m} states below 1 m/s ({len(low)} cycles): CEM plans >= 1 m/s faster at the horizon end in "
                  f"{(low.dv_end >= 1.0).mean():.0%}, slower by >= 1 m/s in {(low.dv_end <= -1.0).mean():.0%}; "
                  f"mean v_end CVAE {low.v_end.mean():.2f}, CEM {low.cem_v_end.mean():.2f} m/s")
    if "learner" in modes:
        print()
        closed_loop(args.name, df)


if __name__ == "__main__":
    main()

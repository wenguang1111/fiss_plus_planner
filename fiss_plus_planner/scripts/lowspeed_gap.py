"""Share of the CVAE-CEM cost gap on Test that comes from low-speed phases (< VLOW m/s), and whether the CVAE is
held back by the clearance requirement there (CVAE-FOP run with log_cycles). Per-cycle J_V = w_V (v_des - v)^2 dt
as in scenario_cost; v_des = the cycle's sampling bound v_max (= v_des when v <= v_des). The CEM side comes from
the Test CEM collection (data/output/cem_test_R8x250_v3, identical to the goalpos CEM run).
    python scripts/lowspeed_gap.py cvae_v3_img                       # data/{measurements,output}/cvae_v3_img
    python scripts/lowspeed_gap.py cvae_warm data/measurements/goalpos/CVAE_warm   # output dir, measurement dir
Run from fiss_plus_planner/. The first line checks the per-cycle J_V sum against the measured J_V (corr ~1)."""
import glob, os, sys
import numpy as np, pandas as pd
W_V, W_T, DT, VLOW = 0.01, 0.1, 0.1, 1.0
run = sys.argv[1] if len(sys.argv) > 1 else "cvae_v3_img"
def load(d):
    m = pd.concat([pd.read_csv(f) for f in glob.glob(d + "/measurement_*.csv")])
    m["scenario"] = m.scenario.str.replace(".xml", "", regex=False)
    m = m.drop_duplicates("scenario", keep="last").set_index("scenario"); m["J"] = m.J_run + m.J_ter
    return m
cem = load("data/measurements/goalpos/CEM")
cv = load(sys.argv[2] if len(sys.argv) > 2 else f"data/measurements/{run}")
ok = [s for s in cem.index if cem.at[s, "termination"] != "failed" and cv.at[s, "termination"] != "failed"
      and os.path.exists(f"data/output/{run}/cvae_cycles/{s}.csv")]
rows, check = [], []
for s in ok:
    c = pd.read_csv(f"data/output/{run}/cvae_cycles/{s}.csv")
    x = pd.read_parquet(f"data/output/cem_test_R8x250_v3/cem_data/{s}/contexts.parquet", columns=["v", "v_des"])
    jv_c = W_V * (c.v_max - c.v) ** 2 * DT
    jv_e = W_V * (x.v_des - x.v) ** 2 * DT
    low_c, low_e = c.v < VLOW, x.v < VLOW
    free = low_c & (c.stage == "samples") & (c.rejected_clearance == 0)          # low speed, clearance not limiting
    held = low_c & ~free
    rows.append(dict(scenario=s, dJ=cv.at[s, "J"] - cem.at[s, "J"], dJV=cv.at[s, "J_V"] - cem.at[s, "J_V"],
                     jv_low_c=jv_c[low_c].sum(), jv_low_e=jv_e[low_e].sum(), jv_free=jv_c[free].sum(), jv_held=jv_c[held].sum(),
                     n_low_c=low_c.sum(), n_low_e=low_e.sum(), n_free=free.sum(), n_held=held.sum(),
                     n_c=len(c), n_e=len(x)))
    check.append((jv_c.sum(), cv.at[s, "J_V"], jv_e.sum(), cem.at[s, "J_V"]))
r = pd.DataFrame(rows).set_index("scenario")
ck = np.array(check)
print(f"{run}: {len(r)} scenarios solved by both | check: per-cycle J_V sum vs measured J_V, "
      f"corr CVAE {np.corrcoef(ck[:,0], ck[:,1])[0,1]:.4f}, CEM {np.corrcoef(ck[:,2], ck[:,3])[0,1]:.4f}, "
      f"ratio {ck[:,0].sum()/ck[:,1].sum():.3f} / {ck[:,2].sum()/ck[:,3].sum():.3f}")
dJ, dJV = r.dJ.sum(), r.dJV.sum()
d_low = (r.jv_low_c - r.jv_low_e).sum()
# time term of the extra low-speed cycles
d_low_t = W_T * DT * (r.n_low_c - r.n_low_e).sum()
print(f"total gap J {dJ:.0f} (mean {dJ/len(r):.2f}), of it J_V {dJV:.0f} ({dJV/dJ:.0%})")
print(f"cycles below {VLOW} m/s: CVAE {r.n_low_c.sum()} ({r.n_low_c.sum()/r.n_c.sum():.1%} of its cycles), "
      f"CEM {r.n_low_e.sum()} ({r.n_low_e.sum()/r.n_e.sum():.1%}); extra {r.n_low_c.sum()-r.n_low_e.sum()} cycles "
      f"vs total extra {r.n_c.sum()-r.n_e.sum()}")
print(f"J_V accumulated below {VLOW} m/s: CVAE {r.jv_low_c.sum():.0f}, CEM {r.jv_low_e.sum():.0f} -> difference {d_low:.0f} "
      f"= {d_low/dJ:.0%} of the J gap; with its time term {d_low + d_low_t:.0f} = {(d_low + d_low_t)/dJ:.0%}")
print(f"CVAE low-speed cycles: clearance not limiting (all samples pass clearance, samples stage) {r.n_free.sum()} "
      f"({r.n_free.sum()/r.n_low_c.sum():.0%}), J_V {r.jv_free.sum():.0f}; limited by clearance / grid {r.n_held.sum()}, J_V {r.jv_held.sum():.0f}")
top = r.assign(d_low=r.jv_low_c - r.jv_low_e).sort_values("d_low", ascending=False)
share10 = top.d_low.head(int(0.1 * len(top))).sum() / d_low
print(f"concentration: the top 10 % of scenarios hold {share10:.0%} of the low-speed J_V difference")
print(top[["dJ", "d_low", "n_low_e", "n_low_c", "n_free", "n_held"]].head(12).round(2).to_string())

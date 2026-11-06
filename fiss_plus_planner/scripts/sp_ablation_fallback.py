"""Sampler ablation of SP with the FOP fallback: CVAE (table_ii_trt/sp) vs uniform / marginal / mean
(sp_ablation_fallback/<sampler>), and the same samplers without fallback where they were run (CVAE: table_ii_trt/cvae,
others: sp_ablation/<sampler>_nofallback). Per sampler and N: completed scenarios, fallback share p_B (share of all
cycles), J over all scenarios, mean planning time without each scenario's first cycle."""
import argparse, ast, glob, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/measurements")
args = ap.parse_args()
NS = [1, 8, 16, 64, 128, 256, 512, 1024]
FB = {"cvae": "table_ii_trt/sp", "uniform": "sp_ablation_fallback/uniform", "marginal": "sp_ablation_fallback/marginal",
      "mean": "sp_ablation_fallback/mean"}
NOFB = {"cvae": "table_ii_trt/cvae", "uniform": "sp_ablation/uniform_nofallback", "marginal": "sp_ablation/marginal_nofallback",
        "mean": "sp_ablation/mean_nofallback"}

def load(folder):
    f = glob.glob(os.path.join(args.root, folder, "measurement_*.csv"))
    if not f or not os.path.exists(os.path.join(args.root, folder, ".complete")):
        return None
    d = pd.read_csv(f[0], skipinitialspace=True); d.columns = [c.strip() for c in d.columns]
    d["ok"] = d.success.astype(str).str.strip() == "True"
    return d

rows = []
for sampler in FB:
    for n in NS:
        r = dict(sampler=sampler, N=n)
        d = load(f"{FB[sampler]}/SP_{n}")
        if d is not None:
            h = np.concatenate([np.array(ast.literal_eval(x))[1:] for x in d["runtime history [s]"]])
            r.update(ok_fb=int(d.ok.sum()), rows_fb=len(d), p_B=100 * d.num_FOP_intervence_for_SP.sum() / d.steps.sum(),
                     J_all=d.final_trajector_cost.mean(), t_ms=1e3 * h.mean(), t_p95_ms=1e3 * np.percentile(h, 95))
        d = load(f"{NOFB[sampler]}/SP_{n}")
        if d is not None:
            r.update(ok_nofb=int(d.ok.sum()))
        rows.append(r)
t = pd.DataFrame(rows)
t.to_csv(os.path.join(args.root, "sp_ablation_fallback", "sp_ablation_fallback.csv"), index=False)
pd.set_option("display.width", 250)
for metric, nd in [("ok_fb", 0), ("p_B", 2), ("J_all", 2), ("t_ms", 2), ("t_p95_ms", 2), ("ok_nofb", 0)]:
    if metric in t:
        print(metric); print(t.pivot_table(index="sampler", columns="N", values=metric).reindex(list(FB)).round(nd).to_string()); print()

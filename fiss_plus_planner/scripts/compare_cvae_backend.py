"""SP / CVAE closed loop with two CVAE inference backends (default: table_ii = graph_fp16, table_ii_trt = trt_fp16):
per N success, J over all scenarios, J over the scenarios both complete (paired), share of scenarios with the same
trajectory cost, planning time without each scenario's first cycle (mean / p95) and the fallback share p_B."""
import argparse, ast, glob, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--a", default="data/measurements/table_ii")
ap.add_argument("--b", default="data/measurements/table_ii_trt")
args = ap.parse_args()

def load(root, sub, n):
    f = glob.glob(os.path.join(root, sub, f"SP_{n}", "measurement_*.csv"))
    if not f or not os.path.exists(os.path.join(root, sub, f"SP_{n}", ".complete")):
        return None
    d = pd.read_csv(f[0], skipinitialspace=True); d.columns = [c.strip() for c in d.columns]
    d["scenario"] = d.scenario.str.replace(".xml", ""); d["ok"] = d.success.astype(str).str.strip() == "True"
    return d.set_index("scenario")

def times(d):
    h = np.concatenate([np.array(ast.literal_eval(x))[1:] for x in d["runtime history [s]"]])
    return 1e3 * h.mean(), 1e3 * np.percentile(h, 95)

rows = []
for sub in ["sp", "cvae"]:
    for n in [1, 8, 16, 64, 128, 256, 512, 1024]:
        a, b = load(args.a, sub, n), load(args.b, sub, n)
        if a is None or b is None:
            continue
        both = sorted(set(a.index[a.ok]) & set(b.index[b.ok]))
        diff = b.loc[both, "final_trajector_cost"] - a.loc[both, "final_trajector_cost"]
        (ta, pa), (tb, pb) = times(a), times(b)
        rows.append(dict(method="SP" if sub == "sp" else "CVAE", N=n, ok_graph=int(a.ok.sum()), ok_trt=int(b.ok.sum()),
                         J_all_graph=a.final_trajector_cost.mean(), J_all_trt=b.final_trajector_cost.mean(), n_both=len(both),
                         J_both_graph=a.loc[both, "final_trajector_cost"].mean(), J_both_trt=b.loc[both, "final_trajector_cost"].mean(),
                         dJ_mean=diff.mean(), dJ_absmax=diff.abs().max(), same_J=int((diff.abs() < 1e-6).sum()),
                         t_graph=ta, t_trt=tb, p95_graph=pa, p95_trt=pb,
                         pB_graph=100 * a.num_FOP_intervence_for_SP.sum() / a.steps.sum(),
                         pB_trt=100 * b.num_FOP_intervence_for_SP.sum() / b.steps.sum()))
t = pd.DataFrame(rows)
pd.set_option("display.width", 300)
print(t.round(3).to_string(index=False))
t.to_csv(os.path.join(args.b, "compare_backend.csv"), index=False)

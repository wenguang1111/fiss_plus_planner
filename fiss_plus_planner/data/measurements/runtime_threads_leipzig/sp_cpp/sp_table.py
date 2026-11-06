"""SP_CPP runtimes on DEU_Leipzig-27_9: rt_<cores>/Sparse_CPP_<backend>_<N>/measurement_Sparse_CPP.csv"""
import sys, glob, os, ast
import numpy as np, pandas as pd
root = sys.argv[1] if len(sys.argv) > 1 else "."
rows = []
for p in glob.glob(os.path.join(root, "rt_*", "Sparse_CPP_*", "measurement_*.csv")):
    d = pd.read_csv(p, skipinitialspace=True); d.columns = [c.strip() for c in d.columns]; r = d.iloc[0]
    h = np.array(ast.literal_eval(r["runtime history [s]"]))
    name = os.path.basename(os.path.dirname(p))
    backend, n = name[len("Sparse_CPP_"):].rsplit("_", 1)
    cores = os.path.basename(os.path.dirname(os.path.dirname(p))).split("_")[1]
    rows.append(dict(backend=backend, N=int(n), cores=int(cores), steps=int(r.steps), ok=str(r.success).strip(),
                     t_warm=1e3 * h[1:].mean(), t_med=1e3 * np.median(h[1:]), t_p95=1e3 * np.percentile(h[1:], 95),
                     p_B=float(r.Percent_FOP_Intervence), J=float(r.final_trajector_cost)))
t = pd.DataFrame(rows)
pd.set_option("display.width", 250)
for c in ["t_warm", "t_med", "t_p95"]:
    print(c, "[ms]"); print(t.pivot_table(index=["backend", "cores"], columns="N", values=c).round(2).to_string()); print()
print("steps / success / p_B [%] / J"); print(t.pivot_table(index=["backend", "cores"], columns="N", values=["steps", "p_B", "J"]).round(3).to_string())
print("success:", set(t.ok))
t.to_csv(os.path.join(root, "sp_cpp_long.csv"), index=False)

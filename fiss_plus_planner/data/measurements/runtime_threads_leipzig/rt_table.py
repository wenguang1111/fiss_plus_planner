import sys, glob, os, ast
import numpy as np, pandas as pd
rows = []
for root in sys.argv[1:]:
    for p in glob.glob(os.path.join(root, "*_*", "measurement_*.csv")):
        d = pd.read_csv(p, skipinitialspace=True); d.columns = [c.strip() for c in d.columns]; r = d.iloc[0]
        h = np.array(ast.literal_eval(r["runtime history [s]"]))
        pl, n = os.path.basename(os.path.dirname(p)).rsplit("_", 1)
        rows.append(dict(planner=pl, N=int(n), cores=root.rsplit("_", 1)[1], steps=int(r.steps), ok=str(r.success).strip(),
                         t_warm=1e3 * h[1:].mean(), t_med=1e3 * np.median(h[1:]), trajs=r.num_trajs_generated))
t = pd.DataFrame(rows)
pd.set_option("display.width", 250)
for c in ["t_warm", "t_med"]:
    print(c, "[ms]"); print(t.pivot_table(index=["planner"], columns=["cores", "N"], values=c).round(2).to_string()); print()
print("steps / success:", t.groupby("planner")[["steps"]].agg(["min", "max"]).to_dict(), set(t.ok))
print(t.pivot_table(index="planner", columns="N", values="trajs").round(1).to_string())

# Training data of the Sparse Planner CVAE (IROS'27 draft)

Copied 2026-11-03 from `~/backup/.local/share/Trash/files/` (laptop, deleted there 2026-02-23 from
`/home/wenguang/workplace/fiss_plus_planner/fiss_plus_planner/data/output/{sampled_vars,conditions}.parquet` and `imgs/`).
Collected 2026-02-15, closed loop with FOP_CPP (J_D not in the planning cost), only scenarios FOP finished.

- `sampled_vars.15.parquet`: 58,427 rows (scenario, time_step, t, d, v) = sampling parameters of the executed (best)
  FOP_CPP trajectory per cycle (top-1), 622 scenarios. Mean t 3.769 s, d 0.0001 m, v 8.928 m/s.
- `conditions.15.parquet`: ego state per cycle (x, y, theta, velocity, ..., Frenet s, d and derivatives).
- `imgs.5/<scenario>/`: 59,049 BEV images (the CVAE condition).

Weights trained on it: `CVAE_efficient_sampling/weights/attn_cvae_zoom_out_zdim_64_sigmoid_1.0_stall_end.pth`.
Note: collected with the C++ spline before the 2026-11-02 fix (transposed matrix, wrong near the path ends).

"""Offline: CVAE samples vs the training target per ego-speed bin of a cache (e.g. Verification v3), K samples per
cycle at the CEM's own states. v bias = mean terminal speed of the samples (clipped to the search space) - mean of
the target (cost-weighted CEM candidates); all_K_slower = share of cycles where every sample is slower than it.
    python fiss_plus_planner/scripts/speed_bins_offline.py <run dir> <cache dir>     # from the repository root"""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from CVAE_efficient_sampling.CVAE import CostAwareCVAE
from CVAE_efficient_sampling.cem_dataset import CEMCache, select_cycles
from CVAE_efficient_sampling.generation_check import generation_metrics
run, cache_dir, n_per_bin, K = sys.argv[1], sys.argv[2], 1500, 16
dev = torch.device("cuda")
cvae = CostAwareCVAE(run, dev); cfg = cvae.config
cache = CEMCache(cache_dir)
rows = select_cycles(cache.cycles, cfg["drop_tail_s"], keep_failed=not cfg["drop_failed"])
v = cache.cycles.ego_v.to_numpy()[rows]
bins = [(-1, 1), (1, 4), (4, 8), (8, 99)]
std = cvae.normalizer.target_scaler.scale_
rng = np.random.default_rng(0)
out = []
for lo, hi in bins:
    sel = rows[(v >= lo) & (v < hi)]
    pick = rng.choice(sel, size=min(n_per_bin, len(sel)), replace=False)
    bias, abserr, spr_c, spr_t, below = [], [], [], [], []
    torch.manual_seed(0)
    for row in pick:
        r = cache.cycles.iloc[row]; s, n = r.cand_start, r.cand_count
        z = np.asarray(cache.z[s:s + n], np.float64)
        logw = -np.asarray(cache.jn[s:s + n], np.float64) / cfg["tau"] - np.asarray(cache.logq[s:s + n], np.float64)
        w = np.exp(logw - logw.max()); w /= w.sum()
        tv = (w * z[:, 1]).sum(); tsd = np.sqrt((w * (z[:, 1] - tv) ** 2).sum())
        frames = [cache.images[r[f"frame_{h}"]] for h in range(3)]
        smp = cvae.generate_samples(frames, K)
        smp[:, 1] = np.clip(smp[:, 1], r.v_min, r.v_max)   # as the planner clips to the search space
        bias.append(smp[:, 1].mean() - tv); abserr.append(abs(smp[:, 1].mean() - tv))
        spr_c.append(smp[:, 1].std()); spr_t.append(tsd)
        below.append((smp[:, 1].max() < tv))   # all K samples slower than the target mean
    m = generation_metrics({"cvae": lambda f, lo_, hi_, k, rg, **kw: cvae.generate_samples(f, k)}, cache, pick, K,
                           cfg["tau"], not cfg["no_density_correction"], std, seed=0)["cvae"]
    out.append(dict(ego_v=f"{lo if lo >= 0 else 0}-{hi if hi < 99 else ''} m/s", cycles=len(sel), share=len(sel) / len(rows),
                    v_bias=np.mean(bias), v_abs_err=np.mean(abserr), v_spread_cvae=np.mean(spr_c),
                    v_spread_target=np.mean(spr_t), all_K_slower=np.mean(below), best_cost=m["best_cost"], mean_dist=m["mean_dist"]))
df = pd.DataFrame(out)
pd.set_option("display.width", 200)
print(f"{run}\n{df.round(3).to_string(index=False)}")

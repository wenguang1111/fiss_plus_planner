"""Train a Sparse Planner sampler p([t, d, v] | 3 BEV frames) on the 2026-02-15 SP data (data/sp_training_2026-02-15:
sampled_vars.15 = top-1 FOP_CPP parameters per cycle, imgs.5 BEV, 622 scenarios, 58,427 cycles).

  --model gaussian  CondGaussian, K = 1 full-covariance Gaussian (planners/sparse_planning/cond_gaussian.py)
  --model mog       CondGaussian, K = --K component Gaussian mixture
  --model cvae      attnCVAE retrained the way of CVAE_efficient_sampling/train_hcvae.py (MSE + beta * KL, sigmoid
                    annealing to --beta_end, the paper weights' name says sigmoid 1.0), for a same-data / same-split
                    comparison; best weights also as a plain state_dict loadable by CVAE_EfficientPaper

Shared by all models (as train_hcvae.py and the paper CVAE): 3 frames [t-2, t-1, t] (clamped at 0), the paper
preprocessing (Resize 256, gray, 1 - x, Normalize 0.5/0.5; cached by build_sp_image_cache.py), targets normalised with
the paper statistics (cvae_paper.TARGET_MEAN / TARGET_STD), AdamW lr 1e-4 wd 1e-2, MultiStepLR [5] x 0.5, batch 64,
20 epochs, grad clip 1.0, ImageNet ResNet18 init, best epoch by validation loss. Split by scenario (--val_frac, --split_seed).

Gaussian / MoG: per-sample NLL. The targets lie on the FOP grid of the collection (t 20 values in 3..5 s, d 34 values in
+-0.828 m), so by default t and d get uniform dequantisation noise of one grid step during training (--no_dequant
turns it off); without it the NLL can grow without bound by shrinking the std onto grid points.

Every epoch all models are also scored by the same sample metrics on --eval_items fixed validation cycles with
--eval_n samples each (normalised space): best_dist (min L2 to the target), mean_dist, spread (mean per-dim std of the
samples) and out_of_range (share of samples with t outside 3..5 s, |d| > 0.828 m or v < 0).

    python fiss_plus_planner/scripts/train_sp_sampler.py --model gaussian --name gaussian_s0
    python fiss_plus_planner/scripts/train_sp_sampler.py --model cvae --eval_only \
        --weights CVAE_efficient_sampling/weights/attn_cvae_zoom_out_zdim_64_sigmoid_1.0_stall_end.pth --name paper_cvae
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from CVAE_efficient_sampling.attn_cvae import attnCVAE, reconstruction_kld  # noqa: E402
from fiss_plus_planner.planners.sparse_planning.cond_gaussian import CondGaussian  # noqa: E402

# cvae_paper.CVAE_EfficientPaper.TARGET_MEAN / TARGET_STD [t, d, v] (not imported: CVAE.py needs the submodule folder
# on sys.path for its script-style imports)
TARGET_MEAN = np.array([3.76989018, -1.97695157e-3, 8.92879236], dtype=np.float32)
TARGET_STD = np.array([0.95395955, 0.14355468, 4.87223273], dtype=np.float32)
GRID_STEP = np.array([2.0 / 19.0, 1.656 / 33.0, 0.0], dtype=np.float32)  # t, d grid steps of the collection, v none
T_RANGE, D_MAX = (3.0, 5.0), 0.828
DATA_DIR = os.path.join(REPO, "fiss_plus_planner/data/sp_training_2026-02-15")
LATENT_DIM = 64  # CVAE_Efficient.latent_dim


class SPData(Dataset):
    """(uint8 frames (3, 256, 256), raw target [t, d, v]) per cycle; the frames are rows of the memory-mapped cache."""

    def __init__(self, images_path: str, hist_rows: np.ndarray, targets: np.ndarray):
        self.images_path, self.hist_rows, self.targets = images_path, hist_rows, targets
        self.images = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        if self.images is None:  # opened per worker
            self.images = np.load(self.images_path, mmap_mode="r")
        return torch.from_numpy(np.ascontiguousarray(self.images[self.hist_rows[i]])), torch.from_numpy(self.targets[i])


def load_data(data_dir: str, cache_dir: str):
    targets = pd.read_parquet(os.path.join(data_dir, "sampled_vars.15.parquet"))
    index = pd.read_parquet(os.path.join(cache_dir, "index.parquet"))
    row_of = {(s, t): r for s, t, r in zip(index.scenario, index.time_step, index.row)}
    hist = np.empty((len(targets), 3), dtype=np.int64)
    for k, (s, t) in enumerate(zip(targets.scenario, targets.time_step)):
        hist[k] = [row_of[(s, max(t - 2, 0))], row_of[(s, max(t - 1, 0))], row_of[(s, t)]]  # as CVAEDataset
    return targets, hist


def to_condition(frames: torch.Tensor) -> torch.Tensor:
    """uint8 gray frames -> the paper's network input: (1 - x - 0.5) / 0.5 = 1 - 2x with x = u8 / 255."""
    return 1.0 - frames.float() / 127.5


def build_model(args) -> torch.nn.Module:
    if args.model == "cvae":
        return attnCVAE(latent_dim=LATENT_DIM, img_channels=3, img_size=256)
    return CondGaussian(n_components=1 if args.model == "gaussian" else args.K, min_std=args.min_std)


def beta_at(epoch: int, args) -> float:
    """KL weight of an epoch (0-based) = BetaAnnealer(0, beta_end, epochs, 'sigmoid', steepness 10) after epoch + 1
    step() calls, as train_hcvae.py steps it at the start of every epoch."""
    r = min((epoch + 1) / args.epochs, 1.0)
    s = lambda x: 1.0 / (1.0 + math.exp(-10.0 * (x - 0.5)))  # noqa: E731
    return args.beta_end * (s(r) - s(0.0)) / (s(1.0) - s(0.0))


def loss_of(model, args, x, c, beta):
    """Mean training loss of a batch and the logged parts."""
    if args.model == "cvae":
        out, kls = model(x, c)
        loss, rec, kl = reconstruction_kld(out, x, kls, beta=beta)
        return loss, {"recon": rec.item(), "kl": kl.item()}
    nll = model(x, c).mean()
    return nll, {"nll": nll.item()}


@torch.no_grad()
def sample(model, args, c: torch.Tensor, n: int) -> torch.Tensor:
    """(B, n, 3) normalised samples for the conditions c (B, 3, H, W), each condition encoded once."""
    if args.model != "cvae":
        return model.sample_per_condition(c, n)
    tokens = model.cond(c).repeat_interleave(n, dim=0)
    mu, logvar = model.pri(tokens)
    out = model.dec(model.lat(model.reparameterize(mu, logvar)), tokens)
    return out.view(c.shape[0], n, 3)


@torch.no_grad()
def evaluate(model, args, loader, eval_loader, device, mean, std):
    model.eval()
    total, parts, count = 0.0, {}, 0
    for frames, raw in loader:
        c, x = to_condition(frames.to(device, non_blocking=True)), ((raw.to(device) - mean) / std)
        loss, p = loss_of(model, args, x, c, beta=1.0)  # the CVAE validates with beta 1 as train_hcvae.py
        b = x.shape[0]
        total += loss.item() * b
        count += b
        for k, v in p.items():
            parts[k] = parts.get(k, 0.0) + v * b
    metrics = {"val_loss": total / count, **{f"val_{k}": v / count for k, v in parts.items()}}

    torch.manual_seed(0)  # same draws every epoch (the samplers draw from the global generator)
    best, mean_d, spread, oor, n_items = 0.0, 0.0, 0.0, 0.0, 0
    for frames, raw in eval_loader:
        c, x = to_condition(frames.to(device, non_blocking=True)), ((raw.to(device) - mean) / std)
        s = sample(model, args, c, args.eval_n)  # (B, n, 3)
        dist = (s - x[:, None]).norm(dim=-1)
        best += dist.min(dim=1).values.sum().item()
        mean_d += dist.mean(dim=1).sum().item()
        spread += s.std(dim=1).mean(dim=-1).sum().item()
        phys = s * std + mean
        bad = (phys[..., 0] < T_RANGE[0]) | (phys[..., 0] > T_RANGE[1]) | (phys[..., 1].abs() > D_MAX) | (phys[..., 2] < 0)
        oor += bad.float().mean(dim=1).sum().item()
        n_items += x.shape[0]
    metrics.update({f"best_dist@{args.eval_n}": best / n_items, "mean_dist": mean_d / n_items,
                    "spread": spread / n_items, "out_of_range": oor / n_items})
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["gaussian", "mog", "cvae"], required=True)
    parser.add_argument("--K", type=int, default=5, help="mixture components of --model mog")
    parser.add_argument("--name", required=True)
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--cache_dir", default=None, help="default <data_dir>/cache_256")
    parser.add_argument("--out_dir", default=os.path.join(REPO, "fiss_plus_planner/data/output/sp_sampler/runs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--beta_end", type=float, default=1.0, help="cvae: final KL weight")
    parser.add_argument("--min_std", type=float, default=1e-3, help="gaussian/mog: std floor (normalised)")
    parser.add_argument("--no_dequant", action="store_true", help="gaussian/mog: no grid dequantisation of t, d")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0, help="initialisation, batch order, subsampling")
    parser.add_argument("--split_seed", type=int, default=0, help="train / val split of the scenarios")
    parser.add_argument("--exclude_dir", action="append", default=[],
                        help="drop the scenarios of these folders (*.xml) from train and val, repeatable")
    parser.add_argument("--max_train", type=int, default=0, help="subsample the training cycles (smoke tests)")
    parser.add_argument("--eval_items", type=int, default=2000)
    parser.add_argument("--eval_n", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval_only", action="store_true", help="score --weights on the validation split and exit")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--no_keep_epochs", action="store_true", help="do not keep epochs/epoch_XX.pt of every epoch")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = args.cache_dir or os.path.join(args.data_dir, "cache_256")
    run_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)

    targets, hist = load_data(args.data_dir, cache_dir)
    scenarios = np.array(sorted(targets.scenario.unique()))
    excluded = set()
    for d in args.exclude_dir:
        excluded |= {f[:-4] for f in os.listdir(d) if f.endswith(".xml")}
    scenarios = np.array([s for s in scenarios if s not in excluded])
    perm = np.random.default_rng(args.split_seed).permutation(len(scenarios))
    n_val = int(round(args.val_frac * len(scenarios)))
    val_scen, train_scen = set(scenarios[perm[:n_val]]), set(scenarios[perm[n_val:]])
    is_train = targets.scenario.isin(train_scen).to_numpy()
    is_val = targets.scenario.isin(val_scen).to_numpy()
    raw = targets[["t", "d", "v"]].to_numpy(np.float32)
    train_idx, val_idx = np.flatnonzero(is_train), np.flatnonzero(is_val)
    if args.max_train:
        train_idx = np.random.default_rng(args.seed).choice(train_idx, min(args.max_train, len(train_idx)), replace=False)
    eval_idx = np.random.default_rng(args.split_seed + 1).choice(val_idx, min(args.eval_items, len(val_idx)), replace=False)
    with open(os.path.join(run_dir, "split.json"), "w") as f:
        json.dump({"train": sorted(train_scen), "val": sorted(val_scen), "excluded": sorted(excluded & set(targets.scenario))}, f, indent=1)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "n_train_cycles": len(train_idx), "n_val_cycles": len(val_idx),
                   "n_train_scenarios": len(train_scen), "n_val_scenarios": len(val_scen),
                   "target_mean": TARGET_MEAN.tolist(), "target_std": TARGET_STD.tolist()}, f, indent=1)
    print(f"train {len(train_scen)} scenarios / {len(train_idx)} cycles, val {len(val_scen)} / {len(val_idx)}, "
          f"excluded {len(excluded & set(targets.scenario))}", flush=True)

    images = os.path.join(cache_dir, "images.npy")
    kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(SPData(images, hist[train_idx], raw[train_idx]), batch_size=args.batch, shuffle=True,
                              drop_last=True, **kw)
    val_loader = DataLoader(SPData(images, hist[val_idx], raw[val_idx]), batch_size=256, **kw)
    eval_loader = DataLoader(SPData(images, hist[eval_idx], raw[eval_idx]), batch_size=64, **kw)

    mean, std = torch.tensor(TARGET_MEAN, device=device), torch.tensor(TARGET_STD, device=device)
    dequant = torch.tensor(GRID_STEP, device=device) if args.model != "cvae" and not args.no_dequant else None
    model = build_model(args).to(device)
    if args.weights:
        state = torch.load(args.weights, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters", flush=True)

    log_path = os.path.join(run_dir, "eval_only.tsv" if args.eval_only else "training_log.tsv")
    if args.eval_only:
        m = evaluate(model, args, val_loader, eval_loader, device, mean, std)
        pd.DataFrame([m]).to_csv(log_path, sep="\t", index=False)
        print(json.dumps(m, indent=1))
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.5)
    best_val, rows = float("inf"), []
    for epoch in range(args.epochs):
        model.train()
        beta = beta_at(epoch, args) if args.model == "cvae" else 0.0
        t0, total, parts, n_batches, skipped = time.time(), 0.0, {}, 0, 0
        for frames, raw_x in train_loader:
            raw_x = raw_x.to(device, non_blocking=True)
            if dequant is not None:
                raw_x = raw_x + (torch.rand_like(raw_x) - 0.5) * dequant
            c, x = to_condition(frames.to(device, non_blocking=True)), (raw_x - mean) / std
            loss, p = loss_of(model, args, x, c, beta)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
            n_batches += 1
            for k, v in p.items():
                parts[k] = parts.get(k, 0.0) + v
        scheduler.step()
        train_time = time.time() - t0
        m = evaluate(model, args, val_loader, eval_loader, device, mean, std)
        row = {"epoch": epoch + 1, "train_loss": total / max(n_batches, 1),
               **{f"train_{k}": v / max(n_batches, 1) for k, v in parts.items()}, **m, "beta": beta,
               "lr": optimizer.param_groups[0]["lr"], "skipped_batches": skipped, "train_s": round(train_time, 1)}
        rows.append(row)
        pd.DataFrame(rows).to_csv(log_path, sep="\t", index=False)
        print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        ckpt = {"model": model.state_dict(), "epoch": epoch + 1, "args": vars(args), "metrics": row,
                "target_mean": TARGET_MEAN.tolist(), "target_std": TARGET_STD.tolist()}
        torch.save(ckpt, os.path.join(run_dir, "last.pt"))
        if not args.no_keep_epochs:  # for a checkpoint choice other than the validation loss (e.g. closed loop)
            os.makedirs(os.path.join(run_dir, "epochs"), exist_ok=True)
            torch.save(ckpt, os.path.join(run_dir, "epochs", f"epoch_{epoch + 1:02d}.pt"))
        if m["val_loss"] < best_val:
            best_val = m["val_loss"]
            torch.save(ckpt, os.path.join(run_dir, "best.pt"))
            if args.model == "cvae":  # plain state_dict as the paper weights (CVAE_EfficientPaper)
                torch.save(model.state_dict(), os.path.join(run_dir, "best_weights.pth"))
            print(f"  best epoch {epoch + 1} (val_loss {best_val:.4f})", flush=True)


if __name__ == "__main__":
    main()

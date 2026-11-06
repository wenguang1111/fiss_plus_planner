"""Choose the checkpoint of each train_sp_sampler.py run: the epoch with the smallest best_dist@16 on the validation
cycles (user's choice 2026-11-06; the validation NLL of the single Gaussian diverges after epoch 1 through over-confident
stds, so it would pick an untrained model). Writes <run>/selected.pt (symlink to epochs/epoch_XX.pt) and
<run>/selected.json, and prints the table.

    python fiss_plus_planner/scripts/select_sp_checkpoint.py [--runs data/output/sp_sampler/runs] [--metric best_dist@16]
"""
import argparse
import json
import os

import pandas as pd

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/output/sp_sampler/runs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=RUNS)
    parser.add_argument("--metric", default="best_dist@16", help="training_log.tsv column, smallest wins")
    parser.add_argument("--epochs", type=int, default=20, help="only runs with this many epochs")
    args = parser.parse_args()
    rows = []
    for name in sorted(os.listdir(args.runs)):
        log = os.path.join(args.runs, name, "training_log.tsv")
        if not os.path.exists(log):
            continue
        t = pd.read_csv(log, sep="\t")
        if len(t) < args.epochs:
            print(f"{name}: {len(t)}/{args.epochs} epochs, skipped")
            continue
        best = t.loc[t[args.metric].idxmin()]
        epoch = int(best.epoch)
        ckpt = os.path.join("epochs", f"epoch_{epoch:02d}.pt")
        if not os.path.exists(os.path.join(args.runs, name, ckpt)):
            print(f"{name}: {ckpt} missing, skipped")
            continue
        link = os.path.join(args.runs, name, "selected.pt")
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(ckpt, link)
        info = {"epoch": epoch, "metric": args.metric, **{k: float(best[k]) for k in
                ["best_dist@16", "mean_dist", "spread", "out_of_range", "val_loss"]}}
        json.dump(info, open(os.path.join(args.runs, name, "selected.json"), "w"), indent=1)
        rows.append({"run": name, **info})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

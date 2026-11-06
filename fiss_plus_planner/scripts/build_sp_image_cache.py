"""One-time cache of the Sparse Planner training BEV images for train_sp_sampler.py: every PNG of <data_dir>/imgs.5/
<scenario>/<time_step>.png preprocessed as the closed loop does (cvae_paper.py: Resize((256, 256)) of the RGB image,
ToTensor, Grayscale) and stored as uint8 round(255 * gray) in <out>/images.npy (N, 256, 256), with
<out>/index.parquet (scenario, time_step, row). The rest of the transform (1 - x, Normalize(0.5, 0.5)) is applied on
the GPU at training time. uint8 differs from the float preprocessing by at most 1/510 before the normalisation.

    python fiss_plus_planner/scripts/build_sp_image_cache.py --data_dir fiss_plus_planner/data/sp_training_2026-02-15
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

IMG_DIM = 256
TRANSFORM = transforms.Compose([transforms.Resize((IMG_DIM, IMG_DIM)), transforms.ToTensor(),
                                transforms.Grayscale(num_output_channels=1)])


def init_worker():
    torch.set_num_threads(1)  # one torch thread per worker (24 per worker made the load 300+)


def load(path: str) -> np.ndarray:
    with Image.open(path) as img:
        gray = TRANSFORM(img.convert("RGB"))[0].numpy()
    return np.clip(np.rint(gray * 255.0), 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--imgs", default="imgs.5")
    parser.add_argument("--out", default=None, help="default <data_dir>/cache_256")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    out = args.out or os.path.join(args.data_dir, f"cache_{IMG_DIM}")
    os.makedirs(out, exist_ok=True)
    root = os.path.join(args.data_dir, args.imgs)
    rows = []
    for scenario in sorted(os.listdir(root)):
        for name in os.listdir(os.path.join(root, scenario)):
            if name.endswith(".png"):
                rows.append((scenario, int(name[:-4])))
    index = pd.DataFrame(rows, columns=["scenario", "time_step"]).sort_values(["scenario", "time_step"], ignore_index=True)
    index["row"] = np.arange(len(index))
    paths = [os.path.join(root, s, f"{t}.png") for s, t in zip(index.scenario, index.time_step)]
    print(f"{len(paths)} images of {index.scenario.nunique()} scenarios -> {out}")

    images = np.lib.format.open_memmap(os.path.join(out, "images.npy.tmp"), mode="w+", dtype=np.uint8,
                                       shape=(len(paths), IMG_DIM, IMG_DIM))
    with Pool(args.workers, initializer=init_worker) as pool:
        for i, img in enumerate(pool.imap(load, paths, chunksize=64)):
            images[i] = img
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(paths)}", flush=True)
    images.flush()
    del images
    os.replace(os.path.join(out, "images.npy.tmp"), os.path.join(out, "images.npy"))
    index.to_parquet(os.path.join(out, "index.parquet"), index=False)
    print("done")


if __name__ == "__main__":
    main()

"""World-model CVAE data (CVAE_trajectory_planning, x / y / yaw output) of a CEM collection in the WorldModel layout.

The CEM collection (Collect_Data_For_ML with CEM_CPP) stores per scenario cem_data/<scenario>/sampled_vars.parquet:
per planning cycle the executed plan (sample 0) and K cost-weighted feasible candidates (planning.wm_samples).
This script merges them, with the reference paths, into the layout of the WorldModel branch's Collect_Data_For_ML
(save_data), next to the images and completion markers of the collection:
    <out>/sampled_vars.parquet  scenario, time_step, x, y, path_length (ego frame of the cycle)
                                + yaw (ego frame), sample, J, weight, d, v, T          (1 + K rows per cycle)
    <out>/conditions.parquet    scenario, time_step, ref_x, ref_y, ref_path_length      (one row per cycle)
    <out>/imgs/<scenario>/<t>.png, <out>/completed/<scenario>.done                       (from the collection)
Only cycles with a plan are written, as in save_data (the failing cycle of a failed scenario has an image only).

python scripts/merge_wm_data.py --data data/output/cem_train_R8x250_v3 [--out <dir>]   # default out: --data
"""
import argparse
import json
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DOUBLES = pa.list_(pa.float64())
SCHEMAS = {
    "sampled_vars": pa.schema([("scenario", pa.string()), ("time_step", pa.int64()), ("x", DOUBLES), ("y", DOUBLES),
                               ("path_length", pa.int64()), ("yaw", DOUBLES), ("sample", pa.int64()),
                               ("J", pa.float64()), ("weight", pa.float64()), ("d", pa.float64()),
                               ("v", pa.float64()), ("T", pa.float64())]),
    "conditions": pa.schema([("scenario", pa.string()), ("time_step", pa.int64()), ("ref_x", DOUBLES),
                             ("ref_y", DOUBLES), ("ref_path_length", pa.int64())]),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="OUTPUT_DIR of the CEM collection")
    p.add_argument("--out", help="output folder (default: --data); imgs/ and completed/ are linked if it differs")
    a = p.parse_args()
    out = a.out or a.data
    paths = {name: os.path.join(out, f"{name}.parquet") for name in SCHEMAS}
    if any(os.path.exists(f) for f in paths.values()):
        sys.exit(f"{' / '.join(paths.values())} exist already")
    os.makedirs(out, exist_ok=True)

    done = {f[:-5] for f in os.listdir(os.path.join(a.data, "completed")) if f.endswith(".done")}
    root = os.path.join(a.data, "cem_data")
    scenarios = sorted(s for s in os.listdir(root) if s in done)
    missing = [s for s in scenarios if not os.path.exists(os.path.join(root, s, "sampled_vars.parquet"))]
    if missing:
        sys.exit(f"{len(missing)} scenarios without sampled_vars.parquet (collected before it existed?), e.g. {missing[0]}")
    writers = {name: pq.ParquetWriter(path, SCHEMAS[name]) for name, path in paths.items()}
    n_rows = n_cycles = 0
    for i, scenario in enumerate(scenarios):
        samples = pd.read_parquet(os.path.join(root, scenario, "sampled_vars.parquet"))
        cond = pd.read_parquet(os.path.join(root, scenario, "conditions.parquet"))
        cond = cond[cond.time_step.isin(samples.time_step)]  # cycles with a plan
        cond.insert(0, "scenario", scenario)
        cond["ref_path_length"] = cond.ref_x.map(len)
        for name, frame in (("sampled_vars", samples), ("conditions", cond)):
            if len(frame):
                writers[name].write_table(pa.Table.from_pandas(frame, schema=SCHEMAS[name], preserve_index=False))
        n_rows, n_cycles = n_rows + len(samples), n_cycles + len(cond)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(scenarios)} scenarios, {n_cycles} cycles, {n_rows} trajectories", flush=True)
    for w in writers.values():
        w.close()

    if os.path.abspath(out) != os.path.abspath(a.data):
        for folder in ("imgs", "completed"):
            os.symlink(os.path.abspath(os.path.join(a.data, folder)), os.path.join(out, folder))
    info = json.load(open(os.path.join(root, "dataset_info.json")))
    json.dump({"source": os.path.abspath(a.data), "scenarios": len(scenarios), "cycles": n_cycles,
               "trajectories": n_rows, "ml_data": info.get("ml_data"), "git_commit": info.get("git_commit")},
              open(os.path.join(out, "wm_data_info.json"), "w"), indent=1)
    print(f"done: {len(scenarios)} scenarios, {n_cycles} cycles, {n_rows} trajectories -> {out}")


if __name__ == "__main__":
    main()

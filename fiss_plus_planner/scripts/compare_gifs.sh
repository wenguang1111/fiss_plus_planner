#!/bin/bash
# GIFs of the same Test scenarios planned by several planners, for a side-by-side look (SAVE_GIF with
# Runtime_Measurement off, which only keeps the trajectories for drawing; the plans are the same).
#     bash fiss_plus_planner/scripts/compare_gifs.sh POL_Krakow-62_1_T-8 GRC_NeaSmyrni-73_1_T-1 ...
# Planners (label=config[,KEY=VALUE...], the configs of the Test comparison); override with PLANNERS="...":
#     CEM, CVAE_warm (CVAE v1 + warm start), v3_img (v3 image-only model)
# Result: data/output/gifs/<scenario>/<label>.gif; one process per planner and scenario, JOBS at a time.
set -u
cd "$(dirname "$0")/.." || exit 1   # fiss_plus_planner/
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'echo "stopped"; pkill -P $$; exit 130' INT TERM HUP
PY=${PY:-$HOME/miniconda3/envs/LUT/bin/python}
JOBS=${JOBS:-6}
PLANNERS=${PLANNERS:-"CEM=cfgs/cem_test.yaml CVAE_warm=cfgs/cvae_fop.yaml \
v3_img=cfgs/cvae_fop.yaml,CVAE_FOP.run=data/output/cvae_local/runs/v3_attn_tau0.25_beta0.1"}
[ $# -gt 0 ] || { echo "usage: compare_gifs.sh SCENARIO [SCENARIO ...]"; exit 1; }
OUT=data/output/gifs

for p in $PLANNERS; do
    label=${p%%=*}; spec=${p#*=}
    for s in "$@"; do
        s=${s%.xml}; dir=$OUT/runs/$label/$s
        mkdir -p "$dir"
        "$PY" - "$spec" "$dir" "$s" <<'EOF' || exit 1
import sys, yaml
spec, out, scenario = sys.argv[1:]
cfg_file, *overrides = spec.split(",")
cfg = yaml.safe_load(open(cfg_file))
cfg.update(OUTPUT_DIR=out + "/", MEASUREMENTS_DIR=out, FILES=[scenario + ".xml"], SAVE_GIF=True,
           Runtime_Measurement=False, SAVE_MEASUREMENTS=True)
if "CVAE_FOP" in cfg:
    cfg["CVAE_FOP"]["log_cycles"] = False
for item in overrides:
    key, value = item.split("=", 1)
    *path, last = key.split(".")
    node = cfg
    for k in path:
        node = node[k]
    node[last] = yaml.safe_load(value)
yaml.safe_dump(cfg, open(out + "/config.yaml", "w"), sort_keys=False)
EOF
        echo "$label $s $dir"
    done
done | xargs -P "$JOBS" -L 1 bash -c '"$0" scripts/demo_cr.py --cfg_file "$3/config.yaml" > "$3/log.txt" 2>&1 \
    && echo "  done $1 $2" || echo "  FAILED $1 $2 (see $3/log.txt)"' "$PY"

for p in $PLANNERS; do
    label=${p%%=*}
    for s in "$@"; do
        s=${s%.xml}
        gif=$(ls "$OUT/runs/$label/$s"/gif/*/"$s.gif" 2> /dev/null | head -1)
        mkdir -p "$OUT/$s"
        [ -n "$gif" ] && cp "$gif" "$OUT/$s/$label.gif" || echo "no GIF: $label $s"
    done
done
echo "GIFs in $(pwd)/$OUT/<scenario>/<planner>.gif"

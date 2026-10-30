#!/bin/bash
# Several planners on one scenario folder, one after the other, then compare_planners.py against CEM.
# Each planner = LABEL=BASE_CONFIG[,KEY=VALUE...]: the base config of a stored run (so the settings are exactly those
# of the earlier comparison) with INPUT_DIR set to INPUT, the output paths and the overrides.
#     bash fiss_plus_planner/scripts/run_test_set.sh                     # defaults below
#     PLANNERS="B_succ" bash fiss_plus_planner/scripts/run_test_set.sh   # a subset of the default labels
#     INPUT=data/10k/Test_cem_success OUT=cem_success PAIR="B_succ:B_v3" (defaults)
# Results: data/measurements/OUT/<label>/ (CSVs, config.yaml), data/output/OUT/<label>/ (cycle logs),
# report data/measurements/OUT/compare/report.txt (+ the pairs in PAIR, first against second).
# Every planner run starts from scratch (its folders are deleted). CVAE runs use CVAE_SHARDS processes (each
# loads CUDA and the model), the others SHARDS.
set -u
cd "$(dirname "$0")/.." || exit 1   # fiss_plus_planner/
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'echo "stopped: ending the planner processes"; pkill -P $$; exit 130' INT TERM HUP
PY=${PY:-$HOME/miniconda3/envs/world-model/bin/python}
INPUT=${INPUT:-data/10k/Test_cem_success}
OUT=${OUT:-cem_success}
SHARDS=${SHARDS:-22}
CVAE_SHARDS=${CVAE_SHARDS:-12}
PAIR=${PAIR:-"B_succ:B_v3"}
G=data/measurements/goalpos
B_CEM=data/measurements/cvae_B_cem/config.yaml   # CVAE + CEM fallback, reference setup of 2026-10-15
declare -A SPEC=(
    [CEM]="$G/CEM/config.yaml"
    [FOP]="$G/FOP/config.yaml"
    [FISS+]="$G/FISS+/config.yaml"
    [B_v3]="$B_CEM,CVAE_FOP.run=data/output/cvae_local/runs/v3_draws2_tau0.05_beta0.1"
    [B_succ]="$B_CEM,CVAE_FOP.run=data/output/cvae_local/runs/succ_draws2_tau0.05_beta0.1"
)
ORDER="CEM FOP FISS+ B_v3 B_succ"
PLANNERS=${PLANNERS:-$ORDER}

if pgrep -f "python.*scripts/demo_cr.py" > /dev/null; then
    echo "demo_cr.py processes are running; stop them first"; exit 1
fi
num_inputs=$(ls "$INPUT" | grep -c '\.xml$')
mkdir -p logs_test "data/measurements/$OUT"
for label in $PLANNERS; do
    spec=${SPEC[$label]:?unknown planner $label (known: $ORDER)}
    mdir=data/measurements/$OUT/$label; odir=data/output/$OUT/$label
    rm -rf "$mdir" "$odir" logs_test/"${OUT}_${label}"_*.log
    mkdir -p "$mdir"
    "$PY" - "$spec" "$INPUT" "$odir" "$mdir" <<'EOF' || exit 1
import sys, yaml
spec, inp, odir, mdir = sys.argv[1:]
base, *overrides = spec.split(",")
cfg = yaml.safe_load(open(base))
cfg.update(INPUT_DIR=inp, FILES=[], OUTPUT_DIR=odir + "/", MEASUREMENTS_DIR=mdir, SAVE_GIF=False)
for item in overrides:
    key, value = item.split("=", 1)
    *path, last = key.split(".")
    node = cfg
    for k in path:
        node = node[k]
    if last not in node:
        sys.exit(f"unknown key {key}")
    node[last] = yaml.safe_load(value)
if "CVAE_FOP" in cfg:
    import os
    if not os.path.exists(os.path.join(cfg["CVAE_FOP"]["run"], "model_best.pth")):
        sys.exit(f"no model_best.pth in {cfg['CVAE_FOP']['run']}")
yaml.safe_dump(cfg, open(mdir + "/config.yaml", "w"), sort_keys=False)
EOF
    n=$SHARDS; grep -q "^PLANNER: CVAE_FOP" "$mdir/config.yaml" && n=$CVAE_SHARDS
    start=$(date +%s)
    echo "== $(date '+%F %T') $label: $n processes, $num_inputs scenarios in $INPUT"
    for k in $(seq 0 $((n - 1))); do
        "$PY" scripts/demo_cr.py --cfg_file "$mdir/config.yaml" --shard "$k" --num_shards "$n" \
            > "logs_test/${OUT}_${label}_$k.log" 2>&1 &
    done
    wait
    done_n=$(cat "$mdir"/measurement_*.csv 2> /dev/null | grep -vc '^scenario')
    crashed=$(grep -l "Traceback" logs_test/"${OUT}_${label}"_*.log | wc -l)
    echo "== $(date '+%F %T') $label done in $(( ($(date +%s) - start) / 60 )) min: $done_n of $num_inputs scenarios," \
         "$crashed logs with a Traceback"
    [ "$done_n" -eq "$num_inputs" ] || echo "!! $label: missing scenarios (dead process? RAM fault): rerun this label"
done

RUNS=()
mkdir -p "data/measurements/$OUT/compare"
for label in $ORDER; do
    [ -d "data/measurements/$OUT/$label" ] && RUNS+=(--run "$label=data/measurements/$OUT/$label")
done
{
    "$PY" scripts/compare_planners.py "${RUNS[@]}" --ref CEM --out "data/measurements/$OUT/compare"
    for p in $PAIR; do
        a=${p%%:*}; b=${p#*:}
        [ -d "data/measurements/$OUT/$a" ] && [ -d "data/measurements/$OUT/$b" ] || continue
        echo; echo "---- $a paired against $b ----"
        "$PY" scripts/compare_planners.py --run "$a=data/measurements/$OUT/$a" --run "$b=data/measurements/$OUT/$b" \
            --ref "$b" | sed -n '/^J_total/,$p'
    done
} 2>&1 | grep -v Warning | tee "data/measurements/$OUT/compare/report.txt"
echo "== $(date '+%F %T') ALL DONE, report: $(pwd)/data/measurements/$OUT/compare/report.txt"

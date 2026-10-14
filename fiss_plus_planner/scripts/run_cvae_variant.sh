#!/bin/bash
# One CVAE-FOP variant on data/10k/Test, compared with stored results of the current code (BASE, default
# the goalpos round of 2026-10-12: CEM, FOP, FISS+, CVAE v1 + warm start; the planners are deterministic,
# so they are not rerun) and paired against REF (default CVAE_warm, the reference CVAE-FOP).
#     bash fiss_plus_planner/scripts/run_cvae_variant.sh NAME [KEY=VALUE ...]
#     bash fiss_plus_planner/scripts/run_cvae_variant.sh warm CVAE_FOP.warm_start=true
#     bash fiss_plus_planner/scripts/run_cvae_variant.sh k32 CVAE_FOP.num_samples=32
# KEY is a (dotted) key of cfgs/cvae_fop.yaml, VALUE is parsed as YAML. The variant runs from scratch
# into data/output/cvae_NAME and data/measurements/cvae_NAME (config.yaml of the run is kept there).
# Report: data/measurements/compare_NAME/report.txt (differences to CEM and to REF).
# EXTRA="warm=../results/ablations/warm_start/measurements ..." adds stored runs to the comparison.
set -u
cd "$(dirname "$0")/.." || exit 1   # fiss_plus_planner/
# One CPU core per planner process: numpy (OpenBLAS) and torch (OpenMP) would otherwise start a thread pool
# per core in every process. Ctrl+C / kill / a closed terminal also stops the planner processes (background
# jobs of a script ignore Ctrl+C and would keep running without it).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'echo "stopped: ending the planner processes"; pkill -P $$; exit 130' INT TERM HUP
PY=${PY:-$HOME/miniconda3/envs/LUT/bin/python}
SHARDS=${SHARDS:-22}
NAME=${1:?usage: run_cvae_variant.sh NAME [KEY=VALUE ...]}
shift
[[ "$NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "NAME must be a plain word: $NAME"; exit 1; }
BASE=${BASE:-data/measurements/goalpos}
REF=${REF:-CVAE_warm}
MDIR=data/measurements/cvae_$NAME
ODIR=data/output/cvae_$NAME
LOGS=logs_test

if pgrep -f "python.*scripts/demo_cr.py" > /dev/null; then
    echo "demo_cr.py processes are running; stop them first:"
    pgrep -fa "python.*scripts/demo_cr.py" | cut -c1-120
    exit 1
fi

rm -rf "$MDIR" "$ODIR" "$LOGS/cvae_${NAME}"_*.log
mkdir -p "$MDIR" "$LOGS"
"$PY" - "$MDIR/config.yaml" "$ODIR" "$MDIR" "$@" <<'EOF' || exit 1
import sys, yaml
out, odir, mdir, overrides = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
cfg = yaml.safe_load(open("cfgs/cvae_fop.yaml"))
cfg["OUTPUT_DIR"], cfg["MEASUREMENTS_DIR"] = odir + "/", mdir
for item in overrides:
    key, value = item.split("=", 1)
    *path, last = key.split(".")
    node = cfg
    for k in path:
        node = node[k]
    if last not in node:
        sys.exit(f"unknown key {key}")
    node[last] = yaml.safe_load(value)
    print(f"  {key} = {node[last]!r}")
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
EOF

num_inputs=$(ls data/10k/Test | grep -c '\.xml$')
start=$(date +%s)
echo "== $(date '+%F %T') cvae_$NAME: $SHARDS processes, $num_inputs scenarios"
for k in $(seq 0 $((SHARDS - 1))); do
    "$PY" scripts/demo_cr.py --cfg_file "$MDIR/config.yaml" --shard "$k" --num_shards "$SHARDS" \
        > "$LOGS/cvae_${NAME}_$k.log" 2>&1 &
done
wait
done_n=$(cat "$MDIR"/measurement_*.csv 2> /dev/null | grep -vc '^scenario')
echo "== $(date '+%F %T') cvae_$NAME done in $(( ($(date +%s) - start) / 60 )) min: $done_n of $num_inputs scenarios"
crashed=$(grep -l "Traceback" "$LOGS/cvae_${NAME}"_*.log | wc -l)
[ "$crashed" -gt 0 ] && echo "!! $crashed logs with a Traceback, e.g. $(grep -l Traceback "$LOGS/cvae_${NAME}"_*.log | head -1)"

OUT=data/measurements/compare_$NAME
mkdir -p "$OUT"
RUNS=(--run "$NAME=$MDIR")
for b in CEM FOP FISS+ "$REF"; do RUNS+=(--run "$b=$BASE/$b"); done
for r in ${EXTRA:-}; do RUNS+=(--run "$r"); done
{
    "$PY" scripts/compare_planners.py "${RUNS[@]}" --ref CEM --out "$OUT"
    echo; echo "---- paired against $REF ----"
    "$PY" scripts/compare_planners.py --run "$NAME=$MDIR" --run "$REF=$BASE/$REF" --ref "$REF" \
        | sed -n '/^J_total/,$p'
} 2>&1 | grep -v Warning | tee "$OUT/report.txt"
echo "== $(date '+%F %T') ALL DONE, report: $(pwd)/$OUT/report.txt"

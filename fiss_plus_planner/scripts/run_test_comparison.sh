#!/bin/bash
# Closed loop on data/10k/Test with CVAE-FOP, CEM (R8x250) and FOP (10x10x10), one planner after the
# other, then the comparison. Run in tmux, from anywhere:
#     bash fiss_plus_planner/scripts/run_test_comparison.sh                  # all three planners
#     PLANNERS="CVAE_FOP CEM" bash fiss_plus_planner/scripts/run_test_comparison.sh
#     SHARDS=16 bash ...                                                     # processes per planner
# Every planner run starts from scratch: its measurement folder, output folder and logs are deleted.
# Configs: cfgs/cvae_fop.yaml, cfgs/cem_test.yaml, cfgs/fop_test.yaml (same SAFETY / MOTION_MODEL).
# Result: data/measurements/compare_test/{report.txt,summary.csv,paired_costs.csv}.
# The shards share CPU and GPU: compare J and success here, measure runtimes with one process.
set -u
cd "$(dirname "$0")/.." || exit 1   # fiss_plus_planner/
# One CPU core per planner process: numpy (OpenBLAS) and torch (OpenMP) would otherwise start a thread pool
# per core in every process. Ctrl+C / kill / a closed terminal also stops the planner processes (background
# jobs of a script ignore Ctrl+C and would keep running without it).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'echo "stopped: ending the planner processes"; pkill -P $$; exit 130' INT TERM HUP
PY=${PY:-$HOME/miniconda3/envs/world-model/bin/python}
PLANNERS=${PLANNERS:-"CVAE_FOP CEM FOP"}
SHARDS=${SHARDS:-22}
LOGS=logs_test
declare -A CFG=([CVAE_FOP]=cfgs/cvae_fop.yaml [CEM]=cfgs/cem_test.yaml [FOP]=cfgs/fop_test.yaml)

if pgrep -f "python.*scripts/demo_cr.py" > /dev/null; then
    echo "demo_cr.py processes are running; stop them first:"
    pgrep -fa "python.*scripts/demo_cr.py" | cut -c1-120
    exit 1
fi

cfg_value() {  # cfg_value CONFIG KEY: a top-level string of the config
    "$PY" -c "import yaml, sys; print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]].rstrip('/'))" "$1" "$2"
}
num_inputs=$(ls "$(cfg_value cfgs/cvae_fop.yaml INPUT_DIR)" | grep -c '\.xml$')

COMPARE=()
for name in $PLANNERS; do
    cfg=${CFG[$name]}
    mdir=$(cfg_value "$cfg" MEASUREMENTS_DIR)
    odir=$(cfg_value "$cfg" OUTPUT_DIR)
    [[ "$mdir" == data/measurements/?* && "$odir" == data/output/?* ]] || { echo "unexpected folders: $mdir $odir"; exit 1; }
    rm -rf "$mdir" "$odir" "$LOGS/${name}"_*.log
    mkdir -p "$LOGS"
    start=$(date +%s)
    echo "== $(date '+%F %T') $name ($cfg): $SHARDS processes, $num_inputs scenarios"
    for k in $(seq 0 $((SHARDS - 1))); do
        "$PY" scripts/demo_cr.py --cfg_file "$cfg" --shard "$k" --num_shards "$SHARDS" > "$LOGS/${name}_$k.log" 2>&1 &
    done
    wait
    done_n=$(cat "$mdir"/measurement_*.csv 2> /dev/null | grep -vc '^scenario')
    echo "== $(date '+%F %T') $name done in $(( ($(date +%s) - start) / 60 )) min: $done_n of $num_inputs scenarios"
    crashed=$(grep -l "Traceback" "$LOGS/${name}"_*.log | wc -l)
    if [ "$crashed" -gt 0 ]; then
        echo "!! $name: $crashed logs with a Traceback, e.g. $(grep -l Traceback "$LOGS/${name}"_*.log | head -1)"
    fi
    COMPARE+=(--run "$name=$mdir")
done

OUT=data/measurements/compare_test
mkdir -p "$OUT"
"$PY" scripts/compare_planners.py "${COMPARE[@]}" --ref CEM --out "$OUT" | tee "$OUT/report.txt"
echo "== $(date '+%F %T') ALL DONE, report: $(pwd)/$OUT/report.txt"

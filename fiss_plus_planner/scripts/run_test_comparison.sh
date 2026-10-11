#!/bin/bash
# Closed loop on data/10k/Test with CVAE-FOP, CEM (R8x250) and FOP (10x10x10), then the comparison.
# Run in tmux, from anywhere:
#     bash fiss_plus_planner/scripts/run_test_comparison.sh            # all three planners
#     RUN_FOP=0 bash fiss_plus_planner/scripts/run_test_comparison.sh  # without FOP
# Configs: cfgs/cvae_fop.yaml, cfgs/cem_test.yaml, cfgs/fop_test.yaml (same SAFETY / MOTION_MODEL).
# A planner whose measurement folder already has CSVs is skipped (delete the folder to run it again);
# if processes with its config are still running (e.g. started by hand), the script waits for them first.
# Result: data/measurements/compare_test/{report.txt,summary.csv,paired_costs.csv}.
# Parallel processes share CPU and GPU: compare J and success here, measure runtimes separately.
set -u
cd "$(dirname "$0")/.." || exit 1   # fiss_plus_planner/
PY=${PY:-$HOME/miniconda3/envs/LUT/bin/python}
RUN_FOP=${RUN_FOP:-1}
LOGS=logs_test
mkdir -p "$LOGS"

# run NAME CONFIG MEASUREMENT_DIR SHARDS
run() {
    local name=$1 cfg=$2 mdir=$3 shards=$4
    if pgrep -f "cfg_file $cfg" > /dev/null; then
        echo "== $(date '+%F %T') $name: processes with $cfg are running, waiting for them"
        while pgrep -f "cfg_file $cfg" > /dev/null; do sleep 30; done
    fi
    if ls "$mdir"/measurement_*.csv > /dev/null 2>&1; then
        echo "== $(date '+%F %T') $name: $mdir has results, skipped"
        return
    fi
    echo "== $(date '+%F %T') $name: $shards processes"
    for k in $(seq 0 $((shards - 1))); do
        "$PY" scripts/demo_cr.py --cfg_file "$cfg" --shard "$k" --num_shards "$shards" > "$LOGS/${name}_$k.log" 2>&1 &
    done
    wait
    local crashed
    crashed=$(grep -l "Traceback" "$LOGS/${name}"_*.log 2> /dev/null | wc -l)
    echo "== $(date '+%F %T') $name done: $(cat "$mdir"/measurement_*.csv | grep -vc '^scenario') scenarios"
    if [ "$crashed" -gt 0 ]; then
        echo "!! WARNING $name: $crashed logs with a Traceback, e.g. $(grep -l Traceback "$LOGS/${name}"_*.log | head -1)"
    fi
}

run CVAE_FOP cfgs/cvae_fop.yaml data/measurements/cvae_fop_test 12   # each process loads CUDA and the model
run CEM cfgs/cem_test.yaml data/measurements/cem_test 22
COMPARE=(--run CVAE_FOP=data/measurements/cvae_fop_test --run CEM=data/measurements/cem_test)
if [ "$RUN_FOP" = 1 ]; then
    run FOP cfgs/fop_test.yaml data/measurements/fop_test 22
    COMPARE+=(--run FOP=data/measurements/fop_test)
fi

OUT=data/measurements/compare_test
mkdir -p "$OUT"
"$PY" scripts/compare_planners.py "${COMPARE[@]}" --ref CEM --out "$OUT" | tee "$OUT/report.txt"
echo "== $(date '+%F %T') ALL DONE, report: $(pwd)/$OUT/report.txt"

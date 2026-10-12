#!/bin/bash
# CEM data collection (Collect_Data_For_ML with CEM_CPP) in SHARDS parallel processes, then the merge of the
# world-model data into the WorldModel layout (scripts/merge_wm_data.py). Needs C_Planner built with
# -DENABLE_DATA_COLLECTION=ON. Finished scenarios (OUTPUT_DIR/completed) are skipped, so the same command
# continues an interrupted collection.
#     bash fiss_plus_planner/scripts/run_collection.sh cfgs/collect_train_v3.yaml          # SHARDS=16 default
#     SHARDS=22 bash fiss_plus_planner/scripts/run_collection.sh cfgs/collect_verification_v3.yaml
# Logs: logs_collect/<config name>_<k>.log
set -u
cd "$(dirname "$0")/.." || exit 1   # fiss_plus_planner/
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one core per process
trap 'echo "stopped: ending the collection processes"; pkill -P $$; exit 130' INT TERM HUP
PY=${PY:-python}
SHARDS=${SHARDS:-16}
CFG=${1:?usage: run_collection.sh CONFIG}
NAME=$(basename "$CFG" .yaml)
OUT=$("$PY" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['OUTPUT_DIR'].rstrip('/'))" "$CFG") || exit 1
INPUT=$("$PY" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['INPUT_DIR'])" "$CFG") || exit 1
"$PY" -c "from fiss_plus_planner.planners.FOP_cpp_wrapper import frenet_planner_cpp as m; assert getattr(m, 'DATA_COLLECTION', False)" \
    2> /dev/null || { echo "C_Planner is not built with -DENABLE_DATA_COLLECTION=ON (see README.md)"; exit 1; }

mkdir -p logs_collect
echo "== $(date '+%F %T') $NAME: $SHARDS processes, $(ls "$INPUT" | grep -c '\.xml$') scenarios in $INPUT -> $OUT"
for k in $(seq 0 $((SHARDS - 1))); do
    "$PY" scripts/demo_cr.py --cfg_file "$CFG" --shard "$k" --num_shards "$SHARDS" > "logs_collect/${NAME}_$k.log" 2>&1 &
done
wait
echo "== $(date '+%F %T') collection done: $(ls "$OUT/completed" | wc -l) scenarios completed" \
     "(scenarios without a global route or ego trajectory have no data), errors: $OUT/collection_errors.log"
rm -f "$OUT/sampled_vars.parquet" "$OUT/conditions.parquet"   # merged files of an earlier run
"$PY" scripts/merge_wm_data.py --data "$OUT"
echo "== $(date '+%F %T') ALL DONE"

#!/bin/bash
# Gap diagnosis (E1) of one CVAE model on data/10k/Verification: the CVAE-FOP and the CEM plan in the same states,
#   learner  CVAE-FOP drives, the CEM plans in its states (DAgger collection: needs C_Planner built with
#            -DENABLE_DATA_COLLECTION=ON)               -> data/{output,measurements}/gap_learner_NAME
#   expert   CEM drives, the CVAE-FOP plans in its states (SHADOW_CVAE)  -> data/{output,measurements}/gap_expert_NAME
# then scripts/gap_decomposition.py NAME. Config: cfgs/gap_diagnosis.yaml (the run's copy is kept in its
# measurement folder); KEY=VALUE overrides a (dotted) key of it, VALUE parsed as YAML. Each mode runs from scratch.
#     bash fiss_plus_planner/scripts/run_gap_diagnosis.sh v3_img data/output/cvae_local/runs/v3_attn_tau0.25_beta0.1
#     MODES=expert SHARDS=12 bash fiss_plus_planner/scripts/run_gap_diagnosis.sh dagger1 data/models/dagger1
#     MODES=expert bash fiss_plus_planner/scripts/run_gap_diagnosis.sh v3_img_nowarm <run> CVAE_FOP.warm_start=false
# Logs: logs_gap/<mode>_NAME_<k>.log
set -u
cd "$(dirname "$0")/.." || exit 1   # fiss_plus_planner/
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one core per process
trap 'echo "stopped: ending the planner processes"; pkill -P $$; exit 130' INT TERM HUP
PY=${PY:-$HOME/miniconda3/envs/world-model/bin/python}
SHARDS=${SHARDS:-12}   # each process loads CUDA and the model
MODES=${MODES:-learner expert}
NAME=${1:?usage: run_gap_diagnosis.sh NAME CVAE_RUN_DIR [KEY=VALUE ...]}
RUN=${2:?usage: run_gap_diagnosis.sh NAME CVAE_RUN_DIR [KEY=VALUE ...]}
shift 2
[[ "$NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "NAME must be a plain word: $NAME"; exit 1; }
[ -f "$RUN/config.json" ] || { echo "no CVAE run in $RUN (config.json missing)"; exit 1; }

if pgrep -f "python.*scripts/demo_cr.py" > /dev/null; then
    echo "demo_cr.py processes are running; stop them first"; exit 1
fi
if [[ " $MODES " == *" learner "* ]] && ! "$PY" -c "from fiss_plus_planner.planners.FOP_cpp_wrapper import \
frenet_planner_cpp as m; assert getattr(m, 'DATA_COLLECTION', False)" 2> /dev/null; then
    echo "the learner mode needs C_Planner built with -DENABLE_DATA_COLLECTION=ON (see README.md)"; exit 1
fi

mkdir -p logs_gap
for mode in $MODES; do
    ODIR=data/output/gap_${mode}_$NAME
    MDIR=data/measurements/gap_${mode}_$NAME
    rm -rf "$ODIR" "$MDIR" logs_gap/${mode}_${NAME}_*.log
    mkdir -p "$MDIR"
    "$PY" - "$mode" "$RUN" "$ODIR" "$MDIR" "$@" <<'EOF' || exit 1
import sys, yaml
mode, run, odir, mdir, overrides = *sys.argv[1:5], sys.argv[5:]
cfg = yaml.safe_load(open("cfgs/gap_diagnosis.yaml"))
cfg["OUTPUT_DIR"], cfg["MEASUREMENTS_DIR"] = odir + "/", mdir
cfg["CVAE_FOP"]["run"] = run
if mode == "expert":
    cfg.update(PLANNER="CEM_CPP", Collect_Data_For_ML=False, SHADOW_CVAE=True)
elif mode != "learner":
    sys.exit(f"unknown mode {mode}")
for item in overrides:
    key, value = item.split("=", 1)
    *path, last = key.split(".")
    node = cfg
    for k in path:
        node = node[k]
    if last not in node:
        sys.exit(f"unknown key {key}")
    node[last] = yaml.safe_load(value)
yaml.safe_dump(cfg, open(f"{mdir}/config.yaml", "w"), sort_keys=False)
EOF
    n=$(ls data/10k/Verification | grep -c '\.xml$')
    start=$(date +%s)
    echo "== $(date '+%F %T') gap_${mode}_$NAME: $SHARDS processes, $n scenarios"
    for k in $(seq 0 $((SHARDS - 1))); do
        "$PY" scripts/demo_cr.py --cfg_file "$MDIR/config.yaml" --shard "$k" --num_shards "$SHARDS" \
            > "logs_gap/${mode}_${NAME}_$k.log" 2>&1 &
    done
    wait
    done_n=$(cat "$MDIR"/measurement_*.csv 2> /dev/null | grep -vc '^scenario')
    echo "== $(date '+%F %T') gap_${mode}_$NAME done in $(( ($(date +%s) - start) / 60 )) min: $done_n of $n scenarios," \
         "$(ls "$ODIR/cvae_cycles" 2> /dev/null | wc -l) cycle logs"
    crashed=$(grep -l "Traceback" logs_gap/${mode}_${NAME}_*.log | wc -l)
    [ "$crashed" -gt 0 ] && echo "!! $crashed logs with a Traceback, e.g. $(grep -l Traceback logs_gap/${mode}_${NAME}_*.log | head -1)"
done
"$PY" scripts/gap_decomposition.py "$NAME" | tee "data/measurements/gap_report_$NAME.txt"
echo "== $(date '+%F %T') ALL DONE, report: $(pwd)/data/measurements/gap_report_$NAME.txt"

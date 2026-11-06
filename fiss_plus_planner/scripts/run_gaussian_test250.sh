#!/bin/bash
# Closed loop of the Gaussian Planner (Gaussian_CPP: conditional Gaussian / MoG instead of the CVAE) on data/demo/Test
# (250), with the settings of the paper-CVAE runs in data/measurements/test250_nostop/{sp,cvae} (2026-11-05): 5x5x5 C++
# FOP fallback on (sp) / off (cvae = SP*), J_D off, no clearance, STOP_ON_STANDSTILL=0, 1 thread, parallel jobs (the
# runtime is not valid). Checkpoints: data/output/sp_sampler/runs/<model>/selected.pt (scripts/select_sp_checkpoint.py).
# Results: data/measurements/test250_nostop/gauss/<model>/{sp,cvae}/SP_<N>/; summary: scripts/summarize_test250_sp.py
#     bash fiss_plus_planner/scripts/run_gaussian_test250.sh
#     MODELS="mog_s0" NS="64" MODES="sp" JOBS=4 bash ...
set -u
cd "$(dirname "$0")/.." || exit 1   # fiss_plus_planner/
MODELS=${MODELS:-"gaussian_s0 gaussian_s1 gaussian_s2 mog_s0 mog_s1 mog_s2"}
NS=${NS:-"1 8 16 64 128 256 512 1024"}
MODES=${MODES:-"sp cvae"}
JOBS=${JOBS:-16}
OUT_ROOT=${OUT_ROOT:-data/measurements/test250_nostop/gauss}
INPUT=${INPUT:-data/demo/Test}
STOP_ON_STANDSTILL=${STOP_ON_STANDSTILL:-0}
BACKEND=${BACKEND:-graph_fp16}   # SP_CVAE_BACKEND: graph_fp16 / graph_fp32 / trt_fp16 / eager
CPUS=${CPUS:-}   # e.g. CPUS=2 JOBS=1: sequential and pinned like table_ii_trt (valid runtimes)
RUNS=$PWD/data/output/sp_sampler/runs

if pgrep -f "python.*scripts/(demo_cr|train_sp_sampler)\.py" > /dev/null; then
    echo "demo_cr.py / train_sp_sampler.py processes are running; stop them or wait."; exit 1
fi
for m in $MODELS; do
    [ -e "$RUNS/$m/selected.pt" ] || { echo "no $RUNS/$m/selected.pt (run scripts/select_sp_checkpoint.py)"; exit 1; }
done

jobs_list() {
    for m in $MODELS; do for mode in $MODES; do for n in $NS; do echo "$m $mode $n"; done; done; done
}
run_one() {  # model mode N
    local m=$1 mode=$2 n=$3 fb=1
    [ "$mode" = cvae ] && fb=0
    local out=$OUT_ROOT/$m/$mode
    mkdir -p "$out"
    PARALLEL=$([ "$JOBS" = 1 ] && echo 0 || echo 1) NO_TABLE=1 STOP_ON_STANDSTILL=$STOP_ON_STANDSTILL INPUT=$INPUT \
        CPUS=$CPUS PLANNERS=SP SP_IMPL=Gaussian_CPP \
        SP_CVAE_BACKEND=$BACKEND SP_FALLBACK=$fb SP_NS=$n THREADS=1 \
        SP_GAUSSIAN_WEIGHTS="$(readlink -f "$RUNS/$m/selected.pt")" OUT=$out \
        bash scripts/run_paper_table.sh > "$out/launch_$n.log" 2>&1
    echo "$(date '+%T') $m $mode N=$n: $(tail -1 "$out/launch_$n.log")"
}
export -f run_one
export OUT_ROOT RUNS INPUT STOP_ON_STANDSTILL CPUS JOBS BACKEND
echo "$(date '+%F %T') $(jobs_list | wc -l) runs, $JOBS in parallel"
jobs_list | xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _
echo "$(date '+%F %T') gaussian test250 done"

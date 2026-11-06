#!/bin/bash
# Table I of the Sparse Planner paper, measured as in the paper (2026-02/03): Python planners on the 26 scenarios of
# data/demo/UsedForSelfEvaluation, cfgs/demo_config.yaml with PLANNING_OBSTACLE_COST False (no J_D in the planning
# cost), one process after the other (the runtime is measured). N = N_W x N_S x N_T; CEM / MPPI use the same budget
# (8 iterations x N / 8 samples). SP (Sparse_FOP) takes N = CVAE samples per cycle (SP_NS) with the 5 x 5 x 5 FOP
# fallback, as in the paper.
#     bash fiss_plus_planner/scripts/run_paper_table.sh                         # planners and N below
#     PLANNERS="CEM MPPI" NS="64 125" bash fiss_plus_planner/scripts/run_paper_table.sh
#     PLANNERS=SP SP_NS="1 64" bash fiss_plus_planner/scripts/run_paper_table.sh
#     THREADS=8 OUT=data/measurements/paper_table_laptop bash ...   # numba threads (default: demo_config.yaml)
#     PARALLEL=1 PLANNERS=CEM bash ...   # next to other runs (failure rates only, the runtime is then not valid)
#     CLEARANCE=1 bash ...   # with the safe following distance (SAFETY in demo_config.yaml), OUT default paper_table_clearance
#     SP_IMPL=Sparse_CPP SP_CVAE_BACKEND=eager bash ...   # SP with the C++ FOP (eager: the same draws and results as Sparse_FOP)
#     SP_FALLBACK=0 PLANNERS=SP OUT=data/measurements/paper_table_sp_nofallback bash ...   # SP without the FOP fallback
#     SP_SAMPLER=marginal SP_IMPL=Sparse_CPP ...   # Sparse_CPP with another sampler (cvae, marginal, uniform, mean)
#     ITER_SPLIT=1 PLANNERS="CEM_CPP MPPI_CPP" NS="1 8 16 64" ...   # CEM / MPPI: N samples in total, clamp(N/8, 1, 8) iterations
#     CPUS=2 THREADS=1 ...   # pin demo_cr.py to these CPUs (taskset); NO_TABLE=1: skip paper_table.py
#     STOP_ON_STANDSTILL=0 ...   # keep simulating at v < 0.01 instead of counting the standstill as failure
#     SP_SHADOW="cvae marginal uniform mean grid" SP_IMPL=Sparse_CPP ...   # same-state evaluation, output/shadow_eval.csv
#     SP_IMPL=Gaussian_CPP SP_GAUSSIAN_WEIGHTS=data/output/sp_sampler/runs/mog_s0/best.pt ...   # conditional Gaussian / mixture instead of the CVAE
# Results (kept as the record): data/measurements/paper_table/<planner>_<N>/{measurement_<planner>.csv, config.yaml,
# log.txt} (OUT=... elsewhere) and machine.txt (CPU, GPU, commit), then scripts/paper_table.py writes table_long.csv and table_paper_layout.csv there.
# A run is skipped when it finished (marker .complete); delete its folder to measure it again.
set -u
cd "$(dirname "$0")/.." || exit 1   # fiss_plus_planner/
trap 'echo "stopped"; pkill -P $$; exit 130' INT TERM HUP
PY=${PY:-$HOME/miniconda3/envs/world-model/bin/python}
PLANNERS=${PLANNERS:-"FOP FISS+ CEM MPPI SP"}
NS=${NS:-"125 216 512 1000"}
SP_NS=${SP_NS:-"1 8 16 64 128 256 512 1000"}
THREADS=${THREADS:-}
INPUT=${INPUT:-data/demo/UsedForSelfEvaluation}
CLEARANCE=${CLEARANCE:-0}
SP_IMPL=${SP_IMPL:-Sparse_FOP}
SP_FALLBACK=${SP_FALLBACK:-1}
SP_CVAE_BACKEND=${SP_CVAE_BACKEND:-graph_fp16}
SP_SAMPLER=${SP_SAMPLER:-cvae}
SP_GAUSSIAN_WEIGHTS=${SP_GAUSSIAN_WEIGHTS:-}
SP_SHADOW=${SP_SHADOW:-}
ITER_SPLIT=${ITER_SPLIT:-0}
CPUS=${CPUS:-}
NO_TABLE=${NO_TABLE:-0}
if [ "$CLEARANCE" = 1 ]; then OUT=${OUT:-data/measurements/paper_table_clearance}; else OUT=${OUT:-data/measurements/paper_table}; fi
PAPER=${PAPER:-$HOME/backup/Desktop/IROS2026/Paper/measurement/CVAE_Comparation_Measurement}
declare -A GRID=([8]="2 2 2" [27]="3 3 3" [64]="4 4 4" [125]="5 5 5" [216]="6 6 6" [343]="7 7 7" [512]="8 8 8"
                 [729]="9 9 9" [1000]="10 10 10" [10648]="22 22 22" [14000]="35 20 20"
                 [210]="7 6 5" [504]="9 8 7" [990]="11 10 9"   # N_W odd: the grid contains d = 0 (lane centre)
                 [12]="3 2 2" [18]="3 3 2" [60]="5 4 3" [252]="7 6 6")
num_inputs=$(ls "$INPUT" | grep -c '\.xml$')

PARALLEL=${PARALLEL:-0}
if [ "$PARALLEL" != 1 ] && pgrep -f "python.*scripts/demo_cr.py" > /dev/null; then
    echo "demo_cr.py processes are running; the runtime would be wrong. Stop them first."; exit 1
fi
mkdir -p "$OUT"
{ date '+%F %T'; hostname; git rev-parse HEAD; git status --short --untracked-files=no | head -20
  echo "THREADS=${THREADS:-config} CLEARANCE=$CLEARANCE PARALLEL=$PARALLEL PLANNERS=$PLANNERS SP_IMPL=$SP_IMPL SP_FALLBACK=$SP_FALLBACK SP_CVAE_BACKEND=$SP_CVAE_BACKEND SP_SAMPLER=$SP_SAMPLER SP_GAUSSIAN_WEIGHTS=$SP_GAUSSIAN_WEIGHTS SP_SHADOW=$SP_SHADOW ITER_SPLIT=$ITER_SPLIT CPUS=$CPUS affinity $(taskset -pc $$ | cut -d: -f2)"; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|MHz'; free -g | head -2
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2> /dev/null
  "$PY" -c 'import torch, numba; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "numba", numba.__version__)'
} >> "$OUT/machine.txt" 2>&1
for p in $PLANNERS; do
    if [ "$p" = SP ]; then list=$SP_NS; else list=$NS; fi
    for n in $list; do
        if [ "$p" = SP ]; then nw=5 ns=5 nt=5; planner=$SP_IMPL
        elif [ "$ITER_SPLIT" = 1 ] && [[ "$p" == CEM* || "$p" == MPPI* ]]; then nw=5 ns=5 nt=5; planner=$p  # N set below
        else read -r nw ns nt <<< "${GRID[$n]:?no N_W x N_S x N_T grid for N=$n (known: ${!GRID[*]})}"; planner=$p; fi
        dir=$OUT/${p}_$n
        if [ -f "$dir/.complete" ]; then echo "-- $p N=$n done, skipped"; continue; fi
        rm -rf "$dir"; mkdir -p "$dir"
        "$PY" - "$planner" "$nw" "$ns" "$nt" "$dir" "$INPUT" "$n" "$THREADS" "$CLEARANCE" "$SP_FALLBACK" "$SP_CVAE_BACKEND" "$SP_SAMPLER" "$SP_SHADOW" "$ITER_SPLIT" "$SP_GAUSSIAN_WEIGHTS" <<'EOF' || exit 1
import os, sys, yaml
p, nw, ns, nt, out, inp, n, threads, clearance, sp_fallback, sp_backend, sp_sampler, sp_shadow, iter_split, sp_gauss = sys.argv[1:]
cfg = yaml.safe_load(open("cfgs/demo_config.yaml"))
cfg["SAFETY"]["check_clearance"] = clearance == "1"
if threads:
    cfg["Num_Threads_For_CollisionChecker"] = int(threads)
cfg.update(PLANNER=p, SP_NUM_SAMPLES=int(n), SP_FALLBACK=sp_fallback == "1", SP_CVAE_BACKEND=sp_backend, SP_SAMPLER=sp_sampler, SP_SHADOW_SAMPLERS=sp_shadow.split(), N_W_SAMPLE=int(nw), N_S_SAMPLE=int(ns), N_T_SAMPLE=int(nt), INPUT_DIR=inp, FILES=[],
           OUTPUT_DIR=out + "/output/", MEASUREMENTS_DIR=out, SAVE_GIF=False, SAVE_MEASUREMENTS=True,
           Runtime_Measurement=True,  # C++ wrappers: no conversion of all feasible trajectories to Python
           PLANNING_OBSTACLE_COST=False, STOP_ON_STANDSTILL=os.environ.get("STOP_ON_STANDSTILL", "1") == "1")
if sp_gauss:
    cfg["SP_GAUSSIAN_WEIGHTS"] = sp_gauss
if iter_split == "1" and p.startswith(("CEM", "MPPI")):
    it = min(8, max(1, int(n) // 8))  # population >= 8 where N allows, N samples in total
    assert int(n) % it == 0, (n, it)
    cfg["CEM" if p.startswith("CEM") else "MPPI"].update(num_iterations=it, population=int(n) // it)
yaml.safe_dump(cfg, open(out + "/config.yaml", "w"), sort_keys=False)
EOF
        echo "== $(date '+%F %T') $p N=$n (grid ${nw}x${ns}x${nt})"
        PIN=(); [ -n "$CPUS" ] && PIN=(taskset -c "$CPUS")
        if "${PIN[@]}" "$PY" scripts/demo_cr.py --cfg_file "$dir/config.yaml" > "$dir/log.txt" 2>&1; then
            touch "$dir/.complete"
            echo "   $(cat "$dir"/measurement_*.csv 2> /dev/null | grep -vc '^scenario') of $num_inputs scenarios with a row"
        else
            echo "!! $p N=$n: demo_cr.py failed (see $dir/log.txt); run the same command again"
        fi
    done
done
[ "$NO_TABLE" = 1 ] || "$PY" scripts/paper_table.py --dir "$OUT" --input "$INPUT" --paper "$PAPER"

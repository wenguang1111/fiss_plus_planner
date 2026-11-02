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
OUT=${OUT:-data/measurements/paper_table}
PAPER=${PAPER:-$HOME/backup/Desktop/IROS2026/Paper/measurement/CVAE_Comparation_Measurement}
declare -A GRID=([8]="2 2 2" [27]="3 3 3" [64]="4 4 4" [125]="5 5 5" [216]="6 6 6" [343]="7 7 7" [512]="8 8 8"
                 [729]="9 9 9" [1000]="10 10 10")
num_inputs=$(ls "$INPUT" | grep -c '\.xml$')

if pgrep -f "python.*scripts/demo_cr.py" > /dev/null; then
    echo "demo_cr.py processes are running; the runtime would be wrong. Stop them first."; exit 1
fi
mkdir -p "$OUT"
{ date '+%F %T'; hostname; git rev-parse HEAD; git status --short --untracked-files=no | head -20
  echo "THREADS=${THREADS:-config}"; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|MHz'; free -g | head -2
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2> /dev/null
  "$PY" -c 'import torch, numba; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "numba", numba.__version__)'
} >> "$OUT/machine.txt" 2>&1
for p in $PLANNERS; do
    if [ "$p" = SP ]; then list=$SP_NS; else list=$NS; fi
    for n in $list; do
        if [ "$p" = SP ]; then nw=5 ns=5 nt=5; planner=Sparse_FOP
        else read -r nw ns nt <<< "${GRID[$n]:?no N_W x N_S x N_T grid for N=$n (known: ${!GRID[*]})}"; planner=$p; fi
        dir=$OUT/${p}_$n
        if [ -f "$dir/.complete" ]; then echo "-- $p N=$n done, skipped"; continue; fi
        rm -rf "$dir"; mkdir -p "$dir"
        "$PY" - "$planner" "$nw" "$ns" "$nt" "$dir" "$INPUT" "$n" "$THREADS" <<'EOF' || exit 1
import sys, yaml
p, nw, ns, nt, out, inp, n, threads = sys.argv[1:]
cfg = yaml.safe_load(open("cfgs/demo_config.yaml"))
if threads:
    cfg["Num_Threads_For_CollisionChecker"] = int(threads)
cfg.update(PLANNER=p, SP_NUM_SAMPLES=int(n), N_W_SAMPLE=int(nw), N_S_SAMPLE=int(ns), N_T_SAMPLE=int(nt), INPUT_DIR=inp, FILES=[],
           OUTPUT_DIR=out + "/output/", MEASUREMENTS_DIR=out, SAVE_GIF=False, SAVE_MEASUREMENTS=True,
           PLANNING_OBSTACLE_COST=False)
yaml.safe_dump(cfg, open(out + "/config.yaml", "w"), sort_keys=False)
EOF
        echo "== $(date '+%F %T') $p N=$n (grid ${nw}x${ns}x${nt})"
        if "$PY" scripts/demo_cr.py --cfg_file "$dir/config.yaml" > "$dir/log.txt" 2>&1; then
            touch "$dir/.complete"
            echo "   $(cat "$dir"/measurement_*.csv 2> /dev/null | grep -vc '^scenario') of $num_inputs scenarios with a row"
        else
            echo "!! $p N=$n: demo_cr.py failed (see $dir/log.txt); run the same command again"
        fi
    done
done
"$PY" scripts/paper_table.py --dir "$OUT" --input "$INPUT" --paper "$PAPER"

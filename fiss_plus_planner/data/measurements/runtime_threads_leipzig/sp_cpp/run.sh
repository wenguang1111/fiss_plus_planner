#!/bin/bash
# run.sh OUT PLANNER N BACKEND CORES   (CORES 1: CPU 2, 1 thread; 16: CPUs 0-15, 16 threads)
set -eu
OUT=$1 P=$2 N=$3 B=$4 C=$5
S=/tmp/claude-1002/-home-xu-workspace-fiss-plus-planner/7003617b-a111-4426-8cd8-0d723ae72fe4/scratchpad
cd /home/xu/workspace/fiss_plus_planner/fiss_plus_planner
PY=$HOME/miniconda3/envs/world-model/bin/python
dir=$OUT/${P}_${B}_$N; rm -rf $dir; mkdir -p $dir
$PY - $P $N $B $C $dir ${INP:-$S/in_leip} <<'PYEOF'
import sys, yaml
p, n, b, c, out, inp = sys.argv[1:]
cfg = yaml.safe_load(open("cfgs/demo_config.yaml"))
cfg["SAFETY"]["check_clearance"] = False
cfg.update(PLANNER=p, SP_NUM_SAMPLES=int(n), SP_CVAE="paper", SP_CVAE_BACKEND=b, N_W_SAMPLE=5, N_S_SAMPLE=5, N_T_SAMPLE=5,
           INPUT_DIR=inp, FILES=[], OUTPUT_DIR=out + "/output/", MEASUREMENTS_DIR=out, SAVE_GIF=False, SAVE_MEASUREMENTS=True,
           Runtime_Measurement=True, PLANNING_OBSTACLE_COST=False, Num_Threads_For_CollisionChecker=int(c))
yaml.safe_dump(cfg, open(out + "/config.yaml", "w"), sort_keys=False)
PYEOF
if [ "$C" = 1 ]; then CPUS=2; else CPUS=0-15; fi
taskset -c $CPUS $PY scripts/demo_cr.py --cfg_file $dir/config.yaml > $dir/log.txt 2>&1 || { echo "FAILED $dir"; tail -20 $dir/log.txt; }

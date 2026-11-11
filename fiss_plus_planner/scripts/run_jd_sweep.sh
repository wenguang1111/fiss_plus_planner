#!/bin/bash
# N-sweep with J_D ON in the planning cost (2026-11-11): C++ FOP / FISS+ / CEM / MPPI, no clearance, sequential on one
# P-core (CPU 2, 1 thread, runtimes valid). Same grids / budgets as Table II (run_table_ii.sh), CEM / MPPI from N = 8:
#   FOP / FISS+ grids (odd N_W): 12 18 60 125 252 504 990; CEM / MPPI: N 8 16 64 128 256 512 1024, clamp(N/8, 1, 8) iterations.
# Part 1: 26 scenarios UsedForSelfEvaluation, STOP_ON_STANDSTILL=1 (as Table II)  -> data/measurements/table_ii_jd/{grid,iter}
# Part 2: data/demo/Test (250), STOP_ON_STANDSTILL=0 (as test250_nostop)           -> data/measurements/test250_jd/{grid,iter}
# Note: FISS+ C++ evaluates cost_total (no J_D term), so FISS+ is identical with J_D on and off.
# Resumable (.complete marker per run folder).
set -u
cd "$(dirname "$0")/.." || exit 1
export JD=1 CPUS=${CPUS:-2} THREADS=${THREADS:-1} NO_TABLE=1
GRID_NS="12 18 60 125 252 504 990"
ITER_NS="8 16 64 128 256 512 1024"
for part in selfeval test250; do
    if [ "$part" = selfeval ]; then export INPUT=data/demo/UsedForSelfEvaluation STOP_ON_STANDSTILL=1; O=data/measurements/table_ii_jd
    else export INPUT=data/demo/Test STOP_ON_STANDSTILL=0; O=data/measurements/test250_jd; fi
    PLANNERS="FOP_CPP FISS+_CPP" NS="$GRID_NS" OUT=$O/grid bash scripts/run_paper_table.sh || exit 1
    ITER_SPLIT=1 PLANNERS="CEM_CPP MPPI_CPP" NS="$ITER_NS" OUT=$O/iter bash scripts/run_paper_table.sh || exit 1
    echo "$part done $(date '+%F %T')"
done
echo "jd_sweep done"

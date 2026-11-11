#!/bin/bash
# Test 250 (data/demo/Test), J_D off, STOP_ON_STANDSTILL=0, as test250_nostop (2026-11-05): the missing FISS+ (grids 12..990)
# and MPPI (N 1..1024, clamp(N/8,1,8) iterations), plus FOP 125 again into test250_recheck (reproducibility of the old runs).
# One process per (planner, N), each pinned to one core, in parallel: failure rates and cost only (runtimes not valid).
set -u
cd "$(dirname "$0")/.." || exit 1
export INPUT=data/demo/Test STOP_ON_STANDSTILL=0 JD=0 THREADS=1 NO_TABLE=1 PARALLEL=1
O=data/measurements/test250_nostop
cpu=2
for n in 12 18 60 125 252 504 990; do
    CPUS=$cpu PLANNERS=FISS+_CPP NS=$n OUT=$O/grid bash scripts/run_paper_table.sh > $O/run_fiss_$n.log 2>&1 & cpu=$((cpu+1))
done
for n in 1 8 16 64 128 256 512 1024; do
    CPUS=$cpu ITER_SPLIT=1 PLANNERS=MPPI_CPP NS=$n OUT=$O/iter bash scripts/run_paper_table.sh > $O/run_mppi_$n.log 2>&1 & cpu=$((cpu+1))
done
CPUS=$cpu PLANNERS=FOP_CPP NS=125 OUT=data/measurements/test250_recheck/grid bash scripts/run_paper_table.sh > $O/run_recheck_fop125.log 2>&1 &
wait
echo "test250_missing done $(date '+%F %T')"

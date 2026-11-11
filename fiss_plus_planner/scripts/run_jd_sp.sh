#!/bin/bash
# J_D on, 26 scenarios (as table_ii_jd from run_jd_sweep.sh): the missing CEM / MPPI N=1 and SP* / SP (Sparse_CPP, trt_fp16,
# without / with the 5x5x5 FOP fallback), CPU 2, 1 thread, sequential. Resumable (.complete).
set -u
cd "$(dirname "$0")/.." || exit 1
export JD=1 CPUS=2 THREADS=1 NO_TABLE=1 STOP_ON_STANDSTILL=1 INPUT=data/demo/UsedForSelfEvaluation
O=data/measurements/table_ii_jd
ITER_SPLIT=1 PLANNERS="CEM_CPP MPPI_CPP" NS=1 OUT=$O/iter bash scripts/run_paper_table.sh || exit 1
SP_IMPL=Sparse_CPP SP_CVAE_BACKEND=trt_fp16 SP_FALLBACK=0 PLANNERS=SP SP_NS="1 8 16 64 128 256 512 1024" OUT=$O/cvae bash scripts/run_paper_table.sh || exit 1
SP_IMPL=Sparse_CPP SP_CVAE_BACKEND=trt_fp16 SP_FALLBACK=1 PLANNERS=SP SP_NS="1 8 16 64 128 256 512 1024" OUT=$O/sp bash scripts/run_paper_table.sh || exit 1
echo "jd_sp done"

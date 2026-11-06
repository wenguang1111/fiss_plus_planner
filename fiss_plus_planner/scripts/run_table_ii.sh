#!/bin/bash
# Table "performance under sample number N" (2026-11-03): all C++ planners and SP_CPP (CUDA graph fp16), one after the
# other on one P-core (CPU 2, 1 thread), 26 scenarios of UsedForSelfEvaluation, J_D off in the planning cost.
# N columns 1 8 16 64 128 256 512 1024; FOP / FISS+ grids with odd N_W (contain d = 0, N_S, N_T >= 2): 12 18 60 125 252 504 990;
# CEM / MPPI: N in total, clamp(N/8, 1, 8) iterations; CVAE = SP_CPP without fallback, SP = with the 5x5x5 C++ FOP fallback.
# Results: data/measurements/table_ii/{grid,iter,sp,cvae}/<planner>_<N>/, table: scripts/table_ii.py
set -u
cd "$(dirname "$0")/.." || exit 1
O=${OUT_ROOT:-data/measurements/table_ii}
export CPUS=${CPUS:-2} THREADS=${THREADS:-1} NO_TABLE=1
PLANNERS="FOP_CPP FISS+_CPP" NS="12 18 60 125 252 504 990" OUT=$O/grid bash scripts/run_paper_table.sh || exit 1
ITER_SPLIT=1 PLANNERS="CEM_CPP MPPI_CPP" NS="1 8 16 64 128 256 512 1024" OUT=$O/iter bash scripts/run_paper_table.sh || exit 1
SP_IMPL=Sparse_CPP SP_CVAE_BACKEND=graph_fp16 SP_FALLBACK=1 PLANNERS=SP SP_NS="1 8 16 64 128 256 512 1024" OUT=$O/sp bash scripts/run_paper_table.sh || exit 1
SP_IMPL=Sparse_CPP SP_CVAE_BACKEND=graph_fp16 SP_FALLBACK=0 PLANNERS=SP SP_NS="1 8 16 64 128 256 512 1024" OUT=$O/cvae bash scripts/run_paper_table.sh || exit 1
echo "table_ii done"

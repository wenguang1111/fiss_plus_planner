#!/bin/bash
# SP with the FOP fallback and the non-learned samplers instead of the CVAE (2026-11-05): uniform (FOP sampling box),
# marginal (rows of the CVAE training targets), mean (N copies of the target mean), N 1..1024; same settings as
# run_table_ii.sh (Sparse_CPP, CPU 2, 1 thread, 5x5x5 C++ FOP fallback, J_D off). Compare with table_ii_trt/sp (CVAE).
# Results: data/measurements/sp_ablation_fallback/<sampler>/SP_<N>/; resumable (.complete)
set -u
cd "$(dirname "$0")/.." || exit 1
O=${OUT_ROOT:-data/measurements/sp_ablation_fallback}
export CPUS=${CPUS:-2} THREADS=${THREADS:-1} NO_TABLE=1
for s in uniform marginal mean; do
    SP_IMPL=Sparse_CPP SP_SAMPLER=$s SP_FALLBACK=1 PLANNERS=SP SP_NS="1 8 16 64 128 256 512 1024" OUT=$O/$s bash scripts/run_paper_table.sh || exit 1
done
echo "sp_ablation_fallback done"

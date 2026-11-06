#!/bin/bash
# Conditional Gaussian (K=1) and Gaussian mixture (K=5) on the 2026-02-15 SP data, seeds 0-2, same split (split_seed 0).
# Three lanes in parallel on the one GPU; logs in data/output/sp_sampler/runs/<name>/train.log.
#   bash fiss_plus_planner/scripts/run_sp_sampler_training.sh
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-$HOME/miniconda3/envs/world-model/bin/python}
RUNS=fiss_plus_planner/data/output/sp_sampler/runs

run() {  # model seed
    local name="$1_s$2"
    mkdir -p "$RUNS/$name"
    "$PY" fiss_plus_planner/scripts/train_sp_sampler.py --model "$1" --K 5 --seed "$2" --name "$name" \
        > "$RUNS/$name/train.log" 2>&1
}

(run gaussian 0; run mog 1) &
(run mog 0; run gaussian 2) &
(run gaussian 1; run mog 2) &
wait
echo "all done"

#!/bin/bash
# All planners on data/10k/Test under the current closed-loop rules, one after the other and from scratch,
# then the comparison. Scenarios lost to crashed processes (RAM fault on this PC) are run again with
# their files read from disk (up to RETRIES passes).
#     bash fiss_plus_planner/scripts/run_eval_suite.sh ROUND [NAME ...]
#     bash fiss_plus_planner/scripts/run_eval_suite.sh goalpos                     # all planners below (FISS+ 10x10x10)
#     bash fiss_plus_planner/scripts/run_eval_suite.sh goalpos CVAE_warm CVAE_egoB   # rerun only these
# Results: data/measurements/ROUND/<NAME> (with the config.yaml of the run), data/output/ROUND/<NAME>,
# logs logs_test/ROUND_<NAME>_*.log, report data/measurements/ROUND/compare/report.txt (all planners
# that have results in ROUND, reference CEM).
set -u
cd "$(dirname "$0")/.." || exit 1   # fiss_plus_planner/
# One CPU core per planner process: numpy (OpenBLAS) and torch (OpenMP) would otherwise start a thread pool
# per core in every process. Ctrl+C / kill / a closed terminal also stops the planner processes (background
# jobs of a script ignore Ctrl+C and would keep running without it).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'echo "stopped: ending the planner processes"; pkill -P $$; exit 130' INT TERM HUP
PY=${PY:-$HOME/miniconda3/envs/LUT/bin/python}
SHARDS=${SHARDS:-22}
RETRIES=${RETRIES:-2}
ROUND=${1:?usage: run_eval_suite.sh ROUND [NAME ...]}
shift
[[ "$ROUND" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "ROUND must be a plain word: $ROUND"; exit 1; }
# NAME -> base config and overrides (KEY=VALUE, dotted keys of the config, YAML values)
declare -A PLAN=(
    [CEM]="cfgs/cem_test.yaml"
    [FOP]="cfgs/fop_test.yaml"
    [FISS+]="cfgs/fiss_test.yaml"
    [CVAE_v1]="cfgs/cvae_fop.yaml CVAE_FOP.run=data/models/cvae_v1 CVAE_FOP.warm_start=false"
    [CVAE_warm]="cfgs/cvae_fop.yaml CVAE_FOP.run=data/models/cvae_v1 CVAE_FOP.warm_start=true"
    [CVAE_egoB]="cfgs/cvae_fop.yaml CVAE_FOP.run=data/output/cvae_local/runs/attn_egoB_tau0.25_beta0.1 CVAE_FOP.warm_start=true"
    [CVAE_ego_full]="cfgs/cvae_fop.yaml CVAE_FOP.run=data/models/ablation_ego_full CVAE_FOP.warm_start=true"
)
NAMES=${*:-CEM FOP FISS+ CVAE_v1 CVAE_warm CVAE_egoB CVAE_ego_full}
LOGS=logs_test
INPUT=${INPUT:-data/10k/Test}   # scenario folder (another one only for tests)
num_inputs=$(ls $INPUT | grep -c '\.xml$')

if pgrep -f "python.*scripts/demo_cr.py" > /dev/null; then
    echo "demo_cr.py processes are running; stop them first:"; pgrep -fa "python.*scripts/demo_cr.py" | cut -c1-120; exit 1
fi

# drop_cache PATH...: drop the cached pages of these files / folders, so they are read from disk again
drop_cache() {
    "$PY" -c "
import os, sys
for root in sys.argv[1:]:
    files = [root] if os.path.isfile(root) else [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
    for f in files:
        fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)" "$@"
}

# write_config NAME FILE [FILES]: the run's config (base + overrides, this round's folders, optional scenario list)
write_config() {
    local spec=${PLAN[$1]}
    "$PY" - "$2" "data/output/$ROUND/$1/" "data/measurements/$ROUND/$1" "$INPUT" "${3:-}" $spec <<'EOF' || exit 1
import sys, yaml
out, odir, mdir, inp, files, base, overrides = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7:]
cfg = yaml.safe_load(open(base))
cfg.update(OUTPUT_DIR=odir, MEASUREMENTS_DIR=mdir, INPUT_DIR=inp, FILES=[f for f in files.split(",") if f])
for item in overrides:
    key, value = item.split("=", 1)
    *path, last = key.split(".")
    node = cfg
    for k in path:
        node = node[k]
    if last not in node:
        sys.exit(f"unknown key {key}")
    node[last] = yaml.safe_load(value)
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
EOF
}

# missing NAME: comma list of Test scenarios without a measurement row
missing() {
    "$PY" - "$INPUT" "data/measurements/$ROUND/$1" <<'EOF'
import glob, os, sys, pandas as pd
done = set()
for f in glob.glob(os.path.join(sys.argv[2], "measurement_*.csv")):
    d = pd.read_csv(f, usecols=[0])
    done |= set(d.iloc[:, 0].astype(str).str.strip())
print(",".join(sorted(f for f in os.listdir(sys.argv[1]) if f.endswith(".xml") and f not in done)))
EOF
}

for name in $NAMES; do
    [ -n "${PLAN[$name]:-}" ] || { echo "unknown planner $name (known: ${!PLAN[*]})"; exit 1; }
    mdir=data/measurements/$ROUND/$name; odir=data/output/$ROUND/$name
    rm -rf "$mdir" "$odir" "$LOGS/${ROUND}_${name}"_*.log
    mkdir -p "$mdir" "$LOGS"
    write_config "$name" "$mdir/config.yaml"
    run=$(sed -n 's/^  run: //p' "$mdir/config.yaml")
    drop_cache "$INPUT" ${run:+"$run"}
    start=$(date +%s)
    echo "== $(date '+%F %T') $name: $SHARDS processes, $num_inputs scenarios"
    for k in $(seq 0 $((SHARDS - 1))); do
        "$PY" scripts/demo_cr.py --cfg_file "$mdir/config.yaml" --shard "$k" --num_shards "$SHARDS" \
            > "$LOGS/${ROUND}_${name}_$k.log" 2>&1 &
    done
    wait
    for pass in $(seq 1 "$RETRIES"); do
        todo=$(missing "$name")
        n=$(echo "$todo" | tr ',' '\n' | grep -c .)
        [ "$n" -eq 0 ] && break   # (the 4 Test scenarios that fail before planning end within seconds)
        echo "   pass $pass: $n scenarios without a result, running them again (files read from disk)"
        write_config "$name" "$mdir/config_retry$pass.yaml" "$todo"
        drop_cache "$INPUT" ${run:+"$run"}
        for k in 0 1 2 3; do
            "$PY" scripts/demo_cr.py --cfg_file "$mdir/config_retry$pass.yaml" --shard "$k" --num_shards 4 \
                > "$LOGS/${ROUND}_${name}_retry${pass}_$k.log" 2>&1 &
        done
        wait
    done
    left=$(missing "$name")
    echo "== $(date '+%F %T') $name done in $(( ($(date +%s) - start) / 60 )) min: $((num_inputs - $(echo "$left" | tr ',' '\n' | grep -c .))) of $num_inputs scenarios${left:+; without a result: $left}"
done

RUNS=()
for name in CEM FOP FISS+ CVAE_v1 CVAE_warm CVAE_egoB CVAE_ego_full; do
    ls data/measurements/$ROUND/$name/measurement_*.csv > /dev/null 2>&1 && RUNS+=(--run "$name=data/measurements/$ROUND/$name")
done
OUT=data/measurements/$ROUND/compare
mkdir -p "$OUT"
"$PY" scripts/compare_planners.py "${RUNS[@]}" --ref CEM --out "$OUT" 2>&1 | grep -v Warning | tee "$OUT/report.txt"
echo "== $(date '+%F %T') ALL DONE, report: $(pwd)/$OUT/report.txt"

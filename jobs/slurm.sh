#!/bin/bash
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J attention
#
# sbatch -A ACCOUNT -p PARTITION [-C FEATURE] -o logs/%x-%j.out jobs/slurm.sh [--quick]
#
# Builds in node-local temporary space, runs the correctness tests (and stops if they
# fail), then the benchmark while nvidia-smi samples the SM clock, and copies
# results/<gpu>/ back to the submit folder. CUDA_MODULE, if set, is loaded first.
set -uo pipefail
SRC=${SLURM_SUBMIT_DIR:-$PWD}
WORK=${TMPDIR:-/tmp}/attn-${SLURM_JOB_ID:-$$}
export CUDA_CACHE_PATH=$WORK/nv
mkdir -p "$WORK" && cp -r "$SRC/src" "$SRC/Makefile" "$WORK/" && cd "$WORK" || exit 1
if [ -n "${CUDA_MODULE:-}" ]; then module load "$CUDA_MODULE" || exit 1; fi
hostname
nvidia-smi --query-gpu=name,driver_version,clocks.sm,temperature.gpu,power.limit --format=csv
if nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | grep -qi reset; then
    echo "the GPU on $(hostname) needs a reset; resubmit with --exclude=$(hostname)"; exit 3
fi
make -j4 all build/ptxas.txt || exit 1
./build/attn_test | tee test.txt || { echo "TESTS FAILED"; exit 1; }
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv -lms 500 > clocks.csv &
smi=$!
./build/attn_bench "$@"
kill $smi
out=$(ls -d results/*/ | head -1)
cp test.txt clocks.csv build/ptxas.txt "$out"
mkdir -p "$SRC/results" && cp -r results/* "$SRC/results/"
echo "results in $SRC/results/$(basename "$out")"
rm -rf "$WORK"

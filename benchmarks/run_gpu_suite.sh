#!/usr/bin/env bash
# Run inside an existing CUDA allocation (local NVIDIA GPU or Slurm GPU job).
# No site-specific Slurm account, queue or GPU type is assumed.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

output_dir="${1:-results/inference-gpu}"
mkdir -p "$output_dir"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable. Allocate an NVIDIA GPU and install a compatible CUDA PyTorch wheel.')
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name())
print('Compute capability:', torch.cuda.get_device_capability())
PY

# Run GPU correctness tests, including CPU/CUDA gradients and hard constraints.
python -m unittest discover -s tests/inference -v

repeats="${INFERENCE_BENCH_REPEATS:-5}"
warmup="${INFERENCE_BENCH_WARMUP:-3}"
compiler="${INFERENCE_BENCH_COMPILE:-none}"
run_cpu="${INFERENCE_BENCH_CPU:-1}"

run_case() {
    local name="$1"
    local variables="$2"
    local factors="$3"
    local arity="$4"
    local graphs="$5"
    local device
    local -a extra
    for device in cuda cpu; do
        if [[ "$device" == cpu && "$run_cpu" != 1 ]]; then
            continue
        fi
        extra=()
        if [[ "$device" == cuda ]]; then
            extra+=(--verify-cpu)
        fi
        python benchmarks/inference_core.py \
            --device "$device" \
            --variables "$variables" --factors "$factors" \
            --arity "$arity" --graphs "$graphs" \
            --iterations 20 --repeats "$repeats" --warmup "$warmup" \
            --compile "$compiler" "${extra[@]}" \
            --json-out "$output_dir/${name}-${device}.json" \
            --csv-out "$output_dir/${name}-${device}.csv"
    done
}

# Factor count is per graph; all cases use matched CPU/CUDA inputs.
run_case small_pairwise 100 250 2 1
run_case medium_ternary 500 1200 3 1
run_case batched_pairwise 100 150 2 16

echo "GPU validation and benchmarks written to: $output_dir"

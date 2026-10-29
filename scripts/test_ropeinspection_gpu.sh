#!/usr/bin/env bash
# Run manually in the Jetson host shell. No CPU fallback or dependency installs.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"

inputs=(
    frayed/images/20260928_101656.jpg
    frayed/images/20260928_101727.jpg
    frayed/images/20260928_110909.jpg
    frayed/images/20260928_111825.jpg
    frayed/images/20260928_111831.jpg
)
mode='five-image representative subset'
if (( $# > 1 )); then
    echo "Usage: $0 [--all]" >&2
    exit 2
fi
case "${1:-}" in
    '') ;;
    --all)
        inputs=(frayed/images)
        mode='all current images in frayed/images'
        ;;
    -h|--help)
        echo "Usage: $0 [--all]"
        echo 'Default: five representative images. --all: all current frayed/images.'
        exit 0
        ;;
    *) echo "Unknown option: $1. Usage: $0 [--all]" >&2; exit 2 ;;
esac

for input in "${inputs[@]}"; do
    if [[ ! -f "$input" && ! -d "$input" ]]; then
        echo "Missing test input: $input" >&2
        exit 1
    fi
done

output='runtime_test_output/fray_yolo_gpu_test'
mkdir -p -- "$output/metadata/config" "$output/metadata/matplotlib"
export YOLO_AUTOINSTALL=false
export YOLO_CONFIG_DIR="$repo_root/$output/metadata/config"
export MPLCONFIGDIR="$repo_root/$output/metadata/matplotlib"
log_file="$output/validation_$(date +%Y%m%d_%H%M%S).log"

echo "Test mode: $mode"
echo "Output: $repo_root/$output"
echo "Console log: $repo_root/$log_file"
# --input already accepts multiple paths. --debug retains the existing useful
# mask/cue/centerline outputs and per-rope strips/geometry; originals stay intact.
# Python verifies CUDA and fails before loading the model if device 0 is unavailable.
python3 -u ropeinspection.py \
    --model frayed/fray_detection.pt \
    --input "${inputs[@]}" \
    --output "$output" \
    --device 0 \
    --imgsz 960 \
    --conf 0.25 \
    --debug 2>&1 | tee "$log_file"

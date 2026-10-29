#!/usr/bin/env bash
# Run manually from the normal Jetson shell; never falls back to CPU.
set -euo pipefail
if (( $# != 0 )); then
    echo "Usage: $0 (deterministic 40-image GPU review)" >&2
    exit 2
fi
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root"
export YOLO_AUTOINSTALL=false

python3 -u - <<'PY'
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

root = Path.cwd()
source = root / 'frayed/images'
output = root / 'runtime_test_output/fray_yolo_40_review'
count = 40
# Direct source photographs only; never recurse into archives or outputs.
# Match the pipeline's generated-image filename exclusions.
generated_suffixes = ('_mask', '_masks', '_overlay', '_prediction',
                      '_fray_score', '_fray_instances')
images = sorted(p for p in source.iterdir()
                if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
                and not p.stem.lower().endswith(generated_suffixes)
                and not p.stem.lower().startswith(('mask_', 'overlay_')))
if len(images) < count:
    sys.exit(f'Need at least {count} source images; found {len(images)} in {source}')
# Integer-spaced indices include both endpoints and are unique when N >= 40.
selected = [images[i * (len(images) - 1) // (count - 1)] for i in range(count)]
if not (root / 'frayed/fray_detection.pt').is_file():
    sys.exit('Missing local checkpoint: frayed/fray_detection.pt')
if output.exists():
    sys.exit(f'Review directory already exists: {output}\n'
             'Move that directory aside before rerunning; existing results are preserved.')
output.mkdir(parents=True)
review = output / 'final_overlays'
review.mkdir()
(output / 'selected_images.txt').write_text(
    ''.join(p.name + '\n' for p in selected), encoding='utf-8')
print(f'Selected exactly {count} of {len(images)} source images:')
for i, path in enumerate(selected, 1):
    print(f'  {i:02d}. {path.name}')
print(f'Review folder: {review}', flush=True)

for variable, directory in [('YOLO_CONFIG_DIR', output / 'metadata/config'),
                            ('MPLCONFIGDIR', output / 'metadata/matplotlib')]:
    directory.mkdir(parents=True)
    os.environ[variable] = str(directory)

# Existing startup prints CUDA information, requires device 0, and selects
# FP16 on supported CUDA hardware. No --debug: keep normal detailed outputs.
command = [sys.executable, '-u', str(root / 'ropeinspection.py'),
           '--model', str(root / 'frayed/fray_detection.pt'),
           '--input', *map(str, selected), '--output', str(output),
           '--device', '0', '--imgsz', '960', '--conf', '0.25']
completed = subprocess.run(command, check=False)

# Collect even if the pipeline reports individual inspection failures.
# Keep its summary.json intact, including timing, environment and diagnostics.
summary_path = output / 'summary.json'
if not summary_path.is_file():
    sys.exit(f'Pipeline exited {completed.returncode} without summary.json; '
             f'any partial outputs remain under {output}')
summary = json.loads(summary_path.read_text())
results = {Path(r['filename']).resolve(): r for r in summary['images']}
columns = ['filename', 'status', 'raw_yolo_rope_count', 'deduped_rope_count',
           'usable_rope_count', 'preliminary_fray_candidates', 'final_fray_count',
           'total_processing_ms', 'final_overlay_path']
successful_statuses = {'FRAY_DETECTED', 'OK_NO_FRAY'}
counts = dict.fromkeys((*sorted(successful_statuses), 'INSPECTION_FAILED'), 0)
times = []
sites = 0
copied = 0
collection_errors = []
with (output / 'review.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for path in selected:
        result = results.get(path.resolve())
        if result is None:
            # Interrupted runs remain incomplete in the original summary;
            # the CSV explicitly marks missing results as inspection failures.
            result = dict(status='INSPECTION_FAILED')
            collection_errors.append(f'No result recorded for {path.name}')
        status = result['status']
        counts[status] = counts.get(status, 0) + 1
        row = {key: result.get(key, '') for key in columns}
        row.update(filename=path.name, final_overlay_path='')
        if status in successful_statuses:
            artifact = result.get('artifacts')
            overlay = Path(artifact) / 'final_overlay.jpg' if artifact else None
            if overlay is None or not overlay.is_file():
                collection_errors.append(f'Missing successful overlay: {path.name}')
            else:
                destination = review / path.name
                if path.suffix.lower() in {'.jpg', '.jpeg'}:
                    shutil.copy2(overlay, destination)
                else:
                    # Preserve a PNG source basename with genuine PNG bytes;
                    # do not mislabel the pipeline's JPEG as a PNG file.
                    import cv2
                    pixels = cv2.imread(str(overlay))
                    if pixels is None or not cv2.imwrite(str(destination), pixels):
                        raise OSError(f'Could not save review overlay: {destination}')
                row['final_overlay_path'] = str(destination)
                copied += 1
        writer.writerow(row)
        sites += result.get('final_fray_count', 0)
        duration = result.get('total_processing_ms')
        if isinstance(duration, (int, float)):
            times.append(duration)

print('\n40-image review summary:')
print(f'  Images attempted/reported: {len(results)} / {count}')
print(f'  Images successfully processed: {sum(counts[s] for s in successful_statuses)}')
for status in ('FRAY_DETECTED', 'OK_NO_FRAY', 'INSPECTION_FAILED'):
    print(f'  Images with {status}: {counts[status]}')
print(f'  Total final fray sites: {sites}')
if times:
    print(f'  Total processing time (ms): mean={statistics.mean(times):.2f}, '
          f'median={statistics.median(times):.2f}, n={len(times)}')
else:
    print('  Total processing time: unavailable (no timed inspections)')
print(f'  Flat overlays: {copied} in {review}')
print(f'  CSV: {output / "review.csv"}')
print(f'  Existing pipeline summary: {summary_path}')
for error in collection_errors:
    print(f'ERROR: {error}', file=sys.stderr)
sys.exit(1 if completed.returncode or collection_errors else 0)
PY

#!/usr/bin/env python3
"""Assemble saved validation artifacts only; no detector imports or inference."""
from pathlib import Path
from collections import Counter
from statistics import mean, median
import csv
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'runtime_test_output'
OUT = R / 'comparison_2x2'
RUNS = {'SAM2': R / 'colleague_sam2_validation',
        'SAM3': R / 'colleague_sam3_lowmem_validation',
        'YOLO': R / 'fray_validation_tensorrt'}
W, H, LABEL, HEADER = 1280, 721, 36, 40
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)


def read_json(path):
    return json.loads(path.read_text())


def load_image(path):
    with Image.open(path) as im:
        return ImageOps.contain(im.convert('RGB'), (W, H), Image.LANCZOS)


def panel(im, label, message=None):
    p = Image.new('RGB', (W, H + LABEL), (24, 27, 32))
    d = ImageDraw.Draw(p)
    d.text((12, 4), label, fill='white', font=FONT)
    if im is not None:
        p.paste(im, ((W-im.width)//2, LABEL+(H-im.height)//2))
    else:
        d.text((40, H//2), message or 'missing artifact', fill='white', font=FONT)
    return p


def composite(panels, name, target):
    canvas = Image.new('RGB', (2*W, HEADER+2*(H+LABEL)), (12, 14, 18))
    ImageDraw.Draw(canvas).text((12, 5), name, fill='white', font=FONT)
    for i, p in enumerate(panels):
        canvas.paste(p, ((i % 2)*W, HEADER+(i//2)*(H+LABEL)))
    canvas.save(target / name, quality=94, subsampling=0)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def main():
    sources = sorted(p for p in (ROOT/'frayed/validation_images').iterdir()
                     if p.is_file() and not p.name.startswith('.') and p.suffix.lower() in {'.jpg','.jpeg','.png'})
    assert len(sources) == 161, len(sources)
    if OUT.exists() and any(p.is_file() for p in OUT.rglob("*")):
        raise SystemExit("Comparison output already contains files; preserve it before rerunning.")
    OUT.mkdir(exist_ok=True)
    for kind in ['final_overlays', 'rope_masks']:
        (OUT/kind).mkdir(exist_ok=True)
    summaries = {k: read_json(v/'summary.json') for k,v in RUNS.items()}
    rows = {k: {Path(x['filename']).name: x for x in s['images']} for k,s in summaries.items()}
    placeholders = {kind: Counter() for kind in ['final_overlays','rope_masks']}
    for index, source in enumerate(sources, 1):
        original = load_image(source)
        panels = {kind: [panel(original, 'ORIGINAL')] for kind in placeholders}
        for method, run in RUNS.items():
            row = rows[method].get(source.name, {})
            failed = row.get('status','').startswith('INSPECTION_FAILED')
            message = 'run failed' if failed else 'missing artifact'
            overlay = run/'final_overlays'/source.name
            if overlay.is_file():
                result = load_image(overlay)
            else:
                result = None
                placeholders['final_overlays'][method] += 1
            panels['final_overlays'].append(panel(result, method, message))
            artifact = Path(row.get('artifacts') or run/'images'/source.name)
            # All these runs save cleaned binary unions, not dedicated mask overlays.
            mask_path = artifact/('cleaned_rope_union.png' if method == 'YOLO' else 'rope_mask.png')
            if mask_path.is_file():
                with Image.open(mask_path) as mask:
                    mask = mask.convert('L').point(lambda x: 255 if x else 0)
                    mask = mask.resize(original.size, Image.NEAREST)
                tint = Image.blend(original, Image.new('RGB', original.size, (0,255,120)), 0.48)
                result = Image.composite(tint, original, mask)
            else:
                result = None
                placeholders['rope_masks'][method] += 1
            panels['rope_masks'].append(panel(result, method, message))
        for kind in panels:
            composite(panels[kind], source.name, OUT/kind)
        if index % 20 == 0 or index == 161:
            print(f'Assembled {index}/161 pairs', flush=True)

    records, timings, statuses = [], [], []
    for method, s in summaries.items():
        allrows = s['images']
        ok = [x for x in allrows if x['status'] in {'OK_NO_FRAY','FRAY_DETECTED'}]
        totals = [x['total_processing_ms'] for x in ok]
        segs = [x['stage_timings_ms']['yolo_inference'] if method=='YOLO' else x['segmentation_ms'] for x in ok]
        label = 'YOLO TensorRT' if method=='YOLO' else method
        counts = Counter(x['status'] for x in allrows)
        rec = dict(method=label, metric_group='runtime', images_evaluated=len(allrows), images_completed=len(ok),
                   failures=len(allrows)-len(ok), fray_detected_images=counts['FRAY_DETECTED'],
                   ok_no_fray_images=counts['OK_NO_FRAY'], final_boxes=sum(x.get('final_fray_count',0) for x in allrows),
                   mean_total_ms=mean(totals), median_total_ms=median(totals),
                   mean_segmentation_ms=mean(segs), median_segmentation_ms=median(segs),
                   source=str(RUNS[method].relative_to(ROOT)/'summary.json'), notes='Timings: successful inspections only; saved per-image wall timers.')
        records.append(rec)
        timings.append([label,len(ok),len(allrows)-len(ok),*[f'{rec[k]:.1f}' for k in ['mean_total_ms','median_total_ms','mean_segmentation_ms','median_segmentation_ms']]])
        statuses.append([label,counts['FRAY_DETECTED'],counts['OK_NO_FRAY'],len(allrows)-len(ok)])

    # Existing manual reviews, with only four TensorRT count-change cases checked separately.
    with (R/'colleague_sam2_pre_fix_visual_review.csv').open() as f:
        sam2review=list(csv.DictReader(f))
    with (R/'colleague_sam3_visual_review.csv').open() as f:
        sam3review=list(csv.DictReader(f))
    assert len(sam2review)==67 and len(sam3review)==161
    quality = [
        dict(method='SAM2',metric_group='quality',images_reviewed=67,images_evaluated=161,actual_visible_sites=35,
             final_boxes=93,true_positives=0,false_positives=93,detected_visible_sites=0,missed_visible_sites=35,
             recall=0,precision=0,source='runtime_test_output/colleague_sam2_pre_fix_visual_review.md',
             notes='QUALITY ONLY: earlier pre-fix successful subset, not current 244-box run; current full-run TP/FP unavailable.'),
        dict(method='SAM3',metric_group='quality',images_reviewed=161,images_evaluated=161,actual_visible_sites=91,
             final_boxes=322,true_positives=21,false_positives=299,uncertain_boxes=2,detected_visible_sites=21,
             missed_visible_sites=68,uncertain_visible_sites=2,recall=21/91,precision=21/322,
             source='runtime_test_output/colleague_sam3_visual_review.md',
             notes='68 definite misses + 2 unresolved sites; 2 uncertain boxes. Confirmed recall/precision shown.'),
        dict(method='YOLO TensorRT',metric_group='quality',images_reviewed=4,images_evaluated=161,actual_visible_sites=91,
             final_boxes=87,true_positives=87,false_positives=0,detected_visible_sites=87,missed_visible_sites=4,
             recall=87/91,precision=1,source='runtime_test_output/fray_validation_tensorrt/comparison.json',
             notes='Provisional transfer of reviewed latest PT 85-site baseline: +1 111515, +1 111516, +2 111825, -2 111831. Four count-changed source/overlay pairs checked; count-neutral shifted boxes not re-reviewed.')]
    records.extend(quality)
    eq = read_json(R/'colleague_equivalence/results.json')
    original = next(x for x in eq if x['case']=='110905')
    before = sum(x['before_geometry_rectification_ms'] for x in original['ropes'])
    bench = read_json(R/'rope_cpu_optimization/benchmark_summary.json')
    after = bench['timing_table']['CPU total']['after_median_ms']
    cpu = [
        ['Colleague CV before optimization',f'{before:.1f} ms', 'Geometry + rectification ONLY, 110905, 4000×2252 / 3 ropes, one memory-safe original pass; colleague_equivalence/results.json'],
        ['My CV after optimization',f'{after:.1f} ms', 'Full CPU inspection, 111831, 4000×2252 / 3 ropes; median of 3 passes, debug saving included; rope_cpu_optimization/benchmark_summary.json']]
    for label, value, basis in cpu:
        records.append(dict(method=label,metric_group='cpu_timing',timing_value=value,notes=basis))
    qrows=[]
    for q in quality:
        reviewed = '67 pre-fix / 161 current' if q['method']=='SAM2' else ('161 baseline + 4 changed checked' if q['method']=='YOLO TensorRT' else '161 / 161')
        qrows.append([q['method'],reviewed,q['actual_visible_sites'],q['final_boxes'],q['true_positives'],q['false_positives'],
                      q['detected_visible_sites'], '68 + 2 unresolved' if q['method']=='SAM3' else q['missed_visible_sites'],
                      f"{q['recall']:.1%}",f"{q['precision']:.1%}",q['notes']])
    note = '\n'.join(f'- {kind}: '+', '.join(f'{m} {placeholders[kind][m]}' for m in RUNS)+' placeholder panels.' for kind in placeholders)
    md = '# Saved validation comparison\n\n### 1. Output counts\n\n- Source images: 161\n- Final-overlay composites: 161\n- Rope-mask composites: 161\n\nEach 2560×1554 JPEG uses ORIGINAL / SAM2 above SAM3 / YOLO, with filename and panel labels. Display resizing only; originals unchanged.\n\n### 2. End-to-end timing comparison\n\n'
    md += table(['Method','Images completed','Failures','Mean total ms/image','Median total ms/image','Mean segmentation ms/image','Median segmentation ms/image'],timings)
    md += '\n\nTimings use successful-image records only (SAM2 143; others 161). SAM3 model startup/restarts excluded from per-image timers; SAM2 first-use startup remains in its recorded segmentation timer. YOLO uses the synchronized wall-time inference stage including backend preprocessing/postprocessing, not just the neural kernel.\n\n'
    md += table(['Method','FRAY_DETECTED','OK_NO_FRAY','Failures / aborted'],statuses)
    md += '\n\n### 3. CV CPU-only comparison\n\n'+table(['Method','Timing value','Timing basis / source artifact'],cpu)
    md += '\n\nDifferent fixtures and scopes: not a like-for-like speedup. Original reference detached retained slices to prevent allocation OOM without changing pixels. For context, the same existing optimized-pipeline benchmark records 669522.0 → 3998.8 ms CPU total before/after optimization, but its before source was already the adapted pipeline, not the untouched colleague algorithm. No new benchmark.\n\n### 4. Detection-quality comparison\n\n'
    md += table(['Method','Reviewed / evaluated images','Actual visible sites','Final boxes','TP','FP','Detected visible sites','Missed sites','Recall','Precision','Notes / limitations'],qrows)
    md += '\n\nManual visual judgments, not formal annotation metrics. SAM2 current full run: 244 boxes; its 93-box quality figures above apply ONLY to the historical reviewed subset. YOLO TensorRT provisional estimate inherits PT site matches for unchanged-count images, not a new full TRT review; count equality alone does not prove spatial equivalence. SAM3 uncertain matches could raise recall to 25.3% and precision to 7.1%.\n\n### 5. Artifact source notes\n\n'
    for method, run in RUNS.items():
        md += f'- {method}: overlays `{run.relative_to(ROOT)}/final_overlays/`; masks from per-image `'+('cleaned_rope_union.png' if method=='YOLO' else 'rope_mask.png')+'`, tinted green over the original. Cleaned unions show masks actually supplied downstream; raw rejected instances are not represented.\n'
    md += note+'\n- Source reviews: `colleague_sam2_pre_fix_visual_review.{md,csv}`, `colleague_sam3_visual_review.{md,csv}`, `colleague_sam3_box_review.csv`, `fray_validation_clean_20260930_095634/codex_visual_review.md`; latest PT 85-site baseline in `fray_validation_tensorrt/metadata/pt_baseline_summary.json`, count changes in `fray_validation_tensorrt/comparison.json`. Paths relative to `runtime_test_output/`.\n'
    (OUT/'summary.md').write_text(md)
    fields=list(dict.fromkeys(k for row in records for k in row))
    with (OUT/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    for kind in placeholders:
        assert {p.name for p in (OUT/kind).iterdir()} == {p.name for p in sources}
    print('COMPLETE',json.dumps({k:dict(v) for k,v in placeholders.items()}),flush=True)
    print(table(['Method','Completed','Failures','Mean total','Median total','Mean seg','Median seg'],timings))


if __name__ == '__main__':
    main()

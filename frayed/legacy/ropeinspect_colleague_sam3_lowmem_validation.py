"""Resumable single-worker orchestration; all detection code imported unchanged.

The master uses only the standard library. No model is constructed on import.
Progress/events are fsync'd JSONL journals; summary/CSV are replaceable views.
"""
import argparse
import csv
import ctypes
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback
import threading

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'runtime_test_output/colleague_sam3_lowmem_validation'
SOURCE = ROOT / 'frayed/validation_images'
MIB = 1024**2
GUARD_EXIT = 75
OOM_EXIT = 86
COLUMNS = ['filename', 'status', 'attempt_count', 'worker_restart_count_at_image',
           'raw_segmentation_mask_count', 'accepted_rope_mask_count', 'usable_rope_count',
           'preliminary_fray_candidates', 'final_fray_count', 'image_load_ms', 'segmentation_ms',
           'mask_processing_ms', 'component_splitting_ms', 'geometry_ms', 'centerline_ms',
           'width_measurement_ms', 'rectification_ms', 'fray_scoring_ms', 'candidate_filtering_ms',
           'visualization_save_ms', 'total_processing_ms', 'final_overlay_path', 'error']


def append(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def read_journal(path):
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    records = []
    for index, line in enumerate(lines):
        if not line.endswith(b'\n'):
            break  # a writer/reboot may have left an uncommitted partial tail
        try:
            records.append(json.loads(line))
        except (ValueError, UnicodeError):
            if index != len(lines)-1 or line.endswith(b'\n'):
                raise RuntimeError(f'Corrupt journal (not a torn last record): {path}')
    return records


def repair_tail(path):
    # Validate first; truncate only an incomplete uncommitted tail after reboot.
    read_journal(path)
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b'\n'):
            try:
                json.loads(data.split(b'\n')[-1])
            except (ValueError, UnicodeError):
                with path.open('r+b') as stream:
                    stream.truncate(data.rfind(b'\n')+1)
                    stream.flush()
                    os.fsync(stream.fileno())
            else:
                with path.open('ab') as stream:
                    stream.write(b'\n')
                    stream.flush()
                    os.fsync(stream.fileno())


def atomic_text(path, text):
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def event(output, kind, **values):
    append(output/'process_events.jsonl', dict(event=kind, timestamp=time.time(), **values))


def memory(torch=None):
    info = {line.split(':')[0]: int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines()}
    rss = next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines()
               if line.startswith('VmRSS:'))
    return dict(mem_available_bytes=info['MemAvailable'], swap_free_bytes=info['SwapFree'],
                swap_used_bytes=info['SwapTotal']-info['SwapFree'], process_rss_bytes=rss,
                cuda_allocated_bytes=torch.cuda.memory_allocated(0) if torch else None,
                cuda_reserved_bytes=torch.cuda.memory_reserved(0) if torch else None)


def snapshot(output, stage, worker_id, torch, **extra):
    record = dict(stage=stage, timestamp=time.time(), pid=os.getpid(), worker_id=worker_id,
                  **memory(torch), **extra)
    append(output/'memory_stages.jsonl', record)
    return record


def oom_error(error):
    text = str(error).lower()
    return 'out of memory' in text or 'outofmemory' in text or 'memoryerror' in text


def stats(values):
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values)) if values else None


def publish(output, names):
    rows = read_journal(output/'progress.jsonl')
    events = read_journal(output/'process_events.jsonl')
    by_name = {r['filename']: r for r in rows}
    if len(by_name) != len(rows):
        raise RuntimeError('Duplicate finalized progress records')
    ordered = [by_name[n] for n in names if n in by_name]
    loads = [e for e in events if e['event']=='MODEL_READY']
    launches = [e for e in events if e['event']=='WORKER_LAUNCH']
    exits = {e['worker_id']: e for e in events if e['event']=='WORKER_EXIT'}
    sessions = [e for e in events if e['event']=='SESSION_START']
    ends = {e['session_id']: e for e in events if e['event']=='SESSION_END'}
    # Prior abruptly ended sessions count through their last durable event.
    wall = 0.0
    for i, start in enumerate(sessions):
        upper = sessions[i+1]['timestamp'] if i+1 < len(sessions) else float('inf')
        last = max((e['timestamp'] for e in events if start['timestamp'] <= e['timestamp'] < upper),
                   default=start['timestamp'])
        end = ends.get(start['session_id'], {}).get('timestamp', last)
        wall += max(0.0, end-start['timestamp'])
    aggregates = {key: stats([r[key] for r in ordered if isinstance(r.get(key), (int,float))])
                  for key in COLUMNS+['image_encoding_ms','prompt_prediction_ms'] if key.endswith('_ms')}
    speeds = {key: stats([speed[key] for r in ordered for speed in r.get('ultralytics_speed_ms', [])
                         if isinstance(speed.get(key), (int,float))])
              for key in ('preprocess','inference','postprocess')}
    load_ids = {e['worker_id'] for e in loads}
    failed_startup = sum(exits[e['worker_id']].get('worker_wall_ms',0) for e in launches
                         if e['worker_id'] not in load_ids and e['worker_id'] in exits)
    summary = dict(variant='SAM3_LOWMEM', benchmark='FRAY-ONLY', rust_evaluation='DISABLED_NOT_EVALUATED',
                   run_status='COMPLETE' if len(ordered)==len(names) else 'IN_PROGRESS',
                   image_count=len(names), images=ordered, worker_launches=len(launches),
                   memory_guard_restarts=sum(e['event']=='MEMORY_GUARD_RESTART' for e in events),
                   unexpected_exit_137_restarts=sum(e['event']=='WORKER_EXIT' and e.get('returncode') in (-9,137) for e in events),
                   oom_worker_exits=sum(e['event']=='WORKER_EXIT' and e.get('returncode') in (-9,137,OOM_EXIT) for e in events),
                   model_load_ms=stats([e['model_load_ms'] for e in loads]),
                   total_worker_startup_overhead_ms=sum(e['worker_startup_ms'] for e in loads)+failed_startup,
                   failed_startup_overhead_ms=failed_startup, timing_aggregates_ms=aggregates,
                   ultralytics_speed_ms=speeds, total_active_wall_seconds=wall,
                   wall_time_note='Includes startup/restarts/cleanup; excludes downtime between sessions. Abrupt session tails end at last durable event.',
                   timing_note='segmentation_ms includes set_image encoding and prompt prediction, excludes preloaded model. Ultralytics speed excludes separate set_image encoding. Geometry sub-stages overlap geometry_ms.')
    atomic_text(output/'summary.json', json.dumps(summary, indent=2))
    import io
    stream = io.StringIO()
    writer = csv.DictWriter(stream, COLUMNS, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(ordered)
    atomic_text(output/'review.csv', stream.getvalue())
    return summary


class TimedPredictor:
    """Transparent acquisition proxy; forwards the exact original API calls."""
    def __init__(self, predictor, torch):
        self.predictor, self.torch = predictor, torch
        self.reset_timings()

    def reset_timings(self):
        self.image_encoding_ms = 0.0
        self.prediction_ms = 0.0
        self.speeds = []

    def set_image(self, image):
        self.torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            return self.predictor.set_image(image)
        finally:
            self.torch.cuda.synchronize()
            self.image_encoding_ms += (time.perf_counter()-start)*1000

    def __call__(self, *args, **kwargs):
        self.torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            results = self.predictor(*args, **kwargs)
            self.speeds.extend(dict(r.speed) for r in results if getattr(r, 'speed', None))
            return results
        finally:
            self.torch.cuda.synchronize()
            self.prediction_ms += (time.perf_counter()-start)*1000


def worker(output, plan_path, parent_pid):
    # Kill this worker if its controller disappears; never leave an orphan GPU job.
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise RuntimeError('Cannot establish parent-death signal')
    if os.getppid() != parent_pid:
        return 1
    plan = json.loads(plan_path.read_text())
    wid = plan['worker_id']
    current = plan['images'][0]
    def begin(item):
        event(output, 'ATTEMPT_START', worker_id=wid, filename=item['filename'],
              attempt_count=item['attempt_count'])
    begin(current)  # associate construction OOM with the first pending image
    try:
        import torch
        import ultralytics
        print(f'Python: {sys.version}\nPyTorch: {torch.__version__}\nCUDA: {torch.version.cuda}\n'
              f'CUDA available: {torch.cuda.is_available()}\nCUDA count: {torch.cuda.device_count()}\n'
              f'Ultralytics: {ultralytics.__version__}', flush=True)
        if not torch.cuda.is_available():
            event(output, 'STARTUP_CHECK_FAILED', worker_id=wid, error='CUDA required; no CPU fallback')
            return 90
        print('GPU:', torch.cuda.get_device_name(0), '\nStartup memory:', memory(torch), flush=True)
        import ropeinspect_colleague_sam3_lowmem as detector
        snapshot(output, 'worker_startup', wid, torch)
        start = time.perf_counter()
        segmenter = detector.create_segmenter(ROOT/'sam3.pt', 'cuda:0')
        segmenter.setup_model()  # genuine same loader, before per-image timer
        torch.cuda.synchronize()
        event(output, 'MODEL_READY', worker_id=wid, model_load_ms=(time.perf_counter()-start)*1000,
              worker_startup_ms=(time.time()-plan['launch_timestamp'])*1000)
        proxy = TimedPredictor(segmenter, torch)
        observed_demand = plan['initial_memory_demand_bytes']
        for index, current in enumerate(plan['images']):
            if index:
                begin(current)
            name = current['filename']
            print(f'[{plan["completed"]+index+1}/{plan["image_count"]}] {name} worker={wid}', flush=True)
            proxy.reset_timings()
            before = snapshot(output, 'before_inspection', wid, torch, filename=name)
            offset = (output/'memory_stages.jsonl').stat().st_size
            record = detector.inspect_image(proxy, SOURCE/name, output, 'cuda:0', torch, False)
            record.update(attempt_count=current['attempt_count'], worker_restart_count_at_image=wid-1,
                          worker_id=wid, image_encoding_ms=proxy.image_encoding_ms,
                          prompt_prediction_ms=proxy.prediction_ms, ultralytics_speed_ms=proxy.speeds)
            if oom_error(record.get('error','')):
                event(output, 'OOM_EXCEPTION', worker_id=wid, filename=name, error=record['error'])
                return OOM_EXIT
            # Commit before cleanup: a cleanup SIGKILL must never lose this image.
            append(output/'progress.jsonl', record)
            restart_after_error = record.get('stop_run', False)
            cleanup_start = time.perf_counter()
            segmenter.reset_image()  # installed official API: im=None, features=None
            for attr in ('results', 'batch', 'dataset', 'plotted_img'):
                setattr(segmenter, attr, None)
            gc.collect()
            torch.cuda.empty_cache()
            after = snapshot(output, 'after_image_cleanup', wid, torch, filename=name)
            with (output/'memory_stages.jsonl').open() as stream:
                stream.seek(offset)
                samples = [json.loads(line) for line in stream if line.strip()]
            available_before = before['mem_available_bytes']+before['swap_free_bytes']
            minimum = min([available_before]+[s['mem_available_bytes']+s['swap_free_bytes'] for s in samples])
            observed_demand = max(observed_demand, available_before-minimum)
            required = max(512*MIB, observed_demand+256*MIB)
            event(output, 'MEMORY_GUARD_CHECK', worker_id=wid, filename=name,
                  cleanup_ms=(time.perf_counter()-cleanup_start)*1000,
                  observed_demand_bytes=observed_demand, required_combined_headroom_bytes=required,
                  **{k:after[k] for k in ('mem_available_bytes','swap_free_bytes','swap_used_bytes',
                                         'process_rss_bytes','cuda_allocated_bytes','cuda_reserved_bytes')})
            print(f'status={record["status"]} frays={record["final_fray_count"]} '
                  f'seg={record.get("segmentation_ms",0):.1f}ms total={record["total_processing_ms"]:.1f}ms '
                  f'MemAvailable={after["mem_available_bytes"]/MIB:.1f}MiB '
                  f'SwapFree={after["swap_free_bytes"]/MIB:.1f}MiB worker={wid}', flush=True)
            del record, samples
            if restart_after_error and index+1 < len(plan['images']):
                event(output, 'ERROR_RESTART', worker_id=wid, filename=name)
                return GUARD_EXIT
            if index+1 < len(plan['images']) and (after['mem_available_bytes'] < 256*MIB or
                    after['mem_available_bytes']+after['swap_free_bytes'] < required):
                event(output, 'MEMORY_GUARD_RESTART', worker_id=wid, filename=name)
                return GUARD_EXIT
        return 0
    except Exception as exc:
        traceback.print_exc()
        event(output, 'WORKER_EXCEPTION', worker_id=wid, filename=current['filename'],
              error=f'{type(exc).__name__}: {exc}')
        return OOM_EXIT if oom_error(exc) else 1


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, text):
        self.logfile.write(text)
        self.terminal.write(text)

    def flush(self):
        self.logfile.flush()
        self.terminal.flush()


def reconcile_exits(output):
    """Idempotent recovery if master died between recording exit and failure."""
    events = read_journal(output/'process_events.jsonl')
    done = {r['filename'] for r in read_journal(output/'progress.jsonl')}
    handled = {e['worker_id'] for e in events if e['event']=='ATTEMPT_FAILURE'}
    for exit_event in events:
        if exit_event['event'] != 'WORKER_EXIT' or exit_event['worker_id'] in handled:
            continue
        wid, code = exit_event['worker_id'], exit_event['returncode']
        if code == 90:
            continue
        starts = [e for e in events if e['event']=='ATTEMPT_START' and e.get('worker_id')==wid]
        launches = [e for e in events if e['event']=='WORKER_LAUNCH' and e.get('worker_id')==wid]
        if not launches:
            continue
        current = starts[-1]['filename'] if starts else launches[-1]['first_filename']
        if current in done:
            continue
        is_oom = code in (-9,137,OOM_EXIT)
        exceptions = [e for e in events if e['event'] in ('WORKER_EXCEPTION','OOM_EXCEPTION') and e.get('worker_id')==wid]
        error = (exceptions[-1]['error'] if exceptions else
                 f'Worker exit {code}; SIGKILL suspected OOM, not kernel-confirmed' if is_oom else
                 f'Worker exited {code} without finalizing image')
        event(output,'ATTEMPT_FAILURE',worker_id=wid,filename=current,oom=is_oom,error=error)


def master(output, fresh):
    # Lock outside output so --fresh cannot bypass another active controller.
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent/(output.name+'.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another low-memory validation controller/worker is active')
        if output.is_symlink():
            raise RuntimeError('Refusing symlink output')
        if fresh and output.exists():
            backup = output.with_name(output.name+time.strftime('_%Y%m%d_%H%M%S'))
            suffix = 0
            while backup.exists():
                suffix += 1
                backup = output.with_name(output.name+time.strftime('_%Y%m%d_%H%M%S')+f'_{suffix}')
            output.rename(backup)
            print('Previous output preserved:', backup, flush=True)
        output.mkdir(exist_ok=True)
        logfile = (output/'console.log').open('a', buffering=1)
        sys.stdout = Tee(sys.stdout, logfile)
        sys.stderr = Tee(sys.stderr, logfile)
        for name in ('final_overlays','metadata'):
            (output/name).mkdir(exist_ok=True)
        names = sorted(p.name for p in SOURCE.iterdir() if p.is_file() and not p.is_symlink()
                       and not p.name.startswith('.') and p.suffix.lower() in {'.jpg','.jpeg','.png'})
        if len(names) != 161 or len({n.casefold() for n in names}) != 161:
            raise RuntimeError(f'Expected exactly 161 unique source images, found {len(names)}')
        for path in (ROOT/'sam3.pt', ROOT/'.deps/ultralytics_clip/clip'):
            if not path.exists():
                raise FileNotFoundError(path)
        manifest = dict(images=[dict(filename=n, size=(SOURCE/n).stat().st_size,
                                    mtime_ns=(SOURCE/n).stat().st_mtime_ns) for n in names],
                        detector_sha256=hashlib.sha256((ROOT/'ropeinspect_colleague_sam3_lowmem.py').read_bytes()).hexdigest(),
                        controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        model_size=(ROOT/'sam3.pt').stat().st_size, model_mtime_ns=(ROOT/'sam3.pt').stat().st_mtime_ns)
        manifest_path = output/'metadata/manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Source/model/code changed since this run; use --fresh')
        atomic_text(manifest_path, json.dumps(manifest, indent=2))
        for file in ('progress.jsonl','process_events.jsonl','memory_stages.jsonl'):
            repair_tail(output/file)
            (output/file).touch(exist_ok=True)
        atomic_text(output/'selected_images.txt', ''.join(n+'\n' for n in names))
        print(f'Model: {ROOT/"sam3.pt"}\nInput: {SOURCE}\nImage count: {len(names)}\n'
              'FRAY-ONLY; rust disabled; exactly one persistent CUDA worker', flush=True)
        print('\n'.join(names), flush=True)
        print('Starting memory:', memory(), flush=True)
        os.environ['SAM3_LOWMEM_OUTPUT'] = str(output)
        os.environ['YOLO_CONFIG_DIR'] = str(output/'metadata/ultralytics')
        os.environ['MPLCONFIGDIR'] = str(output/'metadata/matplotlib')
        session_id = f'{time.time_ns()}-{os.getpid()}'
        event(output, 'SESSION_START', session_id=session_id)
        smoke = read_journal(ROOT/'runtime_test_output/colleague_sam3_lowmem_smoke/memory_stages.jsonl')
        ready = next((s for s in smoke if s['stage']=='model_ready'), None)
        inference = [s for s in smoke if s['stage'] in ('after_first_set_image','after_first_prediction')]
        demand = max(0, ready['mem_available_bytes']+ready['swap_free_bytes']-
                     min(s['mem_available_bytes']+s['swap_free_bytes'] for s in inference)) if ready and inference else 512*MIB
        summary = publish(output, names)
        session_completed_start = len(summary['images'])
        session_clock = time.perf_counter()
        while len(summary['images']) < len(names):
            reconcile_exits(output)
            events = read_journal(output/'process_events.jsonl')
            done = {r['filename'] for r in summary['images']}
            attempts = {n: sum(e['event']=='ATTEMPT_START' and e.get('filename')==n for e in events) for n in names}
            # Durable failure events allow resume even if master died before finalization.
            pending = [n for n in names if n not in done]
            for n in list(pending):
                failures = [e for e in events if e['event']=='ATTEMPT_FAILURE' and e.get('filename')==n]
                oom_count = sum(e['oom'] for e in failures)
                if oom_count >= 2 or any(not e['oom'] for e in failures):
                    last = failures[-1]
                    append(output/'progress.jsonl', dict(filename=n, status='INSPECTION_FAILED_OOM' if oom_count>=2 else 'INSPECTION_FAILED',
                           attempt_count=attempts[n], worker_restart_count_at_image=last['worker_id']-1,
                           final_fray_count=0, final_overlay_path='', error=last['error'], rust_evaluation='DISABLED_NOT_EVALUATED'))
                    pending.remove(n)
            summary = publish(output,names)
            if not pending:
                break
            wid = max([e['worker_id'] for e in events if e['event']=='WORKER_LAUNCH']+[0])+1
            demand = max([demand]+[e['observed_demand_bytes'] for e in events if e['event']=='MEMORY_GUARD_CHECK'])
            launch = time.time()
            plan = dict(worker_id=wid, launch_timestamp=launch, completed=len(summary['images']), image_count=len(names),
                        initial_memory_demand_bytes=demand,
                        images=[dict(filename=n,attempt_count=attempts[n]+1) for n in pending])
            plan_path = output/'metadata'/f'worker_{wid:04d}.json'
            atomic_text(plan_path,json.dumps(plan))
            event(output,'WORKER_LAUNCH',worker_id=wid,first_filename=pending[0])
            child = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(plan_path),
                                      '--parent-pid',str(os.getpid()),'--output',str(output)], pass_fds=(lock.fileno(),), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            def relay():
                for line in child.stdout:
                    if line.startswith('LOWMEM_STAGE '):
                        try:
                            stage = json.loads(line[len('LOWMEM_STAGE '):])
                            line = f'LOWMEM stage={stage["stage"]} MemAvailable={stage["mem_available_bytes"]/MIB:.1f}MiB SwapFree={stage["swap_free_bytes"]/MIB:.1f}MiB\n'
                        except (ValueError,KeyError):
                            pass
                    sys.stdout.write(line)
                    sys.stdout.flush()
            reader = threading.Thread(target=relay, daemon=True)
            reader.start()
            try:
                old_count = len(summary['images'])
                journal_sizes = None
                while child.poll() is None:
                    time.sleep(1)
                    sizes = tuple((output/f).stat().st_size for f in ('progress.jsonl','process_events.jsonl'))
                    if sizes == journal_sizes:
                        continue
                    journal_sizes = sizes
                    summary = publish(output,names)
                    count = len(summary['images'])
                    if count != old_count:
                        rate = (time.perf_counter()-session_clock)/max(1,count-session_completed_start)
                        print(f'Progress {count}/{len(names)}; approximate ETA {rate*(len(names)-count)/60:.1f} min (includes restarts)', flush=True)
                        old_count = count
                code = child.wait()
            except BaseException:
                event(output, 'SESSION_INTERRUPTED', session_id=session_id, worker_id=wid)
                raise
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
            reader.join()
            child.stdout.close()
            for journal in ('progress.jsonl','process_events.jsonl','memory_stages.jsonl'):
                repair_tail(output/journal)
            event(output,'WORKER_EXIT',worker_id=wid,returncode=code,worker_wall_ms=(time.time()-launch)*1000)
            if code == 90:
                event(output,'SESSION_END',session_id=session_id)
                publish(output,names)
                raise RuntimeError('CUDA startup check failed; no inference attempted. Resume on the GPU host.')
            reconcile_exits(output)
            summary = publish(output,names)
            print(f'Worker {wid} exited {code}; committed results preserved.',flush=True)
        event(output,'SESSION_END',session_id=session_id)
        summary = publish(output,names)
        counts = {s:sum(r['status']==s for r in summary['images']) for s in
                  ('FRAY_DETECTED','OK_NO_FRAY','INSPECTION_FAILED','INSPECTION_FAILED_OOM')}
        print('Completed:',json.dumps(counts),'fray sites:',sum(r.get('final_fray_count',0) for r in summary['images']),flush=True)
        print(json.dumps({k:v for k,v in summary.items() if k not in ('images',)},indent=2),flush=True)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh',action='store_true',help='Timestamp-backup old run; default resumes')
    parser.add_argument('--resume',action='store_true',help='Default behavior')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--parent-pid',type=int,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.fresh and args.resume:
        parser.error('--fresh and --resume are mutually exclusive')
    os.environ['YOLO_AUTOINSTALL']='false'
    if args.worker:
        return worker(args.output.resolve(),args.worker,args.parent_pid)
    return master(args.output.resolve(),args.fresh)


if __name__=='__main__':
    raise SystemExit(main())

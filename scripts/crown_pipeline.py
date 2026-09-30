#!/usr/bin/env python3
"""Prepare, run, stop, and inspect resumable CROWN COCO experiments."""

from __future__ import annotations

import argparse
import csv
import errno
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PUBLIC = Path(os.environ.get('CROWN_PUBLIC_ROOT', '/homes/rliuar/work/mnt/nas6/Cytology/Public'))
ARCHIVE = Path(os.environ.get(
    'CROWN_ARCHIVE_ROOT',
    '/homes/rliuar/work/mnt/nas6/Cytology/Cytology/smartcyto_baseline/crown'))
LOCAL = Path(os.environ.get('CROWN_LOCAL_ROOT', '/homes/rliuar/work/2_Temp/crown_runs'))
REMOTE_CKPT = Path(os.environ.get(
    'CROWN_PRETRAINED_CKPT',
    '/homes/rliuar/work/mnt/nas6/Cytology/Private/temp/X_Ckpts/crown_ckpts/CROWN.pth'))
MOUNTPOINT = Path(os.environ.get('CROWN_MOUNTPOINT', '/homes/rliuar/work/mnt/nas6'))
LOCAL_CKPT = LOCAL / 'pretrained/CROWN.pth'
LOCAL_CKPT_VERIFIED = LOCAL_CKPT.with_suffix('.verified')
STATE = Path(os.environ.get('CROWN_STATE_ROOT', REPO / 'work_dirs/crown_pipeline'))
GPUS = (4, 5, 6, 7)
FIELDS = ('task', 'dataset', 'train_images', 'status', 'phase', 'gpu', 'pid',
          'best_ckpt', 'metrics_json', 'message', 'updated_at')


@dataclass(frozen=True)
class Job:
    task: str
    dataset: str
    folder: str
    subdir: str
    external: bool = False
    layout: str = 'standard'

    @property
    def key(self):
        return f'{self.task}_{self.dataset.lower().replace("-", "_")}'

    @property
    def root(self):
        return PUBLIC / self.folder / self.subdir

    @property
    def work(self):
        return LOCAL / self.task / self.dataset

    @property
    def config(self):
        return STATE / 'configs' / f'{self.key}.py'

    @property
    def source_config(self):
        return STATE / 'configs' / f'{self.key}.source.py'

    @property
    def data(self):
        return LOCAL / 'data' / self.key

    @property
    def data_marker(self):
        return self.data / 'stage.done'

    @property
    def archive(self):
        return ARCHIVE / self.task / self.dataset

    @property
    def summary(self):
        return self.work / 'eval' / f'metrics_{"detection" if self.task == "det" else "segmentation"}_summary.json'


JOBS = (
    Job('det', 'APCData', 'APCData', 'det'),
    Job('det', 'Ascites2020', 'Ascites2020', 'det'),
    Job('det', 'BCCD', 'BCCD', 'det'),
    Job('det', 'CCS-Cell', 'CCS-Cell-Det', 'Coco/coco_filtered', layout='ccs'),
    Job('det', 'CDetector', 'CDetector', 'det'),
    Job('det', 'CellDet', 'CellDet', 'det'),
    Job('det', 'NMCD', 'NMCD', 'det'),
    Job('det', 'TXL-PBC', 'TXL-PBC', 'det'),
    Job('det', 'CBC', 'CBC', 'det', external=True),
    Job('seg', 'APACS23', 'APACS23', 'seg'),
    Job('seg', 'BTTFA', 'BTTFA', 'seg'),
    Job('seg', 'CISD', 'CISD', 'seg'),
    Job('seg', 'CNSeg', 'CNSeg', 'seg'),
    Job('seg', 'CPS', 'CPS', 'seg'),
    Job('seg', 'Herlev', 'Herlev', 'seg'),
    Job('seg', 'Jiangxi', 'Jiangxi', 'seg'),
    Job('seg', 'Oral2021', 'Oral2021', 'seg', layout='new_annotations'),
    Job('seg', 'Raabin-WBC', 'Raabin_WBC', 'seg'),
    Job('seg', 'SegPC', 'SegPC', 'seg'),
    Job('seg', 'UFSC_OCPap', 'UFSC_OCPap', 'seg'),
)
BY_KEY = {job.key: job for job in JOBS}


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def timed_run(command, timeout_seconds, **kwargs):
    """Bound FUSE calls even when a dead sshfs makes the child unkillable."""
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, **kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        # A process blocked inside FUSE can stay in D state after SIGKILL.
        # Waiting for it here would freeze the whole experiment controller.
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def detect_mount_layout():
    """Accept mounts rooted at either /jhcnas6 or /jhcnas6/Cytology."""
    global PUBLIC, ARCHIVE, REMOTE_CKPT
    if any(name in os.environ for name in (
            'CROWN_PUBLIC_ROOT', 'CROWN_ARCHIVE_ROOT', 'CROWN_PRETRAINED_CKPT')):
        return
    for attempt in range(3):
        for base in (MOUNTPOINT / 'Cytology', MOUNTPOINT):
            try:
                result = timed_run(
                    ['test', '-d', str(base / 'Public')], 5,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode == 0:
                PUBLIC = base / 'Public'
                REMOTE_CKPT = base / 'Private/temp/X_Ckpts/crown_ckpts/CROWN.pth'
                ARCHIVE = base / 'Cytology/smartcyto_baseline/crown'
                return
        if attempt < 2:
            time.sleep(2)


def mount_healthy():
    try:
        result = timed_run(
            ['findmnt', '-M', str(MOUNTPOINT), '-n', '-o', 'FSTYPE'], 5,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode or result.stdout.strip() != 'fuse.sshfs':
            return False
        result = timed_run(
            ['stat', '-L', str(PUBLIC), str(ARCHIVE.parent)], 8,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def checkpoint_source_healthy():
    try:
        result = timed_run(
            ['stat', '-L', str(REMOTE_CKPT)], 8,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def split_paths(job, split):
    if job.layout == 'ccs':
        return (job.root / 'annotations' / f'instances_{split}2017.json',
                job.root / f'{split}2017')
    if job.layout == 'new_annotations':
        return (job.root / f'new_{split}.json', job.root / split)
    return (job.root / f'{split}.json', job.root / split)


def load_rows():
    path = STATE / 'status.csv'
    if not path.exists():
        return {}
    with path.open(newline='', encoding='utf-8') as handle:
        return {row['task']: row for row in csv.DictReader(handle)}


def save_rows(rows):
    STATE.mkdir(parents=True, exist_ok=True)
    temporary = STATE / 'status.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for job in JOBS:
            if job.key in rows:
                writer.writerow({field: rows[job.key].get(field, '') for field in FIELDS})
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(STATE / 'status.csv')


def update(rows, job, **changes):
    row = rows[job.key]
    row.update({key: str(value) for key, value in changes.items()})
    row['updated_at'] = now()
    save_rows(rows)


def discover(job):
    splits = {}
    for split in ('train', 'val', 'test'):
        ann, prefix = split_paths(job, split)
        with ann.open(encoding='utf-8') as handle:
            data = json.load(handle)
        images = data.get('images', [])
        if not images:
            raise ValueError(f'{ann}: no images')
        sample = images[0].get('file_name')
        if not sample:
            raise ValueError(f'{ann}: image has no file_name')
        if not (prefix / sample).is_file():
            candidates = (job.root, job.root.parent, PUBLIC / job.folder)
            prefix = next((path for path in candidates if (path / sample).is_file()), None)
            if prefix is None:
                raise FileNotFoundError(f'{ann}: cannot resolve sample image {sample}')
        cats = data.get('categories', [])
        if not cats:
            raise ValueError(f'{ann}: no categories')
        splits[split] = (ann, prefix, len(images), cats)
        if job.task == 'seg' and split == 'train':
            if not any(a.get('segmentation') for a in data.get('annotations', [])[:100]):
                raise ValueError(f'{ann}: no instance masks in first 100 annotations')
    classes = tuple(cat['name'] for cat in splits['train'][3])
    if job.external:
        # TXL-PBC head channels are WBC, RBC, Platelet. CBC category names
        # differ and its annotation IDs put RBC first.
        classes = ('WBC', 'RBC', 'Platelets')
    for split in ('val', 'test'):
        names = {cat['name'] for cat in splits[split][3]}
        if set(classes) != names:
            raise ValueError(f'{job.dataset}: {split} categories differ from train: {names}')
    return splits, classes


def discover_bounded(job):
    env = os.environ.copy()
    env['CROWN_PUBLIC_ROOT'] = str(PUBLIC)
    try:
        result = timed_run(
            [sys.executable, str(Path(__file__).resolve()), '--internal-discover', job.key],
            120, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    except subprocess.TimeoutExpired as exc:
        raise MountUnavailable(f'{job.dataset}: dataset discovery timed out on sshfs') from exc
    if result.returncode:
        raise RuntimeError(f'{job.dataset}: discovery worker failed: {result.stderr[-240:]}')
    payload = json.loads(result.stdout)
    if 'error' in payload:
        if payload.get('errno') in (errno.EIO, errno.ENOTCONN, errno.ESTALE, errno.EPERM):
            raise MountUnavailable(f'{job.dataset}: sshfs read failed: {payload["error"]}')
        raise ValueError(payload['error'])
    splits = {name: (Path(row[0]), Path(row[1]), row[2], row[3])
              for name, row in payload['splits'].items()}
    return splits, tuple(payload['classes'])


def run_data_copy(command):
    process = subprocess.Popen(command, stderr=subprocess.STDOUT)
    failed_checks = 0
    last_check = 0.0
    while process.poll() is None:
        if time.monotonic() - last_check >= 15:
            failed_checks = 0 if mount_healthy() else failed_checks + 1
            last_check = time.monotonic()
            if failed_checks >= 3:
                process.terminate()
                raise MountUnavailable('sshfs stopped responding during data staging')
        time.sleep(2)
    if process.returncode:
        raise MountUnavailable(f'data rsync exited {process.returncode}; see data.log')


def retry_data_copy(command, attempts=20):
    for attempt in range(attempts):
        try:
            run_data_copy(command)
            return
        except MountUnavailable as exc:
            if 'stopped responding' in str(exc):
                raise
            if attempt + 1 == attempts:
                raise
            print(f'data copy interrupted; retry {attempt + 2}/{attempts}', flush=True)
            time.sleep(min(30, 5 * (attempt + 1)))


def stage_data(job):
    """Copy only COCO annotations and referenced images before GPU work."""
    from mmcv import Config

    if job.data_marker.exists():
        return
    if not job.source_config.exists():
        shutil.copy2(job.config, job.source_config)
    source_text = job.source_config.read_text(encoding='utf-8')
    if str(PUBLIC) not in source_text:
        for old_root in (MOUNTPOINT / 'Cytology/Public', MOUNTPOINT / 'Public'):
            if str(old_root) in source_text:
                source_text = source_text.replace(str(old_root), str(PUBLIC))
                job.source_config.write_text(source_text, encoding='utf-8')
                break
    source = Config.fromfile(str(job.source_config))
    local_public = job.data / 'Public'
    for split in ('train', 'val', 'test'):
        remote_ann = Path(source.data[split].ann_file)
        remote_prefix = Path(source.data[split].img_prefix)
        local_ann = local_public / remote_ann.relative_to(PUBLIC)
        local_prefix = local_public / remote_prefix.relative_to(PUBLIC)
        local_ann.parent.mkdir(parents=True, exist_ok=True)
        local_prefix.mkdir(parents=True, exist_ok=True)
        retry_data_copy(['rsync', '-t', '--partial', '--append-verify',
                         str(remote_ann), str(local_ann)])
        with local_ann.open(encoding='utf-8') as handle:
            annotation = json.load(handle)
        names = set()
        for image in annotation['images']:
            name = image['file_name']
            path = Path(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'{remote_ann}: unsafe image path {name!r}')
            names.add(name)
        manifest = job.data / f'{split}.files'
        manifest.write_bytes(b''.join(name.encode() + b'\0' for name in sorted(names)))
        retry_data_copy(['rsync', '-rt', '--partial', '--append-verify',
                         '--from0', f'--files-from={manifest}',
                         str(remote_prefix) + '/', str(local_prefix) + '/'])
        missing = next((name for name in names if not (local_prefix / name).is_file()), None)
        if missing is not None:
            raise MountUnavailable(f'{job.dataset}: staged image missing: {missing}')
        print(f'{job.key}: staged {split} ({len(names)} images)', flush=True)
    local_text = source_text.replace(str(PUBLIC), str(local_public))
    temporary = job.config.with_suffix('.py.tmp')
    temporary.write_text(local_text, encoding='utf-8')
    temporary.replace(job.config)
    job.data_marker.write_text(now(), encoding='utf-8')


def render_config(job, splits, classes):
    base = REPO / 'detection/configs' / (
        'faster_rcnn/faster_rcnn_crown_adapter_large_fpn_1x_coco.py'
        if job.task == 'det' else
        'mask_rcnn/mask_rcnn_crown_adapter_large_fpn_1x_coco.py')
    ds_type = 'CrownInstanceCocoDataset' if job.task == 'seg' else 'CocoDataset'
    entries = []
    for split in ('train', 'val', 'test'):
        ann, prefix, _, _ = splits[split]
        kind = ds_type if split == 'val' else 'CocoDataset'
        entries.append(
            f"    {split}=dict(type={kind!r}, ann_file={str(ann)!r}, "
            f"img_prefix={str(prefix) + '/'!r}, classes=classes),")
    monitor = 'bbox_mAP' if job.task == 'det' else 'AJI'
    metric = 'bbox' if job.task == 'det' else 'segm'
    head = f"bbox_head=dict(num_classes={len(classes)})"
    if job.task == 'seg':
        head += f", mask_head=dict(num_classes={len(classes)})"
    return '\n'.join((
        f'_base_ = {str(base)!r}',
        f'classes = {classes!r}',
        f'model = dict(backbone=dict(pretrained={str(LOCAL_CKPT)!r}), '
        f'roi_head=dict({head}))',
        'data = dict(workers_per_gpu=2,', *entries, ')',
        f"evaluation = dict(interval=1, metric={metric!r}, save_best={monitor!r}, "
        f"rule='greater', early_stop_metric={monitor!r}, early_stop_patience=5)",
        'checkpoint_config = dict(interval=1, max_keep_ckpts=2, save_last=True)',
        '',
    ))


def prepare(rows):
    for job in JOBS:
        if job.key not in rows:
            rows[job.key] = dict(task=job.key, dataset=job.dataset,
                                 train_images='', status='queued', phase='', gpu='',
                                 pid='', best_ckpt='', metrics_json='', message='',
                                 updated_at=now())
            save_rows(rows)
    for attempt in range(3):
        if mount_healthy():
            break
        if attempt < 2:
            time.sleep(2)
    else:
        raise MountUnavailable('sshfs mount unavailable; remount it and rerun start')
    (STATE / 'configs').mkdir(parents=True, exist_ok=True)
    for job in JOBS:
        if rows[job.key]['status'] == 'complete':
            continue
        if job.config.exists() and rows[job.key]['train_images']:
            config_text = job.config.read_text(encoding='utf-8')
            if str(job.data / 'Public') in config_text:
                if job.data_marker.exists():
                    continue
                if not job.source_config.exists():
                    raise ValueError(f'{job.dataset}: local config exists without staged data')
                config_text = job.source_config.read_text(encoding='utf-8')
                job.config.write_text(config_text, encoding='utf-8')
            if str(PUBLIC) not in config_text:
                for old_root in (MOUNTPOINT / 'Cytology/Public', MOUNTPOINT / 'Public'):
                    if str(old_root) in config_text:
                        config_text = config_text.replace(str(old_root), str(PUBLIC))
                        job.config.write_text(config_text, encoding='utf-8')
                        break
            if str(PUBLIC) not in config_text:
                raise ValueError(f'{job.config}: cannot update dataset root to {PUBLIC}')
            if str(LOCAL_CKPT) not in config_text:
                job.config.write_text(
                    config_text + f"\nmodel['backbone'] = dict(pretrained={str(LOCAL_CKPT)!r})\n",
                    encoding='utf-8')
            if rows[job.key]['status'] in ('paused_mount', 'invalid_data'):
                update(rows, job, status='queued', message='')
            continue
        try:
            splits, classes = discover_bounded(job)
            text = render_config(job, splits, classes)
            job.config.write_text(text, encoding='utf-8')
            update(rows, job, train_images=splits['train'][2],
                   status='queued', message='external: TXL-PBC checkpoint' if job.external else '')
        except Exception as exc:
            if isinstance(exc, MountUnavailable) or (isinstance(exc, OSError) and exc.errno in (
                    errno.EIO, errno.ENOTCONN, errno.ESTALE, errno.EPERM)) or not mount_healthy():
                raise MountUnavailable('sshfs unavailable while preparing datasets') from exc
            update(rows, job, status='invalid_data', message=str(exc)[:240])
    return rows


def print_status(rows):
    print(f'Progress CSV: {STATE / "status.csv"}')
    print(f'Metrics CSV:  {STATE / "results.csv"}')
    print(f'{"TASK":27} {"N TRAIN":>8} {"STATUS":15} {"PHASE":10} {"GPU":>3}  MESSAGE')
    for job in sorted(JOBS, key=lambda j: (int(rows.get(j.key, {}).get('train_images') or 10**12), j.dataset)):
        row = rows.get(job.key, {})
        print(f'{job.key:27} {row.get("train_images", ""):>8} '
              f'{row.get("status", "unprepared"):15} {row.get("phase", ""):10} '
              f'{row.get("gpu", ""):>3}  {row.get("message", "")[:60]}')


def write_results(rows):
    records = []
    for job in JOBS:
        path = job.summary
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            metrics = payload['results'][0]['metrics']
            for metric, value in metrics.items():
                point = value['point_estimate']
                ci = value['ci95']
                records.append(dict(task=job.key, dataset=job.dataset, metric=metric,
                                    mean=point, ci95_lower=ci['lower'],
                                    ci95_upper=ci['upper']))
        except (OSError, KeyError, IndexError, TypeError, ValueError):
            continue
    temporary = STATE / 'results.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=('task', 'dataset', 'metric',
                                                    'mean', 'ci95_lower', 'ci95_upper'))
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(STATE / 'results.csv')


def _process_matches(pid, fragments):
    try:
        cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        return any(fragment in cmdline for fragment in fragments)
    except (OSError, ValueError):
        return False


def log_reports_mount_error(path, start=0):
    try:
        with path.open('rb') as handle:
            handle.seek(max(start, path.stat().st_size - 131072))
            output = handle.read()
    except OSError:
        return False
    return (b'Input/output error' in output or
            b'Transport endpoint is not connected' in output or
            b'sshfs stopped responding' in output or
            b'data rsync exited' in output)


def stop():
    rows = load_rows()
    for row in rows.values():
        pid = int(row.get('pid') or 0)
        if pid and _process_matches(pid, ('detection/train.py', 'eval_automation.py',
                                          '--internal-stage', 'rsync')):
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    pid_path = STATE / 'controller.pid'
    if pid_path.exists():
        pid = int(pid_path.read_text().strip())
        if _process_matches(pid, ('crown_pipeline.py',)):
            os.kill(pid, signal.SIGTERM)
    print('Stop signal sent to CROWN experiment processes.')


def best_checkpoint(job):
    if job.external:
        # Retain the source checkpoint locally until CBC's external test ends.
        parent = BY_KEY['det_txl_pbc']
        local_best = sorted(parent.work.glob('best_*.pth'))
        if local_best:
            return local_best[-1]
        recorded = load_rows().get('det_txl_pbc', {}).get('best_ckpt')
        if recorded and Path(recorded).is_file():
            return Path(recorded)
    best = sorted(job.work.glob('best_*.pth'))
    if best:
        return best[-1]
    best = sorted(job.archive.glob('best_*.pth'))
    if best:
        return best[-1]
    if job.external:
        parent = BY_KEY['det_txl_pbc']
        best = sorted(parent.archive.glob('best_*.pth'))
        if best:
            return best[-1]
    raise FileNotFoundError(f'{job.dataset}: no validation-selected best checkpoint')


def command_for(job, phase, gpu):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '' if phase == 'data' else str(gpu)
    env['MPLCONFIGDIR'] = str(STATE / 'matplotlib')
    env['PYTHONPATH'] = ':'.join((str(REPO), str(REPO / 'third_party/openmmlab/mmcv'),
                                  str(REPO / 'third_party/openmmlab/mmdet'),
                                  str(REPO / 'third_party/openmmlab/mmseg'),
                                  env.get('PYTHONPATH', '')))
    if phase == 'data':
        env.update(CROWN_PUBLIC_ROOT=str(PUBLIC), CROWN_ARCHIVE_ROOT=str(ARCHIVE),
                   CROWN_PRETRAINED_CKPT=str(REMOTE_CKPT), CROWN_LOCAL_ROOT=str(LOCAL),
                   CROWN_STATE_ROOT=str(STATE), CROWN_MOUNTPOINT=str(MOUNTPOINT))
        cmd = [sys.executable, str(Path(__file__).resolve()),
               '--internal-stage', job.key]
        cwd = REPO
    elif phase == 'train':
        cmd = [sys.executable, str(REPO / 'detection/train.py'), str(job.config),
               '--work-dir', str(job.work), '--auto-resume', '--seed', '42']
        cwd = REPO / 'detection'
    elif phase == 'eval':
        if job.external:
            source = best_checkpoint(job)
            job.work.mkdir(parents=True, exist_ok=True)
            link = job.work / source.name
            if link.is_symlink():
                link.unlink()
            if not link.exists():
                link.symlink_to(source)
        else:
            best_checkpoint(job)
        cmd = [sys.executable, str(REPO / 'detection/eval_automation.py'),
               '--task', 'detection' if job.task == 'det' else 'segmentation',
               '--experiment-dir', str(job.work), '--config', str(job.config),
               '--checkpoint-select', 'final', '--bootstrap-resamples', '1000',
               '--bootstrap-seed', '42', '--bootstrap-jobs', '1', '--low-mem',
               '--samples-per-gpu', '1', '--workers-per-gpu', '1',
               '--cuda-visible-devices', str(gpu)]
        cwd = REPO
    else:
        try:
            mkdir = timed_run(
                ['mkdir', '-p', str(job.archive)], 10,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except subprocess.TimeoutExpired as exc:
            raise MountUnavailable('timed out creating NAS archive directory') from exc
        if mkdir.returncode:
            raise MountUnavailable(f'cannot create NAS archive directory: {mkdir.stderr.strip()}')
        archived_config = job.work / job.config.name
        archived_config.write_text(
            job.config.read_text(encoding='utf-8')
            .replace(str(LOCAL_CKPT), str(REMOTE_CKPT))
            .replace(str(job.data / 'Public'), str(PUBLIC)),
            encoding='utf-8')
        cmd = ['rsync', '-a', '--partial', '--append-verify',
               '--exclude=*.pth' if job.external else '--include=*',
               str(job.work) + '/', str(job.archive) + '/']
        cwd = REPO
    return cmd, cwd, env


def stage(job):
    if not job.data_marker.exists():
        return 'data'
    if not job.external and not (job.work / 'train.done').exists():
        return 'train'
    if not (job.work / 'eval.done').exists():
        return 'eval'
    return 'sync'


class MountUnavailable(RuntimeError):
    pass


class StopRequested(RuntimeError):
    pass


def stage_pretrained(interrupted):
    if LOCAL_CKPT.is_file():
        local_size = LOCAL_CKPT.stat().st_size
        if (LOCAL_CKPT_VERIFIED.is_file() and
                LOCAL_CKPT_VERIFIED.read_text().strip() == str(local_size)):
            return
        try:
            with zipfile.ZipFile(LOCAL_CKPT) as archive:
                if archive.testzip() is None:
                    LOCAL_CKPT_VERIFIED.write_text(str(local_size), encoding='utf-8')
                    return
        except (OSError, zipfile.BadZipFile):
            pass
    if not checkpoint_source_healthy():
        raise MountUnavailable('sshfs unavailable while staging CROWN weights')
    try:
        result = timed_run(
            ['stat', '-Lc', '%s', str(REMOTE_CKPT)], 8,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode:
            raise OSError(result.stderr.strip())
        remote_size = int(result.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise MountUnavailable(f'CROWN source checkpoint inaccessible: {exc}') from exc
    if LOCAL_CKPT.is_file() and LOCAL_CKPT.stat().st_size == remote_size:
        return
    LOCAL_CKPT.parent.mkdir(parents=True, exist_ok=True)
    partial = LOCAL_CKPT.with_suffix('.pth.partial')
    with (STATE / 'pretrained_stage.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(
            ['rsync', '-a', '--partial', '--append-verify', str(REMOTE_CKPT), str(partial)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        last_size = partial.stat().st_size if partial.exists() else 0
        last_progress = time.monotonic()
        while process.poll() is None:
            stopped = interrupted()
            size = partial.stat().st_size if partial.exists() else 0
            if size > last_size:
                last_size = size
                last_progress = time.monotonic()
            stalled = time.monotonic() - last_progress > 180
            if stopped or stalled:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                if stopped:
                    raise StopRequested('CROWN weight staging stopped')
                raise MountUnavailable('CROWN weight staging made no progress for 180 seconds; check sshfs')
            time.sleep(5)
        if process.returncode:
            if not checkpoint_source_healthy():
                raise MountUnavailable('sshfs disconnected during CROWN weight staging')
            raise RuntimeError(f'CROWN weight rsync failed ({process.returncode}); see {log.name}')
    if partial.stat().st_size != remote_size:
        raise RuntimeError('Staged CROWN checkpoint size does not match the NAS source')
    with zipfile.ZipFile(partial) as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(f'Staged CROWN checkpoint is corrupt: {bad_member}')
    partial.replace(LOCAL_CKPT)
    LOCAL_CKPT_VERIFIED.write_text(str(remote_size), encoding='utf-8')


def main_controller(rows):
    STATE.mkdir(parents=True, exist_ok=True)
    lock = (STATE / 'controller.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError('CROWN controller is already running') from exc
    (STATE / 'controller.pid').write_text(str(os.getpid()))
    interrupted = False
    def signal_handler(_signum, _frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    active = {}
    try:
        if all(rows.get(job.key, {}).get('status') == 'complete' for job in JOBS):
            print('All CROWN tasks are complete.')
            return
        try:
            stage_pretrained(lambda: interrupted)
        except MountUnavailable as exc:
            for job in JOBS:
                if rows[job.key]['status'] != 'complete':
                    update(rows, job, status='paused_mount', message=str(exc))
            print(str(exc))
            return
        except StopRequested as exc:
            for job in JOBS:
                if rows[job.key]['status'] != 'complete':
                    update(rows, job, status='stopped', message=str(exc))
            print(str(exc))
            return
        for job in JOBS:
            if rows[job.key]['status'] in ('running', 'stopped', 'paused_mount', 'failed',
                                            'waiting_sync', 'blocked_dependency'):
                update(rows, job, status='queued', phase=stage(job), gpu='', pid='',
                       message='')
        last_health = 0.0
        health_failures = 0
        last_print = 0.0
        while True:
            if interrupted:
                reason = 'stopped'
                break
            if time.monotonic() - last_health >= 15:
                health_failures = 0 if mount_healthy() else health_failures + 1
                if health_failures >= 3:
                    reason = 'paused_mount'
                    break
                last_health = time.monotonic()
            mount_broken = False
            for gpu, (job, phase, process, log, log_start) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                update(rows, job, gpu='', pid='')
                if code:
                    if log_reports_mount_error(job.work / f'{phase}.log', log_start) or not mount_healthy():
                        update(rows, job, status='paused_mount', phase=phase,
                               message='sshfs I/O error; resume after mount is stable')
                        mount_broken = True
                        break
                    update(rows, job, status='failed', phase=phase,
                           message=f'{phase} exited {code}; see {job.work / (phase + ".log")}')
                    continue
                if phase == 'data':
                    if not job.data_marker.exists():
                        update(rows, job, status='failed', phase=phase,
                               message='data staging marker missing')
                        continue
                elif phase == 'train':
                    try:
                        best = best_checkpoint(job)
                    except FileNotFoundError as exc:
                        update(rows, job, status='failed', phase=phase, message=str(exc))
                        continue
                    (job.work / 'train.done').write_text(now())
                    update(rows, job, best_ckpt=best)
                elif phase == 'eval':
                    if not job.summary.exists():
                        update(rows, job, status='failed', phase=phase, message='evaluation summary missing')
                        continue
                    (job.work / 'eval.done').write_text(now())
                    update(rows, job, metrics_json=job.summary)
                    write_results(rows)
                else:
                    archived_best = (rows[BY_KEY['det_txl_pbc'].key]['best_ckpt'] if job.external
                                     else job.archive / Path(rows[job.key]['best_ckpt']).name)
                    if not job.external:
                        for ckpt in job.work.glob('*.pth'):
                            if (job.key == 'det_txl_pbc' and ckpt.name.startswith('best_')
                                    and rows[BY_KEY['det_cbc'].key]['status'] != 'complete'):
                                continue
                            ckpt.unlink()
                    else:
                        for ckpt in job.work.glob('best_*.pth'):
                            if ckpt.is_symlink():
                                ckpt.unlink()
                        for ckpt in BY_KEY['det_txl_pbc'].work.glob('best_*.pth'):
                            ckpt.unlink()
                    update(rows, job, status='complete', phase='done',
                           best_ckpt=archived_best, message='archived on NAS')
                    shutil.rmtree(job.data, ignore_errors=True)
                    continue
                update(rows, job, status='queued', phase=stage(job))
            if mount_broken:
                reason = 'paused_mount'
                break
            busy = set(active)
            candidates = sorted(
                (job for job in JOBS if rows[job.key]['status'] == 'queued'),
                key=lambda job: (int(rows[job.key]['train_images'] or 10**12), job.dataset))
            if -1 not in active:
                data_job = next((job for job in candidates
                                 if stage(job) == 'data' and
                                 (not job.external or
                                  rows[BY_KEY['det_txl_pbc'].key]['status'] == 'complete')), None)
                if data_job is not None:
                    candidates.remove(data_job)
                    data_job.work.mkdir(parents=True, exist_ok=True)
                    try:
                        cmd, cwd, env = command_for(data_job, 'data', None)
                        log = (data_job.work / 'data.log').open('a', encoding='utf-8')
                        log_start = log.tell()
                        process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                        active[-1] = (data_job, 'data', process, log, log_start)
                        update(rows, data_job, status='running', phase='data', gpu='',
                               pid=process.pid, message='')
                    except Exception as exc:
                        if isinstance(exc, MountUnavailable) or not mount_healthy():
                            update(rows, data_job, status='paused_mount', phase='data',
                                   message=str(exc)[:240])
                            mount_broken = True
                        else:
                            update(rows, data_job, status='failed', phase='data',
                                   message=str(exc)[:240])
            if mount_broken:
                reason = 'paused_mount'
                break
            for gpu in GPUS:
                if gpu in busy:
                    continue
                eligible = next((job for job in candidates
                                 if (not job.external or
                                     rows[BY_KEY['det_txl_pbc'].key]['status'] == 'complete')
                                 and stage(job) != 'data'), None)
                if eligible is None:
                    break
                candidates.remove(eligible)
                phase = stage(eligible)
                eligible.work.mkdir(parents=True, exist_ok=True)
                try:
                    cmd, cwd, env = command_for(eligible, phase, gpu)
                    if eligible.external and phase == 'eval':
                        update(rows, eligible, best_ckpt=best_checkpoint(eligible))
                    log = (eligible.work / f'{phase}.log').open('a', encoding='utf-8')
                    log_start = log.tell()
                    process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    active[gpu] = (eligible, phase, process, log, log_start)
                    update(rows, eligible, status='running', phase=phase, gpu=gpu,
                           pid=process.pid, message='')
                except Exception as exc:
                    if isinstance(exc, MountUnavailable) or not mount_healthy():
                        update(rows, eligible, status='paused_mount', phase=phase,
                               message=str(exc)[:240])
                        mount_broken = True
                        break
                    update(rows, eligible, status='failed', phase=phase, message=str(exc)[:240])
            if mount_broken:
                reason = 'paused_mount'
                break
            if time.monotonic() - last_print >= 60:
                print_status(rows)
                last_print = time.monotonic()
            if not active and candidates and all(job.external for job in candidates):
                for job in candidates:
                    update(rows, job, status='blocked_dependency',
                           message='TXL-PBC must complete before external evaluation')
            if not active and not any(row['status'] == 'queued' for row in rows.values()):
                reason = 'finished'
                break
            time.sleep(5)
        for gpu, (job, phase, process, log, _) in active.items():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            log.close()
            update(rows, job, status=reason, phase=phase, gpu='', pid='', message=reason)
        if reason == 'paused_mount':
            for job in JOBS:
                if rows[job.key]['status'] == 'queued':
                    update(rows, job, status='paused_mount', message='sshfs unavailable')
        if all(rows.get(job.key, {}).get('status') == 'complete' for job in JOBS):
            LOCAL_CKPT.unlink(missing_ok=True)
            LOCAL_CKPT_VERIFIED.unlink(missing_ok=True)
        print(f'Controller ended: {reason}')
    finally:
        (STATE / 'controller.pid').unlink(missing_ok=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=('prepare', 'start', 'stop', 'status'), default='status')
    parser.add_argument('--internal-discover', choices=BY_KEY, help=argparse.SUPPRESS)
    parser.add_argument('--internal-stage', choices=BY_KEY, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.internal_discover:
        try:
            splits, classes = discover(BY_KEY[args.internal_discover])
            print(json.dumps({
                'splits': {name: [str(row[0]), str(row[1]), row[2], row[3]]
                           for name, row in splits.items()},
                'classes': classes,
            }))
        except Exception as exc:
            print(json.dumps({'error': str(exc), 'errno': getattr(exc, 'errno', None)}))
        return
    if args.internal_stage:
        stage_data(BY_KEY[args.internal_stage])
        return
    if args.action == 'stop':
        stop()
        return
    rows = load_rows()
    if args.action in ('prepare', 'start'):
        detect_mount_layout()
        try:
            rows = prepare(rows)
        except RuntimeError as exc:
            if 'sshfs' not in str(exc):
                raise
            for job in JOBS:
                if job.key in rows and rows[job.key]['status'] != 'complete':
                    update(rows, job, status='paused_mount', message=str(exc))
            print_status(rows)
            print(str(exc), file=sys.stderr)
            raise SystemExit(75)
    if args.action == 'start':
        main_controller(rows)
    else:
        print_status(rows)


if __name__ == '__main__':
    main()

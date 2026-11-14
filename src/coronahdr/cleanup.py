"""Conservative cleanup of application-owned job directories."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
from datetime import datetime, timezone

from .desktop_jobs import is_published


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _locked(path):
    lock = path/'job.lock'
    if lock.is_symlink():
        return True
    with lock.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def _files(root):
    device = root.stat(follow_symlinks=False).st_dev
    def fail(error):
        raise error
    for current, directories, files in os.walk(root, followlinks=False, onerror=fail):
        for name in directories + files:
            path = Path(current)/name
            info = path.lstat()
            if not stat.S_ISLNK(info.st_mode) and info.st_dev != device:
                raise ValueError(f'Mounted filesystem inside job: {path}')
            yield path, info


def scan_jobs(locations, progress=lambda message: None):
    """List only UUID-named app jobs; unknown and disconnected roots are reported."""
    jobs, unavailable, seen, unsafe = [], [], set(), []
    for location in locations:
        progress(f'Scanning {location}')
        location = Path(location).expanduser()
        if not location.is_dir():
            unavailable.append(str(location))
            continue
        base = (location/'.coronahdr-jobs')
        if base.is_symlink() or not base.is_dir():
            continue
        for root in sorted(base.iterdir()):
            if root.is_symlink() or not root.is_dir() or not re.fullmatch(r'[0-9a-f]{32}', root.name):
                continue
            real = root.resolve()
            if real in seen:
                continue
            seen.add(real)
            try:
                record = root/'job.json'
                if record.is_symlink():
                    raise ValueError('Job record is a symlink')
                if not record.is_file():
                    continue
                job = json.loads(record.read_text())
                if not isinstance(job, dict) or not all(k in job for k in ('result', 'destination', 'status')):
                    raise ValueError('Invalid job record')
                files = list(_files(root))
                references = [job]
                for path, file_info in files:
                    if path.name in ('recipe.json', 'report.json') and stat.S_ISREG(file_info.st_mode):
                        if file_info.st_size > 20_000_000:
                            raise ValueError(f'Job record too large to inspect: {path}')
                        references.append(json.loads(path.read_text()))
                size = sum(info.st_size for _, info in files if stat.S_ISREG(info.st_mode))
                info = record.stat()
                stamp = job.get('created_at')
                if not stamp:
                    birth = getattr(info, 'st_birthtime', None)
                    stamp = datetime.fromtimestamp(birth or info.st_mtime, timezone.utc).isoformat()
                    date_source = 'filesystem creation' if birth else 'record modified'
                else:
                    date_source = 'recorded creation'
                root_info = root.stat(follow_symlinks=False)
                jobs.append(dict(path=str(real), location=str(location.resolve()), job=job, references=references,
                                 identity=(root_info.st_dev, root_info.st_ino),
                                 date=stamp, date_source=date_source, bytes=size, active=_locked(root),
                                 output_available=Path(job['destination']).is_dir()))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                unsafe.append(f'{root}: {error}')
    return dict(jobs=jobs, unavailable=unavailable, unsafe=unsafe)


def _references(job, root):
    for value in _strings(job):
        if not value.startswith('/'):
            continue
        try:
            if Path(value).resolve().is_relative_to(root):
                return True
        except (OSError, RuntimeError):
            continue
    return False


def _preserve_logs(root, destination):
    logs = [(path, info) for path, info in _files(root) if path.suffix.lower() == '.log']
    if not logs:
        return
    if destination.is_symlink() or destination.resolve().is_relative_to(root):
        raise ValueError('Log destination is linked or inside the job')
    destination.mkdir(exist_ok=True)
    for path, info in logs:
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f'Cannot safely preserve linked log: {path}')
        relative = str(path.relative_to(root)).replace('/', '_')
        target = destination/f'CoronaHDR_{root.name}_{relative}'
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        base = target
        number = 1
        while target.exists() or target.is_symlink():
            if not target.is_symlink() and target.is_file():
                with target.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() == digest:
                        break
            target = base.with_name(f'{base.stem}_{number}{base.suffix}')
            number += 1
        if target.exists() and not target.is_symlink():
            continue
        created = None
        try:
            with path.open('rb') as source, target.open('xb') as saved:
                info = os.fstat(saved.fileno())
                created = (info.st_dev, info.st_ino)
                shutil.copyfileobj(source, saved, 1024*1024)
            with target.open('rb') as saved:
                if hashlib.file_digest(saved, 'sha256').hexdigest() != digest:
                    raise OSError(f'Log copy verification failed: {target}')
        except BaseException:
            if created is not None:
                try:
                    info = target.stat(follow_symlinks=False)
                    if (info.st_dev, info.st_ino) == created:
                        target.unlink()
                except FileNotFoundError:
                    pass
            raise


def clean_jobs(locations, selected, progress=lambda message: None):
    """Remove selected, verified jobs; never follow links or cross mounts."""
    snapshot = scan_jobs(locations, progress)
    known = {entry['path']: entry for entry in snapshot['jobs']}
    selected = list(dict.fromkeys(str(Path(path).resolve()) for path in selected))
    outcome, locks, eligible = {}, {}, set()
    if snapshot['unsafe']:
        return {name: dict(status='skipped', reason='A known job could not be inspected: ' + '; '.join(snapshot['unsafe'])) for name in selected}
    try:
        for name in selected:
            try:
                progress(f'Checking {name}')
                root = Path(name)
                if not root.is_dir() or root.is_symlink():
                    raise ValueError('Job folder no longer exists or is unavailable')
                if name not in known:
                    raise ValueError('Job folder is not recognized in the selected locations')
                entry = known[name]
                if root.is_symlink() or root.parent.is_symlink() or root.name != Path(name).name:
                    raise ValueError('Job location changed')
                info = root.stat(follow_symlinks=False)
                if (info.st_dev, info.st_ino) != entry['identity']:
                    raise ValueError('Job location changed')
                lock_path = root/'job.lock'
                if lock_path.is_symlink():
                    raise ValueError('Job lock is a symlink')
                lock = lock_path.open('a')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    lock.close()
                    raise
                locks[name] = lock
                job = json.loads((root/'job.json').read_text())
                entry['job'] = job
                entry['references'][0] = job
                list(_files(root))
                if not shutil.rmtree.avoids_symlink_attacks:
                    raise ValueError('Safe deletion is not available on this platform')
                for other in known.values():
                    destination = other['job'].get('destination')
                    if destination and Path(destination).resolve().is_relative_to(root):
                        raise ValueError('A job output folder is inside this job')
                    for folder in other['job'].get('folders', {}).values():
                        if folder and Path(folder).resolve().is_relative_to(root):
                            raise ValueError('An original input folder is inside this job')
                result = Path(job['result'])
                if job.get('publication') == 'standalone_final_files_v1':
                    if job.get('status') == 'completed' and not is_published(job):
                        raise ValueError('Published final images could not be verified')
                    for filename in job.get('published_files', {}):
                        target = Path(job['destination'])/filename
                        if target.resolve().is_relative_to(root):
                            raise ValueError('Final image resolves inside the job')
                        target_info = target.stat()
                        if any(stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == (target_info.st_dev, target_info.st_ino)
                               for _, info in _files(root)):
                            raise ValueError('Final image is hard-linked to temporary data')
                    if job.get('status') != 'completed' and any(root.rglob('master.fits')):
                        raise ValueError('Unpublished master exists in the job')
                    if job.get('status') != 'completed' and any(root.rglob('master.tiff')):
                        raise ValueError('Unpublished master exists in the job')
                    if Path(job['destination']).resolve().is_relative_to(root):
                        raise ValueError('Final output is inside the job')
                elif result.exists() or result.is_symlink():
                    raise ValueError('Legacy output may depend on this job; it cannot be safely removed')
                _preserve_logs(root, Path(job['destination']))
                eligible.add(name)
            except (KeyError, OSError, ValueError, TypeError) as error:
                outcome[name] = dict(status='skipped', reason=str(error))
        # A skipped job protects its dependencies; repeat until no more jobs are blocked.
        while True:
            blocked = {name for name in eligible if any(
                other != name and other not in eligible and _references(entry['references'], Path(name))
                for other, entry in known.items())}
            if not blocked:
                break
            for name in blocked:
                eligible.remove(name)
                outcome[name] = dict(status='skipped', reason='Another saved job depends on this job')
        # Delete dependents before their dependencies; cycles are protected.
        while eligible:
            blocked = {name for name in eligible if any(
                other != name and other not in eligible and outcome.get(other, {}).get('status') != 'removed'
                and _references(entry['references'], Path(name)) for other, entry in known.items())}
            if blocked:
                for name in blocked:
                    eligible.remove(name)
                    outcome[name] = dict(status='skipped', reason='A saved dependent job was not removed')
                continue
            ready = [name for name in eligible if not any(other != name and _references(known[other]['references'], Path(name))
                                                           for other in eligible)]
            if not ready:
                for name in eligible:
                    outcome[name] = dict(status='skipped', reason='Cyclic saved-job dependencies')
                break
            for name in ready:
                try:
                    progress(f'Removing {name}')
                    current = scan_jobs(locations, progress)
                    if current['unsafe']:
                        raise ValueError('A known job could not be inspected during the final check')
                    if any(other['path'] != name and _references(other['references'], Path(name))
                           for other in current['jobs']):
                        raise ValueError('Another saved job still depends on this job')
                    root = Path(name)
                    for other in current['jobs']:
                        protected = [other['job'].get('destination'), *other['job'].get('folders', {}).values()]
                        if any(value and Path(value).resolve().is_relative_to(root) for value in protected):
                            raise ValueError('A known output or input folder is inside this job')
                    info = root.stat(follow_symlinks=False)
                    if root.is_symlink() or (info.st_dev, info.st_ino) != known[name]['identity']:
                        raise ValueError('Job location changed')
                    list(_files(root))
                    size = known[name]['bytes']
                    shutil.rmtree(root)
                    if root.exists() or root.is_symlink():
                        raise OSError('Job folder still exists after deletion')
                    outcome[name] = dict(status='removed', estimated_bytes=size)
                    eligible.remove(name)
                except (OSError, ValueError) as error:
                    outcome[name] = dict(status='skipped', reason=str(error))
                    eligible.remove(name)
    finally:
        for lock in locks.values():
            lock.close()
    for name in selected:
        outcome.setdefault(name, dict(status='skipped', reason='Job is unavailable or not recognized'))
    return outcome

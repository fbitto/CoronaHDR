"""Isolated desktop jobs; publish completed products without replacing destinations."""
import json
import os
from pathlib import Path
import sys
import fcntl
import shutil
import hashlib
import signal
from uuid import uuid4
from datetime import datetime, timezone

from .aesthetic_pipeline import select_frames


def output_path(destination, *, allow_existing_products=False):
    if not str(destination).strip():
        raise ValueError('Choose an output folder')
    destination = Path(destination).absolute()
    if not destination.parent.is_dir():
        raise ValueError('Choose an existing parent folder for the output')
    if destination.is_dir():
        for name in (() if allow_existing_products else ('master.tiff', 'master.fits', 'report.json')):
            if (destination/name).exists() or (destination/name).is_symlink():
                raise ValueError(f'Output already contains {name}; existing files are never replaced')
        return destination.resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError('Choose an output directory; existing files are never replaced')
    return destination


def job_directories(destination, temporary, *, results_in_temporary=False):
    if temporary is not None and not str(temporary).strip():
        raise ValueError('Choose an existing temporary files folder')
    temporary = Path(temporary).resolve() if temporary else destination.parent.resolve()
    if not temporary.is_dir():
        raise ValueError('Choose an existing temporary files folder')
    if results_in_temporary and (temporary/'.coronahdr-jobs').is_relative_to(destination.resolve()):
        raise ValueError('Choose a temporary files folder outside the output folder')
    identifier = uuid4().hex
    root = temporary/'.coronahdr-jobs'/identifier
    result = root/'result' if results_in_temporary else destination.parent/'.coronahdr-jobs'/identifier/'result'
    root.mkdir(parents=True)
    result.parent.mkdir(parents=True, exist_ok=True)
    return root, result


def create_recipe_job(recipe, base, destination, temporary=None):
    """Freeze an explicitly reviewed recipe; resolve paths before moving its base."""
    recipe = json.loads(json.dumps(recipe, allow_nan=False))
    base, destination = Path(base).resolve(), output_path(destination)
    frames = recipe.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('Import at least one light frame')
    reference = recipe.get('reference')
    if reference is not None and (type(reference) is not int or not 0 <= reference < len(frames)):
        raise ValueError('Reference must be null (automatic) or a valid zero-based light index')
    if reference is None:
        recipe.pop('reference', None)
    calibration = recipe.get('calibration', {})
    if not isinstance(calibration, dict) or set(calibration)-{'bias', 'dark', 'flat', 'dark_flat'}:
        raise ValueError('Calibration roles must be bias, dark, flat or dark_flat')
    for entries in [frames, *calibration.values()]:
        if not isinstance(entries, list):
            raise ValueError('Each calibration role must contain a list of files')
        for item in entries:
            if not isinstance(item, dict) or not isinstance(item.get('path'), str):
                raise ValueError('Each input requires a file path')
            path = (base/item['path']).resolve()
            if not path.is_file():
                raise ValueError(f'Input file does not exist: {path}')
            if not isinstance(item.get('overrides', {}), dict):
                raise ValueError(f'Metadata overrides must be an object: {path}')
            item['path'] = str(path)
    if recipe.get('method', 'weighted') != 'weighted':
        raise ValueError('New import uses weighted HDR; aesthetic HDR requires prepared data and reviewed geometry')
    options = recipe.setdefault('processing', {})
    if not isinstance(options, dict):
        raise ValueError('Processing settings must be an object')
    options['mode'] = 'disk'
    for key in ('cache_from', 'rigid_review', 'registration_review'):
        if options.get(key):
            options[key] = str((base/options[key]).resolve())
    root, result = job_directories(destination, temporary)
    options['work_directory'] = str(root/'work')
    recipe_path = root/'recipe.json'
    recipe_path.write_text(json.dumps(recipe, indent=2)+'\n')
    job = dict(created_at=datetime.now(timezone.utc).isoformat(), destination=str(destination), result=str(result), method='weighted',
               arguments=['run', str(recipe_path), '--output', str(result)], status='ready')
    path = root/'job.json'
    path.write_text(json.dumps(job, indent=2)+'\n')
    return path, job


def output_base_name(value):
    if not isinstance(value, str):
        raise ValueError('Output base name must be text')
    value = value.strip()
    if (len(value) > 150 or len(value.encode('utf-8')) > 200 or value in ('.', '..') or any(c in value for c in '/\\:')
            or any(ord(c) < 32 for c in value)):
        raise ValueError('Output base name must be a single file name without path separators')
    return value


def named_masters(source, destination, base, mode):
    """Keep the engine's master files and add user-named files without replacing anything."""
    base = output_base_name(base)
    if not base or mode not in ('aesthetic', 'scientific'):
        raise ValueError('A valid output base name and HDR mode are required')
    names = []
    for extension in ('fits', 'tiff'):
        original = Path(source)/f'master.{extension}'
        target = Path(destination)/f'{base}_{mode}.{extension}'
        if not original.is_file():
            raise ValueError(f'Missing output master: {original}')
        try:
            os.link(original, target)
        except FileExistsError:
            raise
        except OSError:
            with original.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst, length=1024*1024)
        names.append(target.name)
    return names


class OutputNameConflict(ValueError):
    pass


def final_output_names(mode, base, output_format='both'):
    if mode not in ('scientific', 'aesthetic', 'both'):
        raise ValueError('HDR mode must be aesthetic, scientific, or both')
    modes = ('scientific', 'aesthetic') if mode == 'both' else (mode,)
    if output_format not in ('fits', 'tiff', 'both'):
        raise ValueError('Output format must be FITS, TIFF, or both')
    extensions = ('fits', 'tiff') if output_format == 'both' else (output_format,)
    base = output_base_name(base or 'CoronaHDR')
    return {f'{base}_{item}.{ext}': (f'{item}/master.{ext}' if mode == 'both' else f'master.{ext}')
            for item in modes for ext in extensions}


def final_log_name(base, mode):
    if mode not in ('scientific', 'aesthetic', 'both'):
        raise ValueError('HDR mode must be aesthetic, scientific, or both')
    return f'{output_base_name(base or "CoronaHDR")}_{mode}.log'


def validate_folder_targets(destination, mode, base, *, standalone=False, output_format='both', check_settings=True):
    if not destination.is_dir():
        return
    if standalone:
        names = (*final_output_names(mode, base, output_format), final_log_name(base, mode))
        if check_settings:
            names += (f'{output_base_name(base or "CoronaHDR")}_settings.json',)
        for name in names:
            target = destination/name
            if target.exists() or target.is_symlink():
                raise OutputNameConflict(f'Output already contains {name}. Choose a different base name.')
        return
    modes = ('scientific', 'aesthetic') if mode == 'both' else (mode,)
    base = output_base_name(base or ('CoronaHDR' if mode == 'both' else ''))
    names = list(modes) if mode == 'both' else []
    if base:
        names += [f'{base}_{item}.{ext}' for item in modes for ext in ('fits', 'tiff')]
    for name in names:
        target = destination/name
        if target.exists() or target.is_symlink():
            raise ValueError(f'Output already contains {name}; existing files are never replaced')


def create_folder_job(folders, *, mode='aesthetic', bias_fallback=False, save_intermediates=(), detect_hot_pixels=False, lunar_margin=None, temporary=None, destination=None, base_name='', alignment='conservative', output_format='both'):
    from .lunar import validate_margin
    from .work_paths import default_work_directory, work_exclusion_root
    validate_margin(lunar_margin)
    from .folder_import import ROLES, discover
    artifacts = Path(__file__).resolve().parents[2]/'artifacts'
    if not folders.get('light'):
        raise ValueError('Select the lights folder')
    destination = output_path(destination if destination is not None else Path(folders['light']).absolute().parent/'HDR_output',
                              allow_existing_products=True)
    if mode not in ('aesthetic', 'scientific', 'both'):
        raise ValueError('HDR mode must be aesthetic, scientific, or both')
    if alignment not in ('conservative', 'ael_visual'):
        raise ValueError('Unknown alignment choice')
    if alignment == 'ael_visual' and mode == 'scientific':
        raise ValueError('AEL-style visual alignment is available only for aesthetic output; choose conservative alignment for scientific output')
    base_name = output_base_name(base_name)
    final_output_names(mode, base_name, output_format)
    validate_folder_targets(destination, mode, base_name, standalone=True, output_format=output_format)
    if isinstance(save_intermediates, str) or set(save_intermediates)-{'calibrated', 'debayered', 'registered'}:
        raise ValueError('Unknown intermediate image stage')
    if type(bias_fallback) is not bool:
        raise ValueError('Bias fallback must be an explicit boolean')
    if type(detect_hot_pixels) is not bool:
        raise ValueError('Hot-pixel detection must be an explicit boolean')
    if bias_fallback and not any(folders.get(role) for role in ('bias', 'dark', 'dark_flat')):
        raise ValueError('Bias fallback requires a folder containing validated bias frames')
    for role, folder in folders.items():
        if role not in ROLES or (folder and not Path(folder).is_dir()):
            raise ValueError(f'Invalid {role} folder: {folder}')
    raw_files = discover(folders['light'], work_exclusion_root(artifacts))
    extensions = {p.suffix.lower() for p in raw_files}
    if extensions and extensions <= {'.cr2', '.cr3'}:
        family = 'Canon'
    elif extensions and extensions <= {'.nef', '.nrw'}:
        family = 'Nikon'
    else:
        raise ValueError('Automatic folder preparation requires a single Canon or Nikon RAW acquisition')
    root, result = job_directories(destination, temporary if temporary is not None else default_work_directory(artifacts),
                                   results_in_temporary=True)
    job = dict(created_at=datetime.now(timezone.utc).isoformat(), folders={role: str(Path(folder).resolve()) for role, folder in folders.items() if folder},
               profile_source=None, profile_review=None, camera_family=family,
               work=str(root), destination=str(destination), result=str(result),
               status='ready', kind='folders', hdr_mode=mode, bias_fallback=bias_fallback,
               save_intermediates=list(dict.fromkeys(save_intermediates)), detect_hot_pixels=detect_hot_pixels,
               lunar_margin_px=lunar_margin, output_base_name=base_name, alignment=alignment,
               publication='standalone_final_files_v1', output_format=output_format)
    path = root/'job.json'
    settings_path = destination/f'{base_name or "CoronaHDR"}_settings.json'
    settings = dict(format='coronahdr-folder-settings', version=1, folders=job['folders'], settings=dict(
        mode=mode, bias_fallback=bias_fallback, save_intermediates=job['save_intermediates'],
        detect_hot_pixels=detect_hot_pixels, lunar_margin=lunar_margin, temporary=str(root.parent.parent),
        destination=str(destination), base_name=base_name, alignment=alignment, output_format=output_format))
    created_settings = False
    try:
        destination.mkdir(exist_ok=True)
        with settings_path.open('x', encoding='utf-8') as stream:
            created_settings = True
            json.dump(settings, stream, indent=2)
            stream.write('\n')
        job['settings_path'] = str(settings_path)
        path.write_text(json.dumps(job, indent=2)+'\n')
    except FileExistsError:
        shutil.rmtree(root)
        raise OutputNameConflict(f'Output already contains {settings_path.name}. Choose a different base name.') from None
    except BaseException:
        if created_settings:
            settings_path.unlink(missing_ok=True)
        shutil.rmtree(root)
        raise
    return path, job


def create_job(source, review, destination, indices, method, lunar_margin=None, temporary=None):
    from .lunar import validate_margin
    validate_margin(lunar_margin)
    source, review = Path(source).resolve(), Path(review).resolve()
    destination = output_path(destination)
    if method not in ('aesthetic', 'weighted'):
        raise ValueError('Unsupported desktop HDR method')
    report = json.loads((source/'report.json').read_text())
    geometry = json.loads(review.read_text())
    if geometry['input_source_sha256'] != [i['metadata']['source_sha256'] for i in report['inputs']]:
        raise ValueError('Geometry review does not match this prepared run')
    reference = report.get('reference', report['recipe'].get('reference'))
    selected = select_frames(len(report['inputs']), reference, geometry['matrices_yx'], indices)
    root, result = job_directories(destination, temporary)
    if method == 'aesthetic':
        arguments = ['aesthetic', str(source), '--geometry-review', str(review), '--output', str(result),
                     '--indices', *map(str, selected)]
        if lunar_margin is not None:
            arguments += ['--lunar-margin', str(lunar_margin)]
        arguments += ['--work-directory', str(root/'work')]
    else:
        recipe = dict(report['recipe'], method='weighted')
        recipe['reference'] = reference
        recipe['processing'] = dict(recipe.get('processing', {}), mode='disk', cache_from=str(source),
                                    rigid_review=str(review), include_indices=selected, rejection_policy='exclude',
                                    work_directory=str(root/'work'))
        if lunar_margin is not None:
            recipe['processing']['lunar_margin_px'] = lunar_margin
        recipe_path = root/'recipe.json'
        recipe_path.write_text(json.dumps(recipe, indent=2)+'\n')
        arguments = ['run', str(recipe_path), '--output', str(result)]
    job = dict(created_at=datetime.now(timezone.utc).isoformat(), source=str(source), review=str(review), destination=str(destination), result=str(result),
               indices=selected, method=method, arguments=arguments, status='ready')
    path = root/'job.json'
    path.write_text(json.dumps(job, indent=2)+'\n')
    return path, job


def set_job_status(path, status, message=''):
    path = Path(path)
    job = json.loads(path.read_text())
    job.update(status=status, message=message)
    temporary = path.with_suffix('.new')
    temporary.write_text(json.dumps(job, indent=2)+'\n')
    temporary.replace(path)


def unfinished_jobs(temporary, *, include_completed=False):
    """Inspect only job records, never recursively scan original image folders."""
    found = []
    for path in (Path(temporary)/'.coronahdr-jobs').glob('*/job.json'):
        try:
            job = json.loads(path.read_text())
            if (job.get('kind') != 'folders' or job.get('status') == 'resumed' or
                    (not include_completed and (job.get('status') == 'completed' or is_published(job)))):
                continue
            with path.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            found.append((path, job))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(found, key=lambda item: item[0].stat().st_mtime, reverse=True)


def resume_folder_job(path, *, destination=None, base_name=None):
    """Retain all previous files; unfinished stages write to a fresh attempt."""
    path = Path(path).resolve()
    job = json.loads(path.read_text())
    completed = job.get('status') == 'completed' or is_published(job)
    if job.get('kind') != 'folders' or job.get('status') == 'resumed' or (completed and destination is None):
        raise ValueError('Select an unfinished folder job')
    with path.with_suffix('.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('This job is still processing') from None
        standalone = job.get('publication') == 'standalone_final_files_v1'
        destination = output_path(job['destination'] if destination is None else destination,
                                  allow_existing_products=standalone)
        base = output_base_name(job.get('output_base_name', '') if base_name is None else base_name)
        settings_path = destination/f'{base or "CoronaHDR"}_settings.json'
        existing_settings = job.get('settings_path') == str(settings_path)
        validate_folder_targets(destination, job.get('hdr_mode', 'aesthetic'), base, standalone=standalone,
                                output_format=job.get('output_format', 'both'), check_settings=not existing_settings)
        root, result = job_directories(destination, path.parent.parent.parent, results_in_temporary=standalone)
        # Keep the earliest incomplete acquisition reachable over repeated retries.
        acquisition = path.parent/'acquisition'
        if (acquisition/'prepared/preparation/report.json').is_file() or (acquisition/'prepared/report.json').is_file():
            job['resume_acquisition'] = str(acquisition)
        else:
            job['resume_acquisition'] = job.get('resume_acquisition', str(acquisition))
        job.update(created_at=datetime.now(timezone.utc).isoformat(), result=str(result), work=str(root), status='ready', resumed_from=str(path),
                   previous_result=job['result'], destination=str(destination), output_base_name=base)
        job.pop('published_files', None)
        if completed and not standalone:
            job.pop('previous_result', None)
        if standalone and job.get('settings_path') and not existing_settings:
            old_settings = Path(job['settings_path'])
            settings = json.loads(old_settings.read_text())
            settings['settings'].update(destination=str(destination), base_name=base,
                                        temporary=str(path.parent.parent.parent))
            destination.mkdir(exist_ok=True)
            try:
                with settings_path.open('x', encoding='utf-8') as stream:
                    json.dump(settings, stream, indent=2)
                    stream.write('\n')
            except FileExistsError:
                shutil.rmtree(root)
                raise OutputNameConflict(f'Output already contains {settings_path.name}. Choose a different base name.') from None
            job['settings_path'] = str(settings_path)
        new_path = root/'job.json'
        new_path.write_text(json.dumps(job, indent=2)+'\n')
        if not completed:
            set_job_status(path, 'resumed', f'Continued in {new_path}')
    return new_path, job


def required_result_files(job):
    required = (('report.json', 'scientific/master.tiff', 'scientific/master.fits',
                 'scientific/report.json', 'aesthetic/master.tiff', 'aesthetic/master.fits',
                 'aesthetic/report.json') if job.get('hdr_mode') == 'both' else
                ('master.tiff', 'master.fits', 'report.json'))
    if job.get('publication') == 'standalone_final_files_v1':
        return required
    if job.get('hdr_mode') == 'both':
        base = output_base_name(job.get('output_base_name') or 'CoronaHDR')
        required += tuple(f'{base}_{mode}.{ext}' for mode in ('scientific', 'aesthetic') for ext in ('fits', 'tiff'))
    elif job.get('output_base_name'):
        base = output_base_name(job['output_base_name'])
        required += tuple(f'{base}_{job["hdr_mode"]}.{ext}' for ext in ('fits', 'tiff'))
    return required


def is_published(job):
    destination, result = Path(job['destination']), Path(job['result'])
    if job.get('publication') == 'standalone_final_files_v1':
        recorded = job.get('published_files', {})
        expected = final_output_names(job.get('hdr_mode', 'aesthetic'), job.get('output_base_name', ''), job.get('output_format', 'both'))
        if set(recorded) != set(expected):
            return False
        for name, digest in recorded.items():
            target = destination/name
            if target.is_symlink() or not target.is_file():
                return False
            with target.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    return False
        return True
    required = required_result_files(job)
    if not all((result/name).is_file() for name in required):
        return False
    return all((destination/entry.name).resolve() == entry.resolve() for entry in result.iterdir())


def publish_job(path):
    path = Path(path)
    job = json.loads(path.read_text())
    result, destination = Path(job['result']), Path(job['destination'])
    for filename in required_result_files(job):
        if not (result/filename).is_file():
            raise ValueError(f'Incomplete job: missing {filename}')
    json.loads((result/'report.json').read_text())
    if job.get('publication') == 'standalone_final_files_v1':
        return publish_final_files(path, job)
    if destination.is_dir() and not destination.is_symlink():
        entries = list(result.iterdir())
        for entry in entries:
            target = destination/entry.name
            if target.exists() or target.is_symlink():
                raise FileExistsError(f'Existing output is never replaced: {target}')
        published = []
        try:
            for entry in entries:
                target = destination/entry.name
                target.symlink_to(os.path.relpath(entry, destination), target_is_directory=entry.is_dir())
                published.append((target, entry))
        except BaseException:
            # Roll back only links created by this publication, preserving other files.
            for target, entry in reversed(published):
                if target.is_symlink() and target.resolve() == entry.resolve():
                    target.unlink()
            raise
    else:
        destination.symlink_to(os.path.relpath(result, destination.parent), target_is_directory=True)
    set_job_status(path, 'completed')


def publish_final_files(path, job):
    """Copy only final images; never overwrite or depend on the temporary drive."""
    result, destination = Path(job['result']), Path(job['destination'])
    mode, base = job.get('hdr_mode', 'aesthetic'), job.get('output_base_name', '')
    output_format = job.get('output_format', 'both')
    files = final_output_names(mode, base, output_format)
    validate_folder_targets(destination, mode, base, standalone=True, output_format=output_format, check_settings=False)
    required = sum((result/relative).stat().st_size for relative in files.values())
    location = destination if destination.is_dir() else destination.parent
    if required > shutil.disk_usage(location).free:
        raise OSError('Not enough free space in the output location for the selected final files')
    destination.mkdir(exist_ok=True)
    created, digests = [], {}
    previous = signal.getsignal(signal.SIGTERM)

    def cancelled(signum, frame):
        raise InterruptedError('Publication cancelled')

    signal.signal(signal.SIGTERM, cancelled)
    try:
        for name, relative in files.items():
            source, target = result/relative, destination/name
            try:
                with source.open('rb') as src, target.open('xb') as dst:
                    stat = os.fstat(dst.fileno())
                    created.append((target, stat.st_dev, stat.st_ino))
                    shutil.copyfileobj(src, dst, length=1024*1024)
            except FileExistsError:
                raise OutputNameConflict(f'Output already contains {name}. Choose a different base name.') from None
            with source.open('rb') as src, target.open('rb') as dst:
                digest = hashlib.file_digest(src, 'sha256').hexdigest()
                if hashlib.file_digest(dst, 'sha256').hexdigest() != digest:
                    raise OSError(f'Copied output failed verification: {name}')
            digests[name] = digest
        settings_path = Path(job['settings_path']) if job.get('settings_path') else None
        if settings_path and settings_path.is_file():
            from .corona_filters import job_filter_context
            import base64
            import zlib
            import numpy as np
            from astropy.io import fits
            settings = json.loads(settings_path.read_text(encoding='utf-8'))
            records = {}
            for name, relative in files.items():
                try:
                    context = job_filter_context(result/relative)
                    with fits.open(context['coverage'], memmap=True) as hdus:
                        coverage = np.asarray(hdus[0].data) >= .5*context['count']
                        shape = list(coverage.shape)
                        bits = base64.b64encode(zlib.compress(np.packbits(coverage).tobytes())).decode('ascii')
                    saved = dict(coverage_bits=bits, coverage_shape=shape, moon=context['moon'],
                                 sun=context['sun'], margin=context['margin'])
                    records[name] = dict(sha256=digests[name], context=saved)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    print(f'WARNING: Could not preserve filter coverage for {name}: {error}', flush=True)
            if records:
                settings['filter_data'] = records
                temp_settings = settings_path.with_name(f'.{settings_path.name}.{uuid4().hex}.tmp')
                try:
                    with temp_settings.open('x', encoding='utf-8') as stream:
                        json.dump(settings, stream, indent=2)
                        stream.write('\n')
                    temp_settings.replace(settings_path)
                finally:
                    temp_settings.unlink(missing_ok=True)
        job.update(status='completed', published_files=digests)
        temporary = Path(path).with_suffix('.new')
        temporary.write_text(json.dumps(job, indent=2)+'\n')
        temporary.replace(path)
    except BaseException:
        for target, device, inode in reversed(created):
            try:
                stat = target.stat(follow_symlinks=False)
                if (stat.st_dev, stat.st_ino) == (device, inode):
                    target.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def save_completed_log(job_path, job):
    """Save the complete drained worker log without replacing existing files."""
    source = Path(job_path).parent/'process.log'
    mode = job.get('hdr_mode') or ('scientific' if job.get('method') == 'weighted' else 'aesthetic')
    target = Path(job['destination'])/final_log_name(job.get('output_base_name', ''), mode)
    created = None
    try:
        with source.open('rb') as src, target.open('xb') as dst:
            stat = os.fstat(dst.fileno())
            created = (stat.st_dev, stat.st_ino)
            shutil.copyfileobj(src, dst, length=1024*1024)
        with source.open('rb') as src, target.open('rb') as dst:
            if hashlib.file_digest(src, 'sha256').hexdigest() != hashlib.file_digest(dst, 'sha256').hexdigest():
                raise OSError('Copied job log failed verification')
    except BaseException:
        if created is not None:
            try:
                stat = target.stat(follow_symlinks=False)
                if (stat.st_dev, stat.st_ino) == created:
                    target.unlink()
            except FileNotFoundError:
                pass
        raise
    return target


def main():
    if sys.version_info[:3] != (3, 14, 8):
        raise SystemExit('Desktop jobs require Python 3.14.8')
    path = Path(sys.argv[1])
    lock = path.with_suffix('.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('This job is already processing') from None
    job = json.loads(path.read_text())
    set_job_status(path, 'running')
    try:
        from .folder_import import progress_stage
        if job.get('kind') == 'folders':
            from .folder_import import run_folders
            progress_stage('prepare')
            run_folders(path)
        else:
            progress_stage('build')
            from .cli import main as cli_main
            sys.argv = ['coronahdr', *job['arguments']]
            cli_main()
        from .performance import stage
        progress_stage('publish')
        with stage('publication'):
            publish_job(path)
    except BaseException as error:
        set_job_status(path, 'failed', str(error))
        raise


if __name__ == '__main__':
    main()

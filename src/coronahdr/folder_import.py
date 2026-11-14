"""Folder processing with fresh preparation or an explicit legacy acquisition profile."""
import hashlib
import json
import os
from pathlib import Path

from .raw import RAW_EXTENSIONS
from .performance import stage
from .work_paths import work_exclusion_root

ROLES = ('light', 'bias', 'dark', 'flat', 'dark_flat')


def progress_stage(name):
    if os.environ.get('CORONAHDR_INTERACTIVE') == '1':
        print('CORONAHDR_STAGE ' + name, flush=True)


def preview_ready(role, path=None, reason=None):
    if os.environ.get('CORONAHDR_INTERACTIVE') == '1':
        print('CORONAHDR_PREVIEW_READY ' + json.dumps(dict(role=role, path=str(path) if path else None, reason=reason)), flush=True)


def related_folders(lights, work=None):
    """Search the nearby tree, never working storage or symlinked directories."""
    lights = Path(lights).resolve()
    aliases = {'bias': {'bias', 'biases', 'offsets'}, 'dark': {'dark', 'darks'},
               'flat': {'flat', 'flats'}, 'dark_flat': {'darkflat', 'darkflats', 'flatdarks'}}
    result = {role: [] for role in aliases}
    work = Path(work).resolve() if work is not None else work_exclusion_root()
    for root, directories, _ in os.walk(lights.parent, followlinks=False):
        depth = len(Path(root).relative_to(lights.parent).parts)
        directories[:] = sorted(d for d in directories if depth < 2 and d not in ('CoronaHDR_Work', '.coronahdr-jobs')
            and not (Path(root)/d).is_symlink() and not (Path(root)/d).resolve().is_relative_to(work))
        for name in directories:
            path = Path(root)/name
            normalized = name.lower().replace('_', '').replace('-', '').replace(' ', '')
            for role, names in aliases.items():
                if normalized in names and path != lights:
                    result[role].append(path)
    return result


def discover(folder, work):
    folder, work = Path(folder).resolve(), Path(work).resolve()
    if not folder.is_dir() or folder.is_relative_to(work):
        raise ValueError(f'Select an original image folder outside the working directory: {folder}')
    extensions = RAW_EXTENSIONS | {'.fits', '.fit', '.fts', '.tif', '.tiff'}
    files = []
    for root, directories, names in os.walk(folder, followlinks=False):
        directories[:] = sorted(d for d in directories if d not in ('CoronaHDR_Work', '.coronahdr-jobs')
            and not (Path(root)/d).is_symlink() and not (Path(root)/d).resolve().is_relative_to(work))
        files.extend(Path(root)/name for name in sorted(names)
                     if Path(name).suffix.lower() in extensions and not (Path(root)/name).is_symlink())
    return files


def match_profile(folders, source, work):
    source = Path(source)
    report = json.loads((source/'report.json').read_text())
    known = {'light': [item['metadata']['source_sha256'] for item in report['inputs']]}
    for role in ROLES[1:]:
        known[role] = [item['metadata']['source_sha256']
            for master in report.get('calibration', {}).get(role, []) for item in master['sources']]
    selected, audit, role_hashes = [], {}, {}
    for role in ROLES:
        paths = discover(folders[role], work) if folders.get(role) else []
        family = report.get('recipe', {}).get('camera_family')
        if family in ('Canon', 'Nikon'):
            extensions = ('.cr2', '.cr3') if family == 'Canon' else ('.nef', '.nrw')
            paths = [p for p in paths if p.suffix.lower() in extensions]
        if role == 'light' and not paths:
            raise ValueError('The lights folder contains no supported images')
        hashes = []
        for path in paths:
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            bias_alias = role in ('dark', 'dark_flat') and not known[role] and digest in known['bias']
            if digest not in known[role] and not bias_alias:
                raise ValueError(f'No validated automatic profile for {role} file: {path}. '
                                 'Prepare and validate this acquisition before reusing its profile.')
            hashes.append(digest)
        if len(hashes) != len(set(hashes)):
            raise ValueError(f'Duplicate image content in {role} folder')
        if role == 'light':
            selected = sorted(known[role].index(digest) for digest in hashes)
        role_hashes[role] = set(hashes)
        audit[role] = [dict(path=str(path), source_sha256=digest, **(
            dict(effective_role='bias', selected_role=role) if role in ('dark', 'dark_flat') and digest in known['bias'] and not known[role] else {}))
            for path, digest in zip(paths, hashes)]
    aliases = {role for role in ('dark', 'dark_flat') if audit[role] and all(item.get('effective_role') == 'bias' for item in audit[role])}
    for role in ROLES[1:]:
        expected = set(known['bias'] if role in aliases else known[role])
        supplied = role_hashes[role]
        if role == 'bias' and not supplied and aliases:
            supplied = set.union(*(role_hashes[name] for name in aliases))
        if supplied != expected:
            raise ValueError(f'The selected {role} frames differ from the verified preparation. '
                             'Select the original calibration folder used for this dataset.')
    return selected, audit


def run_both_folders(job_path, job):
    """Run each policy independently, sharing the original preparation and geometry."""
    from .desktop_jobs import named_masters, required_result_files, final_output_names
    if job.get('resumed_from') and job.get('previous_result'):
        previous = Path(job['previous_result'])
        if all((previous/name).is_file() for name in required_result_files(job)):
            completed = json.loads((previous/'report.json').read_text())
            if completed.get('folder_processing', {}).get('hdr_mode') == 'both':
                from .streaming import load_cached_run
                source = Path(job['profile_source'])
                saved = json.loads((source/'report.json').read_text())
                load_cached_run(source, saved['recipe'])
                job['result'] = str(previous)
                job_path.write_text(json.dumps(job, indent=2)+'\n')
                print('Resumed completed HDR pair; publishing saved results', flush=True)
                return
    root, output = job_path.parent, Path(job['result'])
    output.mkdir(parents=True)
    shared_source = job.get('profile_source')
    shared_review = job.get('profile_review')
    requested_alignment = job.get('alignment', 'conservative')
    results, warnings = {}, []
    for mode in ('scientific', 'aesthetic'):
        print(f'Starting {mode} HDR', flush=True)
        if results:
            progress_stage(mode)
        child_root = root/mode
        child_root.mkdir()
        child = dict(job, hdr_mode=mode, work=str(child_root), result=str(output/mode),
                     profile_source=shared_source, profile_review=shared_review,
                     alignment='conservative' if mode == 'scientific' else requested_alignment,
                     output_base_name='', _shared_geometry=bool(results), _preview_main=(mode == 'aesthetic'))
        if mode == 'scientific':
            child.pop('ael_visual_review', None)
        for key in ('resumed_from', 'previous_result', 'defect_detection', 'defect_selection', 'original_profile_source'):
            child.pop(key, None)
        if not shared_source and not child.get('resume_acquisition'):
            job['resume_acquisition'] = str(child_root/'acquisition')
            job_path.write_text(json.dumps(job, indent=2)+'\n')
        child_path = child_root/'job.json'
        child_path.write_text(json.dumps(child, indent=2)+'\n')
        run_folders(child_path)
        child = json.loads(child_path.read_text())
        shared_source = child.get('original_profile_source', child['profile_source'])
        if mode == 'scientific' or requested_alignment == 'conservative':
            shared_review = child['profile_review']
        elif child.get('ael_visual_review'):
            job['ael_visual_review'] = child['ael_visual_review']
        job.update(profile_source=shared_source, profile_review=shared_review)
        job_path.write_text(json.dumps(job, indent=2)+'\n')
        report = json.loads((output/mode/'report.json').read_text())
        results[mode] = dict(report=f'{mode}/report.json', accepted_indices=report['accepted_indices'],
                             defect_detection=report.get('defect_detection'))
        warnings.extend(report.get('warnings', []))
    base = job.get('output_base_name') or 'CoronaHDR'
    for mode in ('scientific', 'aesthetic'):
        results[mode]['named_masters'] = (list(final_output_names(mode, base, job.get('output_format', 'both')))
            if job.get('publication') == 'standalone_final_files_v1' else named_masters(output/mode, output, base, mode))
    summary = dict(folder_processing=dict(hdr_mode='both', output_base_name=base, alignment=requested_alignment,
                                          output_format=job.get('output_format', 'both')), results=results,
                   warnings=list(dict.fromkeys(warnings)))
    (output/'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    job['warnings'] = summary['warnings']
    job_path.write_text(json.dumps(job, indent=2)+'\n')


def run_folders(job_path):
    from .aesthetic_pipeline import build_aesthetic_hdr
    job_path = Path(job_path)
    job = json.loads(job_path.read_text())
    mode = job.get('hdr_mode', 'aesthetic')
    alignment = job.get('alignment', 'conservative')
    if alignment not in ('conservative', 'ael_visual') or (alignment == 'ael_visual' and mode == 'scientific'):
        raise ValueError('AEL-style visual alignment is available only for aesthetic output')
    if mode == 'both':
        return run_both_folders(job_path, job)
    if mode not in ('aesthetic', 'scientific'):
        raise ValueError('HDR mode must be aesthetic or scientific')
    warnings = []

    def warn(message):
        if message not in warnings:
            warnings.append(message)
            print(f'WARNING: {message}', flush=True)

    if mode == 'aesthetic':
        warn('Aesthetic HDR uses empirical exposure/background matching; the output is not calibrated radiance.')
    else:
        warn('Scientific mode enforces radiometric rejection; it does not provide absolute radiometric calibration or a measured noise model.')
    print('Inspecting image folders and checking acquisition profile…', flush=True)
    reused_registration = bool(job.get('profile_source'))
    if not job.get('profile_source'):
        from .canon import prepare_canon
        if job.get('bias_fallback') and not job['folders'].get('bias'):
            raise ValueError('For a new acquisition, select actual bias frames in the bias folder; dark roles are not inferred')
        family = job.get('camera_family', 'Canon')
        with stage('preparation and initial registration'):
            source, review = prepare_canon(job['folders'], job['work'], job_path.parent/'acquisition',
                                           camera_family=family,
                                           **({'resume_from': job['resume_acquisition']} if job.get('resume_acquisition') else {}))
        job.update(profile_source=str(source.resolve()), profile_review=str(review.resolve()))
        job_path.write_text(json.dumps(job, indent=2)+'\n')
    indices, audit = match_profile(job['folders'], job['profile_source'], job['work'])
    saved_report = json.loads((Path(job['profile_source'])/'report.json').read_text())
    rejected = saved_report.get('rejected_frames', [])
    if rejected and reused_registration and alignment == 'conservative' and not job.get('_shared_geometry'):
        from .decisions import ask
        choice = ask(f'Saved registration excluded {len(rejected)} frames. Retry registration with larger shifts '
            'using the cached preparation? Quality checks remain enforced; a smaller shared field may be needed.',
            ['Retry registration with larger shifts', 'Keep saved registration'], 'Keep saved registration')
        if choice == 'Retry registration with larger shifts':
            from uuid import uuid4
            from .pipeline import run_manifest
            from .canon import write_translation_review
            retry = job_path.parent/f'registration-retry-{uuid4().hex[:12]}'
            recipe = json.loads(json.dumps(saved_report['recipe']))
            recipe['max_shift'] = max(recipe.get('max_shift', 256), max(saved_report['output']['shape_yx']))
            options = recipe.setdefault('processing', {})
            for key in ('rigid_review', 'registration_review'):
                options.pop(key, None)
            options.update(mode='disk', cache_from=job['profile_source'], include_indices=indices,
                           work_directory=str(retry/'work'), rejection_policy='exclude', decision_stage='Registration retry')
            manifest = retry.with_suffix('.json')
            manifest.write_text(json.dumps(recipe, indent=2)+'\n')
            updated = run_manifest(manifest, retry)
            review_path = retry/'geometry.json'
            write_translation_review(updated, recipe, review_path)
            job.update(profile_source=str(retry.resolve()), profile_review=str(review_path.resolve()))
            job.pop('ael_visual_review', None)
            # A changed registration requires a fresh HDR, even if an older result completed.
            job.pop('previous_result', None)
            job_path.write_text(json.dumps(job, indent=2)+'\n')
    if job.get('resumed_from') and job.get('previous_result'):
        from .streaming import load_cached_run
        saved = json.loads((Path(job['profile_source'])/'report.json').read_text())
        load_cached_run(Path(job['profile_source']), saved['recipe'])
        previous = Path(job['previous_result'])
        if (previous/'report.json').is_file() and all((previous/name).is_file() for name in ('master.fits', 'master.tiff')):
            try:
                completed = json.loads((previous/'report.json').read_text())
            except json.JSONDecodeError:
                completed = {}
            if 'folder_processing' in completed and (not job.get('save_intermediates') or 'intermediates' in completed):
                job['result'] = str(previous)
                job_path.write_text(json.dumps(job, indent=2)+'\n')
                print('Resumed completed HDR; publishing saved result', flush=True)
                return
    aliases = [role for role in ('dark', 'dark_flat') if audit.get(role) and
               all(item.get('effective_role') == 'bias' for item in audit[role])]
    if aliases:
        job['bias_fallback'] = True
        warn(f'Validated bias files selected as {", ".join(aliases)} are used as pedestal-only substitutes, once; no thermal-dark or hot-pixel correction is measured.')
    if job.get('bias_fallback'):
        if not job['folders'].get('bias') and not aliases:
            raise ValueError('Bias fallback requires validated bias frames')
        warn('Bias used as the missing-dark fallback: pedestal correction only, applied once; bias alone does not correct thermal dark signal or hot pixels.')
    else:
        if not job['folders'].get('dark'):
            warn('No thermal-dark correction. Bias subtraction, when supplied, removes only the pedestal.')
        if not job['folders'].get('dark_flat'):
            warn('No dark-flat correction. Flat thermal signal may remain after bias correction.')
    job['bias_folder_aliases'] = aliases
    if alignment == 'ael_visual':
        warn('AEL-style visual alignment is experimental; it does not include independent graph or lens-distortion validation and is not scientific geometry.')
        if not job.get('ael_visual_review'):
            from uuid import uuid4
            from .ael_visual import write_review
            review_path = job_path.parent/f'ael-visual-geometry-{uuid4().hex[:12]}.json'
            with stage('visual registration and frame review'):
                write_review(job['profile_source'], review_path, indices=indices)
            job['ael_visual_review'] = str(review_path.resolve())
        job['profile_review'] = job['ael_visual_review']
        job_path.write_text(json.dumps(job, indent=2)+'\n')
    review = json.loads(Path(job['profile_review']).read_text())
    accepted = [i for i in indices if str(i) in review['matrices_yx']]
    overrides = [i for i in accepted if review.get('metrics', {}).get(str(i), {}).get('user_decision') == 'Include experimental frame']
    report = json.loads((Path(job['profile_source'])/'report.json').read_text())
    reference = report.get('reference', report['recipe'].get('reference'))
    if reference not in accepted:
        raise ValueError(f'The verified reference frame {reference} is absent from the lights folder')
    if alignment == 'conservative':
        from .decisions import select_qualified_frames
        reasons = {item['index']: item for item in report.get('rejected_frames', [])}
        rows = []
        for i in indices:
            rejected = reasons.get(i, {})
            diagnostic = rejected.get('diagnostic', {})
            metric = review.get('metrics', {}).get(str(i), {})
            radiometry = diagnostic or report.get('registration', {}).get('reference_radiometry', {}).get(str(i), {})
            rows.append(dict(index=i, filename=Path(report['inputs'][i]['path']).name,
                exposure_s=report['inputs'][i]['metadata'].get('exposure_s'), reference=i == reference,
                status='accepted' if i in accepted else 'rejected',
                reason='Passed preparation checks' if i in accepted else rejected.get('reason', 'No qualified geometry'),
                measurements=dict(correlation=metric.get('correlation'),
                    median_error=radiometry.get('raw_relative_error'),
                    regional_error=max(radiometry['raw_regional_relative_errors'], default=None)
                        if radiometry.get('raw_regional_relative_errors') else None)))
        accepted = select_qualified_frames(rows)
        if len(accepted) < 2 and len(indices) > 1:
            raise ValueError('Select at least two qualified frames to build an HDR')
    if job.get('detect_hot_pixels'):
        from .defects import build_profile
        warn('Automatic defect detection is conservative and requires sensor-fixed recurrence plus scene displacement. Selective repair is interpolated data, not thermal calibration.')
        if job.get('defect_detection') is not None:
            previous_selection = job.get('defect_selection', job.get('accepted_indices'))
            if previous_selection is None or set(previous_selection) != set(accepted):
                raise ValueError('Saved hot-pixel detection used a different or unrecorded frame selection. Start a new job to change it.')
            detection = job['defect_detection']
            print('Reused completed defect handling', flush=True)
        else:
            with stage('hot-pixel detection and preparation'):
                profile, detection = build_profile(job['profile_source'], job['profile_review'], accepted,
                                                  job_path.parent/'defect-profile', mode, work_directory=job['work'])
            job['original_profile_source'] = job['profile_source']
            job['profile_source'] = str(profile.resolve())
            job['defect_detection'] = detection
            job['defect_selection'] = accepted
            job_path.write_text(json.dumps(job, indent=2)+'\n')
        warn(f'{detection["count"]} persistent detector defects confirmed; '
             f'{"masked" if mode == "scientific" else "selectively repaired where valid"} in the native grid before geometric resampling.')
        if detection.get('skipped_reason'):
            warn(detection['skipped_reason'])
        if detection['unconfirmed_motion_count']:
            warn(f'{detection["unconfirmed_motion_count"]} recurring candidates left unchanged because independent scene displacement is insufficient.')
    job.update(folder_inputs=audit, accepted_indices=accepted,
               rejected_indices=[i for i in indices if i not in accepted],
               method='aesthetic' if mode == 'aesthetic' else 'weighted', warnings=warnings)
    job_path.write_text(json.dumps(job, indent=2)+'\n')
    print(f'Using {len(accepted)} selected frames out of {len(indices)} inputs', flush=True)
    if len(accepted) != len(indices):
        warn(f'{len(indices)-len(accepted)} frames excluded by preparation checks or user selection; see the frame review and report for reasons.')
    progress_stage(mode)
    if mode == 'aesthetic':
        with stage('aesthetic HDR'):
            result = build_aesthetic_hdr(job['profile_source'], job['profile_review'], job['result'], indices=accepted,
                **({'lunar_margin': job['lunar_margin_px']} if job.get('lunar_margin_px') is not None else {}),
                work_directory=job_path.parent/'work')
    else:
        from .pipeline import run_manifest
        recipe = json.loads(json.dumps(report['recipe']))
        recipe.update(method='weighted', reference=reference)
        recipe['processing'] = dict(recipe.get('processing', {}), mode='disk',
            cache_from=job['profile_source'], rigid_review=job['profile_review'],
            include_indices=accepted, rejection_policy='exclude', work_directory=str(job_path.parent/'work'),
            decision_stage='Scientific HDR')
        if job.get('lunar_margin_px') is not None:
            recipe['processing']['lunar_margin_px'] = job['lunar_margin_px']
        path = job_path.parent/'recipe.json'
        path.write_text(json.dumps(recipe, indent=2)+'\n')
        print('Checking scientific radiometric consistency; incompatible exposures will be excluded…', flush=True)
        with stage('scientific HDR'):
            result = run_manifest(path, job['result'])
    included = result['accepted_indices']
    if job.get('_preview_main', True):
        preview_ready('corona', Path(job['result'])/'master.tiff')
    if set(accepted)-set(included):
        warn(f'{len(set(accepted)-set(included))} additional frames rejected by radiometric quality checks.')
    from astropy.io import fits
    import numpy as np
    for filename, description in [('unresolved_saturation', 'pixels saturated in every eligible observation; no values invented'),
                                  ('support', 'pixels without valid observations; consult the support mask')]:
        with fits.open(Path(job['result'])/f'{filename}.fits', memmap=True) as hdus:
            count = int(np.count_nonzero(hdus[0].data if filename != 'support' else hdus[0].data == 0))
        if count:
            warn(f'{count} {description}.')
    for message in result.get('warnings', []):
        warn(str(message))
    if overrides:
        warn(f'{len(overrides)} frame(s) were included by user despite AEL-style alignment criteria; '
             'inspect coronal detail and hot-pixel handling carefully. See the geometry review for measurements.')
    if alignment == 'ael_visual':
        warn(f'AEL-style visual result uses {len(included)} of {len(indices)} selected frames. '
             'Inspect coronal detail and the lunar limb for misalignment or unsupported gaps; '
             'the alignment has no independent graph validation.')
    job.update(warnings=warnings, accepted_indices=included,
               rejected_indices=sorted(set(indices)-set(included)))
    job_path.write_text(json.dumps(job, indent=2)+'\n')
    result['folder_processing'] = dict(hdr_mode=mode, bias_fallback=job.get('bias_fallback', False),
        alignment=alignment,
        bias_folder_aliases=aliases,
        calibration_policy='bias pedestal only; no thermal-dark substitution' if job.get('bias_fallback') else 'verified preparation',
        warnings=warnings)
    if job.get('publication') == 'standalone_final_files_v1':
        from .desktop_jobs import final_output_names
        base = job.get('output_base_name') or 'CoronaHDR'
        result['folder_processing']['output_base_name'] = base
        result['folder_processing']['output_format'] = job.get('output_format', 'both')
        result['folder_processing']['named_masters'] = list(final_output_names(mode, base, job.get('output_format', 'both')))
    elif job.get('output_base_name'):
        from .desktop_jobs import named_masters
        result['folder_processing']['output_base_name'] = job['output_base_name']
        result['folder_processing']['named_masters'] = named_masters(job['result'], job['result'],
                                                                    job['output_base_name'], mode)
    if 'defect_detection' in job:
        result['defect_detection'] = job['defect_detection']
        result['defect_profile'] = job['profile_source']
    result['warnings'] = list(dict.fromkeys([*result.get('warnings', []), *warnings]))
    if job.get('save_intermediates'):
        from .intermediates import export_intermediates
        print('Saving requested linear intermediate images in the temporary job folder…'
              if job.get('publication') == 'standalone_final_files_v1' else
              'Saving requested linear intermediate images in the output folder…', flush=True)
        result['intermediates'] = export_intermediates(job['profile_source'], result, job['result'],
            indices, job['save_intermediates'], warn=warn)
        job_path.write_text(json.dumps(job, indent=2)+'\n')
        result['warnings'] = list(dict.fromkeys([*result.get('warnings', []), *warnings]))
    (Path(job['result'])/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

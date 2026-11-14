"""Reusable empirical HDR from a prepared run and reviewed rigid geometry."""
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np

from coronahdr.aesthetic import apply_match, feather_reliability, fit_green_gain
from coronahdr.empirical import fit_empirical_match
from coronahdr.io import write_image
from coronahdr.rigid import apply_points, lower_matrix
from coronahdr.recovery import analysis_image
from coronahdr.streaming import accumulate_tiled, cleanup_scratch, load_cached_run, mapped, release_pages, sample_band
from coronahdr.performance import stage


def mean4(a):
    h, w = a.shape[:2]
    h, w = h//4*4, w//4*4
    return a[:h, :w].reshape(h//4, 4, w//4, 4, *a.shape[2:]).mean(axis=(1, 3))


def select_frames(count, reference, matrices, indices=None):
    selected = sorted(int(i) for i in matrices) if indices is None else list(indices)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Select at least one frame; duplicate indices are not allowed')
    if reference not in selected:
        raise ValueError(f'Selection must include the prepared reference frame {reference}')
    for i in selected:
        if type(i) is not int or not 0 <= i < count or str(i) not in matrices:
            raise ValueError(f'Frame {i} has no reliable reviewed geometry or is outside the input list')
        matrix = np.asarray(matrices[str(i)], dtype=float)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or
                not np.allclose(matrix[2], [0, 0, 1]) or
                not np.allclose(matrix[:2, :2].T@matrix[:2, :2], np.eye(2), atol=1e-5) or
                not np.isclose(np.linalg.det(matrix[:2, :2]), 1, atol=1e-5)):
            raise ValueError(f'Frame {i}: expected a finite rigid source-to-reference matrix')
    return selected


def build_aesthetic_hdr(source, review_path, output, *, indices=None, feather=192, weight_sigma=8, lunar_margin=None, work_directory=None):
    from .lunar import apply_margin, validate_margin
    validate_margin(lunar_margin)
    if sys.version_info[:3] != (3, 14, 8):
        raise ValueError('CoronaHDR requires Python 3.14.8')
    source, review_path, output = Path(source).resolve(), Path(review_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    old = json.loads((source/'report.json').read_text())
    if lunar_margin is None:
        lunar_margin = old.get('output', {}).get('lunar_margin_px')
        validate_margin(lunar_margin)
    if not np.isfinite([feather, weight_sigma]).all() or feather <= 0 or weight_sigma < 0:
        raise ValueError('Feather must be positive and weight sigma nonnegative; both must be finite')
    review = json.loads(review_path.read_text())
    if review['input_source_sha256'] != [a['metadata']['source_sha256'] for a in old['inputs']]:
        raise ValueError('Geometry source hashes do not match')
    reference_index = old.get('reference', old['recipe'].get('reference'))
    selected = select_frames(len(old['inputs']), reference_index, review['matrices_yx'], indices)
    preparation = Path(old.get('preparation_directory', source/'work'))
    if any(not (preparation/f'prepared_{i:03d}'/'data.npy').exists() for i in selected):
        raise ValueError('Aesthetic export requires an existing disk-backed prepared cache')
    frames, _, _ = load_cached_run(source, old['recipe'])
    poses = {i: np.linalg.inv(review['matrices_yx'][str(reference_index)])@np.asarray(review['matrices_yx'][str(i)]) for i in selected}
    images, masks = {}, {}
    for i in selected:
        if frames[i].metadata.get('state') != 'linear_rgb' or frames[i].data.ndim != 3 or frames[i].data.shape[-1] != 3:
            raise ValueError(f'Frame {i}: aesthetic matching currently requires prepared linear RGB')
        if frames[i].metadata.get('lunar_disk') is None:
            raise ValueError(f'Frame {i}: declare lunar_disk geometry before coronal matching')
        images[i], masks[i] = analysis_image(frames[i], 8)
        release_pages(frames[i])
        print('Prepared matching auxiliary', i, flush=True)
    y, x = np.indices(masks[reference_index].shape)
    ny, nx = y*8+3.5, x*8+3.5
    cx, cy, radius = frames[reference_index].metadata['lunar_disk']
    ref_radius = np.hypot(ny-cy, nx-cx)/radius
    reference = images[reference_index]
    params, diagnostics = [], []
    domain = [reference.shape[0]//4*32, reference.shape[1]//4*32]
    for i in selected:
        h, w = frames[i].data.shape[:2]
        low = lower_matrix(poses[i], 8, [0, 0, w, h])
        aligned = sample_band(images[i], low, 0, len(y), y.shape[1])
        good = (sample_band(masks[i], low, 0, len(y), y.shape[1]) > 1-1e-9) & masks[reference_index]
        mx, my, mr = frames[i].metadata['lunar_disk']
        my, mx = apply_points(poses[i], [[my, mx]])[0]
        r = np.hypot(ny-my, nx-mx)/mr
        good &= (r > 1.5) & (ref_radius > 1.5) & (r < 4) & (ref_radius < 4)
        if i == reference_index:
            p = np.array([[1., 0, 0, 0]]*3)
            diagnostic = dict(reference=True)
        else:
            # Fit green directly so a noisy unused blue fit cannot veto shared gain.
            shared_gain, diagnostic = fit_green_gain(mean4(reference), mean4(aligned), mean4(good.astype(float)) == 1)
            _, plane_fit = fit_empirical_match(mean4(reference), mean4(aligned)*shared_gain,
                                                mean4(good.astype(float)) == 1, 'plane')
            diagnostic = dict(green_gain_fit=diagnostic, shared_gain_source='green channel', plane_fit=plane_fit)
            p = np.column_stack([np.full(3, shared_gain), np.asarray(plane_fit['parameters'])])
            if np.any(p[:, 0] < .7) or np.any(p[:, 0] > 1.4):
                raise ValueError(f'Excessive empirical gain in input {i}: {p[:, 0]}')
        params.append(p)
        diagnostics.append(dict(input_index=i, parameters=p.tolist(), fit=diagnostic))
        print('Empirical match', i, p[:, 0], flush=True)
    del images, masks
    output.mkdir(parents=True)
    work = Path(work_directory).resolve() if work_directory is not None else output/'work'
    work.mkdir(parents=True)
    frames, lunar_masks = apply_margin(frames, lunar_margin, work)
    (output/'matching.json').write_text(json.dumps(dict(domain_yx=domain, records=diagnostics), indent=2)+'\n')
    chosen = []
    for i in selected:
        frame = frames[i]
        eligible = frame.auxiliary['coverage'] & ~frame.auxiliary['lunar'] & ~frame.auxiliary['saturation']
        weights = mapped(work/f'feather_{i:03d}.npy', eligible.shape, np.float32)
        weights[:] = feather_reliability(frame.auxiliary['soft_weight'], eligible, feather, weight_sigma)
        weights.flush()
        chosen.append(replace(frame, auxiliary=dict(frame.auxiliary, soft_weight=weights)))
    if review.get('alignment') == 'ael_visual':
        from .streaming import output_canvas
        origin, shape = output_canvas(chosen, [poses[i] for i in selected], 'union')
        origin = np.array(origin)
        canvas = 'union'
    else:
        origin = np.array(old['output']['origin_reference_yx'])
        shape = tuple(old['output']['shape_yx'])
        canvas = 'fixed_source_run'
    if canvas == 'fixed_source_run' and all(np.allclose(poses[i][:2, :2], np.eye(2), atol=1e-10) for i in selected):
        from .streaming import output_canvas
        from .decisions import ask
        try:
            crop_origin, crop_shape = output_canvas(chosen, [poses[i][:2, 2] for i in selected], 'intersection')
        except ValueError:
            pass
        else:
            choice = ask(f'Aesthetic HDR: the saved geometry supports a shared detector field of {crop_shape[1]} × {crop_shape[0]} pixels. '
                'Crop the final HDR? Lunar and saturation gaps can remain.',
                ['Crop to shared field', 'Keep saved canvas'], 'Keep saved canvas')
            if choice == 'Crop to shared field':
                origin, shape, canvas = np.array(crop_origin), crop_shape, 'intersection'
    def correct(index, values, start, end):
        yy, xx = np.meshgrid(np.arange(start, end)+origin[0], np.arange(shape[1])+origin[1], indexing='ij')
        return apply_match(values, params[index], yy, xx, domain)
    with stage('aesthetic integration and mask export'):
        master = accumulate_tiled(chosen, [poses[i] for i in selected], origin, shape, output, work, 96, radiance_correction=correct)
    metadata = dict(old.get('output', {}), method='aesthetic_weighted_empirical',
                    canvas=canvas, origin_reference_yx=origin.tolist(), shape_yx=list(shape),
                    qualification='Visual-quality priority; empirical response and background matching, not calibrated radiance',
                    matching='shared green-derived gain plus RGB additive planes at bin32',
                    feather_native_px=feather, weight_smoothing_sigma_native_px=weight_sigma,
                    reference=reference_index, normalization='none', stretch='none')
    with stage('aesthetic master export'):
        for extension in ('fits', 'tiff'):
            write_image(output/f'master.{extension}', master, metadata)
    transforms = [dict(index=i, matrix_yx=poses[i].tolist()) for i in selected]
    warnings = [*old.get('warnings', []), 'Aesthetic priority explicitly authorized; strict radiometric matching gates do not qualify this mode.',
                'Empirical planes may change broad coronal brightness and colour; original master retained.',
                'Only measured coverage contributes; no filled lunar or saturated gaps; no absolute radiance claim.',
                ('Canvas cropped to the shared detector footprint; masked gaps can remain.' if canvas == 'intersection' else
                 'Canvas includes the selected transformed detector footprints; masked gaps can remain.' if canvas == 'union' else
                 'Fixed canvas matches the source run, not the full selected-frame footprint union.'),
                'Noise improvement and fine-detail retention require measured comparison; no pixel denoising applied.']
    rejected = [dict(index=i, reason='No reliable reviewed coronal geometry') for i in range(len(old['inputs'])) if str(i) not in review['matrices_yx']]
    result = dict(old, reference=reference_index, output=metadata, accepted_indices=selected, requested_indices=selected, rejected_frames=rejected, transformations=transforms,
                  registration=dict(reference=reference_index, accepted_indices=selected,
                                    transformations={str(t['index']): t for t in transforms}, geometry_source=str(review_path)),
                  excluded_indices=[i for i in range(len(old['inputs'])) if i not in selected],
                  source_product=str(source), aesthetic_matching=diagnostics,
                  recipe=dict(old['recipe'], method='aesthetic_weighted_empirical',
                              aesthetic=dict(source_run=str(source), geometry_review=str(review_path),
                                             feather_native_px=feather, weight_smoothing_sigma_native_px=weight_sigma,
                                             work_directory=str(work)),
                              processing=dict(old['recipe'].get('processing', {}), include_indices=selected)),
                  memory=dict(old.get('memory', {}), tile_rows=96, work_directory=str(work)), warnings=warnings,
                  geometry_audit=str(review_path), geometry_audit_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
                  source_report_sha256=hashlib.sha256((source/'report.json').read_bytes()).hexdigest(),
                  auxiliary_provenance='Rebuilt from source-hash-verified prepared RGB; factor8 means, factor32 matching',
                  python='3.14.8', executable=sys.executable,
                  versions={name: version(name) for name in ('coronahdr', 'numpy', 'scipy', 'scikit-image', 'astropy', 'tifffile', 'rawpy')})
    if lunar_margin is not None:
        result['output']['lunar_margin_px'] = lunar_margin
        result['recipe']['aesthetic']['lunar_margin_px'] = lunar_margin
        result['lunar_margin_masks'] = lunar_masks
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'recipe.json').write_text(json.dumps(result['recipe'], indent=2, allow_nan=False)+'\n')
    (output/'registration.json').write_text(json.dumps(result['registration'], indent=2)+'\n')
    master.flush()
    master._mmap.close()
    cleanup_scratch(work)
    for i in selected:
        (work/f'feather_{i:03d}.npy').unlink()
    return result

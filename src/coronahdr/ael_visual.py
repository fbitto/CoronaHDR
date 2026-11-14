"""Create an AEL-style visual registration review from prepared camera frames."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from .ael_ecc import register
from .decisions import review_frames
from .registration import valid_mask
from .rigid import apply_points, control_points, lower_matrix, matrix_distance, measure_rigid
from .performance import detector_workers
from .streaming import load_cached_run, registration_auxiliary, release_pages


def check_rigid_agreement(reference, moving, matrix, factor, roi):
    """Check an ECC pose against an independently fitted, spatially split rigid pose."""
    reduced = lower_matrix(matrix, factor, roi)
    center = (np.array(reference.data.shape[:2])-1)/2
    initial = (apply_points(reduced, [center])[0]-center)*factor
    measured, quality = measure_rigid(reference, moving, initial, factor, roi, 1.)
    error = factor*matrix_distance(reduced, lower_matrix(measured, factor, roi), control_points(reference.data.shape[:2]))
    return dict(disagreement_native_px=error, fit=quality, passed=error <= 1.)


def write_review(source, output, *, indices=None):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((source/'report.json').read_text())
    frames, _, _ = load_cached_run(source, report['recipe'])
    roi = report['recipe']['processing']['registration_roi_xywh']
    ref = report['reference']
    requested = list(range(len(frames))) if indices is None else list(indices)
    if ref not in requested or len(set(requested)) != len(requested) or any(type(i) is not int or not 0 <= i < len(frames) for i in requested):
        raise ValueError('Visual registration selection must contain unique valid indices and the reference')
    if any(frame.data.ndim != 3 or frame.data.shape[-1] != 3 for frame in frames):
        raise ValueError('AEL-style folder alignment requires prepared RGB camera frames')
    def auxiliary(index, factor=2):
        frame = frames[index]
        green = replace(frame, data=frame.data[..., 1], metadata=dict(frame.metadata, state='linear_mono'))
        result = registration_auxiliary(green, factor, roi)
        release_pages(frame)
        return result
    def moon(index):
        disk = report['inputs'][index]['metadata']['lunar_disk']
        if disk is None:
            raise ValueError('AEL-style alignment requires a measured lunar exclusion for each frame')
        return dict(xy=[(disk[0]-roi[0]-.5)/2, (disk[1]-roi[1]-.5)/2], r=disk[2]/2)
    reference = auxiliary(ref)
    ref_valid = valid_mask(reference)
    check_factor = report['recipe']['processing']['registration_bin']
    check_reference = auxiliary(ref, check_factor)
    swap = np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]])
    poses, metrics, edges, rejected, candidates = {}, {}, [], [], {}
    workers = detector_workers(reference.data.shape, len(requested))
    print(f'Visual registration workers: {workers}', flush=True)
    def fit(index):
        try:
            moving = reference if index == ref else auxiliary(index)
            moving_valid = ref_valid if index == ref else valid_mask(moving)
            xy, quality = register(reference.data, moving.data, moon(ref), moon(index),
                is_reference=index == ref, reference_valid=ref_valid, moving_valid=moving_valid)
            affine = np.vstack([xy,[0.,0.,1.]])
            origin = np.array([roi[0],roi[1]])
            affine[:2,2] += origin-affine[:2,:2]@origin
            matrix = np.linalg.inv(swap@affine@swap)
            # OpenCV returns float32 rotations; project roundoff onto a proper rigid matrix.
            u, _, vt = np.linalg.svd(matrix[:2,:2])
            matrix[:2,:2] = u@vt
            if np.linalg.det(matrix[:2,:2]) <= 0:
                raise ValueError('Invalid AEL-style rigid rotation')
        except ValueError as error:
            if index == ref:
                raise ValueError(f'AEL-style reference registration failed: {error}') from error
            return dict(error=str(error))
        if index != ref:
            try:
                check = check_rigid_agreement(check_reference, auxiliary(index, check_factor), matrix, check_factor, roi)
            except ValueError as error:
                check = dict(passed=False, reason=str(error))
            quality['independent_check'] = check
        else:
            check = dict(passed=True)
        return dict(matrix=matrix, quality=quality, check=check)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, result in zip(requested, executor.map(fit, requested)):
            if 'error' in result:
                reason = result['error']
                rejected.append(dict(index=index, reason=reason))
                edges.append(dict(parent=ref, child=index, accepted=False, reason=reason))
                print(f'WARNING: AEL-style registration excluded frame {index}: {reason}', flush=True)
                continue
            matrix, quality, check = result['matrix'], result['quality'], result['check']
            if quality['status'] != 'registered' or not check['passed']:
                if index == ref:
                    raise ValueError('AEL-style reference registration did not pass its checks')
                reason = (f'held-out correlation {quality["heldout_correlation"]:.3f} (minimum 0.650), '
                          f'rotation {quality["angle_deg"]:.3f}° (maximum absolute 0.500°)')
                if not check['passed']:
                    reason += ('; independent rigid check: ' + (check.get('reason') or
                        f'{check["disagreement_native_px"]:.3f} native pixels disagreement (maximum 1.000)'))
                candidates[index] = (matrix, quality, reason)
                print(f'AEL-style visual registration {index+1}/{len(frames)}: candidate — {reason}', flush=True)
                continue
            poses[str(index)] = matrix.tolist()
            metrics[str(index)] = dict(path=[index] if index == ref else [index, ref],
                reference=index == ref, matrix_yx=matrix.tolist(), **quality)
            if index != ref:
                edges.append(dict(parent=ref, child=index, accepted=True, quality=quality))
            print(f'AEL-style visual registration {index+1}/{len(frames)}: accepted', flush=True)
    reasons = {entry['index']: entry['reason'] for entry in rejected}
    rows = [dict(index=i, filename=Path(report['inputs'][i]['path']).name,
                 status='candidate' if i in candidates else 'accepted' if str(i) in poses else 'rejected',
                 reason=candidates[i][2] if i in candidates else reasons.get(i, 'Passed visual alignment checks'),
                 exposure_s=report['inputs'][i]['metadata'].get('exposure_s'),
                 reference=i == ref,
                 measurements=candidates[i][1] if i in candidates else metrics.get(str(i), {}))
            for i in requested]
    selected_candidates = set(review_frames(rows))
    for index, (matrix, quality, reason) in candidates.items():
        choice = 'Include experimental frame' if index in selected_candidates else 'Exclude frame'
        quality = dict(quality, user_decision=choice, rejection_criteria=reason)
        if index in selected_candidates:
            poses[str(index)] = matrix.tolist()
            metrics[str(index)] = dict(path=[index, ref], reference=False, matrix_yx=matrix.tolist(), **quality)
            edges.append(dict(parent=ref, child=index, accepted=True, quality=quality))
            print(f'WARNING: Frame {index} included by user despite AEL-style alignment criteria: {reason}', flush=True)
        else:
            rejected.append(dict(index=index, reason=reason, candidate_matrix_yx=matrix.tolist(), quality=quality))
            edges.append(dict(parent=ref, child=index, accepted=False, reason=reason, quality=quality))
            print(f'WARNING: AEL-style registration excluded frame {index}: {reason}', flush=True)
    if len(poses) < 2 and len(requested) > 1:
        raise ValueError('Fewer than two frames passed AEL-style visual registration')
    review = dict(input_source_sha256=[item['metadata']['source_sha256'] for item in report['inputs']],
        reference=ref, matrices_yx=poses, metrics=metrics, edges=edges, rejected_frames=rejected,
        factor=report['recipe']['processing']['registration_bin'], roi_xywh=roi,
        alignment='ael_visual',
        qualification='AEL-style visual alignment; no independent graph or distortion correction; not scientific geometry')
    with output.open('x') as stream:
        stream.write(json.dumps(review, indent=2, allow_nan=False)+'\n')
    return output

"""Conservative persistent detector outliers in native CFA, mono and RGB grids."""
import json
import shutil
import hashlib
import platform
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, median_filter, uniform_filter


def candidates(data, valid, *, cfa=True):
    """Test each phase/channel independently; isolated peaks only."""
    if data.shape != valid.shape or data.ndim not in (2, 3) or (data.ndim == 3 and data.shape[-1] != 3) or (cfa and data.ndim != 2):
        raise ValueError("Defect detection requires matching native samples and validity masks")
    if data.ndim == 3:
        pairs = [candidates(data[..., c], valid[..., c], cfa=False) for c in range(data.shape[-1])]
        return tuple(np.stack([pair[k] for pair in pairs], axis=-1) for k in range(2))
    stride = 2 if cfa else 1
    found = np.zeros(data.shape, bool)
    eligible = np.zeros(data.shape, bool)
    footprint = np.ones((5, 5), bool)
    footprint[2, 2] = False
    for y in range(stride):
        for x in range(stride):
            a = np.asarray(data[y::stride, x::stride])
            v = valid[y::stride, x::stride] & np.isfinite(a)
            safe = np.where(v, a, 0.)
            baseline = median_filter(safe, footprint=footprint, mode='constant')
            residual = a-baseline
            good = uniform_filter(v.astype(float), size=5, mode='constant') >= 1-1e-12
            if not good.any():
                continue
            center = np.median(residual[good])
            noise = 1.4826*np.median(np.abs(residual[good]-center))
            local = 1.4826*median_filter(np.abs(np.nan_to_num(residual)), size=5, mode='constant')
            floor = np.finfo(float).eps*max(float(np.median(np.abs(a[good]))), 1.)*32
            threshold = np.maximum(np.maximum(12*local, 12*noise), np.maximum(.02*np.abs(baseline), floor))
            neighbor_max = maximum_filter(safe, footprint=footprint, mode='constant')
            found[y::stride, x::stride] = good & (residual > threshold) & (a-neighbor_max > threshold)
            eligible[y::stride, x::stride] = good
    return found, eligible


def confirm(records, matrices, shape, min_views=3, fraction=.6, min_motion=8.):
    """Require recurring sensor coordinates and scene displacement among actual hits."""
    if len(records) != len(matrices) or any(np.asarray(a).shape != (3, 3) or not np.isfinite(a).all() for a in matrices):
        raise ValueError('Each defect observation requires a finite source-to-reference 3x3 matrix')
    hits = np.zeros(shape, np.uint16)
    support = np.zeros(shape, np.uint16)
    sparse = []
    for detected, valid in records:
        hits += detected
        support += valid
        sparse.append(np.flatnonzero(detected))
    ids = np.flatnonzero((hits >= min_views) & (hits >= fraction*support))
    coordinates = np.unravel_index(ids, shape)
    y, x = coordinates[:2]
    points = np.array([y, x, np.ones(len(ids))])
    seen = [np.isin(ids, item, assume_unique=True) for item in sparse]
    positions = [np.asarray(matrix)@points for matrix in matrices]
    motion = np.zeros(len(ids))
    for i in range(len(records)):
        for j in range(i):
            distance = np.linalg.norm(positions[i][:2]-positions[j][:2], axis=0)
            motion = np.maximum(motion, np.where(seen[i] & seen[j], distance, 0))
    keep = motion >= min_motion
    pixels = np.column_stack([x[keep], y[keep]])
    return dict(pixels_xy=[list(point) for point in dict.fromkeys(map(tuple, pixels.tolist()))],
        samples_xyc=(np.column_stack([pixels, coordinates[2][keep]]).tolist() if len(shape) == 3 else []),
        hits=hits.flat[ids[keep]].tolist(), eligible_views=support.flat[ids[keep]].tolist(),
        motion_px=motion[keep].tolist(), unconfirmed_motion_count=int((~keep).sum()),
        parameters=dict(min_views=min_views, min_fraction=fraction, min_scene_motion_px=min_motion,
                        phase_neighborhood='5x5 same-phase samples, center excluded', threshold_sigma=12,
                        minimum_relative_peak=.02))


def confirm_by_exposure(records, matrices, shape, exposures, indices):
    """Each exposure tier must independently establish recurrence and motion."""
    if not (len(records) == len(matrices) == len(exposures) == len(indices)):
        raise ValueError('Exposure confirmation requires one exposure and index per observation')
    if any(not np.isfinite(t) or t <= 0 for t in exposures):
        raise ValueError('Defect exposure times must be finite and positive')
    pooled = confirm(records, matrices, shape)
    groups, union = [], {}
    for exposure in sorted(set(exposures)):
        members = [i for i, t in enumerate(exposures) if t == exposure]
        result = confirm([records[i] for i in members], [matrices[i] for i in members], shape)
        result.update(exposure_s=exposure, analyzed_indices=[indices[i] for i in members],
                      skipped_reason='Fewer than three observations in this exposure tier' if len(members) < 3 else None)
        groups.append(result)
        samples = result['samples_xyc'] if len(shape) == 3 else result['pixels_xy']
        for n, sample in enumerate(samples):
            union.setdefault(tuple(sample), (result['hits'][n], result['eligible_views'][n], result['motion_px'][n]))
    samples = list(union)
    pixels = list(dict.fromkeys(sample[:2] for sample in samples))
    return dict(pixels_xy=list(map(list, pixels)),
        samples_xyc=list(map(list, samples)) if len(shape) == 3 else [],
        hits=[union[s][0] for s in samples], eligible_views=[union[s][1] for s in samples],
        motion_px=[union[s][2] for s in samples],
        unconfirmed_motion_count=sum(g['unconfirmed_motion_count'] for g in groups),
        parameters=dict(pooled['parameters'], confirmation='independent exact exposure tiers; union of confirmed samples'),
        exposure_groups=groups, pooled_comparison=pooled,
        added_vs_pooled_xy=[list(p) for p in pixels if list(p) not in pooled['pixels_xy']],
        pooled_only_xy=[p for p in pooled['pixels_xy'] if tuple(p) not in pixels])


def repair_cfa(data, pixels, blocked, *, stride=2, require_outlier=False, donor_pixels=()):
    if data.ndim != 2 or blocked.shape != data.shape:
        raise ValueError('CFA repair requires matching two-dimensional samples and masks')
    for x, y in [*pixels, *donor_pixels]:
        if type(x) is not int or type(y) is not int or not (0 <= x < data.shape[1] and 0 <= y < data.shape[0]):
            raise ValueError('Defect repair coordinates must be in-bounds integer [x,y] pairs')
    result = data.copy()
    unusable = blocked | ~np.isfinite(data)
    for x, y in [*pixels, *donor_pixels]:
        unusable[y, x] = True
    detected = candidates(data, ~blocked, cfa=stride == 2)[0] if require_outlier else None
    repaired = []
    for x, y in pixels:
        if detected is not None and not detected[y, x]:
            continue
        if blocked[y, x] or not np.isfinite(data[y, x]):
            continue
        ys = np.arange(y-2*stride, y+2*stride+1, stride)
        xs = np.arange(x-2*stride, x+2*stride+1, stride)
        ys, xs = ys[(ys >= 0) & (ys < data.shape[0])], xs[(xs >= 0) & (xs < data.shape[1])]
        values = data[np.ix_(ys, xs)]
        valid = ~unusable[np.ix_(ys, xs)]
        if valid.sum() >= 8:
            result[y, x] = np.median(values[valid])
            repaired.append([x, y])
        else:
            result[y, x] = np.nan
    return result, repaired


def repair_linear(data, pixels, blocked, *, require_outlier=False, donor_pixels=(), samples=None):
    """Repair only currently isolated channels at confirmed native-grid pixels."""
    if data.ndim == 2:
        return repair_cfa(data, pixels, blocked, stride=1, require_outlier=require_outlier, donor_pixels=donor_pixels)
    result = data.copy()
    repaired = set()
    detected = candidates(data, ~blocked, cfa=False)[0] if samples is None else None
    for channel in range(data.shape[-1]):
        selected = ([[x, y] for x, y in pixels if detected[y, x, channel]] if samples is None else
                    [[x, y] for x, y, c in samples if c == channel])
        result[..., channel], done = repair_cfa(data[..., channel], selected, blocked[..., channel],
                                                stride=1, donor_pixels=donor_pixels)
        repaired.update(map(tuple, done))
    return result, sorted(map(list, repaired))


def build_profile(source, review, indices, destination, mode, *, apply_policy=True, work_directory=None):
    """Keep old preparations immutable; persist a separately qualified overlay."""
    from .calibration import calibrate, select_dark
    from .pipeline import load_item
    from .sensor import SensorFrame, level_map
    from .streaming import CachedFrame, cache_prepared, load_cached_run, release_pages
    source, destination = Path(source), Path(destination)
    from .work_paths import work_exclusion_root
    work = Path(work_directory).resolve() if work_directory is not None else work_exclusion_root()
    resolved_destination = destination.resolve()
    # macOS can expose the same accented folder with different Unicode spellings.
    if not work.is_dir() or not any(parent.is_dir() and parent.samefile(work)
            for parent in (resolved_destination, *resolved_destination.parents)):
        raise ValueError('Detector analysis and preparation must be stored in the selected work directory (external work directory by default)')
    if mode not in ('scientific', 'aesthetic'):
        raise ValueError('Defect policy must be scientific or aesthetic')
    destination.mkdir()
    old = json.loads((source/'report.json').read_text())
    geometry = json.loads(Path(review).read_text())
    if geometry.get('input_source_sha256') != [item['metadata']['source_sha256'] for item in old['inputs']]:
        raise ValueError('Defect geometry hashes do not match the original inputs')
    frames, _, masters = load_cached_run(source, old['recipe'])
    reliable = [i for i in indices if str(i) in geometry['matrices_yx']]
    if len(reliable) < 3:
        summary = dict(count=0, applied=False, mode=mode, analyzed_indices=reliable,
            pixels_xy=[], unconfirmed_motion_count=0,
            skipped_reason='Fewer than three reliably registered lights: automatic defect handling skipped; samples unchanged')
        (destination/'detection.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(f'WARNING: {summary["skipped_reason"]}', flush=True)
        return source, summary
    records, poses, signature, measurements = [], [], None, []
    from .performance import detector_workers
    detector_shape = frames[reliable[0]].data.shape
    if old['inputs'][reliable[0]]['metadata'].get('state') == 'linear_cfa':
        detector_shape = detector_shape[:2]
    workers = detector_workers(detector_shape, len(reliable))
    if shutil.disk_usage(destination).free < np.prod(detector_shape)*2*len(reliable):
        workers = 1
    print(f'Detector candidate scan workers: {workers}', flush=True)
    def scan(index):
        item = dict(old['recipe']['frames'][index], path=old['inputs'][index]['path'])
        frame = load_item(item, source)
        if not isinstance(frame, SensorFrame) or (frame.metadata['state'] != 'linear_cfa'
                and frame.metadata.get('defect_grid') != 'native_unregistered'):
            return None
        current = [frame.metadata.get(key) for key in ('sensor_id', 'iso', 'gain', 'active_area_xywh', 'cfa_pattern', 'state', 'defect_grid')]
        values, _ = calibrate(frame, masters.get('bias', [None])[0], select_dark(frame, masters.get('dark', [])), masters.get('flat', [None])[0])
        valid = np.isfinite(values) & (frame.data < level_map(frame, 'white_level')) & (values > frame.metadata['low_signal'])
        if frame.metadata.get('lunar_disk') is not None:
            cx, cy, radius = frame.metadata['lunar_disk']
            yy, xx = np.ogrid[:values.shape[0], :values.shape[1]]
            outside = (xx-cx)**2+(yy-cy)**2 > (radius+frame.metadata.get('limb_margin_px', 4)+4)**2
            valid &= outside[..., None] if values.ndim == 3 else outside
        detected, eligible = candidates(values, valid, cfa=frame.metadata['state'] == 'linear_cfa')
        # Boolean maps stay on the external disk between frames.
        pair = []
        for name, array in [('candidates', detected), ('eligible', eligible)]:
            path = destination/f'{name}_{index:03d}.npy'
            np.save(path, array)
            pair.append(path)
        return current, pair, dict(index=index, exposure_s=frame.metadata['exposure_s'],
            candidates=int(detected.sum()), eligible_samples=int(eligible.sum()),
            source_sha256=frame.metadata['source_sha256'])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, result in zip(reliable, executor.map(scan, reliable)):
            if result is None:
                summary = dict(count=0, applied=False, mode=mode, pixels_xy=[],
                    skipped_reason='Native detector metadata required; mono/RGB exports must declare defect_grid=native_unregistered. Registered, resized or stacked images cannot establish a sensor map')
                (destination/'detection.json').write_text(json.dumps(summary, indent=2)+'\n')
                return source, summary
            current, pair, measurement = result
            if signature is not None and current != signature:
                raise ValueError('Defect confirmation requires matching sensor, state, native grid origin/shape and gain')
            signature = current
            pair = [np.load(path, mmap_mode='r') for path in pair]
            records.append(pair)
            poses.append(geometry['matrices_yx'][str(index)])
            measurements.append(measurement)
            print(f'Defect candidates frame {index}: {measurement["candidates"]}', flush=True)
    shape = records[0][0].shape
    summary = confirm_by_exposure(records, poses, shape,
        [m['exposure_s'] for m in measurements], reliable)
    for result in [summary, summary['pooled_comparison'], *summary['exposure_groups']]:
        result['parameters']['phase_neighborhood'] = ('5x5 same-phase samples, center excluded'
            if signature[5] == 'linear_cfa' else '5x5 same-channel samples, center excluded')
    count = len(summary['pixels_xy'])
    limit = max(16, min(5000, int(np.prod(shape[:2])*.0005)))
    if count > limit:
        raise ValueError(f'{count} confirmed outliers exceed the conservative safety limit {limit}; no automatic repair applied')
    summary.update(mode=mode, applied=False, analyzed_indices=reliable, count=count,
        classification='Persistent isolated detector outliers; thermal origin is not established',
        sensor_signature=signature, measurements=measurements, python=platform.python_version(),
        geometry_review_sha256=hashlib.sha256(Path(review).read_bytes()).hexdigest())
    repairs = {}
    if mode == 'aesthetic' and apply_policy:
        for index, (detected, _) in zip(reliable, records):
            if len(shape) == 3:
                samples = [p for p in summary['samples_xyc'] if detected[p[1], p[0], p[2]]]
                repairs[index] = (list(map(list, dict.fromkeys(tuple(p[:2]) for p in samples))), samples)
            else:
                repairs[index] = ([p for p in summary['pixels_xy'] if detected[p[1], p[0]]], [])
    (destination/'detection.json').write_text(json.dumps(summary, indent=2)+'\n')
    from .io import write_image
    mask = np.zeros(shape[:2], np.float32)
    for x, y in summary['pixels_xy']:
        mask[y, x] = 1
    write_image(destination/'detected_defects.fits', mask, dict(unit='mask', geometry='native detector array',
        active_area_xywh=signature[3], cfa_pattern=signature[4], sensor_id=signature[0],
        coordinate_convention='pixels_xy use zero-based array centers; sensor origin is active_area_xywh[:2]'))
    for pair in records:
        for array in pair:
            path = Path(array.filename)
            array._mmap.close()
            path.unlink()
    print(f'WARNING: Confirmed {count} persistent detector defects; requested {mode} policy, apply={apply_policy}. Not a thermal-noise correction.', flush=True)
    if not count or not apply_policy:
        return source, summary
    estimated = sum(frames[i].data.nbytes+np.prod(frames[i].data.shape[:2])*16 for i in indices)
    if estimated*1.05 > shutil.disk_usage(destination).free:
        raise OSError(f'Insufficient external space for immutable defect-handled preparation: {estimated/2**30:.2f} GiB needed')
    print(f'Defect-handled preparation needs approximately {estimated/2**30:.2f} GiB on external storage', flush=True)
    updated = dict(old, prepared_overlays=dict(old.get('prepared_overlays', {})), defect_detection=summary)
    summary['per_frame'] = {}
    for index in indices:
        item = dict(old['recipe']['frames'][index], path=old['inputs'][index]['path'])
        frame = load_item(item, source)
        metadata = dict(frame.metadata, auto_defect_pixels_xy=summary['pixels_xy'], defect_policy=mode)
        if [frame.metadata.get(key) for key in ('sensor_id', 'iso', 'gain', 'active_area_xywh', 'cfa_pattern', 'state', 'defect_grid')] != signature:
            raise ValueError('Cannot apply the defect map to a different sensor, state, native grid origin/shape or gain')
        if mode == 'scientific':
            metadata['defective_pixels_xy'] = [*metadata.get('defective_pixels_xy', []), *summary['pixels_xy']]
        else:
            metadata['repair_defects_xy'], metadata['repair_defects_xyc'] = repairs.get(index, ([], []))
            metadata['repair_defects_prequalified'] = True
        cached = CachedFrame(frame.data, metadata, frame.path, detector=True)
        folder = destination/f'prepared_{index:03d}'
        prepared = cache_prepared(cached, masters.get('bias', [None])[0], select_dark(cached, masters.get('dark', [])), masters.get('flat', [None])[0], folder, 96)
        updated['prepared_overlays'][str(index)] = str(folder.resolve())
        summary['per_frame'][str(index)] = dict(repaired_samples=int(prepared.auxiliary['repaired'].sum()),
            detected_samples=int(prepared.auxiliary['detected_defects'].sum()),
            excluded_footprint_pixels=int(prepared.auxiliary['defective'].sum()))
        if mode == 'aesthetic':
            summary['per_frame'][str(index)]['qualified_samples'] = len(metadata['repair_defects_xy'])
        release_pages(prepared)
        print(f'Prepared defect-handled frame {index}', flush=True)
    # Calibration masters stay in the verified original preparation directory.
    updated['preparation_directory'] = str(Path(old.get('preparation_directory', source/'work')).resolve())
    summary['applied'] = True
    (destination/'detection.json').write_text(json.dumps(summary, indent=2)+'\n')
    (destination/'report.json').write_text(json.dumps(updated, indent=2)+'\n')
    return destination, summary

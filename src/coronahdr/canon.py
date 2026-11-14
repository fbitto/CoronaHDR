"""Build fresh camera-specific eclipse recipes and translation geometry."""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, label, map_coordinates
from scipy.optimize import least_squares

from .aesthetic_pipeline import mean4
from .folder_import import discover
from .raw import RAW_EXTENSIONS, camera_exif, decode_raw


def lunar_disk(data):
    """Measure a closed lunar limb for exclusion only, never for registration."""
    image = gaussian_filter(mean4(data), 1.)
    candidates = []
    for percentile in (90, 95, 97, 98, 99, 99.5):
        bright = binary_closing(image > np.percentile(image, percentile), iterations=2)
        holes, _ = label(binary_fill_holes(bright) & ~bright)
        areas = np.bincount(holes.ravel())
        for component in np.flatnonzero((areas > 64) & (areas < image.size/4)):
            if component == 0:
                continue
            yy, xx = np.nonzero(holes == component)
            radius = np.sqrt(len(xx)/np.pi)
            cx, cy = float(xx.mean()), float(yy.mean())
            if min(cx, cy, image.shape[1]-1-cx, image.shape[0]-1-cy) > 1.4*radius:
                candidates.append((len(xx), cx, cy, radius))
    if not candidates:
        raise ValueError('No closed lunar limb found; supply an explicit recipe with measured lunar_disk')
    _, cx, cy, radius = max(candidates)
    angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
    radii = np.arange(.65*radius, 1.35*radius, .25)
    for _ in range(3):
        profiles = map_coordinates(image, [cy+np.sin(angles[:, None])*radii,
                                          cx+np.cos(angles[:, None])*radii], order=1, mode='nearest')
        gradient = np.gradient(profiles, axis=1)
        peak = gradient.argmax(axis=1)
        strength = gradient[np.arange(len(angles)), peak]
        good = (strength > 0) & (peak > 2) & (peak < len(radii)-3)
        if good.sum() < .8*len(angles):
            raise ValueError('Incomplete lunar limb; supply explicit measured geometry')
        px = cx+np.cos(angles[good])*radii[peak[good]]
        py = cy+np.sin(angles[good])*radii[peak[good]]
        fit = least_squares(lambda p: np.hypot(px-p[0], py-p[1])-p[2], [cx, cy, radius],
                            loss='soft_l1', f_scale=.5,
                            bounds=([cx-radius/4, cy-radius/4, radii[0]],
                                    [cx+radius/4, cy+radius/4, radii[-1]]))
        cx, cy, radius = fit.x
    residual = float(np.median(np.abs(np.hypot(px-cx, py-cy)-radius))*4)
    if not fit.success or residual > max(2., .02*radius*4):
        raise ValueError(f'Unreliable lunar exclusion fit: {residual:.3f} native pixels')
    # A bin4 centre is native coordinate 4*i+1.5; preparation adds limb_margin_px once.
    return [float(cx*4+1.5), float(cy*4+1.5), float(radius*4)], residual


def canon_recipe(folders, work, camera_family='Canon'):
    paths = {role: [p for p in discover(folder, work) if p.suffix.lower() in RAW_EXTENSIONS]
             for role, folder in folders.items() if folder}
    extensions = {'Canon': ('.cr2', '.cr3'), 'Nikon': ('.nef', '.nrw')}[camera_family]
    if any(p.suffix.lower() not in extensions for items in paths.values() for p in items):
        raise ValueError(f'Keep {camera_family} and other camera RAWs in separate acquisition folders')
    if not paths.get('light'):
        raise ValueError(f'No {camera_family} RAW light files found')
    for role, items in paths.items():
        if not items:
            raise ValueError(f'No {camera_family} RAW files in selected {role} folder')
    reference_metadata = camera_exif(paths['light'][0])
    group_keys = ('sensor_id', 'iso', 'aperture', 'focal_length_mm')
    if any(key not in reference_metadata for key in group_keys):
        raise ValueError(f'{camera_family} camera identity, ISO, aperture and focal length are required; supply explicit metadata in a recipe')
    group = {key: reference_metadata[key] for key in group_keys}
    entries, calibration, disks, exposures = [], {}, [], []
    for role, items in paths.items():
        for path in items:
            metadata = camera_exif(path)
            keys = group_keys if role in ('light', 'flat') else ('sensor_id', 'iso')
            if any(metadata.get(key) != group[key] for key in keys):
                raise ValueError(f'Incompatible {camera_family} {role} acquisition: {path}')
            overrides = dict(state='linear_cfa', role=role)
            if role == 'light':
                data, _ = decode_raw(path)
                disk, residual = lunar_disk(data)
                shape = data.shape
                del data
                exposures.append(metadata['exposure_s'])
                disks.append(disk)
                overrides.update(response_group=json.dumps(group, sort_keys=True), lunar_disk=disk,
                    low_signal=2., limb_margin_px=8., lunar_mask_fit_median_residual_px=residual,
                    lunar_mask_provenance='closed-limb bin4 radial-gradient fit; exclusion only; unpadded radius; 8px limb_margin_px applied during preparation')
                print(f'Measured {camera_family} exclusion: {path.name}, residual {residual:.3f}px', flush=True)
            elif role in ('dark', 'dark_flat'):
                overrides['dark_contains_bias'] = True
            item = dict(path=str(path), overrides=overrides)
            (entries if role == 'light' else calibration.setdefault(role, [])).append(item)
    # Keep the full output canvas; restrict only the registration auxiliary to corona.
    low = np.floor(np.min([[x-4*r, y-4*r] for x, y, r in disks], axis=0)).astype(int)
    high = np.ceil(np.max([[x+4*r, y+4*r] for x, y, r in disks], axis=0)).astype(int)
    low = np.maximum(low, 0); high = np.minimum(high, shape[::-1])
    roi = [int(low[0]), int(low[1]), int(high[0]-low[0]), int(high[1]-low[1])]
    return dict(camera_family=camera_family, method='weighted', reference=int(np.argsort(exposures)[len(exposures)//2]),
        max_shift=1024, frames=entries, calibration=calibration,
        processing=dict(mode='disk', tile_rows=96, registration_bin=4,
                        registration_roi_xywh=roi, canvas='union', rejection_policy='exclude'))


def prepare_canon(folders, work, output, camera_family='Canon', *, resume_from=None):
    """Use the existing detector, registration, calibration and export engine."""
    from .pipeline import run_manifest
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    previous = Path(resume_from) if resume_from is not None else None
    if previous is not None and (previous/'recipe.json').is_file():
        recipe = json.loads((previous/'recipe.json').read_text())
    else:
        recipe = (canon_recipe(folders, work) if camera_family == 'Canon'
                  else canon_recipe(folders, work, camera_family))
    recipe.setdefault('processing', {})['decision_stage'] = 'Preparation and registration'
    output.mkdir(parents=True)
    manifest = output/'recipe.json'
    manifest.write_text(json.dumps(recipe, indent=2)+'\n')
    source = output/'prepared'
    report = None
    if previous is not None and (previous/'prepared/report.json').is_file():
        try:
            report = json.loads((previous/'prepared/report.json').read_text())
        except json.JSONDecodeError:
            pass
    if report is not None:
        from .streaming import load_cached_run
        source = previous/'prepared'
        load_cached_run(source, report['recipe'])
        print('Resumed completed acquisition and registration', flush=True)
    else:
        recipe['max_shift'] = max(1024, recipe.get('max_shift', 1024))
        if previous is not None and (previous/'prepared/preparation/report.json').is_file():
            recipe['processing']['cache_from'] = str((previous/'prepared/preparation').resolve())
            print('Resuming from completed detector preparation', flush=True)
        manifest.write_text(json.dumps(recipe, indent=2)+'\n')
        report = run_manifest(manifest, source)
    review_path = output/'geometry.json'
    write_translation_review(report, recipe, review_path)
    return source, review_path


def write_translation_review(report, recipe, review_path):
    matrices, metrics = {}, {}
    # shortcut: automatic geometry fits translations, use a reviewed rigid recipe when rotation is measurable.
    for item in report['transformations']:
        matrix = np.eye(3)
        matrix[:2, 2] = item['shift_dy_dx']
        matrices[str(item['index'])] = matrix.tolist()
        metrics[str(item['index'])] = item
    review = dict(input_source_sha256=[i['metadata']['source_sha256'] for i in report['inputs']],
        reference=report['reference'], matrices_yx=matrices, metrics=metrics, edges=[],
        factor=4, roi_xywh=recipe['processing']['registration_roi_xywh'],
        qualification='Automatically qualified coronal translations; no rotation fit or manual geometry review')
    review_path.write_text(json.dumps(review, indent=2)+'\n')

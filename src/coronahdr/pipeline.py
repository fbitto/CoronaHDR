"""One linear interpolation per input; conservative support and soft HDR weights."""
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy as np
from .io import read_frame, write_image
from .registration import register, translate, valid_mask
from .model import Frame
from .sensor import SensorFrame
from .calibration import build_master, normalize_flat, select_dark
from .prepare import prepare_light


def suggest_reference(frames):
    return max(range(len(frames)), key=lambda i: int(valid_mask(frames[i]).sum()))


def combine(frames, reference=None, max_shift=24):
    if not frames:
        raise ValueError('No input frames')
    if any(not isinstance(frame, Frame) for frame in frames):
        raise ValueError('Calibrate detector frames and demosaic CFA before HDR combination')
    if reference is None:
        reference = suggest_reference(frames)
    if not isinstance(reference, int) or not 0 <= reference < len(frames):
        raise ValueError('reference must be a valid zero-based frame index')
    ref = frames[reference]
    for frame in frames:
        if frame.data.shape != ref.data.shape or frame.metadata['state'] != ref.metadata['state']:
            raise ValueError('All frames must share shape and declared channel state')
        if frame.metadata['response_group'] != ref.metadata['response_group']:
            raise ValueError('Different response groups require independently validated relative response factors')
        for key in ('iso', 'gain', 'aperture'):
            if frame.metadata.get(key) != ref.metadata.get(key):
                raise ValueError(f'Changed {key}: the current engine requires identical acquisition response')
    shape = ref.data.shape[:2]
    numerator = np.zeros(ref.data.shape)
    weights = np.zeros(shape)
    counts = np.zeros(shape, np.uint32)
    maps = {k: np.zeros(shape, np.uint32) for k in ('coverage', 'lunar', 'saturation', 'low_signal', 'defective')}
    eligible = np.zeros(shape, np.uint32)
    unsaturated = np.zeros(shape, np.uint32)
    metrics = []
    per_frame = []
    for i, frame in enumerate(frames):
        delta, quality = (np.zeros(2), {'reference': True, 'shift_dy_dx': [0., 0.]}) if i == reference else register(ref, frame, max_shift)
        masks = frame.masks()
        # Full bilinear footprint must be valid, including at image boundaries.
        coverage = translate(masks['coverage'], delta) >= 1-1e-9
        aligned_masks = {'coverage': coverage}
        for key in ('lunar', 'saturation', 'low_signal', 'defective'):
            aligned_masks[key] = translate(masks[key], delta) > 1e-9
        valid = coverage.copy()
        for key in ('lunar', 'saturation', 'low_signal', 'defective'):
            valid &= ~aligned_masks[key]
        maximum = frame.data.max(axis=-1) if frame.data.ndim == 3 else frame.data
        m = frame.metadata
        high = np.clip((m['white_level']-maximum)/(.2*m['white_level']), 0, 1)
        low = np.clip((frame.mono-m['low_signal'])/(.05*(m['white_level']-m['low_signal'])), 0, 1)
        high_weight = frame.auxiliary.get('high_weight', high*high*(3-2*high))
        soft = high_weight*(low*low*(3-2*low))
        w = np.where(valid, translate(np.nan_to_num(soft), delta), 0)
        data = translate(np.where(np.isfinite(frame.data), frame.data, 0), delta)/frame.scale
        numerator += data*w[..., None] if data.ndim == 3 else data*w
        weights += w
        counts += w > 0
        for key in maps:
            maps[key] += aligned_masks[key]
        observable = coverage & ~aligned_masks['lunar'] & ~aligned_masks['defective']
        eligible += observable
        unsaturated += observable & ~aligned_masks['saturation']
        quality.update(index=i, path=str(frame.path))
        metrics.append(quality)
        per_frame.append({**aligned_masks, 'weight': w})
    master = np.full(ref.data.shape, np.nan, dtype=np.float64)
    np.divide(numerator, weights[..., None] if master.ndim == 3 else weights,
              out=master, where=(weights > 0)[..., None] if master.ndim == 3 else weights > 0)
    return master.astype(np.float32), dict(**maps, contributions=counts, weight=weights,
              support=weights > 0, unresolved_saturation=(eligible > 0) & (unsaturated == 0)), metrics, per_frame, reference


def load_item(item, base):
    frame = read_frame(base / item['path'], item.get('overrides'))
    if 'crop_xywh' in item:
        if not isinstance(frame, SensorFrame):
            raise ValueError('Detector crop requires explicit detector metadata and role=light/calibration role')
        frame = frame.crop(*item['crop_xywh'])
    return frame


def prepare_recipe(recipe, base):
    calibration = recipe.get('calibration', {})
    unknown = set(calibration)-{'bias', 'dark', 'flat', 'dark_flat'}
    if unknown:
        raise ValueError(f'Unknown calibration roles: {sorted(unknown)}')
    masters = {}
    for role, entries in calibration.items():
        inputs = [load_item(item, base) for item in entries]
        if any(not isinstance(frame, SensorFrame) for frame in inputs):
            raise ValueError(f'Calibration inputs require explicit role={role} and detector metadata')
        if role in ('dark', 'dark_flat'):
            groups = {}
            for frame in inputs:
                key = (frame.metadata['exposure_s'], frame.metadata.get('temperature_c'),
                       frame.metadata.get('dark_contains_bias'))
                groups.setdefault(key, []).append(frame)
            masters[role] = [build_master(group, role) for group in groups.values()]
        else:
            masters[role] = [build_master(inputs, role)]
    bias = masters.get('bias', [None])[0]
    flat = masters.get('flat', [None])[0]
    if masters.get('dark_flat') and flat is None:
        raise ValueError('dark_flat supplied without flat frames')
    if flat is not None:
        flat = normalize_flat(flat, bias, select_dark(flat, masters.get('dark_flat', [])))
        masters['flat'] = [flat]
    frames = []
    for item in recipe['frames']:
        frame = load_item(item, base)
        if calibration and not isinstance(frame, SensorFrame):
            frame = SensorFrame(frame.data, dict(frame.metadata, role='light'), frame.path)
        if isinstance(frame, SensorFrame):
            frame = prepare_light(frame, bias, select_dark(frame, masters.get('dark', [])), flat)
        frames.append(frame)
    return frames, masters


def run_manifest(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    recipe = json.loads(manifest_path.read_text())
    if recipe.get('method', 'weighted') != 'weighted':
        raise ValueError('The current engine implements only method=weighted')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite output directory {output}')
    mode = recipe.get('processing', {}).get('mode', 'memory')
    if mode not in ('memory', 'disk'):
        raise ValueError('processing.mode must be memory or disk')
    if mode == 'disk':
        from .streaming import run_disk_manifest
        return run_disk_manifest(recipe, manifest_path.parent, output)
    if recipe.get('processing', {}).get('canvas', 'reference') != 'reference':
        raise ValueError('Union canvas requires processing.mode=disk')
    frames, calibration_masters = prepare_recipe(recipe, manifest_path.parent)
    master, maps, metrics, per_frame, reference = combine(frames, recipe.get('reference'), recipe.get('max_shift', 24))
    output.mkdir(parents=True)
    metadata = dict(state=frames[0].metadata['state'], scale='relative response per second; no normalization',
                    missing='NaN; support=0', channel_order='RGB' if master.ndim == 3 else 'mono')
    for extension in ('fits', 'tiff'):
        write_image(output / f'master.{extension}', master, metadata)
    for key, data in maps.items():
        write_image(output / f'{key}.fits', data, {'meaning': key, 'unit': '1'})
    for i, diagnostics in enumerate(per_frame):
        folder = output / f'frame_{i:03d}'
        folder.mkdir()
        for key, data in diagnostics.items():
            write_image(folder / f'{key}.fits', data, {'meaning': key, 'unit': '1'})
    if calibration_masters:
        folder = output / 'calibration'
        folder.mkdir()
        for role, masters in calibration_masters.items():
            for index, master in enumerate(masters):
                unit = '1' if role == 'flat' else 'detector response'
                write_image(folder / f'{role}_{index:03d}.fits', master.data, dict(master.metadata, unit=unit))
    report = dict(recipe=recipe, reference=reference, suggested_reference=suggest_reference(frames),
                  inputs=[dict(path=str(f.path), metadata=f.metadata) for f in frames],
                  versions={name: importlib.metadata.version(name) for name in
                            ('coronahdr', 'numpy', 'scipy', 'scikit-image', 'astropy', 'tifffile', 'rawpy')},
                  python=platform.python_version(), transformations=metrics, output=metadata,
                  conventions='source to reference: output[y,x] = source[y-dy,x-dx]; pixel centers at integers; fixed reference canvas; outside coverage is invalid',
                  corrections=[dict(path=str(f.path), steps=f.metadata.get('calibration_steps',
                      ['none; input explicitly declared black-subtracted']), demosaic=f.metadata.get('demosaic')) for f in frames],
                  calibration={role: [dict(metadata=m.metadata, sources=m.provenance) for m in masters]
                               for role, masters in calibration_masters.items()},
                  warnings=['Relative radiance, not absolute calibration', 'Real-data qualification is limited to documented datasets',
                            'Reference canvas crops shifted borders explicitly via coverage maps',
                            'Weights are smooth reliability weights, not inverse noise variance',
                            'response_group certifies unchanged response; regional residual checks cannot prove cloud-free data'])
    if any('raw_validation' in frame.metadata for frame in frames) or any(
            'raw_validation' in source['metadata'] for masters in calibration_masters.values()
            for master in masters for source in master.provenance):
        report['warnings'].append('Experimental Bayer RAW ingestion; camera/file validation coverage is documented separately')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report

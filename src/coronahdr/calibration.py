"""Detector-coordinate calibration, with no dark scaling or geometric transforms."""
from dataclasses import dataclass
import warnings
import numpy as np
from .sensor import SensorFrame, PHASES, phase_map, level_map


@dataclass
class Master:
    data: np.ndarray
    metadata: dict
    provenance: list


def compatible(target, source, dark=False):
    if target.data.shape != source.data.shape:
        raise ValueError('Calibration shape mismatch')
    for key in ('state', 'sensor_id', 'iso', 'gain', 'active_area_xywh'):
        if target.metadata.get(key) != source.metadata.get(key):
            raise ValueError(f'Calibration {key} mismatch')
    if target.metadata['state'] == 'linear_cfa' and target.metadata['cfa_pattern'] != source.metadata.get('cfa_pattern'):
        raise ValueError('Calibration CFA phase/pattern mismatch')
    if source.metadata.get('role') == 'flat':
        for key in ('aperture', 'focal_length_mm', 'optical_train_id'):
            if target.metadata.get(key) != source.metadata.get(key):
                raise ValueError(f'Flat optical configuration {key} mismatch')
    if dark:
        for key in ('exposure_s', 'temperature_c'):
            if key not in target.metadata or key not in source.metadata or not np.isfinite(target.metadata[key]) or not np.isfinite(source.metadata[key]):
                raise ValueError(f'Dark compatibility requires measured or explicitly declared {key}')
            if not np.isclose(target.metadata[key], source.metadata[key], rtol=0, atol=1e-9):
                raise ValueError(f'Dark {key} mismatch; automatic scaling is disabled')


def sample_defective(frame):
    defective = ~np.isfinite(frame.data)
    for x, y in frame.metadata.get('defective_pixels_xy', []):
        if not (isinstance(x, int) and isinstance(y, int) and 0 <= x < frame.data.shape[1] and 0 <= y < frame.data.shape[0]):
            raise ValueError('Invalid detector defect coordinates')
        defective[y, x] = True
    return defective


def sample_valid(frame):
    return ~sample_defective(frame) & (frame.data < level_map(frame, 'white_level'))


def build_master(frames, role):
    if not frames:
        raise ValueError(f'No {role} calibration frames')
    first = frames[0]
    for frame in frames:
        if frame.metadata['role'] != role:
            raise ValueError(f'Expected calibration role={role}')
        compatible(first, frame, dark=role in ('dark', 'dark_flat'))
        for key in ('exposure_s', 'black_subtracted', 'dark_contains_bias', 'black_level'):
            if first.metadata.get(key) != frame.metadata.get(key):
                raise ValueError(f'Calibration stack {key} mismatch; do not mix pedestal conventions or exposures')
    samples = np.stack([np.where(sample_valid(f), f.data, np.nan) for f in frames])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        master = np.nanmedian(samples, axis=0)
    provenance = [dict(path=str(f.path), metadata=f.metadata) for f in frames]
    metadata = dict(first.metadata, master_integration='per-sample median of finite unsaturated samples',
                    master_frames=len(frames), master_missing_samples=int((~np.isfinite(master)).sum()))
    return Master(master, metadata, provenance)


def subtract_detector(frame, bias=None, dark=None):
    """Return corrected samples and an exact audit of the pedestal/dark route."""
    data = frame.data.copy()
    m = frame.metadata
    steps = []
    if bias is not None:
        compatible(frame, bias)
        if bias.metadata['role'] != 'bias':
            raise ValueError('Pedestal master must have role=bias')
    if dark is not None:
        compatible(frame, dark, dark=True)
        if dark.metadata['role'] not in ('dark', 'dark_flat'):
            raise ValueError('Dark master role must be dark or dark_flat')
    if dark is not None and dark.metadata['dark_contains_bias']:
        if m['black_subtracted']:
            raise ValueError('A full dark would subtract bias twice from already black-subtracted input')
        data -= dark.data
        steps.append('subtract full dark including bias; no separate bias or black-level subtraction')
        if bias is not None:
            steps.append('provided bias intentionally not subtracted because full dark contains bias')
    else:
        if m['black_subtracted']:
            if bias is not None:
                raise ValueError('A bias would subtract the pedestal twice from already black-subtracted input')
            steps.append('input pedestal already subtracted; no second subtraction')
        elif bias is not None:
            data -= bias.data
            steps.append('subtract full bias master; no additional black-level subtraction')
        else:
            data -= level_map(frame, 'black_level')
            steps.append('subtract declared black-level map only; no measured bias master')
        if dark is not None:
            data -= dark.data
            steps.append('subtract bias-subtracted dark without exposure or temperature scaling')
        else:
            steps.append('no dark correction')
    # Saturated lights retain finite storage; their separate mask excludes them.
    invalid = sample_defective(frame) if m['role'] == 'light' else ~sample_valid(frame)
    data[invalid] = np.nan
    return data, steps


def normalize_flat(flat, bias=None, dark_flat=None):
    corrected, steps = subtract_detector(flat, bias, dark_flat)
    normalized = np.full(flat.data.shape, np.nan)
    factors = {}
    if flat.metadata['state'] == 'linear_cfa':
        labels = phase_map(flat.data.shape, flat.metadata)
        groups = [(phase, labels == phase) for phase in PHASES]
    elif flat.metadata['state'] == 'linear_rgb':
        channels = np.indices(flat.data.shape)[-1]
        groups = [(phase, channels == i) for i, phase in enumerate(('R', 'G', 'B'))]
    else:
        groups = [('mono', np.ones(flat.data.shape, bool))]
    for name, mask in groups:
        values = corrected[mask & np.isfinite(corrected) & (corrected > 0)]
        if values.size < 16:
            raise ValueError(f'Insufficient positive corrected flat samples for {name}')
        factor = float(np.median(values))
        factors[name] = factor
        normalized[mask] = corrected[mask]/factor
    normalized[normalized <= 0] = np.nan
    metadata = dict(flat.metadata, flat_normalization=factors,
                    flat_normalization_method='one median per detector phase/channel after pedestal/dark-flat correction; no light normalization',
                    flat_corrections=steps)
    return Master(normalized, metadata, flat.provenance)


def select_dark(frame, darks):
    if not darks:
        return None
    matches = []
    reasons = []
    for dark in darks:
        try:
            compatible(frame, dark, dark=True)
            matches.append(dark)
        except ValueError as error:
            reasons.append(str(error))
    if len(matches) != 1:
        raise ValueError(f'Exactly one compatible dark is required when darks are supplied; found {len(matches)}: {reasons}')
    return matches[0]


def calibrate(frame, bias=None, dark=None, flat=None):
    if frame.metadata['role'] != 'light':
        raise ValueError('Only light frames can enter the HDR path')
    if flat is not None:
        compatible(frame, flat)
        if flat.metadata['role'] != 'flat' or 'flat_normalization' not in flat.metadata:
            raise ValueError('Flat must be explicitly corrected and normalized')
    data, steps = subtract_detector(frame, bias, dark)
    if flat is not None:
        np.divide(data, flat.data, out=data, where=np.isfinite(flat.data) & (flat.data > 0))
        data[~np.isfinite(flat.data) | (flat.data <= 0)] = np.nan
        steps.append('divide by corrected phase/channel-normalized master flat')
    else:
        steps.append('no flat correction')
    metadata = dict(frame.metadata, black_subtracted=True, calibration_steps=steps,
                    calibration_masters={name: master.metadata for name, master in
                        [('bias', bias), ('dark', dark), ('flat', flat)] if master is not None})
    if frame.metadata.get('repair_defects_xy'):
        from .defects import repair_cfa, repair_linear
        blocked = sample_defective(frame) | (frame.data >= level_map(frame, 'white_level'))
        if frame.metadata.get('lunar_disk') is not None:
            cx, cy, radius = frame.metadata['lunar_disk']
            yy, xx = np.ogrid[:data.shape[0], :data.shape[1]]
            lunar = (xx-cx)**2+(yy-cy)**2 <= (radius+frame.metadata.get('limb_margin_px', 4))**2
            blocked |= lunar[..., None] if data.ndim == 3 else lunar
        repair = repair_cfa if frame.metadata['state'] == 'linear_cfa' else repair_linear
        qualified = frame.metadata.get('repair_defects_prequalified', False)
        options = dict(require_outlier=bool(frame.metadata.get('auto_defect_pixels_xy')) and not qualified,
                       donor_pixels=frame.metadata.get('auto_defect_pixels_xy', []))
        if qualified and data.ndim == 3:
            options['samples'] = frame.metadata['repair_defects_xyc']
        data, repaired = repair(data, frame.metadata['repair_defects_xy'], blocked,
                                **options)
        metadata['repaired_defects_xy'] = repaired
        steps.append('selective phase/channel median repair of confirmed detector outliers; no global filtering')
    return data, metadata

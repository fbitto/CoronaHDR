"""Calibrate first, demosaic second; geometry is handled later by registration."""
import numpy as np
from scipy.ndimage import convolve, maximum_filter, minimum_filter
from .calibration import calibrate, level_map, sample_defective
from .sensor import phase_map
from .model import Frame


def demosaic_linear(data, metadata):
    """Bilinear camera RGB, no white balance/color matrix/gamma/clipping."""
    if metadata['state'] != 'linear_cfa' or data.ndim != 2:
        raise ValueError('Linear demosaicing requires a calibrated Bayer mosaic')
    labels = phase_map(data.shape, metadata)
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float)
    rgb = np.empty((*data.shape, 3))
    for index, color in enumerate(('R', 'G', 'B')):
        selected = np.char.startswith(labels, color)
        samples = np.where(selected & np.isfinite(data), data, 0.)
        denominator = convolve(selected.astype(float), kernel, mode='constant')
        channel = np.divide(convolve(samples, kernel, mode='constant'), denominator,
                            out=np.full(data.shape, np.nan), where=denominator > 0)
        channel[selected] = data[selected]
        rgb[..., index] = channel
    return rgb


def prepare_light(frame, bias=None, dark=None, flat=None):
    original = frame.metadata
    data, metadata = calibrate(frame, bias, dark, flat)
    white = level_map(frame, 'white_level')
    black = level_map(frame, 'black_level')
    saturation = frame.data >= white
    defective = sample_defective(frame)
    coverage = np.ones(data.shape[:2], bool)
    y, x = np.indices(data.shape[:2])
    lunar = np.zeros(coverage.shape, bool)
    if original['lunar_disk'] is not None:
        cx, cy, radius = original['lunar_disk']
        lunar = (x-cx)**2+(y-cy)**2 <= (radius+original.get('limb_margin_px', 4))**2
    headroom_range = white if original['black_subtracted'] else white-black
    headroom = np.clip((white-frame.data)/(.2*headroom_range), 0, 1)
    high_weight = headroom*headroom*(3-2*headroom)
    if frame.metadata['state'] == 'linear_cfa':
        defective |= ~np.isfinite(data)
        data = demosaic_linear(data, metadata)
        # All interpolation footprints must be observed and unocculted, not filled.
        lunar = maximum_filter(lunar, size=3, mode='constant', cval=0)
        saturation = maximum_filter(saturation, size=3, mode='constant', cval=0)
        defective = maximum_filter(defective, size=3, mode='constant', cval=0)
        high_weight = minimum_filter(high_weight, size=3, mode='constant', cval=0)
        coverage[[0, -1], :] = False
        coverage[:, [0, -1]] = False
        metadata['state'] = 'linear_rgb'
        metadata['demosaic'] = 'linear bilinear camera RGB; R,G1,G2,B retained during calibration; no white balance or color matrix'
        metadata['cfa_tile_at_active_origin'] = phase_map((2, 2), original).tolist()
    elif data.ndim == 3:
        saturation = saturation.any(axis=-1)
        defective = defective.any(axis=-1) | ~np.isfinite(data).all(axis=-1)
        high_weight = high_weight.min(axis=-1)
    else:
        defective |= ~np.isfinite(data)
    metadata['source_detector_metadata'] = original
    # Actual rejection and high-end weights use pre-calibration detector limits.
    metadata['white_level'] = float(np.min(headroom_range))
    metadata['saturation_basis'] = 'original detector samples and declared source white levels, propagated over demosaic footprint'
    metadata['defective_pixels_xy'] = []
    metadata['preparation'] = 'detector calibration then optional linear demosaic; no geometric resampling'
    auxiliary = dict(coverage=coverage, lunar=lunar, saturation=saturation,
                     defective=defective, high_weight=high_weight)
    if original.get('auto_defect_pixels_xy'):
        for name, pixels in [('detected_defects', original['auto_defect_pixels_xy']),
                             ('repaired', metadata.get('repaired_defects_xy', []))]:
            mask = np.zeros(coverage.shape, bool)
            for px, py in pixels:
                mask[py, px] = True
            auxiliary[name] = mask
        auxiliary['repair_footprint'] = (maximum_filter(auxiliary['repaired'], size=3, mode='constant', cval=0)
                                         if original['state'] == 'linear_cfa' else auxiliary['repaired'].copy())
    return Frame(data, metadata, frame.path, auxiliary)

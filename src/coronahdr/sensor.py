"""Unrendered detector samples and Bayer phase in full-sensor coordinates."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

BAYER_PATTERNS = ('RGGB', 'BGGR', 'GRBG', 'GBRG')
PHASES = ('R', 'G1', 'G2', 'B')


def phase_tile(pattern, origin_xy):
    if pattern not in BAYER_PATTERNS:
        raise ValueError('Only declared 2x2 RGB Bayer CFA patterns are implemented')
    labels, green = [], 0
    for color in pattern:
        if color == 'G':
            green += 1
            labels.append(f'G{green}')
        else:
            labels.append(color)
    tile = np.array(labels).reshape(2, 2)
    x, y = origin_xy
    return np.roll(tile, (-y % 2, -x % 2), axis=(0, 1))


def phase_map(shape, metadata):
    tile = phase_tile(metadata['cfa_pattern'], metadata['active_area_xywh'][:2])
    height, width = shape[:2]
    return np.tile(tile, ((height+1)//2, (width+1)//2))[:height, :width]


def level_map(frame, key):
    value = np.asarray(frame.metadata[key], dtype=float)
    if value.ndim == 0:
        return np.full(frame.data.shape, value)
    if frame.metadata['state'] == 'linear_cfa':
        labels = phase_tile(frame.metadata['cfa_pattern'], frame.metadata['active_area_xywh'][:2])
        tile = np.array([value[PHASES.index(phase)] for phase in labels.flat]).reshape(2, 2)
        height, width = frame.data.shape
        return np.tile(tile, ((height+1)//2, (width+1)//2))[:height, :width]
    return np.broadcast_to(value, frame.data.shape).copy()


@dataclass
class SensorFrame:
    data: np.ndarray
    metadata: dict
    path: Path

    def __post_init__(self):
        m = self.metadata
        required = ('state', 'role', 'sensor_id', 'exposure_s', 'iso', 'gain',
                    'white_level', 'black_level', 'black_subtracted', 'active_area_xywh')
        missing = [k for k in required if k not in m]
        if missing:
            raise ValueError(f'{self.path}: explicit detector metadata required: {", ".join(missing)}')
        state = m['state']
        if state not in ('linear_cfa', 'linear_mono', 'linear_rgb'):
            raise ValueError('Detector inputs must be explicitly linear; no response inversion is implemented')
        if m['role'] not in ('light', 'bias', 'dark', 'flat', 'dark_flat'):
            raise ValueError('role must be light, bias, dark, flat, or dark_flat')
        a = np.asarray(self.data)
        if a.dtype.kind not in 'uif' or min(a.shape[:2], default=0) < 32 or not (
                a.ndim == 3 and a.shape[-1] == 3 if state == 'linear_rgb' else a.ndim == 2):
            raise ValueError('Detector shape must match declared linear mono, CFA, or interleaved RGB state')
        area = m['active_area_xywh']
        if (len(area) != 4 or any(not isinstance(v, int) or isinstance(v, bool) for v in area)
                or min(area[:2]) < 0 or area[2:] != [a.shape[1], a.shape[0]]):
            raise ValueError('active_area_xywh must contain integer sensor origin and the exact array width/height')
        if state == 'linear_cfa':
            phase_tile(m.get('cfa_pattern'), area[:2])
        for key in ('iso', 'gain'):
            if key == 'gain' and m[key] is None:
                continue
            if not isinstance(m[key], (int, float)) or not np.isfinite(m[key]) or m[key] <= 0:
                raise ValueError(f'{key} must be explicitly finite and positive')
        exposure = m['exposure_s']
        if not isinstance(exposure, (int, float)) or not np.isfinite(exposure) or exposure < 0 or (exposure == 0 and m['role'] != 'bias'):
            raise ValueError('exposure_s must be positive, except zero is allowed for bias')
        if not isinstance(m['black_subtracted'], bool) or not isinstance(m['sensor_id'], str) or not m['sensor_id'].strip():
            raise ValueError('black_subtracted must be boolean and sensor_id must identify the detector')
        for key in ('black_level', 'white_level'):
            values = np.asarray(m[key], dtype=float)
            channels = 4 if state == 'linear_cfa' else 3 if state == 'linear_rgb' else 1
            if (values.shape not in ((), (channels,)) or not np.isfinite(values).all()
                    or (key == 'white_level' and np.any(values <= 0))):
                raise ValueError(f'{key} must be a finite scalar or {channels} explicitly ordered channel levels')
        if m['role'] in ('dark', 'dark_flat'):
            if not isinstance(m.get('dark_contains_bias'), bool):
                raise ValueError('Darks require explicit dark_contains_bias')
            if m['dark_contains_bias'] == m['black_subtracted']:
                raise ValueError('dark_contains_bias contradicts black_subtracted')
        if m['role'] == 'bias' and m['black_subtracted']:
            raise ValueError('A bias master must contain the complete pedestal')
        if m['role'] == 'light':
            for key in ('response_group', 'lunar_disk', 'low_signal'):
                if key not in m:
                    raise ValueError(f'Light requires explicit {key}')
        self.data = a.astype(np.float64)
        if not m['black_subtracted'] and np.any(np.asarray(m['white_level']) <= np.asarray(m['black_level'])):
            raise ValueError('Uncorrected white levels must exceed black levels')

    def crop(self, x, y, width, height):
        if any(not isinstance(v, int) for v in (x, y, width, height)) or min(x, y) < 0 or min(width, height) < 32 or x+width > self.data.shape[1] or y+height > self.data.shape[0]:
            raise ValueError('Crop must be an explicit in-bounds integer region, at least 32x32')
        metadata = self.metadata.copy()
        ax, ay, *_ = metadata['active_area_xywh']
        metadata['active_area_xywh'] = [ax+x, ay+y, width, height]
        metadata['crop_history'] = [*metadata.get('crop_history', []), [x, y, width, height]]
        if metadata.get('lunar_disk') is not None:
            cx, cy, r = metadata['lunar_disk']
            metadata['lunar_disk'] = [cx-x, cy-y, r]
        metadata['defective_pixels_xy'] = [[px-x, py-y] for px, py in metadata.get('defective_pixels_xy', [])
                                               if x <= px < x+width and y <= py < y+height]
        return SensorFrame(self.data[y:y+height, x:x+width].copy(), metadata, self.path)

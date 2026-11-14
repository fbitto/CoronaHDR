"""Linear samples and explicit acquisition declarations; no image normalization."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class Frame:
    data: np.ndarray
    metadata: dict
    path: Path
    auxiliary: dict = field(default_factory=dict)

    def __post_init__(self):
        m = self.metadata
        required = ('state', 'exposure_s', 'white_level', 'black_level',
                    'black_subtracted', 'response_group', 'lunar_disk', 'low_signal')
        missing = [k for k in required if k not in m]
        if missing:
            raise ValueError(f'{self.path}: explicit metadata required: {", ".join(missing)}')
        if m['state'] not in ('linear_mono', 'linear_rgb'):
            raise ValueError(f'{self.path}: prepared HDR data require linear_mono or linear_rgb; calibrate and demosaic CFA first')
        a = np.asarray(self.data)
        expected = a.ndim == 2 if m['state'] == 'linear_mono' else a.ndim == 3 and a.shape[-1] == 3
        if not expected or min(a.shape[:2]) < 32 or a.dtype.kind not in 'uif':
            raise ValueError(f'{self.path}: invalid numeric shape for declared state: {a.shape}')
        for key in ('exposure_s', 'white_level', 'relative_scale'):
            value = m.get(key, m['exposure_s'])
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{self.path}: {key} must be finite and positive')
        if 'relative_scale' in m and not m.get('scale_provenance'):
            raise ValueError('relative_scale requires scale_provenance documenting its effective-exposure calibration')
        if not m['black_subtracted'] or not isinstance(m['black_subtracted'], bool):
            raise ValueError(f'{self.path}: prepared HDR data require black-subtracted input; declare detector role to request calibration')
        if not np.isfinite(m['black_level']).all() or not np.isfinite(m['low_signal']) or m['low_signal'] >= m['white_level']:
            raise ValueError(f'{self.path}: invalid black/low/white levels')
        if not isinstance(m['response_group'], str) or not m['response_group'].strip():
            raise ValueError('response_group must explicitly certify a common acquisition response')
        disk = m['lunar_disk']
        if disk is not None and (len(disk) != 3 or not np.isfinite(disk).all() or disk[2] <= 0):
            raise ValueError('lunar_disk must be null (no occultation) or [center_x, center_y, radius]')
        self.data = a.astype(np.float64)

    @property
    def scale(self):
        return self.metadata.get('relative_scale', self.metadata['exposure_s'])

    @property
    def mono(self):
        return self.data.mean(axis=-1) if self.data.ndim == 3 else self.data

    def masks(self):
        a, m = self.data, self.metadata
        finite = np.isfinite(a).all(axis=-1) if a.ndim == 3 else np.isfinite(a)
        maximum = a.max(axis=-1) if a.ndim == 3 else a
        y, x = np.indices(a.shape[:2])
        lunar = np.zeros(finite.shape, bool)
        if m['lunar_disk'] is not None:
            cx, cy, radius = m['lunar_disk']
            lunar = (x-cx)**2 + (y-cy)**2 <= (radius + m.get('limb_margin_px', 4))**2
        defective = ~finite
        for px, py in m.get('defective_pixels_xy', []):
            if not (0 <= px < a.shape[1] and 0 <= py < a.shape[0]):
                raise ValueError('defective pixel outside image')
            defective[py, px] = True
        masks = dict(coverage=np.ones(finite.shape, bool), lunar=lunar,
                    saturation=maximum >= m['white_level'],
                    low_signal=self.mono <= m['low_signal'], defective=defective)
        for key in masks:
            if key in self.auxiliary:
                override = np.asarray(self.auxiliary[key], dtype=bool)
                if override.shape != finite.shape:
                    raise ValueError(f'Invalid auxiliary {key} mask shape')
                masks[key] = override
        masks['defective'] = masks['defective'] | ~finite
        return masks

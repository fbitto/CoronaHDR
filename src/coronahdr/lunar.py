"""Explicit limb margins relative to the fitted, unpadded lunar radius."""
from dataclasses import replace
import re

import numpy as np
from scipy.ndimage import maximum_filter


def validate_margin(margin):
    if margin is not None and (type(margin) is not int or not -4 <= margin <= 4):
        raise ValueError('Lunar margin must be an integer from -4 to +4 pixels')


def fitted_disk(metadata):
    disk = metadata.get('lunar_disk')
    if disk is None:
        return None
    cx, cy, radius = disk
    provenance = metadata.get('lunar_mask_provenance', '')
    # Historical Nikon and Canon recipes embedded 8px in the measured radius.
    # Only historical, manually declared profiles used this exact response format.
    response = metadata.get('response_group', '')
    legacy_nikon = isinstance(response, str) and bool(re.fullmatch(
        r'Nikon Z50 II serial \d+; ISO800; f8; 250mm; fixed response declaration', response))
    if 'radius padded by 8px' in provenance or provenance == 'closed-limb bin4 radial-gradient fit; exclusion only; 8px padding' or (not provenance and legacy_nikon):
        radius -= 8
    return cx, cy, radius


def apply_margin(frames, margin, work):
    validate_margin(margin)
    if margin is None:
        return frames, {}
    result, paths = [], {}
    for i, frame in enumerate(frames):
        disk = fitted_disk(frame.metadata)
        if disk is None:
            result.append(frame)
            continue
        cx, cy, radius = disk
        if not np.isfinite([cx, cy, radius]).all() or radius+margin <= 0:
            raise ValueError('Invalid lunar geometry for selected margin')
        path = work/f'lunar_margin_{i:03d}.npy'
        shape = frame.data.shape[:2]
        mask = np.lib.format.open_memmap(path, mode='w+', dtype=bool, shape=shape)
        demosaiced = bool(frame.metadata.get('demosaic'))
        for start in range(0, shape[0], 128):
            end = min(start+128, shape[0])
            lo, hi = max(0, start-1), min(shape[0], end+1)
            y, x = np.ogrid[lo:hi, :shape[1]]
            tile = (x-cx)**2+(y-cy)**2 <= (radius+margin)**2
            if demosaiced:
                tile = maximum_filter(tile, size=3, mode='constant', cval=0)
            mask[start:end] = tile[start-lo:end-lo]
        mask.flush()
        result.append(replace(frame, auxiliary=dict(frame.auxiliary, lunar=mask)))
        paths[str(i)] = str(path.resolve())
    return result, paths

"""Experimental AEL-inspired registration features and independent structure checks.

The masked log-band and rotated-null concepts follow Agentic Eclipse Library,
Pere Guerra Serra, MIT, revision 63c16820ccf703badfcbd4d14bfe2fb59db0a63c.
This independent implementation uses SciPy and CoronaHDR matrix conventions.
"""
import numpy as np
from dataclasses import replace
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import fftconvolve

from .registration import valid_mask
from .rigid import apply_points, control_points, lower_matrix, matrix_distance, matrix_yx, measure_rigid


def log_band(frame, small=1., large=8.):
    """Remove smooth radial light with normalized convolution of valid samples."""
    image = frame.mono / frame.scale
    valid = valid_mask(frame) & np.isfinite(image) & (image > 0)
    values = np.zeros_like(image, dtype=float)
    values[valid] = np.log(image[valid])
    weights = valid.astype(float)
    a, b = gaussian_filter(weights, small), gaussian_filter(weights, large)
    band = (gaussian_filter(values, small)/np.maximum(a, 1e-12)
            - gaussian_filter(values, large)/np.maximum(b, 1e-12))
    keep = valid & (b > .9)
    return np.where(keep, band, 0.), keep


def correlation_seed(reference, moving, max_shift):
    """Unwhitened linear FFT correlation of independently masked log bands."""
    r, rm = log_band(reference)
    m, mm = log_band(moving)
    if r.shape != m.shape or max_shift <= 0:
        raise ValueError('Correlation requires matching shapes and a positive shift limit')
    if min(rm.sum(), mm.sum()) < 256 or min(np.std(r[rm]), np.std(m[mm])) < 1e-10:
        raise ValueError('Insufficient coronal contrast for correlation')
    window = np.outer(np.hanning(r.shape[0]), np.hanning(r.shape[1]))
    a = np.where(rm, r-r[rm].mean(), 0.)*window
    b = np.where(mm, m-m[mm].mean(), 0.)*window
    correlation = fftconvolve(a, b[::-1, ::-1], mode='full')
    center = np.array(r.shape)-1
    limits = np.minimum(np.floor(max_shift).astype(int), center)
    lo, hi = center-limits, center+limits+1
    region = correlation[lo[0]:hi[0], lo[1]:hi[1]]
    peak = np.array(np.unravel_index(np.argmax(region), region.shape))+lo
    if np.any(np.abs(peak-center) >= limits):
        raise ValueError('Correlation peak reaches shift-search boundary')
    shift = (peak-center).astype(float)
    for axis in range(2):
        before, after = peak.copy(), peak.copy()
        before[axis] -= 1
        after[axis] += 1
        a, b, c = correlation[tuple(before)], correlation[tuple(peak)], correlation[tuple(after)]
        curvature = a-2*b+c
        if curvature < 0:
            shift[axis] += np.clip(.5*(a-c)/curvature, -.5, .5)
    return shift


def structure_check(reference, moving, matrix, center):
    """Compare the fitted correlation with quarter-turn coronal nulls."""
    r, rm = log_band(reference)
    m, mm = log_band(moving)
    y, x = np.indices(r.shape, dtype=float)
    points = np.stack([y.ravel(), x.ravel()], axis=1)
    correlations = []
    inverse = np.linalg.inv(matrix)
    for angle in (0, 90, 270):
        turn = matrix_yx(r.shape, [0., 0.], angle)
        turn[:2, 2] += np.asarray(center) - apply_points(turn, [center])[0]
        sample = apply_points(inverse @ turn, points)
        coords = [sample[:, 0].reshape(r.shape), sample[:, 1].reshape(r.shape)]
        observed = map_coordinates(m, coords, order=1, mode='constant', cval=0.)
        support = map_coordinates(mm.astype(float), coords, order=1, mode='constant', cval=0.) > .999
        mask = rm & support
        if mask.sum() < 100:
            raise ValueError('Insufficient coronal support for rotated-null check')
        a, b = r[mask], observed[mask]
        a, b = a-a.mean(), b-b.mean()
        norm = np.linalg.norm(a)*np.linalg.norm(b)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError('No coronal contrast for rotated-null check')
        correlations.append(float(a@b/norm))
    null = max(correlations[1:])
    score = (correlations[0]-null)/(1-null) if null < 1 else float('nan')
    if not np.isfinite(score):
        raise ValueError('Invalid rotated-null structure score')
    return dict(correlation=correlations[0], rotated_null_correlations=correlations[1:], structure_score=float(score))


def validate_pair(reference, moving, shift, factor, roi, center):
    """Independently validate fits from two starts and coronal structure."""
    first, quality = measure_rigid(reference, moving, shift, factor, roi, 1.)
    second, _ = measure_rigid(reference, moving, np.asarray(shift)+np.array([1.5, -1.5])*factor,
                              factor, roi, 1.)
    a, b = lower_matrix(first, factor, roi), lower_matrix(second, factor, roi)
    disagreement = factor*matrix_distance(a, b, control_points(reference.mono.shape))
    if disagreement > 1.:
        raise ValueError(f'Rigid fit depends on initialization: {disagreement:.3f} native pixels')
    check = structure_check(reference, moving, a, center)
    if check['structure_score'] < .25:
        raise ValueError(f"Rotated-null structure score {check['structure_score']:.3f} < 0.25")
    quality.update(initialization_disagreement_px=disagreement, **check)
    return first, quality


def validate_band_pair(reference, moving, shift, factor, roi, center):
    """Experimental fit on the log-band auxiliaries; never updates source samples."""
    def band_frame(frame):
        data, valid = log_band(frame)
        metadata = dict(frame.metadata, exposure_s=1., white_level=1e300,
                        low_signal=-1e300, lunar_disk=None)
        metadata.pop('relative_scale', None)
        return replace(frame, data=data, metadata=metadata,
                       auxiliary=dict(coverage=valid, defective=~valid))
    first, quality = measure_rigid(band_frame(reference), band_frame(moving), shift, factor, roi, 1.)
    second, _ = measure_rigid(band_frame(reference), band_frame(moving),
                              np.asarray(shift)+np.array([1.5, -1.5])*factor, factor, roi, 1.)
    a, b = lower_matrix(first, factor, roi), lower_matrix(second, factor, roi)
    disagreement = factor*matrix_distance(a, b, control_points(reference.mono.shape))
    if disagreement > 1.:
        raise ValueError(f'Band fit depends on initialization: {disagreement:.3f} native pixels')
    check = structure_check(reference, moving, a, center)
    if check['structure_score'] < .25:
        raise ValueError(f"Rotated-null structure score {check['structure_score']:.3f} < 0.25")
    from .rigid import geometry_condition
    quality['original_geometry_condition'] = geometry_condition(reference, moving, np.asarray(shift)/factor, a)
    quality.update(initialization_disagreement_px=disagreement, **check)
    return first, quality

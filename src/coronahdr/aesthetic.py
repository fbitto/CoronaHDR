"""Explicit empirical matching and weight feathering; no science-image smoothing."""
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from .empirical import fit_empirical_match


def fit_green_gain(reference, moving, mask):
    """Estimate shared gain from green without fitting unused noisy red/blue gains."""
    green_reference = np.repeat(reference[..., 1, None], 3, axis=-1)
    green_moving = np.repeat(moving[..., 1, None], 3, axis=-1)
    _, diagnostic = fit_empirical_match(green_reference, green_moving, mask, 'gain_plane')
    return float(diagnostic['parameters'][1][0]), diagnostic


def feather_reliability(reliability, eligible, width=64, weight_sigma=0):
    """Taper source reliability toward geometry, lunar and saturation boundaries."""
    if not np.isfinite(width) or width <= 0 or reliability.shape != eligible.shape:
        raise ValueError('Feathering requires matching masks and a positive width')
    if not np.isfinite(reliability).all() or (reliability < 0).any():
        raise ValueError('Invalid source reliability')
    if not np.isfinite(weight_sigma) or weight_sigma < 0:
        raise ValueError('Weight smoothing must be nonnegative')
    if weight_sigma:
        support = gaussian_filter(eligible.astype(float), weight_sigma, mode='constant', cval=0)
        reliability = np.divide(gaussian_filter(reliability*eligible, weight_sigma, mode='constant', cval=0),
                                support, out=np.zeros_like(support), where=support > 1e-12)
    distance = distance_transform_edt(np.pad(eligible.astype(bool), 1))[1:-1, 1:-1]
    u = np.clip(distance/width, 0, 1)
    return reliability*(u*u*(3-2*u))


def apply_match(values, parameters, reference_y, reference_x, domain_yx):
    """Apply gain plus plane in declared reference coordinates, preserving negatives."""
    p = np.asarray(parameters, dtype=float)
    if p.shape != (3, 4) or not np.isfinite(p).all() or np.any(p[:, 0] <= 0):
        raise ValueError('RGB matching requires finite positive gains and three plane coefficients')
    height, width = domain_yx
    plane_x = (reference_x-(width-1)/2)/width
    plane_y = (reference_y-(height-1)/2)/height
    return values*p[:, 0]+p[:, 1]+plane_x[..., None]*p[:, 2]+plane_y[..., None]*p[:, 3]

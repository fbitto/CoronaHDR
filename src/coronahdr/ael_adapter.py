"""Diagnostic bridge from AEL's reference-to-input registration to rigid reviews."""
import cv2
import numpy as np

from . import ael_registration as ael
from .registration import valid_mask
from .rigid import apply_points, control_points, lift_matrix, lower_matrix, matrix_distance, measure_rigid


def ael_matrix_yx(center_xy, shift_xy, angle):
    """Invert AEL's reference-to-input affine to source-to-reference [y,x,1]."""
    cx, cy = np.asarray(center_xy, dtype=float)
    tx, ty = np.asarray(shift_xy, dtype=float)
    c, s = np.cos(angle), np.sin(angle)
    forward = np.array([[c, s, cy-c*cy-s*cx+ty],
                        [-s, c, cx+s*cy-c*cx+tx], [0., 0., 1.]])
    return np.linalg.inv(forward)


def estimate_pair(reference, moving, reference_moon, moving_moon, factor, roi, *, require_independent=True):
    """Run AEL's exact pair fitter, then independently check native rigid geometry."""
    def prepared(frame, moon):
        image = np.asarray(frame.mono/frame.scale, dtype=np.float32)
        valid = valid_mask(frame) & np.isfinite(image)
        image = np.where(valid, image, 0).astype(np.float32)
        return image, ~valid, moon
    r, rs, rm = prepared(reference, reference_moon)
    m, ms, mm = prepared(moving, moving_moon)
    radius = float(rm['r'])
    if radius <= 0:
        raise ValueError('Measured lunar radius must be positive')
    small = float(np.clip(3.*radius/227., 1.5, 6.))
    sigma = (small, small*10/3)
    rb, rk = ael.band_image(r, rs, rm, sigma)
    mb, mk = ael.band_image(m, ms, mm, sigma)
    y, x = np.indices(r.shape)
    radial = np.hypot(x-rm['xy'][0], y-rm['xy'][1])/radius
    ring = rk & (radial >= 1.36) & (radial <= 3.)
    weight = np.where(ring, cv2.GaussianBlur(np.maximum(r, 0), (0, 0), 3), 0).astype(np.float32)
    weight /= max(float(weight.max()), 1e-30)
    half = int(3*radius+2*sigma[1]+8)
    offset = np.rint(np.asarray(mm['xy'])-np.asarray(rm['xy'])).astype(int)
    try:
        coarse = ael.coarse_shift(rb, ring, mb, mk, rm['xy'], half, offset)
        shift, angle = ael.fine_rigid(rb, ring, weight, mb, mk, rm['xy'], half, coarse)
        second, theta2 = ael.fine_rigid(rb, ring, weight, mb, mk, rm['xy'], half, coarse+[1.5, -1.5])
        spread = max(float(np.linalg.norm(second-shift)), abs(theta2-angle)*half)*factor
        structure = ael.rigid_structure(rb, ring, mb, mk, rm['xy'], half, shift, angle)
    except (RuntimeError, np.linalg.LinAlgError) as error:
        raise ValueError(str(error)) from error
    if spread > .05*factor:
        raise ValueError(f'AEL fit depends on start: {spread:.3f} native pixels')
    if (not np.isfinite([structure['correlation'], *structure['turned'], structure['structure_score']]).all()
            or structure['structure_score'] < .25):
        raise ValueError(f"AEL structure score {structure['structure_score']:.3f} < 0.25")
    estimate = ael_matrix_yx(rm['xy'], shift, angle)
    quality = dict(ael_start_spread_native_px=spread, ael_structure=structure,
                   ael_shift_xy=shift.tolist(), ael_angle_rad=float(angle))
    if not require_independent:
        quality['qualification'] = 'AEL gates only; independent CoronaHDR geometry checks not applied'
        return lift_matrix(estimate, factor, roi), quality
    center = (np.array(r.shape)-1)/2
    initial = (apply_points(estimate, [center])[0]-center)*factor
    checked, independent_quality = measure_rigid(reference, moving, initial, factor, roi, 1.)
    disagreement = factor*matrix_distance(estimate, lower_matrix(checked, factor, roi), control_points(r.shape))
    if disagreement > 1.:
        raise ValueError(f'AEL and independent rigid fit disagree: {disagreement:.3f} native pixels')
    quality.update(independent_disagreement_native_px=disagreement, independent_fit=independent_quality)
    return lift_matrix(estimate, factor, roi), quality

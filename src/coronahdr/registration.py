"""Source-to-reference shifts are [dy, dx], with integer pixel centers at (y, x)."""
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, map_coordinates, shift
from scipy.optimize import least_squares
from skimage.registration import phase_cross_correlation


def translate(array, delta, order=1):
    displacement = tuple(delta) + ((0,) if array.ndim == 3 else ())
    return shift(array.astype(float), displacement, order=order, mode='constant', cval=0, prefilter=order > 1)


def valid_mask(frame):
    if frame.metadata['state'] not in ('linear_mono', 'linear_rgb') or not frame.metadata['black_subtracted']:
        raise ValueError('Detector samples must be calibrated and demosaiced before registration')
    masks = frame.masks()
    return masks['coverage'] & ~(masks['lunar'] | masks['saturation'] | masks['low_signal'] | masks['defective'])


def register(reference, moving, max_shift=24, fit_photometry=False):
    rm, mm = valid_mask(reference), valid_mask(moving)
    # Only auxiliary registration images are smoothed; linear source samples stay intact.
    r = gaussian_filter(np.where(rm, reference.mono/reference.scale, 0), 1)
    m = gaussian_filter(np.where(mm, moving.mono/moving.scale, 0), 1)
    coarse, _, _ = phase_cross_correlation(r, m, reference_mask=rm, moving_mask=mm,
                                            overlap_ratio=.4)
    if np.max(np.abs(coarse)) > max_shift:
        raise ValueError('Registration exceeds max_shift; increase the shift limit or select a closer reference')
    common = binary_erosion(rm & (translate(mm, coarse, 0) > .5), iterations=8)
    y, x = np.nonzero(common)
    if len(y) < 256:
        raise ValueError('Insufficient common valid coronal structure (minimum 256 pixels)')
    # Bound work while retaining samples across the independent field.
    stride = max(1, len(y)//15000)
    y, x = y[::stride], x[::stride]
    target = r[y, x]
    spread = np.std(target)
    if spread <= 1e-12:
        raise ValueError('No measurable coronal contrast')

    center = float(np.median(target))

    def residual(parameters):
        dy, dx = parameters[:2]
        sample = map_coordinates(m, [y-dy, x-dx], order=3, mode='constant', cval=0)
        if fit_photometry:
            gain, offset = parameters[2:]
            return gain*(sample-center)/spread+offset-(target-center)/spread
        return (sample-target)/spread

    initial = np.r_[coarse, 1., 0.] if fit_photometry else coarse
    lower = np.r_[coarse-1.5, .25, -20.] if fit_photometry else coarse-1.5
    upper = np.r_[coarse+1.5, 4., 20.] if fit_photometry else coarse+1.5
    fit = least_squares(residual, initial, bounds=(lower, upper),
                        loss='soft_l1', f_scale=.1, diff_step=1e-3,
                        xtol=1e-10, ftol=1e-10, gtol=1e-10)
    errors = residual(fit.x)
    sample = target + errors*spread
    correlation = float(np.corrcoef(target, sample)[0, 1])
    relative_residual = float(np.median(np.abs(sample-target))/max(np.median(np.abs(target)), 1e-12))
    jacobian = fit.jac[:, :2]
    if fit_photometry:
        nuisance = fit.jac[:, 2:]
        jacobian = jacobian-nuisance @ np.linalg.lstsq(nuisance, jacobian, rcond=None)[0]
    condition = float(np.linalg.cond(jacobian.T @ jacobian))
    if (not fit.success or not np.isfinite(correlation) or correlation < .97
            or relative_residual > .15 or condition > 1000
            or np.max(np.abs(fit.x[:2]-coarse)) > 1.45):
        raise ValueError(f'Unreliable coronal registration: correlation={correlation:.4f}, relative residual={relative_residual:.4f}, condition={condition:.1f}')
    # Check regional radiometric agreement: a global ratio cannot correct variable clouds.
    regional_errors = []
    for region in ((y < r.shape[0]/2) & (x < r.shape[1]/2),
                   (y < r.shape[0]/2) & (x >= r.shape[1]/2),
                   (y >= r.shape[0]/2) & (x < r.shape[1]/2),
                   (y >= r.shape[0]/2) & (x >= r.shape[1]/2)):
        if region.sum() >= 64:
            regional_errors.append(float(np.median(np.abs(sample[region]-target[region])) /
                                         max(np.median(np.abs(target[region])), 1e-12)))
    if regional_errors and max(regional_errors) > .2:
        raise ValueError('Spatial radiometric disagreement; possible clouds or incompatible response')
    auxiliary_photometry = (dict(gain=float(fit.x[2]), offset_reference_units=float(fit.x[3]*spread+center*(1-fit.x[2])),
        applied_to_master=False) if fit_photometry else None)
    return fit.x[:2], dict(auxiliary_photometry=auxiliary_photometry, correlation=correlation, relative_residual=relative_residual,
                      condition=condition, common_samples=len(y), regional_errors=regional_errors,
                      coarse_dy_dx=coarse.tolist(), shift_dy_dx=fit.x[:2].tolist())

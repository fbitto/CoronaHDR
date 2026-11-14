"""Translation paths measured on corona only; radiometric fits are diagnostics."""
import numpy as np
from scipy.ndimage import binary_erosion, map_coordinates
from scipy.optimize import least_squares
from .registration import register, valid_mask, translate


def radiometry(reference, moving, displacement):
    common = binary_erosion(valid_mask(reference) & (translate(valid_mask(moving), displacement, 0) > .5), iterations=3)
    y, x = np.nonzero(common)
    stride = max(1, len(y)//20000)
    y, x = y[::stride], x[::stride]
    if len(y) < 64:
        return dict(samples=len(y), status='insufficient overlap')
    target = reference.mono[y, x]/reference.scale
    sample = map_coordinates(moving.mono/moving.scale, [y-displacement[0], x-displacement[1]], order=1, mode='constant', cval=0)
    valid = np.isfinite(target) & np.isfinite(sample)
    target, sample, y, x = target[valid], sample[valid], y[valid], x[valid]
    spread = max(float(np.std(sample)), 1e-12)
    center = float(np.median(sample))
    fit = least_squares(lambda p: (p[0]*(sample-center)+p[1]-target)/spread,
                        [1., float(np.median(target))], loss='soft_l1', f_scale=.1)
    slope = float(fit.x[0])
    offset = float(fit.x[1]-slope*center)
    raw_error = float(np.median(np.abs(target-sample))/max(np.median(np.abs(target)), 1e-12))
    fitted_error = float(np.median(np.abs(target-(slope*sample+offset)))/max(np.median(np.abs(target)), 1e-12))
    regional = []
    raw_regional = []
    for quadrant in ((y < reference.data.shape[0]/2)&(x < reference.data.shape[1]/2),
                     (y < reference.data.shape[0]/2)&(x >= reference.data.shape[1]/2),
                     (y >= reference.data.shape[0]/2)&(x < reference.data.shape[1]/2),
                     (y >= reference.data.shape[0]/2)&(x >= reference.data.shape[1]/2)):
        if quadrant.sum() >= 64:
            raw_regional.append(float(np.median(np.abs(target[quadrant]-sample[quadrant])) /
                max(np.median(np.abs(target[quadrant])), 1e-12)))
            residual = target[quadrant]-(slope*sample[quadrant]+offset)
            regional.append(float(np.median(residual)))
    return dict(samples=len(target), raw_relative_error=raw_error, diagnostic_slope=slope,
                diagnostic_offset_reference_units=offset, diagnostic_fitted_relative_error=fitted_error,
                quadrant_median_residuals=regional, raw_regional_relative_errors=raw_regional, applied=False,
                interpretation='Slope/offset fit is diagnostic only; it does not establish a causal correction')


def solve_translations(auxiliaries, reference, factor, max_shift, bridges=True, cycle_tolerance=.5, geometry_photometry=False):
    n = len(auxiliaries)
    if not 0 <= reference < n:
        raise ValueError('Invalid registration reference')
    solved = {reference: np.zeros(2)}
    metrics = {reference: dict(reference=True, shift_dy_dx=[0., 0.], path=[reference])}
    failures, edges, attempts = {}, {}, []
    pending = set(range(n))-{reference}
    while pending:
        progress = False
        for child in sorted(pending, key=lambda i: abs(np.log(auxiliaries[i].scale/auxiliaries[reference].scale))):
            parents = [reference] if not bridges else sorted(solved, key=lambda i: abs(np.log(auxiliaries[i].scale/auxiliaries[child].scale)))
            candidates = []
            for parent in parents:
                if (parent, child) in edges:
                    outcome = edges[parent, child]
                else:
                    try:
                        delta, quality = register(auxiliaries[parent], auxiliaries[child], max_shift/factor, fit_photometry=True) if geometry_photometry else register(auxiliaries[parent], auxiliaries[child], max_shift/factor)
                        delta = delta*factor
                        quality['radiometry'] = radiometry(auxiliaries[parent], auxiliaries[child], delta/factor)
                        outcome = (delta, quality)
                        attempts.append(dict(parent=parent, child=child, accepted=True, **quality))
                    except ValueError as error:
                        outcome = str(error)
                        attempts.append(dict(parent=parent, child=child, accepted=False, reason=str(error)))
                    edges[parent, child] = outcome
                if isinstance(outcome, str):
                    failures[child] = outcome
                    continue
                delta, quality = outcome
                composed = solved[parent]+delta
                candidates.append((parent, composed, quality))
                if len(candidates) == 2:
                    break
            if not candidates:
                continue
            disagreement = None
            if len(candidates) > 1:
                disagreement = float(np.linalg.norm(candidates[0][1]-candidates[1][1]))
                if disagreement > cycle_tolerance:
                    failures[child] = f'Inconsistent translation paths: {disagreement:.4f} px > {cycle_tolerance}'
                    continue
            parent, composed, quality = min(candidates, key=lambda c: c[2]['relative_residual'])
            solved[child] = composed
            metrics[child] = dict(quality, shift_dy_dx=composed.tolist(), parent=parent,
                path=[child, *metrics[parent]['path']], cycle_disagreement_px=disagreement,
                consistency_status='two paths checked' if disagreement is not None else 'single path; no independent cycle available')
            pending.remove(child)
            progress = True
        if not progress:
            break
    return solved, metrics, [dict(index=i, reason=failures.get(i, 'No reliable bridge path')) for i in sorted(pending)], attempts


def qualify_radiometry(auxiliaries, solved, metrics, reference, factor, attempts):
    """Recheck each composed transform against unmodified reference radiance."""
    rejected, checks = [], {}
    accepted = {reference: solved[reference]}
    for index, displacement in solved.items():
        if index == reference:
            continue
        check = radiometry(auxiliaries[reference], auxiliaries[index], displacement/factor)
        checks[index] = check
        if check.get('status') == 'insufficient overlap':
            path = metrics[index]['path']
            edges = [next((edge.get('radiometry', {}) for edge in attempts
                           if edge['parent'] == parent and edge['child'] == child and edge['accepted']), {})
                     for child, parent in zip(path[:-1], path[1:])]
            good = bool(edges) and all(edge.get('raw_relative_error', 1) <= .15 and
                       max(edge.get('raw_regional_relative_errors', [1]), default=1) <= .2 for edge in edges)
            check['qualification'] = 'path radiometry only; direct reference overlap insufficient'
        else:
            good = check['raw_relative_error'] <= .15 and max(check.get('raw_regional_relative_errors', []), default=0) <= .2
            check['qualification'] = 'direct composed-transform comparison to reference'
        if good:
            accepted[index] = displacement
        else:
            rejected.append(dict(index=index, reason='Reference-level radiometric disagreement', diagnostic=check))
    return accepted, rejected, checks

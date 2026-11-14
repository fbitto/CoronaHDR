"""Conservative recovery audits; spatial sky differences never change camera gain."""
import numpy as np
from scipy.ndimage import map_coordinates, binary_erosion
from scipy.optimize import least_squares
from .registration import register
from .streaming import slice_input


def analysis_image(frame, factor=8):
    """Unbiased RGB area means: low/negative sky samples must not be truncated."""
    height, width = frame.data.shape[:2]
    oh, ow = height//factor, width//factor
    image = np.empty((oh, ow, 3))
    usable = np.zeros((oh, ow), bool)
    for row in range(0, oh, 16):
        count = min(16, oh-row)
        start = row*factor
        size = max(32, count*factor)
        lo = min(start, height-size)
        tile = slice_input(frame, lo, lo+size)
        section = slice(start-lo, start-lo+count*factor)
        data = tile.data[section, :ow*factor]
        if data.ndim != 3 or data.shape[-1] != 3:
            raise ValueError('Sky recovery currently requires explicit linear RGB')
        masks = tile.masks()
        good = masks['coverage'] & ~masks['saturation'] & ~masks['defective']
        image[row:row+count] = np.nan_to_num(data).reshape(count, factor, ow, factor, 3).mean(axis=(1, 3))/frame.scale
        usable[row:row+count] = good[section, :ow*factor].reshape(count, factor, ow, factor).all(axis=(1, 3))
    return image, usable


def align_analysis(image, mask, displacement, factor):
    y, x = np.indices(mask.shape)
    coordinates = [y-displacement[0]/factor, x-displacement[1]/factor]
    result = np.stack([map_coordinates(image[..., c], coordinates, order=1, mode='constant', cval=0) for c in range(3)], axis=-1)
    valid = map_coordinates(mask.astype(float), coordinates, order=1, mode='constant', cval=0) >= 1-1e-9
    return result, binary_erosion(valid, iterations=2)


def robust_plane(design, values):
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values-design @ coefficients
    scale = np.maximum(1.4826*np.median(np.abs(residual-np.median(residual, axis=0)), axis=0), 1e-6)
    for channel in range(3):
        fit = least_squares(lambda p: (design @ p-values[:, channel])/scale[channel],
                            coefficients[:, channel], loss='soft_l1', f_scale=1.)
        if not fit.success:
            raise ValueError('Sky plane did not converge')
        coefficients[:, channel] = fit.x
    return coefficients


def block_rms(residual, mask, pixels_per_block):
    means = []
    for row in range(0, mask.shape[0], pixels_per_block):
        for col in range(0, mask.shape[1], pixels_per_block):
            selected = mask[row:row+pixels_per_block, col:col+pixels_per_block]
            if selected.sum() >= 16:
                values = residual[row:row+pixels_per_block, col:col+pixels_per_block][selected]
                means.append(values.mean(axis=0))
    if len(means) < 8:
        return np.full(3, np.inf), len(means)
    return np.sqrt(np.mean(np.asarray(means)**2, axis=0)), len(means)


def audit_sky_difference(reference, moving, reference_mask, moving_mask, displacement,
                         reference_metadata, moving_metadata, full_shape, factor=8):
    """Fit far-field differences; validate independent sky, lunar and coronal data."""
    aligned, moving_good = align_analysis(moving, moving_mask, displacement, factor)
    valid = binary_erosion(reference_mask & moving_good, iterations=1)
    height, width = full_shape[:2]
    yy, xx = np.indices(valid.shape, dtype=float)
    x, y = xx*factor+(factor-1)/2, yy*factor+(factor-1)/2
    center = [(width-1)/2, (height-1)/2]
    scales = [width/2, height/2]
    design = np.stack([np.ones(valid.shape), (x-center[0])/scales[0], (y-center[1])/scales[1]], axis=-1)
    if not reference_metadata.get('lunar_disk') or not moving_metadata.get('lunar_disk'):
        raise ValueError('Independent sky validation requires explicit lunar disks for both frames')
    rcx, rcy, radius = reference_metadata['lunar_disk']
    mcx, mcy, moving_radius = moving_metadata['lunar_disk']
    mcx, mcy = mcx+displacement[1], mcy+displacement[0]
    r = np.hypot(x-rcx, y-rcy)/radius
    mr = np.hypot(x-mcx, y-mcy)/moving_radius
    # Spatial blocks, not alternating neighboring samples, define independent validation.
    blocks = ((x//128+y//128).astype(int) % 2) == 0
    far = valid & (r > 6) & (mr > 6)
    train, held = far & blocks, far & ~blocks
    result = dict(model='differential additive RGB plane in reference radiance units',
        factor=factor, source_to_reference_shift_dy_dx=np.asarray(displacement).tolist(),
        normalizing_center_xy=center, normalizing_scale_xy=scales,
        training_pixels=int(train.sum()), heldout_sky_pixels=int(held.sum()),
        gain_applied=False, status='rejected', rejection_reasons=[])
    if min(train.sum(), held.sum()) < 1000:
        result['rejection_reasons'].append('Insufficient independent far-field support')
        return result
    difference = reference-aligned
    coefficients = robust_plane(design[train], difference[train])
    plane = design @ coefficients
    corrected = aligned+plane
    residual = corrected-reference
    sky_scale = np.maximum(np.median(np.abs(reference[held]), axis=0), 1e-6)
    sky_rmse = np.sqrt(np.mean(residual[held]**2, axis=0))/sky_scale
    blocked_sky, sky_blocks = block_rms(residual, held, max(1, 128//factor))
    if sky_blocks < 8:
        result['rejection_reasons'].append('Insufficient independent sky blocks')
        return result
    blocked_sky /= sky_scale
    result.update(coefficients=coefficients.tolist(), heldout_sky_relative_rmse=sky_rmse.tolist(),
        heldout_sky_block_relative_rmse=blocked_sky.tolist(), heldout_sky_blocks=sky_blocks)
    if np.max(blocked_sky) > .03:
        result['rejection_reasons'].append('Held-out sky RMS exceeds 3%')
    # The deep common lunar disk supplies an independently occulted sky control.
    lunar = valid & (r < .45) & (mr < .45)
    result['lunar_validation_pixels'] = int(lunar.sum())
    if lunar.sum() < 100:
        result['rejection_reasons'].append('Insufficient deep common lunar sky control')
    else:
        lunar_bias = np.abs(np.median(residual[lunar], axis=0))/np.maximum(np.median(np.abs(reference[lunar]), axis=0), 1e-6)
        result['lunar_relative_median_bias'] = lunar_bias.tolist()
        if np.max(lunar_bias) > .03:
            result['rejection_reasons'].append('Independent lunar sky bias exceeds 3%')
    corona = valid & (r >= 1.5) & (r < 6) & (mr >= 1.5)
    result['coronal_validation_pixels'] = int(corona.sum())
    if corona.sum() < 1000:
        result['rejection_reasons'].append('Insufficient independent coronal validation')
        return result
    spread = np.maximum(np.std(reference[corona], axis=0), 1e-6)
    coronal_rmse = np.sqrt(np.mean(residual[corona]**2, axis=0))/spread
    gains = []
    for c in range(3):
        target, sample = reference[..., c][corona], corrected[..., c][corona]
        gains.append(float(np.cov(target, sample, ddof=0)[0, 1]/max(np.var(target), 1e-12)))
    blocked_corona, corona_blocks = block_rms(residual, corona, max(1, 64//factor))
    if corona_blocks < 8:
        result['rejection_reasons'].append('Insufficient independent coronal blocks')
        return result
    blocked_corona /= spread
    result.update(coronal_relative_contrast_rmse=coronal_rmse.tolist(), coronal_contrast_ratios=gains,
        coronal_block_relative_contrast_rmse=blocked_corona.tolist(), coronal_blocks=corona_blocks)
    if np.max(blocked_corona) > .05:
        result['rejection_reasons'].append('Coronal contrast RMS exceeds 5%')
    if np.max(np.abs(np.asarray(gains)-1)) > .05:
        result['rejection_reasons'].append('Coronal contrast ratio differs by more than 5%')
    angle = (np.arctan2(y-rcy, x-rcx) % (2*np.pi))
    cells = []
    for inner, outer in ((1.5, 2.5), (2.5, 4.), (4., 6.)):
        for sector in range(8):
            region = corona & (r >= inner) & (r < outer) & (angle >= sector*np.pi/4) & (angle < (sector+1)*np.pi/4)
            if region.sum() < 20:
                continue
            bias = np.abs(np.median(residual[region], axis=0))/spread
            cell_rmse, cell_blocks = block_rms(residual, region, max(1,64//factor))
            cell_rmse /= spread
            if cell_blocks < 8:
                cell_rmse = np.sqrt(np.mean(residual[region]**2, axis=0))/spread
            cells.append(dict(validation_blocks=cell_blocks, radius_range=[inner, outer], sector=sector, samples=int(region.sum()),
                relative_median_bias=bias.tolist(), relative_contrast_rmse=cell_rmse.tolist(),
                passed=bool(np.max(bias) <= .03 and np.max(cell_rmse) <= .05)))
    result['coronal_cells'] = cells
    if not cells or not all(cell['passed'] for cell in cells):
        result['rejection_reasons'].append('One or more independent coronal sectors failed')
    result['status'] = 'validated' if not result['rejection_reasons'] else 'rejected'
    result['applied'] = False
    return result


def review_registration(auxiliaries, solved, child, factor, max_shift, tolerance=1.):
    """Require a three-parent consensus rather than accepting a disputed pair."""
    candidates = []
    parents = sorted((i for i in solved if i != child),
        key=lambda i: (abs(np.log(auxiliaries[i].scale/auxiliaries[child].scale)), abs(i-child)))[:10]
    for parent in parents:
        try:
            edge, quality = register(auxiliaries[parent], auxiliaries[child], max_shift/factor, fit_photometry=True)
            candidate = solved[parent]+edge*factor
            candidates.append(dict(parent=parent, shift_dy_dx=candidate.tolist(), edge_quality=quality))
        except ValueError as error:
            candidates.append(dict(parent=parent, rejected=str(error)))
    good = [candidate for candidate in candidates if 'shift_dy_dx' in candidate]
    if len(good) < 3:
        return dict(status='rejected', reason='Fewer than three valid parent measurements', candidates=candidates)
    points = np.array([candidate['shift_dy_dx'] for candidate in good])
    clusters = [[j for j in range(len(points)) if np.linalg.norm(points[j]-point) <= tolerance/2] for point in points]
    cluster = max(clusters, key=len)
    if len(cluster) < 3:
        return dict(status='rejected', reason='No three-parent consensus within the existing tolerance', candidates=candidates)
    selected = points[cluster]
    center = np.median(selected, axis=0)
    competing = [group for group in clusters if len(group) >= 3 and not set(group)&set(cluster)]
    if competing:
        return dict(status='rejected', reason='Competing registration consensuses', candidates=candidates)
    return dict(status='consensus', shift_dy_dx=center.tolist(), consensus_parents=[good[j]['parent'] for j in cluster],
                max_consensus_residual_px=float(np.linalg.norm(selected-center, axis=1).max()), candidates=candidates)


def reviewed_translations(path, input_records, reference, indices):
    """Express audited full-field translations relative to a local bracket reference."""
    import json
    from pathlib import Path
    audit = json.loads(Path(path).read_text())
    expected = [item['metadata']['source_sha256'] for item in input_records]
    if audit.get('input_source_sha256') != expected:
        raise ValueError('Registration review does not match the prepared inputs')
    available = {int(i):np.asarray(shift, dtype=float) for i,shift in audit['transformations'].items()}
    if reference not in available:
        raise ValueError('Local reference has no audited translation')
    reviewed = {item['index']:item for item in audit['reviews']}
    solved, metrics, rejected = {}, {}, []
    for index in indices:
        if index not in available:
            rejected.append(dict(index=index, reason='No audited coronal translation'))
            continue
        if str(index) not in audit['original_metrics']:
            consensus = reviewed.get(index, {})
            if consensus.get('status') != 'consensus' or len(consensus.get('consensus_parents', [])) < 3:
                rejected.append(dict(index=index, reason='Registration review lacks three-parent consensus'))
                continue
        displacement = available[index]-available[reference]
        solved[index] = displacement
        original = audit['original_metrics'].get(str(index), {})
        metrics[index] = dict(original, shift_dy_dx=displacement.tolist(), reference=index==reference,
            path=[reference] if index==reference else [],
            path_to_review_reference=original.get('path'), review_reference=audit['reference'],
            registration_review=str(path),
            shift_derivation='difference of audited source-to-common-reference translations; no image resampling',
            review_consensus=reviewed.get(index),
            consistency_status='audited global paths; local radiometry is checked independently')
    return solved, metrics, rejected, []


def read_noise_from_bias(samples, labels):
    """Paired robust temporal noise in source DN, not an electron-gain model."""
    from .sensor import PHASES
    if samples.ndim != 2 or len(samples) < 4 or len(samples) % 2 or samples.shape[1] != len(labels):
        raise ValueError('Bias noise needs an even number of at least four matched sample vectors')
    difference = samples[::2].astype(float)-samples[1::2].astype(float)
    result = {}
    for phase in PHASES:
        values = difference[:, labels == phase]
        if values.shape[1] < 100:
            raise ValueError('Insufficient bias phase samples')
        centers = np.median(values, axis=1)
        sigmas = 1.4826*np.median(np.abs(values-centers[:,None]), axis=1)/np.sqrt(2)
        result[phase] = dict(temporal_noise_dn=float(np.median(sigmas)), pair_estimates_dn=sigmas.tolist(),
                            pair_median_offsets_dn=centers.tolist())
    return result

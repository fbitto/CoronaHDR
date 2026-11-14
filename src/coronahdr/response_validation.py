"""Conservative local controls for experimental response matching, not calibration."""
import hashlib
from pathlib import Path

import numpy as np


POLICY = dict(version=1, sector_count=8, guard_native_px=32, minimum_cell_pixels=64,
              maximum_standard_error_fraction=.005, effective_sample_divisor=16,
              maximum_cell_level_bias=.02, maximum_local_error_ratio=1.03,
              maximum_gain_split=.03, maximum_offset_split_over_sky=.01,
              minimum_validated_cells=32, minimum_validated_fold_cells=16,
              radial_edges=[1.12, 1.5, 2., 3., 5., 8.])


def sector_split(y, x, center_yx, guard=32):
    """Alternate complete 45-degree sectors, excluding a native-distance guard."""
    dy, dx = y-center_yx[0], x-center_yx[1]
    angle = np.mod(np.arctan2(dy, dx), 2*np.pi)
    width = np.pi/4
    within = np.mod(angle, width)
    sectors = (angle//width).astype(int)
    distance = np.hypot(dy, dx)*np.sin(np.minimum(within, width-within))
    guarded = distance >= guard
    return sectors, guarded & (sectors % 2 == 0), guarded & (sectors % 2 == 1)


def coefficient_stability(first, second, sky):
    gains = np.abs(np.array([c['gain'] for c in first])-np.array([c['gain'] for c in second]))
    offset = np.abs(np.array([c['offset_xy_plane'] for c in first])-np.array([c['offset_xy_plane'] for c in second]))
    # Bound the whole normalized field, including plane slopes if present.
    bound = offset[:, 0]+.5*offset[:, 1:].sum(axis=1)
    relative = bound/np.maximum(np.asarray(sky), 1)
    return dict(gain_difference_rgb=gains.tolist(), offset_field_bound_over_sky_rgb=relative.tolist(),
                passed=bool(np.all(gains <= POLICY['maximum_gain_split']) and
                            np.all(relative <= POLICY['maximum_offset_split_over_sky'])))


def local_controls(reference, moving, corrected, mask, radius, sectors, *, fold=False):
    """Keep every cell result; a failed sector cannot be hidden by other sectors."""
    records = []
    for low, high in zip(POLICY['radial_edges'][:-1], POLICY['radial_edges'][1:]):
        for sector in range(POLICY['sector_count']):
            valid = mask & (radius >= low) & (radius < high) & (sectors == sector)
            count = int(valid.sum())
            item = dict(radius=[low, high], sector=sector, pixels=count, status='unvalidated')
            if count >= POLICY['minimum_cell_pixels']:
                ref, before, after = reference[valid], moving[valid]-reference[valid], corrected[valid]-reference[valid]
                level = np.maximum(np.abs(np.median(ref, axis=0)), 100.)
                proxy = 1.4826*np.median(np.abs(before-np.median(before, axis=0)), axis=0)
                standard_error = proxy/np.sqrt(count/POLICY['effective_sample_divisor'])/level
                item['standard_error_proxy_fraction_rgb'] = standard_error.tolist()
                if np.all(standard_error <= POLICY['maximum_standard_error_fraction']):
                    signed = np.median(after, axis=0)/level
                    before_error = np.median(np.abs(before), axis=0)
                    after_error = np.median(np.abs(after), axis=0)
                    passed = np.all(np.abs(signed) <= POLICY['maximum_cell_level_bias']) and np.all(after_error <= POLICY['maximum_local_error_ratio']*before_error+1e-7)
                    item.update(status='passed' if passed else 'failed', level_bias_fraction_rgb=signed.tolist(),
                                before_absolute_error_rgb=before_error.tolist(), after_absolute_error_rgb=after_error.tolist())
            records.append(item)
    validated = [r for r in records if r['status'] != 'unvalidated']
    failed = [r for r in records if r['status'] == 'failed']
    return dict(cells=records, validated_cells=len(validated), failed_cells=len(failed),
                unvalidated_cells=len(records)-len(validated),
                passed=bool(len(validated) >= POLICY['minimum_validated_fold_cells' if fold else 'minimum_validated_cells'] and not failed))


def require_strict_receipt(receipt, audit_path, gains, offsets):
    """An old or mismatched preliminary report cannot authorize a new export."""
    digest = hashlib.sha256(Path(audit_path).read_bytes()).hexdigest()
    if receipt.get('policy') != POLICY or receipt.get('source_audit_sha256') != digest:
        raise ValueError('Missing, obsolete or mismatched strict response validation')
    if not receipt.get('accepted') or not receipt.get('synthetic_controls_passed'):
        raise ValueError('Strict response validation rejected this correction; retain the original master')
    if not np.array_equal(receipt.get('gains_rgb'), gains) or not np.array_equal(receipt.get('offsets_rgb'), offsets):
        raise ValueError('Strict response receipt coefficients do not match the export')

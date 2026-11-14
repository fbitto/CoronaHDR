"""Known-truth falsification controls for the separate response experiment."""
from pathlib import Path
import runpy

import numpy as np
import pytest
from coronahdr.response_validation import POLICY, coefficient_stability, local_controls, require_strict_receipt, sector_split


if not (Path(__file__).parents[1]/'scripts/evaluate_short_response.py').is_file():
    pytest.skip('Private research script is not included in the public checkout', allow_module_level=True)


fit_response = runpy.run_path(str(Path(__file__).parents[1]/'scripts/evaluate_short_response.py'))['fit_response']


def test_affine_recovers_response_and_preserves_negative_values_and_source():
    rng = np.random.default_rng(8)
    moving = rng.normal(50, 100, (64, 64, 3))
    source = moving.copy()
    truth = moving*np.array([1.07, 1.06, 1.04])+[18, 48, 70]
    result, coefficients = fit_response(truth, moving, np.ones((64, 64), bool), 'affine')
    np.testing.assert_allclose(result, truth, atol=1e-8)
    np.testing.assert_array_equal(moving, source)
    assert (result < 0).any()
    np.testing.assert_allclose([c['gain'] for c in coefficients], [1.07, 1.06, 1.04], atol=1e-10)


def test_coronal_gain_can_fail_an_independent_background_control():
    rng = np.random.default_rng(18)
    reference = rng.uniform(1000, 3000, (64, 64, 3))
    reference[:16] = 50
    moving = reference-50
    train = np.ones((64, 64), bool)
    train[:16] = False
    gain, _ = fit_response(reference, moving, train, 'gain')
    offset, _ = fit_response(reference, moving, train, 'offset')
    assert np.median(np.abs(gain[:16]-reference[:16])) == 50
    np.testing.assert_allclose(offset, reference, atol=1e-9)


def test_response_requires_supported_model_and_sufficient_samples():
    a = np.ones((32, 32, 3))
    with pytest.raises(ValueError, match='Unknown'):
        fit_response(a, a, np.ones((32, 32), bool), 'polynomial')
    with pytest.raises(ValueError, match='Insufficient'):
        fit_response(a, a, np.zeros((32, 32), bool), 'affine')


def test_a_local_failure_cannot_hide_in_a_good_global_median():
    y, x = np.indices((256, 256))
    radius = np.hypot(y-128, x-128)/25
    sectors, first, second = sector_split(y, x, [128, 128], 0)
    assert not (first & second).any()
    reference = np.full((*y.shape, 3), 1000.)
    moving = reference-100
    candidate = reference.copy()
    candidate[(sectors == 0) & (radius > 2) & (radius < 3)] += 50
    assert np.median(candidate-reference) == 0
    result = local_controls(reference, moving, candidate, radius > 1.12, radius, sectors)
    assert result['failed_cells'] > 0 and not result['passed']


def test_offset_stability_matters_even_when_gains_agree():
    first = [dict(gain=1.06, offset_xy_plane=[0, 0, 0]) for _ in range(3)]
    second = [dict(gain=1.06, offset_xy_plane=[20, 0, 0]) for _ in range(3)]
    assert not coefficient_stability(first, second, [1000]*3)['passed']


def test_export_rejects_a_preliminary_pass_without_strict_acceptance(tmp_path):
    import hashlib
    audit = tmp_path/'audit.json'
    audit.write_text('{}')
    receipt = dict(policy=POLICY, source_audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
                   accepted=False, synthetic_controls_passed=True, gains_rgb=[1]*3, offsets_rgb=[0]*3)
    with pytest.raises(ValueError, match='rejected'):
        require_strict_receipt(receipt, audit, np.ones(3), np.zeros(3))
    receipt['accepted'] = True
    require_strict_receipt(receipt, audit, np.ones(3), np.zeros(3))
    with pytest.raises(ValueError, match='coefficients'):
        require_strict_receipt(receipt, audit, np.ones(3)*1.1, np.zeros(3))
    audit.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='mismatched'):
        require_strict_receipt(receipt, audit, np.ones(3), np.zeros(3))

"""Controls for temporal response comparison and noisy gain estimation."""
from pathlib import Path
import runpy

import numpy as np
import pytest


if not (Path(__file__).parents[1]/'scripts/audit_bracket_response.py').is_file():
    pytest.skip('Private research script is not included in the public checkout', allow_module_level=True)


audit = runpy.run_path(str(Path(__file__).parents[1]/'scripts/audit_bracket_response.py'))


def test_binning_exposes_gain_attenuation_without_changing_inputs():
    y, x = np.indices((256, 256))
    signal = 2000+100*np.sin(x/20)+50*np.cos(y/15)
    reference = np.repeat(signal[..., None], 3, axis=-1)
    moving = reference/1.1+np.random.default_rng(44).normal(0, 50, reference.shape)
    original = moving.copy()
    results = audit['resolution_check'](reference, moving, np.ones(y.shape, bool))
    gains = np.array([r['gains_rgb'] for r in results])
    assert np.all(gains[-1] > gains[0]+.1)
    assert np.all(np.abs(gains[-1]-1.1) < np.abs(gains[0]-1.1))
    np.testing.assert_array_equal(moving, original)


def test_noiseless_response_is_invariant_on_matched_footprints():
    y, x = np.indices((128, 128))
    reference = np.repeat((100+2*x+3*y)[..., None], 3, axis=-1).astype(float)
    mask = np.ones(y.shape, bool)
    mask[:13, :19] = False
    results = audit['resolution_check'](reference, (reference-12)/1.08, mask)
    for result in results:
        np.testing.assert_allclose(result['gains_rgb'], 1.08, atol=1e-9)


def test_repeat_summary_preserves_spatial_failures_and_unmeasured_cases():
    records = [dict(status='measured', exposures_s=[.004, .25], indices=[1, 2],
                    gains_rgb=[1, 1, 1], spatial_stability=dict(passed=True)),
               dict(status='measured', exposures_s=[.004, .25], indices=[3, 4],
                    gains_rgb=[1.01, 1, 1], spatial_stability=dict(passed=False)),
               dict(status='unmeasured', exposures_s=[.004, .25], indices=[5, 6])]
    result = audit['repeat_summary'](records)[0]
    assert result['within_existing_3_percent_gain_bound']
    assert not result['all_spatial_splits_stable']
    assert result['pairs'] == [[1, 2], [3, 4]]

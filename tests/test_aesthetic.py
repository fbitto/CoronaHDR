"""Known-truth checks for empirical matching and seamless masked integration."""
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from coronahdr.aesthetic import apply_match, feather_reliability, fit_green_gain
from coronahdr.streaming import CachedFrame, accumulate_tiled


def test_feather_never_invents_support_and_has_a_smooth_boundary():
    eligible = np.ones((160, 160), bool)
    eligible[:, :40] = False
    weight = feather_reliability(np.ones(eligible.shape), eligible, 32)
    assert not weight[:, :40].any()
    assert np.all(np.diff(weight[80, 40:100]) >= 0)
    assert np.max(np.diff(weight[80, 40:100])) < .05
    assert weight[80, 80] == 1


def test_match_changes_only_declared_gain_and_plane_and_preserves_negative_detail():
    y, x = np.indices((64, 64))
    values = np.repeat((-20+10*np.sin(x))[..., None], 3, axis=-1)
    before = values.copy()
    parameters = [[1.1, 4, 3, -2]]*3
    result = apply_match(values, parameters, y, x, y.shape)
    expected = 1.1*values+4+((x-31.5)/64*3-(y-31.5)/64*2)[..., None]
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(values, before)
    assert (result < 0).any()


def test_smoothing_changes_weights_only_and_excludes_invalid_samples():
    eligible = np.ones((64, 64), bool)
    eligible[:, :16] = False
    reliability = np.ones(eligible.shape)
    reliability[:, :16] = 1e6
    original = reliability.copy()
    result = feather_reliability(reliability, eligible, 8, weight_sigma=4)
    np.testing.assert_array_equal(reliability, original)
    assert not result[:, :16].any()
    np.testing.assert_allclose(result[32, 32:48], 1, atol=1e-8)


def test_nonfinite_weight_parameters_cannot_silently_remove_support():
    a = np.ones((16, 16))
    with pytest.raises(ValueError):
        feather_reliability(a, a > 0, np.nan)
    with pytest.raises(ValueError):
        feather_reliability(a, a > 0, 8, np.inf)


def test_shared_gain_is_unaffected_by_unusable_blue_data():
    y, x = np.indices((96, 96))
    reference = np.repeat((1000+100*np.sin(x/3)+50*np.cos(y/7)+4*y)[..., None], 3, axis=-1).astype(float)
    moving = (reference-20)/1.08
    moving[..., 2] = 0
    gain, _ = fit_green_gain(reference, moving, np.ones(y.shape, bool))
    np.testing.assert_allclose(gain, 1.08, atol=1e-8)


def test_empirical_tiled_blend_keeps_injected_filaments_and_missing_pixels(tmp_path):
    y, x = np.indices((64, 64))
    truth = np.repeat((-4+20*np.sin(x/3)+y)[..., None], 3, axis=-1)
    sources = [truth, (truth-10)/1.2]
    frames = []
    for i, data in enumerate(sources):
        coverage = np.ones(y.shape, bool)
        coverage[:4] = False
        auxiliary = {name: np.zeros(y.shape, bool) for name in ('lunar', 'saturation', 'low_signal', 'defective')}
        auxiliary['coverage'] = coverage
        auxiliary['soft_weight'] = feather_reliability(np.ones(y.shape), coverage, 8)
        frames.append(CachedFrame(data.copy(), dict(exposure_s=1), Path(str(i)), auxiliary))
    output, work = tmp_path/'output', tmp_path/'work'
    output.mkdir(); work.mkdir()
    def correct(i, values, start, end):
        return values*1.2+10 if i else values
    result = accumulate_tiled(frames, [np.eye(3)]*2, np.zeros(2), y.shape, output, work, 32, radiance_correction=correct)
    np.testing.assert_allclose(result[4:], truth[4:], atol=1e-5)
    assert np.isnan(result[:4]).all()
    assert not fits.getdata(output/'support.fits')[:4].any()

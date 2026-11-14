"""Known-truth checks for the experimental rigid registration guards."""
from pathlib import Path

import numpy as np
import pytest

from coronahdr.empirical import warp_rigid
from coronahdr.model import Frame
from coronahdr.rigid import control_points, matrix_distance, matrix_yx
from coronahdr.rigid_features import correlation_seed, log_band, structure_check, validate_band_pair, validate_pair
from coronahdr.synthetic import corona


def frame(data, mask):
    return Frame(data, dict(state='linear_mono', exposure_s=1., white_level=1e6,
        black_level=0., black_subtracted=True, response_group='synthetic',
        lunar_disk=None, low_signal=-1.), Path('synthetic'), dict(coverage=mask))


@pytest.mark.parametrize('angle', [-.7, .7])
def test_rigid_feature_checks_known_rotation_and_moving_moon(angle):
    y, x = np.indices((256, 256), dtype=float)
    source = corona(y, x, 256)
    mask = np.hypot(y-120, x-120) > 30
    target, target_mask = warp_rigid(source, mask, [1.2, -2.1], angle)
    target_mask &= np.hypot(y-137, x-140) > 30
    reference, moving = frame(target, target_mask), frame(source, mask)
    seed = correlation_seed(reference, moving, 20)
    assert np.linalg.norm(seed-[1.2, -2.1]) < 2
    transform, quality = validate_pair(reference, moving, seed, 1, [0, 0, 256, 256], [128, 128])
    assert matrix_distance(transform, matrix_yx(source.shape, [1.2, -2.1], angle), control_points(source.shape)) < .25
    assert quality['initialization_disagreement_px'] < 1.
    assert quality['structure_score'] >= .25
    band_transform, _ = validate_band_pair(reference, moving, seed, 1, [0, 0, 256, 256], [128, 128])
    assert matrix_distance(band_transform, matrix_yx(source.shape, [1.2, -2.1], angle), control_points(source.shape)) < .25


def test_flat_and_moon_only_do_not_pass_structure_check():
    y, x = np.indices((128, 128))
    mask = np.hypot(y-64, x-64) > 20
    flat = frame(np.full((128, 128), 20.), mask)
    _, keep = log_band(flat)
    assert keep.any()
    with pytest.raises(ValueError, match='contrast'):
        structure_check(flat, flat, np.eye(3), [64, 64])
    with pytest.raises(ValueError, match='contrast'):
        correlation_seed(flat, flat, 20)


def test_rotated_null_rejects_a_wrong_quarter_turn():
    y, x = np.indices((256, 256), dtype=float)
    image = corona(y, x, 256)
    mask = np.hypot(y-128, x-128) > 30
    reference = frame(image, mask)
    turned, covered = warp_rigid(image, mask, [0, 0], 90)
    check = structure_check(reference, frame(turned, covered), np.eye(3), [127.5, 127.5])
    assert check['structure_score'] < .25


@pytest.mark.parametrize('shift', [[23.4, -17.6], [-22.1, 18.3]])
def test_seed_handles_large_motion_exposure_noise_and_partial_overlap(shift):
    y, x = np.indices((256, 256), dtype=float)
    image = corona(y, x, 256)
    mask = (np.hypot(y-126, x-126) > 25) & (image < 350)
    moved, support = warp_rigid(image, mask, shift, .45)
    support &= np.hypot(y-137, x-139) > 25
    rng = np.random.default_rng(41)
    reference = frame(moved*1.7+rng.normal(0, .01, moved.shape), support)
    seed = correlation_seed(reference, frame(image, mask), 50)
    assert np.linalg.norm(seed-shift) < 4


def test_band_fit_rejects_flat_field():
    flat = frame(np.full((128, 128), 20.), np.ones((128, 128), bool))
    with pytest.raises(ValueError):
        validate_band_pair(flat, flat, [0., 0.], 1, [0, 0, 128, 128], [64, 64])


def test_native_patch_sampler_preserves_masks_and_channel_mean():
    import importlib.util
    from types import SimpleNamespace
    if not (Path(__file__).parents[1]/'scripts/diagnose_rigid_pairs.py').is_file():
        pytest.skip('Private research script is not included in the public checkout')
    spec = importlib.util.spec_from_file_location('diagnose_rigid_pairs',
        Path(__file__).parents[1]/'scripts/diagnose_rigid_pairs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = np.ones((40, 40, 3))*np.array([2., 4., 6.])
    masks = {name: np.zeros((40, 40), bool) for name in ('lunar', 'saturation', 'low_signal', 'defective')}
    masks['coverage'] = np.ones((40, 40), bool)
    masks['lunar'][10, 12] = True
    test_frame = SimpleNamespace(data=image, auxiliary=masks, scale=2.)
    y, x = np.indices((4, 4), dtype=float)
    values, valid = module.patch(test_frame, y+10, x+10)
    np.testing.assert_allclose(values, 2.)
    assert valid.sum() == 15 and not valid[0, 2]

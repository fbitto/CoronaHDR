"""AEL transform conversion and original registration on known solar structure."""
import numpy as np
import pytest

from coronahdr.ael_adapter import ael_matrix_yx, estimate_pair
from coronahdr.empirical import warp_rigid
from coronahdr.rigid import apply_points, control_points, matrix_distance, matrix_yx
from coronahdr.synthetic import corona
from test_rigid_features import frame


def test_ael_xy_reference_to_input_is_inverted_without_half_pixel_shift():
    center = [89.4, 65.2]
    t = [2.5, -3.25]
    angle = np.deg2rad(.7)
    points = np.array([[20., 30.], [70., 90.], [120., 140.]])
    c, s = np.cos(angle), np.sin(angle)
    y, x = points.T
    target = np.column_stack([center[1]+s*(x-center[0])+c*(y-center[1])+t[1],
                              center[0]+c*(x-center[0])-s*(y-center[1])+t[0]])
    np.testing.assert_allclose(apply_points(ael_matrix_yx(center, t, angle), target), points, atol=1e-12)


@pytest.mark.parametrize('angle', [-.7, .7])
def test_ael_recovers_synthetic_rotation_with_separate_moon(angle):
    y, x = np.indices((256, 256), dtype=float)
    source = corona(y, x, 256)
    mask = np.hypot(y-120, x-120) > 30
    target, target_mask = warp_rigid(source, mask, [1.2, -2.1], angle)
    target_mask &= np.hypot(y-130, x-131) > 30
    transform, q = estimate_pair(frame(target, target_mask), frame(source, mask),
        dict(xy=[131.,130.], r=30.), dict(xy=[120.,120.], r=30.), 1, [0,0,256,256])
    assert matrix_distance(transform, matrix_yx(source.shape, [1.2,-2.1],angle), control_points(source.shape)) < .5
    assert q['independent_disagreement_native_px'] <= 1
    unqualified, q = estimate_pair(frame(target, target_mask), frame(source, mask),
        dict(xy=[131.,130.], r=30.), dict(xy=[120.,120.], r=30.), 1, [0,0,256,256], require_independent=False)
    assert matrix_distance(unqualified, transform, control_points(source.shape)) < 1e-9
    assert 'not applied' in q['qualification']


def test_ael_refuses_flat_moon_only_scene():
    y, x = np.indices((256, 256), dtype=float)
    mask = np.hypot(y-128, x-128) > 30
    flat = frame(np.full((256, 256), 20.), mask)
    with pytest.raises(ValueError):
        estimate_pair(flat, flat, dict(xy=[128.,128.], r=30.), dict(xy=[128.,128.], r=30.),
                      1, [0,0,256,256])
    # The copied AEL-only gates admit this flat control; the independent gate above rejects it.
    _, result = estimate_pair(flat, flat, dict(xy=[128.,128.], r=30.), dict(xy=[128.,128.], r=30.),
                              1, [0,0,256,256], require_independent=False)
    assert result['ael_structure']['structure_score'] >= .25


def test_ael_seed_handles_approximately_700_native_pixel_displacement():
    y, x = np.indices((512, 512), dtype=float)
    source = corona(y+170, x+160, 512)
    mask = np.hypot(y-86, x-96) > 35
    target, target_mask = warp_rigid(source, mask, [170., 160.], .2)
    target_mask &= np.hypot(y-258, x-255) > 35
    transform, _ = estimate_pair(frame(target, target_mask), frame(source, mask),
        dict(xy=[255.,258.], r=35.), dict(xy=[96.,86.], r=35.), 4, [0,0,2048,2048])
    from coronahdr.rigid import lower_matrix
    error = matrix_distance(lower_matrix(transform, 4, [0,0,2048,2048]),
                            matrix_yx(source.shape, [170.,160.], .2), control_points(source.shape))*4
    assert error < 1.


def test_ael_rejects_insufficient_overlap():
    y, x = np.indices((256, 256), dtype=float)
    source = corona(y, x, 256)
    left = x < 40
    right = x > 215
    with pytest.raises(ValueError):
        estimate_pair(frame(source, left), frame(source, right),
            dict(xy=[128.,128.], r=30.), dict(xy=[128.,128.], r=30.), 1, [0,0,256,256])

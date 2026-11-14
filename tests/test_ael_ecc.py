"""Known-truth and failure controls for the saved AEL workflow's ECC fitter."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from coronahdr.ael_ecc import register
from coronahdr.empirical import warp_rigid


def test_ecc_recovers_shift_and_keeps_native_pixel_centres():
    rng = np.random.default_rng(23)
    y, x = np.indices((1024,1024), dtype=float)
    image = gaussian_filter(rng.normal(size=(1024,1024)), 4)*100+1000
    image *= np.exp(-np.hypot(y-512,x-512)/500)
    moving, _ = warp_rigid(image, np.ones(image.shape, bool), [10.,-12.], 0.)
    affine, result = register(image, moving, dict(xy=[512.,512.],r=100.), dict(xy=[500.,522.],r=100.))
    np.testing.assert_allclose(affine[:2,:2], np.eye(2), atol=.001)
    np.testing.assert_allclose(affine[:,2], [-24.,20.], atol=.3)
    assert result['heldout_correlation'] > .95


def test_ecc_rejects_crop_outside_image():
    image = np.ones((512,512))
    with pytest.raises(ValueError, match='crop'):
        register(image,image,dict(xy=[256.,256.],r=50.),dict(xy=[256.,256.],r=50.))


def test_ecc_rejects_flat_corona():
    image = np.full((1024,1024),20.)
    with pytest.raises(ValueError):
        register(image,image,dict(xy=[512.,512.],r=100.),dict(xy=[512.,512.],r=100.))


@pytest.mark.parametrize('angle', [-.3, .3])
def test_ecc_recovers_small_rotations(angle):
    from coronahdr.rigid import apply_points, control_points, lift_matrix, matrix_distance, matrix_yx
    rng = np.random.default_rng(24)
    y, x = np.indices((1024,1024), dtype=float)
    image = (gaussian_filter(rng.normal(size=(1024,1024)), 4)*100+1000)*np.exp(-np.hypot(y-512,x-512)/500)
    moving, _ = warp_rigid(image, np.ones(image.shape, bool), [10.,-12.], angle)
    truth = matrix_yx(image.shape, [10.,-12.], angle)
    moon = apply_points(truth, [[512.,512.]])[0]
    affine, _ = register(image, moving, dict(xy=[512.,512.],r=100.),
                         dict(xy=[float(moon[1]),float(moon[0])],r=100.))
    swap = np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]])
    measured = swap@np.vstack([affine,[0,0,1]])@swap
    expected = lift_matrix(truth,2,[0,0,2048,2048])
    assert matrix_distance(measured,expected,control_points((2048,2048))) < 1.


def test_ecc_recovers_680_native_pixel_displacement():
    rng = np.random.default_rng(25)
    y,x = np.indices((1600,1600),dtype=float)
    image = (gaussian_filter(rng.normal(size=(1600,1600)),4)*100+1000)*np.exp(-np.hypot(y-700,x-700)/500)
    moving,_ = warp_rigid(image,np.ones(image.shape,bool),[340.,0.],0.)
    affine, quality = register(image,moving,dict(xy=[700.,700.],r=100.),dict(xy=[700.,1040.],r=100.))
    np.testing.assert_allclose(affine[:,2], [0.,680.], atol=.4)
    assert quality['heldout_correlation'] > .95


def test_ecc_excludes_invalid_saturated_regions():
    rng = np.random.default_rng(26)
    y,x = np.indices((1024,1024),dtype=float)
    image = (gaussian_filter(rng.normal(size=(1024,1024)),4)*100+1000)*np.exp(-np.hypot(y-512,x-512)/500)
    moving,_ = warp_rigid(image,np.ones(image.shape,bool),[10.,-12.],0.)
    valid = np.ones(image.shape,bool)
    valid[440:500,600:660] = False
    moving[~valid] = 1e12
    affine, quality = register(image,moving,dict(xy=[512.,512.],r=100.),dict(xy=[500.,522.],r=100.),
                               reference_valid=np.ones(image.shape,bool), moving_valid=valid)
    np.testing.assert_allclose(affine[:,2], [-24.,20.], atol=.4)
    assert 'validity masks' in quality['qualification']


def test_ecc_rejects_missing_mask_partner():
    image = np.ones((1024,1024))
    with pytest.raises(ValueError, match='Both reference'):
        register(image,image,dict(xy=[512.,512.],r=100.),dict(xy=[512.,512.],r=100.),
                 reference_valid=np.ones(image.shape,bool))

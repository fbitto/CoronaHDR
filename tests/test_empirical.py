"""Known-truth controls for experimental geometry and sky matching."""
import numpy as np
from coronahdr.empirical import inverse_rigid,warp_rigid,audit_rotation,fit_empirical_match,compare_merges


def test_rigid_inverse_and_borders():
    image=np.arange(64*64.).reshape(64,64);mask=np.ones((64,64),bool)
    result,valid=warp_rigid(image,mask,[2,3],0)
    np.testing.assert_array_equal(result[3:60,4:60],image[1:58,1:57])
    assert not valid[:2].any() and not valid[:,:3].any()
    y,x=inverse_rigid((65,65),[0,0],90,np.array([32.]),np.array([42.]))
    np.testing.assert_allclose([y[0],x[0]],[22,32])


def test_empirical_match_recovers_gain_and_plane_without_modifying_input():
    rng=np.random.default_rng(25);y,x=np.indices((96,96));r=rng.normal(1000,100,(96,96,3))
    m=(r-(40+.3*x-.2*y)[...,None])/1.2;before=m.copy()
    out,report=fit_empirical_match(r,m,np.ones((96,96),bool))
    np.testing.assert_allclose(out,r,atol=1e-7);np.testing.assert_array_equal(m,before)
    assert max(report['heldout_contrast_rmse'])<1e-9


def test_merge_excludes_invalid_and_preserves_negative_and_missing_samples():
    a=np.array([[[[-2.,-2,-2],[4,4,4],[np.nan,np.nan,np.nan]]],[[[6.,6,6],[8,8,8],[999,999,999]]]])
    valid=np.array([[[True,False,False]],[[False,True,False]]])
    products,support=compare_merges(a,valid,[1,4])
    for p in products.values():
        np.testing.assert_equal(p[0,0],[-2,-2,-2]);np.testing.assert_equal(p[0,1],[8,8,8]);assert np.isnan(p[0,2]).all()
    np.testing.assert_array_equal(support,[[1,1,0]])


def test_rotation_recovers_asymmetric_known_truth():
    from scipy.ndimage import gaussian_filter
    rng=np.random.default_rng(6);m=gaussian_filter(rng.normal(size=(256,256)),3)+10
    mask=np.ones(m.shape,bool);r,rm=warp_rigid(m,mask,[1.1,-.8],.45)
    report=audit_rotation(r,m,rm,mask,[1,-1],1.)
    assert abs(report['angle_deg']-.45)<.03
    assert report['heldout_improvement']>.5
    assert report['angle_split_difference_deg']<.03


def test_empirical_models_reject_unknown_method():
    import pytest
    a=np.ones((64,64,3));mask=np.ones((64,64),bool)
    with pytest.raises(ValueError,match='model must be'):
        fit_empirical_match(a,a,mask,'unvalidated_polynomial')


def test_longest_preference_and_feather_keep_common_radiance():
    rng=np.random.default_rng(15);truth=rng.normal(10,1,(32,32,3))
    images=np.stack([truth,truth,truth]);masks=np.ones((3,32,32),bool);masks[2,:,:8]=False
    products,count=compare_merges(images,masks,[1,4,16])
    for product in products.values():np.testing.assert_allclose(product,truth,atol=1e-12)
    np.testing.assert_array_equal(count[:,:8],2)

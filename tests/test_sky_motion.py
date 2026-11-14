"""Coverage motion must not manufacture sky disagreement in common regions."""
import numpy as np
from coronahdr.sky_motion import align_with_guard,compare_region,annular_check
from coronahdr.rigid import matrix_yx,apply_points


def test_large_rigid_motion_preserves_a_linear_sky_plane_in_common_coverage():
    shape=(192,224);y,x=np.indices(shape,dtype=float);matrix=matrix_yx(shape,[43.25,-52.4],.8)
    q=apply_points(matrix,np.stack([y,x],axis=-1))
    reference=(100+.2*x+.1*y)[...,None]*np.array([.8,1.2,1.])
    moving=(100+.2*q[...,1]+.1*q[...,0])[...,None]*np.array([.8,1.2,1.])
    aligned,mask=align_with_guard(moving,np.ones(shape,bool),matrix,1,8)
    result=compare_region(reference,aligned,mask)
    assert max(result['relative_absolute_error'])<1e-12
    assert not mask[:8].any() and result['samples']<y.size


def test_small_motion_does_not_excuse_real_sky_offset():
    y,x=np.indices((128,128));reference=(100+.2*x+.1*y)[...,None]*np.ones(3)
    matrix=matrix_yx(y.shape,[1.,2.],0);q=apply_points(matrix,np.stack([y,x],axis=-1))
    moving=(120+.2*q[...,1]+.1*q[...,0])[...,None]*np.ones(3)
    aligned,mask=align_with_guard(moving,np.ones(y.shape,bool),matrix,1,8)
    result=compare_region(reference,aligned,mask)
    np.testing.assert_allclose(result['signed_median_difference'],20,atol=1e-12)
    assert min(result['relative_median_bias'])>.1


def test_corona_only_check_does_not_use_uncovered_or_outer_sky_pixels():
    y,x=np.indices((160,160),dtype=float);positions=np.stack([y,x],axis=-1)
    reference=np.ones((160,160,3))*100;moving=reference.copy()
    moving[np.hypot(y-80,x-80)>60]=1e6
    result=annular_check(reference,moving,np.ones(y.shape,bool),positions,[80,80],20)
    assert result['passed'] and result['mono_relative_absolute_error']==0
    assert not annular_check(reference,moving,np.zeros(y.shape,bool),positions,[80,80],20)['passed']

"""Rigid conventions, composition, coverage, and redundant known-truth controls."""
import numpy as np
import pytest
from coronahdr.rigid import matrix_yx,lift_matrix,lower_matrix,apply_points,solve_rigid_graph,control_points,matrix_distance,measure_rigid


def test_matrix_pixel_centers_binning_roi_and_composition():
    m=matrix_yx((65,65),[2,3],90)
    np.testing.assert_allclose(apply_points(m,[[32,42]]),[[44,35]])
    native=lift_matrix(m,4,[1800,900,260,260])
    np.testing.assert_allclose(lower_matrix(native,4,[1800,900,260,260]),m,atol=1e-12)
    p=np.array([[20.5,10.5],[32,32]])
    np.testing.assert_allclose(apply_points(np.linalg.inv(m)@m,p),p,atol=1e-12)


def test_redundant_graph_composes_rotations_and_excludes_disconnected():
    shape=(128,160);truth=[np.eye(3),matrix_yx(shape,[5,-3],.8),matrix_yx(shape,[-8,2],-.3)]
    edges=[]
    for p,c in [(0,1),(1,2),(0,2)]:
        edges.append(dict(parent=p,child=c,accepted=True,matrix_yx=(np.linalg.inv(truth[p])@truth[c]).tolist(),quality=dict(split_control_error_px=.01)))
    poses,metrics,rejected,removed=solve_rigid_graph(4,0,edges,shape,.1)
    assert rejected==[dict(index=3,reason='No reliable rigid path to reference')] and not removed
    for i in poses:np.testing.assert_allclose(poses[i],truth[i],atol=1e-10)
    assert metrics[1]['incident_constraints']==2
    assert metrics[1]['two_edge_disjoint_paths_to_reference'] is True
    assert metrics[0]['two_edge_disjoint_paths_to_reference'] is None


def test_graph_redundancy_requires_two_routes_to_reference():
    # The second triangle has a cycle but only one connection to the reference.
    links=[(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,3),(5,6)]
    edges=[dict(parent=p,child=c,accepted=True,matrix_yx=np.eye(3).tolist(),
                quality=dict(split_control_error_px=.01)) for p,c in links]
    _,metrics,_,_=solve_rigid_graph(7,0,edges,(40,40),.1)
    assert metrics[1]['two_edge_disjoint_paths_to_reference'] is True
    assert metrics[2]['two_edge_disjoint_paths_to_reference'] is True
    for i in (3,4,5,6):
        assert metrics[i]['two_edge_disjoint_paths_to_reference'] is False
        assert metrics[i]['consistency_status']=='no two edge-disjoint paths to reference'
    assert metrics[5]['incident_constraints']==3


def test_graph_parallel_links_are_distinct_routes():
    edges=[dict(parent=0,child=1,accepted=True,matrix_yx=np.eye(3).tolist(),
                quality=dict(split_control_error_px=.01)) for _ in range(2)]
    _,metrics,_,_=solve_rigid_graph(2,0,edges,(40,40),.1)
    assert metrics[1]['two_edge_disjoint_paths_to_reference'] is True


def test_rigid_measurement_ignores_independent_occulting_disk():
    from scipy.ndimage import gaussian_filter
    from pathlib import Path
    from coronahdr.model import Frame
    from coronahdr.empirical import warp_rigid
    from test_stage1 import metadata
    rng=np.random.default_rng(25);a=gaussian_filter(rng.normal(size=(256,256)),3)+20
    y,x=np.indices(a.shape);mm=np.hypot(y-122,x-116)>24
    target,rm=warp_rigid(a,mm,[1.2,-.7],.7)
    rm &= np.hypot(y-130,x-135)>28
    m=metadata();m['lunar_disk']=None;m['white_level']=1e5
    moving=Frame(a,m,Path('moving'),dict(coverage=mm))
    reference=Frame(target,m,Path('reference'),dict(coverage=rm))
    transform,q=measure_rigid(reference,moving,np.array([1.,-1.]),1,[0,0,256,256],.2)
    truth=matrix_yx(a.shape,[1.2,-.7],.7)
    assert matrix_distance(transform,truth,control_points(a.shape))<.1
    from coronahdr.ael_visual import check_rigid_agreement
    assert check_rigid_agreement(reference,moving,truth,1,[0,0,256,256])['passed']
    wrong=truth.copy();wrong[0,2]+=3
    assert not check_rigid_agreement(reference,moving,wrong,1,[0,0,256,256])['passed']


def test_tiled_rigid_sampling_matches_direct_inverse_and_has_no_seams():
    from coronahdr.streaming import sample_band,output_canvas
    from scipy.ndimage import map_coordinates
    from types import SimpleNamespace
    rng=np.random.default_rng(19);a=rng.normal(size=(95,117,3));m=matrix_yx(a.shape[:2],[8.3,-13.7],1.2)
    y,x=np.indices(a.shape[:2]);inv=np.linalg.inv(m);coords=[inv[0,0]*y+inv[0,1]*x+inv[0,2],inv[1,0]*y+inv[1,1]*x+inv[1,2]]
    expected=np.stack([map_coordinates(a[...,c],coords,order=1,mode='constant',cval=0) for c in range(3)],axis=-1)
    actual=np.concatenate([sample_band(a,m,0,33,117),sample_band(a,m,33,95,117)])
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    frames=[SimpleNamespace(data=a)]*2
    origin,shape=output_canvas(frames,[np.eye(3),m],'union')
    corners=np.array([[-.5,-.5],[-.5,116.5],[94.5,-.5],[94.5,116.5]])
    transformed=apply_points(m,corners)-origin
    assert (transformed>=-.5).all() and (transformed<=np.array(shape)-.5).all()


def test_review_rejects_scale_and_hash_mismatch(tmp_path):
    import json
    from coronahdr.rigid import reviewed_rigid
    a=np.eye(3);a[0,0]=1.01
    audit=dict(input_source_sha256=['abc'],matrices_yx={'0':a.tolist()},metrics={'0':dict(path=[0])},edges=[])
    p=tmp_path/'audit.json';p.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='proper rigid'):
        reviewed_rigid(p,[dict(metadata=dict(source_sha256='abc'))],0,[0])
    with pytest.raises(ValueError,match='hashes mismatch'):
        reviewed_rigid(p,[dict(metadata=dict(source_sha256='changed'))],0,[0])


def test_full_rigid_hdr_analytic_radiance_and_independent_moon(tmp_path):
    import importlib.util
    from pathlib import Path
    if not (Path(__file__).parents[1]/'scripts/demo_rigid.py').is_file():
        pytest.skip('Private research script is not included in the public checkout')
    spec=importlib.util.spec_from_file_location('demo_rigid',Path(__file__).parents[1]/'scripts/demo_rigid.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.demo(tmp_path/'rigid')
    assert result['accepted_indices']==[0,1,2,3] and result['relative_radiance_rms']<.01


def test_longest_full_export_preserves_masks_and_float32(tmp_path):
    import importlib.util
    from pathlib import Path
    from astropy.io import fits
    from coronahdr.alternatives import export_longest
    root=Path(__file__).parents[1]
    if not (root/'scripts/demo_rigid.py').is_file():
        pytest.skip('Private research script is not included in the public checkout')
    spec=importlib.util.spec_from_file_location('demo_rigid',root/'scripts/demo_rigid.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.demo(tmp_path/'scene')
    source=tmp_path/'scene/output';destination=tmp_path/'longest'
    report=export_longest(source,destination)
    spec=importlib.util.spec_from_file_location('verify',root/'scripts/verify_disk_output.py');verify=importlib.util.module_from_spec(spec);spec.loader.exec_module(verify)
    checks=verify.verify(destination)
    assert checks['fits_tiff_float32_identical'] and report['hdr_method']=='longest_valid_soft'
    assert (destination/'lunar.fits').is_symlink()
    with pytest.raises(FileExistsError):export_longest(source,destination)


def test_flat_field_does_not_provide_rigid_geometry():
    from coronahdr.model import Frame
    from coronahdr.rigid import geometry_condition
    from test_stage1 import metadata
    from pathlib import Path
    flat=Frame(np.ones((64,64))*100,metadata(),Path('flat'))
    with pytest.raises(ValueError,match='contrast|geometry'):
        geometry_condition(flat,flat,[0,0],np.eye(3))


def test_redundant_graph_discards_an_inconsistent_constraint():
    shape=(128,160);truth=[matrix_yx(shape,[i*2.,-i],i*.1) for i in range(5)];edges=[]
    for p in range(5):
        for c in range(p+1,5):
            edges.append(dict(parent=p,child=c,accepted=True,matrix_yx=(np.linalg.inv(truth[p])@truth[c]).tolist(),quality=dict(split_control_error_px=.01)))
    bad=np.linalg.inv(truth[0])@truth[4];bad[:2,2]+=[8,0]
    edges.append(dict(parent=0,child=4,accepted=True,matrix_yx=bad.tolist(),quality=dict(split_control_error_px=.1)))
    poses,metrics,rejected,removed=solve_rigid_graph(5,0,edges,shape,.2)
    assert not rejected and removed
    for i in poses:assert matrix_distance(poses[i],truth[i],control_points(shape))<.01


def test_analytic_corona_registration_below_tenth_pixel_with_saturation():
    from coronahdr.synthetic import corona
    from coronahdr.model import Frame
    from pathlib import Path
    from test_stage1 import metadata
    size=256;y,x=np.indices((size,size),dtype=float);truth=matrix_yx((size,size),[1.35,-2.42],.7)
    q=apply_points(truth,np.stack([y,x],axis=-1))
    reference=corona(y,x,size)*.25;moving=corona(q[...,0],q[...,1],size)*.5
    reference[np.hypot(y-128,x-128)<30]=0;moving[np.hypot(y-120,x-140)<30]=0
    r=Frame(np.minimum(reference,1000),metadata(exposure_s=.25,lunar_disk=[128,128,30],low_signal=1.),Path('reference'))
    m=Frame(np.minimum(moving,1000),metadata(exposure_s=.5,lunar_disk=[140,120,30],low_signal=1.),Path('moving'))
    measured,quality=measure_rigid(r,m,np.array([1.2,-2.3]),1,[0,0,size,size],.2)
    assert matrix_distance(measured,truth,control_points((size,size)))<.1

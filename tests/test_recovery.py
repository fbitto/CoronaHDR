"""Known-truth controls for differential sky fitting and independent rejection."""
import numpy as np
import pytest
from coronahdr.recovery import audit_sky_difference, review_registration


def scene():
    factor = 4
    y, x = np.indices((208, 208), dtype=float)
    x, y = x*factor+1.5, y*factor+1.5
    r = np.hypot(x-416, y-416)
    angle = np.arctan2(y-416, x-416)
    corona = 2000*np.exp(-r/100)*(1+.2*np.sin(5*angle)+.1*np.cos(9*angle))
    corona[r < 64] = 0
    rgb = corona[..., None]*np.array([.8, 1.2, 1.])
    sky = 500+(x-416)*.1+(y-416)*.03
    reference = rgb+sky[..., None]
    delta = np.stack([100+.05*(x-416), 160+.03*(y-416), 80+.02*(x+y-832)], axis=-1)
    metadata = dict(lunar_disk=[416,416,64])
    return reference, reference-delta, rgb, delta, metadata


def test_independent_sky_model_recovers_only_additive_difference():
    reference, moving, corona, delta, metadata = scene()
    before = moving.copy()
    mask = np.ones(reference.shape[:2], bool)
    result = audit_sky_difference(reference, moving, mask, mask, [0,0], metadata, metadata, (832,832), 4)
    assert result['status'] == 'validated', result['rejection_reasons']
    y,x=np.indices(mask.shape);x=x*4+1.5;y=y*4+1.5
    design=np.stack([np.ones(mask.shape),(x-415.5)/416,(y-415.5)/416],axis=-1)
    np.testing.assert_allclose(design @ result['coefficients'],delta,atol=1e-9)
    assert not result['gain_applied'] and not result['applied']
    np.testing.assert_array_equal(moving,before)


@pytest.mark.parametrize('defect', ['cloud', 'response', 'lunar_sky'])
def test_independent_controls_reject_invalid_corrections(defect):
    reference, moving, corona, delta, metadata=scene()
    y,x=np.indices(moving.shape[:2]);x=x*4+1.5;y=y*4+1.5
    r=np.hypot(x-416,y-416)
    if defect=='cloud':
        moving[(x>416)&(r<300)&(r>96)] *= .8
    elif defect=='response':
        moving-=.15*corona
    else:
        moving[r<30]+=100
    mask=np.ones(moving.shape[:2],bool)
    result=audit_sky_difference(reference,moving,mask,mask,[0,0],metadata,metadata,(832,832),4)
    assert result['status']=='rejected' and result['rejection_reasons']


def test_registration_consensus_rejects_isolated_bad_parent(monkeypatch):
    import coronahdr.recovery as module
    from types import SimpleNamespace
    frames=[SimpleNamespace(scale=1.) for _ in range(5)]
    solved={i:np.array([float(i),0]) for i in range(4)}
    def measure(reference,moving,limit,fit_photometry):
        parent=next(i for i,frame in enumerate(frames) if frame is reference)
        point=np.array([10.,20.])+([8.,0.] if parent==0 else [0.,0.])
        return point-solved[parent],{}
    monkeypatch.setattr(module,'register',measure)
    result=review_registration(frames,solved,4,1,100,1.)
    assert result['status']=='consensus' and len(result['consensus_parents'])==3
    np.testing.assert_allclose(result['shift_dy_dx'],[10,20])


def test_bias_noise_estimator_on_known_independent_frames():
    from coronahdr.recovery import read_noise_from_bias
    rng=np.random.default_rng(916)
    labels=np.tile(np.array(['R','G1','G2','B']),10000)
    samples=1008+rng.normal(0,4.,(20,len(labels)))
    result=read_noise_from_bias(samples,labels)
    for phase,values in result.items():
        assert abs(values['temporal_noise_dn']-4)<.1


def test_local_review_transform_changes_reference_without_resampling(tmp_path):
    import json
    from coronahdr.recovery import reviewed_translations
    inputs=[dict(metadata=dict(source_sha256=str(i))) for i in range(3)]
    audit=dict(input_source_sha256=['0','1','2'],reference=0,
        transformations={'0':[0,0],'1':[10,20],'2':[11,22]},reviews=[],
        original_metrics={'0':dict(path=[0]),'1':dict(path=[1,0]),'2':dict(path=[2,1,0])})
    path=tmp_path/'geometry.json';path.write_text(json.dumps(audit))
    solved,metrics,rejected,_=reviewed_translations(path,inputs,1,[1,2])
    assert not rejected and set(solved)=={1,2}
    np.testing.assert_array_equal(solved[2],[1,2])
    assert metrics[2]['path']==[]
    inputs[0]['metadata']['source_sha256']='changed'
    with pytest.raises(ValueError,match='does not match'):
        reviewed_translations(path,inputs,1,[1,2])


def test_spatial_cloud_cannot_be_hidden_by_a_sky_plane():
    reference,moving,corona,delta,metadata=scene()
    y,x=np.indices(reference.shape[:2]);x=x*4+1.5;y=y*4+1.5
    moving[(x>450)&(y>400)] += 300
    mask=np.ones(reference.shape[:2],bool)
    result=audit_sky_difference(reference,moving,mask,mask,[0,0],metadata,metadata,(832,832),4)
    assert result['status']=='rejected'
    assert not result.get('applied',False)


def test_temporal_subset_is_explicit_and_global_indices_are_preserved(tmp_path):
    import json
    from coronahdr.synthetic import make_scene
    from coronahdr.pipeline import run_manifest
    from astropy.io import fits
    manifest=make_scene(tmp_path/'inputs')
    recipe=json.loads(manifest.read_text())
    recipe.update(reference=1,processing=dict(mode='disk',tile_rows=35,registration_bin=1,
        include_indices=[1,2],canvas='union',bridges=True,rejection_policy='error'))
    manifest.write_text(json.dumps(recipe))
    report=run_manifest(manifest,tmp_path/'subset')
    assert report['requested_indices']==[1,2]
    assert report['accepted_indices']==[1,2]
    assert report['not_requested_indices']==[0,3] and not report['rejected_frames']
    assert fits.getdata(tmp_path/'subset/contributions.fits').max()<=2
    assert fits.getdata(tmp_path/'subset/contributions.fits').dtype.itemsize==1


def test_insufficient_reference_overlap_never_passes_an_empty_review_path(monkeypatch):
    import coronahdr.registration_graph as graph
    monkeypatch.setattr(graph,'radiometry',lambda *args:dict(status='insufficient overlap',samples=0))
    solved={0:np.zeros(2),1:np.ones(2)}
    metrics={0:dict(path=[0]),1:dict(path=[])}
    accepted,rejected,_=graph.qualify_radiometry([None,None],solved,metrics,0,1,[])
    assert set(accepted)=={0} and rejected[0]['index']==1


def test_sky_audit_requires_explicit_lunar_geometry():
    reference,moving,_,_,metadata=scene()
    mask=np.ones(reference.shape[:2],bool)
    with pytest.raises(ValueError,match='explicit lunar disks'):
        audit_sky_difference(reference,moving,mask,mask,[0,0],{},metadata,(832,832),4)


def test_sky_audit_rejects_too_few_independent_blocks(monkeypatch):
    import json
    import coronahdr.recovery as module
    reference,moving,_,_,metadata=scene()
    mask=np.ones(reference.shape[:2],bool)
    monkeypatch.setattr(module,'block_rms',lambda *args:(np.full(3,np.inf),2))
    result=audit_sky_difference(reference,moving,mask,mask,[0,0],metadata,metadata,(832,832),4)
    assert result['status']=='rejected'
    assert 'Insufficient independent sky blocks' in result['rejection_reasons']
    json.dumps(result,allow_nan=False)

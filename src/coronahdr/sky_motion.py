"""Diagnostic separation of coverage artefacts, detector motion and sky changes."""
import numpy as np
from .streaming import sample_band
from .rigid import lower_matrix


def align_with_guard(image, usable, matrix, factor, edge_margin_px=128):
    """Keep only interpolated observed pixels far from each source detector edge."""
    h,w=usable.shape;mask=np.array(usable,copy=True)
    margin=int(np.ceil(edge_margin_px/factor))
    if margin:
        mask[:margin]=False;mask[-margin:]=False;mask[:,:margin]=False;mask[:,-margin:]=False
    a=lower_matrix(matrix,factor,[0,0,w*factor,h*factor])
    return sample_band(image,a,0,h,w),sample_band(mask,a,0,h,w)>1-1e-9


def compare_region(reference,moving,mask,minimum_pixels=64):
    """Never measure padded exterior pixels or select only positive sky noise."""
    valid=mask&np.isfinite(reference).all(axis=-1)&np.isfinite(moving).all(axis=-1)
    count=int(valid.sum())
    if count<minimum_pixels:return dict(samples=count,status='insufficient overlap')
    target,sample=reference[valid],moving[valid]
    scale=np.maximum(np.median(np.abs(target),axis=0),1e-12)
    difference=sample-target
    mono_target=target.mean(axis=-1);mono_sample=sample.mean(axis=-1)
    mono_scale=max(float(np.median(np.abs(mono_target))),1e-12)
    return dict(samples=count,reference_median= np.median(target,axis=0).tolist(),
        moving_median=np.median(sample,axis=0).tolist(),
        signed_median_difference=np.median(difference,axis=0).tolist(),
        relative_median_bias=(np.median(difference,axis=0)/scale).tolist(),
        relative_absolute_error=(np.median(np.abs(difference),axis=0)/scale).tolist(),
        mono_relative_absolute_error=float(np.median(np.abs(mono_sample-mono_target))/mono_scale),
        mono_relative_median_bias=float(np.median(mono_sample-mono_target)/mono_scale))


def annular_check(reference,moving,mask,positions,center_xy,radius,inner=1.5,outer=3.):
    cx,cy=center_xy;y,x=positions[...,0],positions[...,1]
    r=np.hypot(y-cy,x-cx)/radius
    common=mask&(r>=inner)&(r<outer)
    result=compare_region(reference,moving,common)
    sectors=[];angle=np.arctan2(y-cy,x-cx)%(2*np.pi)
    for sector in range(8):
        check=compare_region(reference,moving,common&(angle>=sector*np.pi/4)&(angle<(sector+1)*np.pi/4),32)
        if 'mono_relative_absolute_error' in check:sectors.append(dict(sector=sector,**check))
    result['sectors']=sectors
    result['passed']=bool(result.get('mono_relative_absolute_error',1)<=.15 and len(sectors)>=4
        and max((s['mono_relative_absolute_error'] for s in sectors),default=1)<=.2)
    return result

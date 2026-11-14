"""Opt-in diagnostic experiments, separate from the qualified master pipeline."""
import numpy as np
from scipy.ndimage import map_coordinates, binary_erosion, gaussian_filter, distance_transform_edt
from scipy.optimize import least_squares


def inverse_rigid(shape, shift_yx, angle_deg, y=None, x=None):
    """Forward: q = R(theta)(p-center)+center+shift in image x/y coordinates."""
    if y is None:
        y,x=np.indices(shape,dtype=float)
    cy,cx=(np.array(shape)-1)/2
    theta=np.deg2rad(angle_deg);c,s=np.cos(theta),np.sin(theta)
    qx=x-cx-shift_yx[1];qy=y-cy-shift_yx[0]
    return cy-s*qx+c*qy,cx+c*qx+s*qy


def warp_rigid(image, mask, shift_yx, angle_deg):
    coordinates=inverse_rigid(mask.shape,shift_yx,angle_deg)
    if image.ndim==3:
        result=np.stack([map_coordinates(image[...,c],coordinates,order=1,mode='constant',cval=0) for c in range(3)],axis=-1)
    else:
        result=map_coordinates(image,coordinates,order=1,mode='constant',cval=0)
    valid=map_coordinates(mask.astype(float),coordinates,order=1,mode='constant',cval=0)>1-1e-9
    return result,binary_erosion(valid,iterations=2)


def audit_rotation(reference, moving, reference_mask, moving_mask, initial_shift, max_angle=2., guard_pixels=None, minimum_samples=1024):
    """Compare nested translation/rigid fits on held-out coronal spatial blocks."""
    r=gaussian_filter(np.where(reference_mask,reference,0),1)
    m=gaussian_filter(np.where(moving_mask,moving,0),1)
    _,good=warp_rigid(m,moving_mask,initial_shift,0)
    # Large guard covers the bounded trial rotation and translation refinement.
    guard=int(np.ceil(np.hypot(*r.shape)*np.deg2rad(max_angle)/2))+8
    guard=guard if guard_pixels is None else guard_pixels
    common=binary_erosion(reference_mask&good,iterations=guard)
    y,x=np.nonzero(common)
    if len(y)<minimum_samples:
        raise ValueError('Insufficient independent coronal support for rotation audit')
    stride=max(1,len(y)//20000);y,x=y[::stride],x[::stride]
    target=r[y,x];scale=max(np.std(target),1e-12);center=np.median(target)
    blocks=((y//8+x//8)%2)==0
    if min(blocks.sum(),(~blocks).sum())<minimum_samples//4:
        raise ValueError('Insufficient held-out coronal spatial blocks')
    def fit(selected,rotate):
        def error(p,selection=selected):
            angle=p[2] if rotate else 0
            sy,sx=inverse_rigid(r.shape,p[:2],angle,y[selection],x[selection])
            v=map_coordinates(m,[sy,sx],order=3,mode='constant',cval=0)
            gain,offset=p[-2:]
            return (gain*(v-center)+offset-(target[selection]-center))/scale
        initial=[*initial_shift,*([0.] if rotate else []),1.,0.]
        lower=[*(np.array(initial_shift)-3),*([-max_angle] if rotate else []),.25,-20*scale]
        upper=[*(np.array(initial_shift)+3),*([max_angle] if rotate else []),4.,20*scale]
        result=least_squares(error,initial,bounds=(lower,upper),loss='soft_l1',f_scale=.1,diff_step=1e-4,max_nfev=150)
        if not result.success:
            raise ValueError('Rigid audit optimizer did not converge')
        return result,error
    translation,te=fit(blocks,False);rigid,re=fit(blocks,True);independent,_=fit(~blocks,True)
    terr=float(np.median(np.abs(te(translation.x,~blocks))))
    rerr=float(np.median(np.abs(re(rigid.x,~blocks))))
    sy,sx=inverse_rigid(r.shape,rigid.x[:2],rigid.x[2],y,x)
    final_valid=map_coordinates(moving_mask.astype(float),[sy,sx],order=1,mode='constant',cval=0)>1-1e-9
    return dict(shift_yx=rigid.x[:2].tolist(),angle_deg=float(rigid.x[2]),
        independent_shift_yx=independent.x[:2].tolist(), valid_sample_fraction=float(final_valid.mean()),
        independent_angle_deg=float(independent.x[2]),
        angle_split_difference_deg=float(abs(rigid.x[2]-independent.x[2])),
        translation_heldout_error=terr,rigid_heldout_error=rerr,
        heldout_improvement=float(1-rerr/max(terr,1e-12)),samples=len(y),
        rotation_at_bound=bool(abs(rigid.x[2])>.98*max_angle),
        applied_to_master=False)


def fit_empirical_match(reference, moving, mask, model='gain_plane'):
    """RGB affine matching with independent held-out blocks; no radiometric claim."""
    if model not in ('plane','gain_plane'):
        raise ValueError('Empirical model must be plane or gain_plane')
    y,x=np.indices(mask.shape,dtype=float)
    plane=np.stack([np.ones(mask.shape),(x-(x.shape[1]-1)/2)/x.shape[1],
                    (y-(y.shape[0]-1)/2)/y.shape[0]],axis=-1)
    train=mask&(((x//16+y//16)%2)==0);held=mask&~train
    if min(train.sum(),held.sum())<256:
        raise ValueError('Insufficient empirical matching support')
    output=np.empty_like(moving);parameters=[]
    for c in range(3):
        d=np.concatenate([moving[...,c,None],plane],axis=-1) if model=='gain_plane' else plane
        target=reference[...,c] if model=='gain_plane' else reference[...,c]-moving[...,c]
        stride=max(1,int(train.sum())//12000)
        a=d[train][::stride];b=target[train][::stride];scale=max(np.std(b),1e-6)
        initial=np.linalg.lstsq(a,b,rcond=None)[0]
        result=least_squares(lambda p:(a@p-b)/scale,initial,loss='soft_l1',f_scale=.1)
        if not result.success or (model=='gain_plane' and not .25<result.x[0]<4):
            raise ValueError('Unstable empirical gain/sky fit')
        output[...,c]=d@result.x+(moving[...,c] if model!='gain_plane' else 0)
        parameters.append(result.x.tolist())
    scale=np.maximum(np.std(reference[held],axis=0),1e-6)
    residual=output-reference
    return output,dict(model=model,parameters=parameters,training_pixels=int(train.sum()),
        heldout_pixels=int(held.sum()),heldout_contrast_rmse=(np.sqrt(np.mean(residual[held]**2,axis=0))/scale).tolist(),
        empirical=True,absolute_radiometry=False)


def compare_merges(images, masks, exposures):
    """Two linear-domain experiments with identical validity and scale inputs."""
    a=np.asarray(images);valid=np.asarray(masks);t=np.asarray(exposures)[:,None,None]
    # No measured noise model: exposure preference is heuristic, never inverse variance.
    valid=valid & np.isfinite(a).all(axis=-1)
    a=np.where(valid[...,None],a,0)
    feather=np.stack([np.minimum(distance_transform_edt(np.pad(m,1))[1:-1,1:-1]/3,1) for m in valid])
    w=feather*t
    total=w.sum(axis=0)
    mean=np.divide((a*w[...,None]).sum(axis=0),total[...,None],
        out=np.full(a.shape[1:],np.nan),where=total[...,None]>0)
    # Concentrate on the longest valid exposure with smooth 10% exposure transitions.
    preference=soft_longest_weights(feather,exposures)
    norm=preference.sum(axis=0)
    selected=np.divide((a*preference[...,None]).sum(axis=0),norm[...,None],
        out=np.full(a.shape[1:],np.nan),where=norm[...,None]>0)
    return dict(exposure_weighted=mean,longest_soft=selected),valid.sum(axis=0)


def soft_longest_weights(reliability,exposures):
    """Longest valid exposure preference; reliability is not a noise variance."""
    w=np.asarray(reliability,dtype=float);times=np.asarray(exposures,dtype=float)
    if len(times)!=len(w) or not np.isfinite(times).all() or (times<=0).any():
        raise ValueError('Exposure preferences need matching positive finite times')
    if not np.isfinite(w).all() or (w<0).any():
        raise ValueError('Reliability weights must be finite and nonnegative')
    t=times.reshape((len(times),)+(1,)*(w.ndim-1))
    longest=np.max(np.where(w>0,t,0),axis=0)
    return w*np.exp(np.clip((t-longest)/np.maximum(.1*longest,1e-12),-700,0))

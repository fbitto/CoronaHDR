"""Rigid coronal registration with native-pixel matrices and redundant graph checks."""
import numpy as np
from scipy.ndimage import map_coordinates,binary_erosion,gaussian_filter
from scipy.optimize import least_squares
from .empirical import audit_rotation
from .registration import valid_mask


def matrix_yx(shape, shift, angle_deg):
    """Source-to-target homogeneous matrix on [y,x,1], integer pixel centers."""
    theta=np.deg2rad(angle_deg);c,s=np.cos(theta),np.sin(theta)
    rotation=np.array([[c,s],[-s,c]])
    center=(np.array(shape)-1)/2
    result=np.eye(3);result[:2,:2]=rotation;result[:2,2]=center+shift-rotation@center
    return result


def apply_points(matrix, points):
    return np.asarray(points)@matrix[:2,:2].T+matrix[:2,2]


def lift_matrix(matrix, factor, roi):
    """Area-mean centers: native = ROI origin + factor*aux + (factor-1)/2."""
    mapping=np.eye(3);mapping[:2,:2]*=factor
    mapping[:2,2]=[roi[1]+(factor-1)/2,roi[0]+(factor-1)/2]
    return mapping@matrix@np.linalg.inv(mapping)


def lower_matrix(matrix, factor, roi):
    mapping=np.eye(3);mapping[:2,:2]*=factor
    mapping[:2,2]=[roi[1]+(factor-1)/2,roi[0]+(factor-1)/2]
    return np.linalg.inv(mapping)@matrix@mapping


def control_points(shape):
    center=(np.array(shape)-1)/2;radius=min(shape)/3
    return np.array([center,center+[radius,0],center-[radius,0],center+[0,radius],center-[0,radius]])


def matrix_distance(a,b,points):
    return float(np.linalg.norm(apply_points(a,points)-apply_points(b,points),axis=1).max())


def measure_rigid(reference,moving,initial_shift,factor,roi,tolerance=1.):
    initial_shift=np.asarray(initial_shift,dtype=float)
    rm,mm=valid_mask(reference),valid_mask(moving)
    quality=audit_rotation(reference.mono/reference.scale,moving.mono/moving.scale,rm,mm,
        initial_shift/factor,max_angle=1.5,guard_pixels=8,minimum_samples=256)
    measured=matrix_yx(rm.shape,quality['shift_yx'],quality['angle_deg'])
    independent=matrix_yx(rm.shape,quality['independent_shift_yx'],quality['independent_angle_deg'])
    split=factor*matrix_distance(measured,independent,control_points(rm.shape))
    quality['geometry_condition']=geometry_condition(reference,moving,initial_shift/factor,measured)
    quality['split_control_error_px']=split
    if (quality['rotation_at_bound'] or quality['valid_sample_fraction']<.995
            or quality['rigid_heldout_error']>.15 or split>tolerance):
        raise ValueError(f'Unreliable rigid edge: split={split:.3f}px, heldout={quality["rigid_heldout_error"]:.4f}, valid={quality["valid_sample_fraction"]:.4f}')
    return lift_matrix(measured,factor,roi),quality


def solve_rigid_graph(count,reference,edges,shape,tolerance=1.):
    """Robust pose optimization; discard inconsistent edges and disconnected nodes."""
    points=control_points(shape);active=[e for e in edges if e.get('accepted')]
    removed=[]
    for iteration in range(5):
        poses={reference:np.eye(3)};paths={reference:[reference]}
        while True:
            progress=False
            for e in sorted(active,key=lambda e:e['quality']['split_control_error_px']):
                p,c=e['parent'],e['child'];a=np.array(e['matrix_yx'])
                if p in poses and c not in poses:
                    poses[c]=poses[p]@a;paths[c]=[c,*paths[p]];progress=True
                elif c in poses and p not in poses:
                    poses[p]=poses[c]@np.linalg.inv(a);paths[p]=[p,*paths[c]];progress=True
            if not progress:break
        nodes=sorted(set(poses)-{reference})
        if not nodes:break
        connected=[e for e in active if e['parent'] in poses and e['child'] in poses]
        center=(np.array(shape)-1)/2
        def parameters(a):
            return [*(apply_points(a,[center])[0]-center),np.arctan2(a[0,1],a[0,0])]
        initial=np.array([parameters(poses[i]) for i in nodes]).ravel()
        def unpack(v):
            return {reference:np.eye(3),**{i:matrix_yx(shape,p[:2],np.rad2deg(p[2])) for i,p in zip(nodes,v.reshape(-1,3))}}
        def residual(v):
            g=unpack(v)
            return np.concatenate([(apply_points(g[e['child']],points)-apply_points(g[e['parent']]@np.array(e['matrix_yx']),points)).ravel() for e in connected])
        fit=least_squares(residual,initial,loss='soft_l1',f_scale=.25,max_nfev=100)
        if not fit.success:raise ValueError('Rigid pose graph did not converge')
        poses=unpack(fit.x)
        errors=[matrix_distance(poses[e['child']],poses[e['parent']]@np.array(e['matrix_yx']),points) for e in connected]
        bad=[e for e,error in zip(connected,errors) if error>tolerance]
        if not bad:break
        # Remove the largest inconsistent constraint, then re-estimate the network.
        worst=connected[int(np.argmax(errors))];active.remove(worst)
        removed.append(dict(parent=worst['parent'],child=worst['child'],residual_px=max(errors)))
    else:
        raise ValueError('Rigid graph remains inconsistent after outlier removal')
    connected=[e for e in active if e['parent'] in poses and e['child'] in poses]
    adjacency={i:[] for i in poses}
    for edge_id,e in enumerate(connected):
        adjacency[e['parent']].append((e['child'],edge_id))
        adjacency[e['child']].append((e['parent'],edge_id))
    redundant=set(poses)
    for excluded in range(len(connected)):
        reachable={reference};pending=[reference]
        while pending:
            for neighbor,edge_id in adjacency[pending.pop()]:
                if edge_id!=excluded and neighbor not in reachable:
                    reachable.add(neighbor);pending.append(neighbor)
        redundant.intersection_update(reachable)
    checks={}
    for i in poses:
        incident=[e for e in active if i in (e['parent'],e['child']) and e['parent'] in poses and e['child'] in poses]
        errors=[matrix_distance(poses[e['child']],poses[e['parent']]@np.array(e['matrix_yx']),points) for e in incident]
        checks[i]=dict(matrix_yx=poses[i].tolist(),path=paths[i],reference=i==reference,
            incident_constraints=len(incident),max_graph_residual_px=max(errors,default=0),
            two_edge_disjoint_paths_to_reference=None if i==reference else i in redundant,
            consistency_status=('reference frame' if i==reference else
                'two edge-disjoint paths to reference' if i in redundant else
                'no two edge-disjoint paths to reference'))
    rejected=[dict(index=i,reason='No reliable rigid path to reference') for i in range(count) if i not in poses]
    return poses,checks,rejected,removed


def rigid_radiometry(reference,moving,matrix):
    """Use the existing direct-radiometry diagnostics on one rigidly warped auxiliary."""
    from .model import Frame
    from .registration_graph import radiometry
    y,x=np.indices(reference.data.shape[:2],dtype=float)
    inv=np.linalg.inv(matrix);sy=inv[0,0]*y+inv[0,1]*x+inv[0,2];sx=inv[1,0]*y+inv[1,1]*x+inv[1,2]
    image=map_coordinates(moving.mono,[sy,sx],order=1,mode='constant',cval=0)
    good=map_coordinates(valid_mask(moving).astype(float),[sy,sx],order=1,mode='constant',cval=0)>1-1e-9
    good=binary_erosion(good,iterations=1)
    metadata=dict(moving.metadata,lunar_disk=None,white_level=1e300,low_signal=-1e300)
    warped=Frame(image,metadata,moving.path,dict(coverage=good,defective=~good))
    return radiometry(reference,warped,[0,0])


def reviewed_rigid(path,input_records,reference,indices):
    import json
    from pathlib import Path
    audit=json.loads(Path(path).read_text())
    if audit['input_source_sha256'] != [i['metadata']['source_sha256'] for i in input_records]:
        raise ValueError('Rigid review source hashes mismatch')
    poses={int(i):np.asarray(a,dtype=float) for i,a in audit['matrices_yx'].items()}
    if reference not in poses:raise ValueError('Reference has no reviewed rigid pose')
    target=np.linalg.inv(poses[reference]);selected={};metrics={};rejected=[]
    for i in indices:
        if i not in poses:
            rejected.append(dict(index=i,reason='No reliable rigid path to reference'));continue
        a=poses[i]
        if (a.shape!=(3,3) or not np.isfinite(a).all() or not np.allclose(a[2],[0,0,1])
                or not np.allclose(a[:2,:2].T@a[:2,:2],np.eye(2),atol=1e-8) or np.linalg.det(a[:2,:2])<0):
            raise ValueError('Reviewed transformation must be a finite proper rigid matrix')
        selected[i]=target@a
        metrics[i]=dict(audit['metrics'][str(i)],matrix_yx=selected[i].tolist(),reference=i==reference,
            registration_review=str(path),path_to_review_reference=audit['metrics'][str(i)]['path'],path=[])
    return selected,metrics,rejected,audit['edges'],audit


def geometry_condition(reference,moving,initial_shift,matrix):
    """Reject flat fields and nuisance-degenerate geometric measurements."""
    from .empirical import warp_rigid
    rm,mm=valid_mask(reference),valid_mask(moving)
    _,good=warp_rigid(moving.mono,mm,initial_shift,0)
    common=binary_erosion(rm&good,iterations=8)
    y,x=np.nonzero(common);stride=max(1,len(y)//15000);y,x=y[::stride],x[::stride]
    if len(y)<256:raise ValueError('Insufficient coronal information')
    r=gaussian_filter(np.where(rm,reference.mono/reference.scale,0),1)
    m=gaussian_filter(np.where(mm,moving.mono/moving.scale,0),1)
    if np.std(r[y,x])<1e-8*max(float(np.mean(np.abs(r[y,x]))),1):
        raise ValueError('No measurable coronal contrast')
    inv=np.linalg.inv(matrix);sy=inv[0,0]*y+inv[0,1]*x+inv[0,2];sx=inv[1,0]*y+inv[1,1]*x+inv[1,2]
    gy,gx=np.gradient(m)
    dy=map_coordinates(gy,[sy,sx],order=1,mode='constant',cval=0);dx=map_coordinates(gx,[sy,sx],order=1,mode='constant',cval=0)
    center=(np.array(m.shape)-1)/2
    jac=np.stack([-dy,-dx,-dy*(sx-center[1])+dx*(sy-center[0])],axis=-1)
    nuisance=np.stack([map_coordinates(m,[sy,sx],order=1,mode='constant',cval=0),np.ones(len(y))],axis=-1)
    jac-=nuisance@np.linalg.lstsq(nuisance,jac,rcond=None)[0]
    norm=np.linalg.norm(jac,axis=0)
    if norm.min()<max(norm.max()*1e-8,1e-10):raise ValueError('Unidentifiable rigid geometry')
    normalized=jac/norm
    condition=float(np.linalg.cond(normalized.T@normalized))
    if not np.isfinite(condition) or condition>1000:raise ValueError(f'Ill-conditioned rigid geometry: {condition:.1f}')
    return condition

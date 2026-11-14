"""Registration used by the user's saved AEL Nikon stack, as a diagnostic."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def feature(image, valid=None):
    if valid is None:
        z = np.log(np.maximum(image, 0)+20)
        return (gaussian_filter(z, 4)-gaussian_filter(z, 25)).astype(np.float32)
    valid = np.asarray(valid, bool) & np.isfinite(image)
    z = np.where(valid, np.log(np.maximum(np.where(valid, image, 0), 0)+20), 0.)
    weight = valid.astype(float)
    small, large = gaussian_filter(weight, 4), gaussian_filter(weight, 25)
    band = gaussian_filter(z, 4)/np.maximum(small, 1e-12)-gaussian_filter(z, 25)/np.maximum(large, 1e-12)
    return np.where(valid & (large > .995), band, 0.).astype(np.float32), valid & (large > .995)


def register(reference, moving, reference_moon, moving_moon, *, is_reference=False,
             reference_valid=None, moving_valid=None):
    """Fit reference-to-input on half-resolution green, with held-out sectors."""
    cx, cy = reference_moon['xy']
    radius = reference_moon['r']
    x0, y0, size = int(cx)-400, int(cy)-400, 800
    if (x0 < 0 or y0 < 0 or y0+size > reference.shape[0] or x0+size > reference.shape[1]
            or moving.shape != reference.shape):
        raise ValueError('AEL ECC reference crop must fit matching green images')
    y, x = np.indices((size,size), dtype=np.float32)
    rr = np.hypot(x+x0-cx, y+y0-cy)
    sectors = np.floor((np.arctan2(y+y0-cy, x+x0-cx)+np.pi)/(np.pi/4)).astype(int)%8
    annulus = (rr>radius+18)&(rr<radius*2.2)
    train, held = annulus&(sectors%2==0), annulus&(sectors%2==1)
    if min(train.sum(), held.sum()) < 256:
        raise ValueError('Insufficient AEL ECC coronal support')
    if (reference_valid is None) != (moving_valid is None):
        raise ValueError('Both reference and moving validity masks are required')
    masked = reference_valid is not None
    if masked:
        if np.shape(reference_valid) != reference.shape or np.shape(moving_valid) != moving.shape:
            raise ValueError('AEL ECC validity masks must match the images')
        r_feature, r_valid = feature(reference, reference_valid)
        b_full, b_valid = feature(moving, moving_valid)
        a = r_feature[y0:y0+size, x0:x0+size].copy()
        r_valid = r_valid[y0:y0+size, x0:x0+size]
    else:
        a = feature(reference)[y0:y0+size, x0:x0+size].copy()
        b_full = feature(moving)
    dx, dy = np.asarray(moving_moon['xy'])-np.asarray(reference_moon['xy'])
    initial = np.array([[1,0,dx],[0,1,dy]], np.float32)
    premapped = cv2.warpAffine(b_full, initial, (moving.shape[1], moving.shape[0]),
                               flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
    b = premapped[y0:y0+size, x0:x0+size].copy()
    if masked:
        support = cv2.warpAffine(b_valid.astype(np.float32), initial, (moving.shape[1], moving.shape[0]),
                                 flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        support = support[y0:y0+size, x0:x0+size] > .999
        common = r_valid & cv2.erode(support.astype(np.uint8), np.ones((7,7), np.uint8)).astype(bool)
        train &= common
        held &= common
        if min(train.sum(), held.sum()) < 256:
            raise ValueError('Insufficient valid AEL ECC coronal support')
    w = np.eye(2,3,dtype=np.float32)
    if is_reference:
        quality = 1.
    else:
        try:
            quality, w = cv2.findTransformECC(a,b,w,cv2.MOTION_EUCLIDEAN,
                (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,200,1e-7),train.astype(np.uint8)*255,3)
        except cv2.error as error:
            raise ValueError('AEL ECC did not converge: '+str(error).splitlines()[-1]) from error
    aligned = cv2.warpAffine(b,w,(size,size),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT)
    if masked:
        aligned_support = cv2.warpAffine(support.astype(np.float32),w,(size,size),
            flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT) > .999
        if np.mean(aligned_support[train]) < .995:
            raise ValueError('AEL ECC fit moved outside valid training support')
        held &= aligned_support
        if held.sum() < 256:
            raise ValueError('Insufficient held-out valid AEL ECC support')
    corr = float(np.corrcoef(a[held], aligned[held])[0,1])
    if not np.isfinite(corr):
        raise ValueError('AEL ECC held-out correlation is undefined')
    full = w.astype(float)
    full[:,2] += np.array([x0,y0])-full[:,:2]@np.array([x0,y0])+np.array([dx,dy])
    native = full.copy()
    native[:,2] = 2*full[:,2]+.5-.5*full[:,:2].sum(axis=1)
    angle_deg = float(np.degrees(np.arctan2(w[1,0],w[0,0])))
    return native, dict(ecc=float(quality), heldout_correlation=corr, angle_deg=angle_deg,
        status='candidate' if corr < .65 or abs(angle_deg) > .5 else 'registered',
        qualification=('AEL-style ECC with supplied validity masks; independent graph not applied' if masked else
                       'saved AEL workflow check; native input masks and independent graph not applied'))

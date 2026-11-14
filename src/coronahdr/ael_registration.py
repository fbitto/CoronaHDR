"""AEL registration routines, revision 63c16820ccf703badfcbd4d14bfe2fb59db0a63c.

MIT License

Copyright (c) 2026 Pere Guerra Serra

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import math
import cv2
import numpy as np


def band_image(green: np.ndarray, saturated: np.ndarray, moon: dict, sigma: tuple[float, float]):
    """Band-pass of the log corona with normalised convolution: no saturated, empty or lunar pixel counts.

    The small scale stays above the sensor's own pattern (interpolated greens, hot pixels), the large one removes the
    smooth light that the Moon uncovers as it moves. Nothing here is centred anywhere: the band only needs the data.
    """
    valid = np.isfinite(green) & (green > 0) & ~saturated
    yy, xx = np.mgrid[:green.shape[0], :green.shape[1]]
    valid &= np.hypot(xx - moon["xy"][0], yy - moon["xy"][1]) > moon["r"] + max(5.0, 0.02 * moon["r"])
    lg = np.where(valid, np.log(np.where(valid, green, 1)), 0).astype(np.float32)
    v = valid.astype(np.float32)
    blur = lambda a, s: cv2.GaussianBlur(a, (0, 0), s)
    small, large = blur(v, sigma[0]), blur(v, sigma[1])
    band = blur(lg * v, sigma[0]) / np.maximum(small, 1e-6) - blur(lg * v, sigma[1]) / np.maximum(large, 1e-6)
    keep = valid & (large > 0.9)
    return np.where(keep, band, 0).astype(np.float32), keep


# --- registration on the Sun ---------------------------------------------------------------------------------
def _box(center, half: int, shape):
    cx, cy = int(round(center[0])), int(round(center[1]))
    return cy - half, cy + half, cx - half, cx + half


def _crop(a: np.ndarray, box, fill=0):
    y0, y1, x0, x1 = box
    out = np.full((y1 - y0, x1 - x0), fill, a.dtype)
    ys, ye, xs, xe = max(y0, 0), min(y1, a.shape[0]), max(x0, 0), min(x1, a.shape[1])
    if ye > ys and xe > xs:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = a[ys:ye, xs:xe]
    return out


def coarse_shift(ref_band, ref_use, mov_band, mov_keep, center, half, offset):
    """Cross-correlation (not whitened: a band-pass leaves frequencies empty that whitening would inflate) of the
    reference ring against the moving frame, in boxes of the same size; the moving box is moved by ``offset``."""
    box = _box(center, half, ref_band.shape)
    moved = (box[0] + offset[1], box[1] + offset[1], box[2] + offset[0], box[3] + offset[0])
    window = np.outer(np.hanning(2 * half), np.hanning(2 * half)).astype(np.float32)
    a = _crop(np.where(ref_use, ref_band, 0).astype(np.float32), box) * window
    b = _crop(np.where(mov_keep, mov_band, 0).astype(np.float32), moved) * window
    spectrum = np.fft.rfft2(b) * np.conj(np.fft.rfft2(a))
    corr = np.fft.irfft2(spectrum, s=a.shape)
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)

    def refine(k, n, axis):
        values = [corr[(peak[0] + d) % n, peak[1]] if axis == 0 else corr[peak[0], (peak[1] + d) % n] for d in (-1, 0, 1)]
        den = values[0] - 2 * values[1] + values[2]
        frac = 0.5 * (values[0] - values[2]) / den if den < 0 else 0.0
        k = k + frac
        return k - n if k > n / 2 else k
    dy, dx = refine(peak[0], corr.shape[0], 0), refine(peak[1], corr.shape[1], 1)
    return np.array([offset[0] + dx, offset[1] + dy], np.float64)


def _axes(shape, centre):
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    return xx - np.float32(centre[0]), yy - np.float32(centre[1])


def _rigid_sampler(shape, centre, theta, shift, axes=None):
    """Coordinates where reference box pixel q falls in the moving box: R(theta) (q - centre) + centre + shift."""
    u, v = axes if axes is not None else _axes(shape, centre)
    c, s = np.float32(math.cos(theta)), np.float32(math.sin(theta))
    mx = (centre[0] + c * u - s * v + shift[0]).astype(np.float32)
    my = (centre[1] + s * u + c * v + shift[1]).astype(np.float32)
    sample = lambda a: cv2.remap(np.ascontiguousarray(a, np.float32), mx, my, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return sample, u, v


def solve_rigid(ref, ref_use, weight, mov, mov_use, centre, start, *, iterations=150):
    """Weighted Gauss-Newton for mov(R(theta)(x - centre) + centre + t) = a*ref(x) + b: the translation and the turn
    of the field about the Sun (a mount that does not track the sky turns the field during the totality)."""
    gy, gx = np.gradient(mov)
    grad_use = cv2.erode(mov_use.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
    t = np.asarray(start, np.float64).copy()
    theta = 0.0
    reach = max(1.0, 0.5 * max(ref.shape))
    axes = _axes(ref.shape, centre)
    for _ in range(iterations):
        sample, u, v = _rigid_sampler(ref.shape, centre, theta, t, axes)
        m = ref_use & (sample(grad_use) > 0.999)
        n = int(m.sum())
        if n < 100:
            raise RuntimeError("Insufficient overlap for solar registration")
        c, s = math.cos(theta), math.sin(theta)
        sx, sy = sample(gx)[m].astype(np.float64), sample(gy)[m].astype(np.float64)
        jt = sx * (-s * u[m] - c * v[m]) + sy * (c * u[m] - s * v[m])
        root = np.sqrt(weight[m]).astype(np.float64)
        a = np.column_stack([sx, sy, jt, -ref[m], -np.ones(n)]) * root[:, None]
        step = np.linalg.lstsq(a, -sample(mov)[m] * root, rcond=None)[0][:3]
        move = float(np.hypot(step[0], step[1]))
        step[:2] *= min(1.0, 1.0 / max(move, 1e-12))
        step[2] = float(np.clip(step[2], -1.0 / reach, 1.0 / reach))
        t += step[:2]
        theta += step[2]
        if move < 1e-4 and abs(step[2]) * reach < 1e-4:
            return t, theta
    raise RuntimeError("Solar registration did not converge")


def fine_rigid(ref_band, ref_use, weight, mov_band, mov_keep, centre, half, start):
    """``solve_rigid`` on the band images in a box around the reference Sun; returns (translation, turn) that take
    reference pixels to the frame's pixels about ``centre`` (half resolution)."""
    whole = np.round(start).astype(int)
    box = _box(centre, half, ref_band.shape)
    moved = (box[0] + whole[1], box[1] + whole[1], box[2] + whole[0], box[3] + whole[0])
    r, ru, w = _crop(ref_band, box), _crop(ref_use, box, False), _crop(weight, box)
    m, mu = _crop(mov_band, moved), _crop(mov_keep, moved, False)
    local = (centre[0] - box[2], centre[1] - box[0])
    t, theta = solve_rigid(r, ru, w, m, mu, local, np.asarray(start, np.float64) - whole)
    return t + whole, theta


def rigid_structure(ref_band, ref_use, mov_band, mov_keep, centre, half, t, theta) -> dict:
    """The registration checked on coronal structure, against the same frame turned 90 and 270 degrees about the
    Sun: those keep the radial statistics and lose the structure. Score = share of what the turn leaves unexplained
    that the registration explains."""
    whole = np.round(t).astype(int)
    box = _box(centre, half, ref_band.shape)
    moved = (box[0] + whole[1], box[1] + whole[1], box[2] + whole[0], box[3] + whole[0])
    r, ru = _crop(ref_band, box), _crop(ref_use, box, False)
    m, mu = _crop(mov_band, moved), _crop(mov_keep, moved, False).astype(np.float32)
    local = (centre[0] - box[2], centre[1] - box[0])
    values = {}
    for turn in (0, 90, 270):
        sample, _, _ = _rigid_sampler(r.shape, local, theta + math.radians(turn), np.asarray(t) - whole)
        use = ru & (sample(mu) > 0.999)
        if use.sum() < 100:
            values[turn] = float("nan")
            continue
        x, y = r[use].astype(np.float64), sample(m)[use].astype(np.float64)
        x, y = x - x.mean(), y - y.mean()
        values[turn] = float((x * y).sum() / max(math.sqrt((x * x).sum() * (y * y).sum()), 1e-30))
    null = max(values[90], values[270])
    score = (values[0] - null) / (1.0 - null) if null < 1 else float("nan")
    return dict(correlation=values[0], turned=[values[90], values[270]], structure_score=score)


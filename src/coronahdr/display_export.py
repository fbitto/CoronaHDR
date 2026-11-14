"""16-bit HDR display export with an independent per-pixel validity record."""
from contextlib import contextmanager
import os
import errno
import shutil
from pathlib import Path
import tempfile

import cv2
import numpy as np
import tifffile
from astropy.io import fits
from scipy import ndimage

from .desktop_previews import hdr_display


@contextmanager
def source_image(path):
    path = Path(path)
    if path.suffix.lower() in ('.fits', '.fit', '.fts'):
        hdus = fits.open(path, memmap=True)
        try:
            try:
                data = next(h.data for h in hdus if h.data is not None and h.data.ndim >= 2)
            except ValueError as error:
                if 'memory-mapped' not in str(error):
                    raise
                hdus.close()
                hdus = fits.open(path, memmap=False)
                data = next(h.data for h in hdus if h.data is not None and h.data.ndim >= 2)
            if data.ndim == 3 and data.shape[0] == 3:
                data = np.moveaxis(data, 0, -1)
            yield data
        finally:
            hdus.close()
    else:
        try:
            data = tifffile.memmap(path)
        except ValueError:
            data = tifffile.imread(path)
        yield data


def valid_and_gaps(array):
    valid = np.isfinite(array).all(axis=-1) if array.ndim == 3 else np.isfinite(array)
    labels, count = ndimage.label(ndimage.binary_fill_holes(valid) & ~valid)
    sizes = np.bincount(labels.ravel())
    small = np.flatnonzero((sizes <= 16) & (np.arange(len(sizes)) > 0))
    gaps = np.isin(labels, small) if count else np.zeros(valid.shape, bool)
    return valid, gaps


def analyze(path, crop=None):
    path = Path(path)
    before = path.stat()
    with source_image(path) as array:
        if crop is not None:
            from .crop_export import validate_crop
            x, y, width, height = validate_crop(crop, array.shape[:2])
            array = array[y:y+height, x:x+width]
        valid, gaps = valid_and_gaps(array)
        counts = dict(missing=int((~valid).sum()), eligible=int(gaps.sum()))
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('Source image changed during inspection')
    counts['identity'] = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    return counts


def export(path, destination, levels, gains, repair, identity, filter_config=None, crop=None):
    path, destination = Path(path), Path(destination)
    if filter_config:
        from .corona_filters import check_filter_source
        check_filter_source(path, filter_config)
    extension = destination.suffix.lower()
    if extension not in ('.png', '.tif', '.tiff', '.fits', '.fit', '.fts'):
        raise ValueError('Choose a PNG, TIFF or FITS filename')
    validity_path = destination.with_name(destination.stem+'_validity.png')
    if destination.exists() or destination.is_symlink() or validity_path.exists() or validity_path.is_symlink():
        raise FileExistsError('The output image or validity mask already exists; choose another name')
    info = path.stat()
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != tuple(identity):
        raise ValueError('Source image changed after the export check')
    gains = np.asarray(gains if gains is not None else (1., 1., 1.))
    with source_image(path) as array:
        if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[-1] != 3):
            raise ValueError('Export requires a mono or RGB image')
        sample = np.asarray(array[::10, ::10], dtype=np.float64)
        if sample.ndim == 2:
            sample = np.repeat(sample[..., None], 3, axis=-1)
        good = np.isfinite(sample).all(axis=-1)
        if not good.any():
            raise ValueError('Image has no finite sampled observations')
        high = max(float(np.percentile(sample[good]*gains, 99.8)), 1e-6)
        filter_data = None
        if filter_config:
            from .corona_filters import image_filter_layers
            filter_data = image_filter_layers(array, filter_config)
        x, y = 0, 0
        if crop is not None:
            from .crop_export import validate_crop
            x, y, width, height = validate_crop(crop, array.shape[:2])
            array = array[y:y+height, x:x+width]
            if filter_data:
                filtered, layers = filter_data
                filter_data = (filtered[y:y+height, x:x+width],
                               [(layer[y:y+height, x:x+width], mode, opacity) for layer, mode, opacity in layers])
        valid, gaps = valid_and_gaps(array)
        nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True) if repair and gaps.any() else None
        output = np.empty((*array.shape[:2], 3), np.uint16)
        mask = valid.astype(np.uint8)*255
        if nearest is not None:
            mask[gaps] = 127
        for start in range(0, array.shape[0], 128):
            end = min(start+128, array.shape[0])
            block = np.array(array[start:end], dtype=np.float64, copy=True)
            if block.ndim == 2:
                block = np.repeat(block[..., None], 3, axis=-1)
            if nearest is not None:
                yy, xx = np.nonzero(gaps[start:end])
                source_y, source_x = nearest[0, yy+start, xx], nearest[1, yy+start, xx]
                replacement = np.asarray(array[source_y, source_x])
                if replacement.ndim == 1:
                    replacement = np.repeat(replacement[:, None], 3, axis=-1)
                block[yy, xx] = replacement
            display, observed = hdr_display(block, high, gains, levels)
            if filter_data:
                from .corona_filters import blend_layers, lunar_blend_transition
                filtered, layers = filter_data
                transition = lunar_blend_transition(filter_config, observed.shape, (y+start, x))
                display = blend_layers(display, filtered[start:end] & observed,
                    [(layer[start:end], mode, opacity) for layer, mode, opacity in layers], filter_config['strength'], transition)
            output[start:end] = np.rint(display*65535).astype(np.uint16)
            output[start:end][~observed] = 0
        after = path.stat()
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != tuple(identity):
            raise ValueError('Source image changed during export')
    temporary = []
    created = []
    try:
        for _ in range(2):
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.coronahdr-export-', delete=False) as stream:
                temporary.append(Path(stream.name))
        with temporary[0].open('wb') as stream:
            if extension == '.png':
                ok, encoded = cv2.imencode('.png', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise OSError('PNG encoding failed')
                stream.write(encoded.tobytes())
            elif extension in ('.fits', '.fit', '.fts'):
                header = fits.Header()
                header['CHDRTYPE'] = 'NONLINEAR DISPLAY'
                fits.PrimaryHDU(np.moveaxis(output, -1, 0), header).writeto(stream)
            else:
                tifffile.imwrite(stream, output, photometric='rgb', metadata={'description': 'Nonlinear display image'})
        ok, encoded = cv2.imencode('.png', mask)
        if not ok:
            raise OSError('Validity mask encoding failed')
        temporary[1].write_bytes(encoded.tobytes())
        for source, target in zip(temporary, (destination, validity_path)):
            try:
                os.link(source, target)
            except OSError as error:
                if error.errno not in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV):
                    raise
                with source.open('rb') as src, target.open('xb') as dst:
                    info = os.fstat(dst.fileno())
                    created.append((target, info.st_dev, info.st_ino))
                    shutil.copyfileobj(src, dst, 1024*1024)
            else:
                info = target.stat()
                created.append((target, info.st_dev, info.st_ino))
    except BaseException:
        for target, device, inode in created:
            try:
                info = target.stat(follow_symlinks=False)
                if (info.st_dev, info.st_ino) == (device, inode):
                    target.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for name in temporary:
            name.unlink(missing_ok=True)
    return dict(image=str(destination), validity=str(validity_path), measured=int(valid.sum()),
                interpolated=int(gaps.sum()) if repair else 0, missing=int((~valid).sum())-int(gaps.sum() if repair else 0))

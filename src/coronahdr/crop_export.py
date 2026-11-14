"""Save a cropped, linear CoronaHDR master without replacing existing files."""
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import tifffile
from astropy.io import fits

from .display_export import source_image


def validate_crop(crop, shape):
    if len(crop) != 4 or any(type(v) is not int for v in crop):
        raise ValueError('Crop must contain integer x, y, width and height')
    x, y, width, height = crop
    if min(x, y) < 0 or min(width, height) < 1 or x+width > shape[1] or y+height > shape[0]:
        raise ValueError('Crop must be inside the image and have a positive size')
    return x, y, width, height


def source_metadata(path):
    path = Path(path)
    header = None
    if path.suffix.lower() in ('.fits', '.fit', '.fts'):
        with fits.open(path, memmap=False, lazy_load_hdus=True) as hdus:
            hdu = next(h for h in hdus if h.header.get('NAXIS', 0) >= 2)
            header = hdu.header.copy()
            raw = header.get('CHDRMETA', '{}')
            metadata = json.loads(raw)
    else:
        with tifffile.TiffFile(path) as tif:
            description = json.loads(tif.pages[0].description or '{}')
            metadata = description.get('coronahdr', {})
    if not isinstance(metadata, dict) or 'origin_reference_yx' not in metadata or 'shape_yx' not in metadata or 'reference' not in metadata:
        raise ValueError('This image lacks the CoronaHDR geometry metadata needed to save a crop')
    if header is None and metadata.get('crop_fits_header'):
        header = fits.Header.fromstring(metadata['crop_fits_header'], sep='\n')
    if header is not None:
        if any(k.startswith(('CPDIS', 'D2IM', 'DP', 'DET2IM', 'LTV', 'LTM', 'WAT')) or k in ('DETSEC', 'CCDSEC', 'DATASEC', 'TRIMSEC') for k in header) or any('-TAB' in str(header.get(k, '')) for k in header if k.startswith('CTYPE')):
            raise ValueError('This image contains coordinate distortions that cannot be safely cropped')
    return metadata, header


def save_crop(path, destinations, crop, expected_identity=None):
    path = Path(path)
    destinations = [Path(p) for p in destinations]
    if not destinations or len(set(destinations)) != len(destinations) or any(p.suffix.lower() not in ('.fits', '.tiff', '.tif') for p in destinations):
        raise ValueError('Choose distinct FITS and/or TIFF output files')
    if any(p.exists() or p.is_symlink() for p in destinations):
        raise FileExistsError('A cropped output already exists; choose a different name')
    before = path.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if expected_identity is not None and tuple(expected_identity) != identity:
        raise ValueError('Source image changed since the preview was loaded')
    metadata, header = source_metadata(path)
    with source_image(path) as array:
        x, y, width, height = validate_crop(crop, array.shape[:2])
        if list(array.shape[:2]) != list(metadata['shape_yx']):
            raise ValueError('Image dimensions do not match its CoronaHDR metadata')
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        new = dict(metadata)
        oy, ox = metadata['origin_reference_yx']
        new['origin_reference_yx'] = [oy+y, ox+x]
        new['shape_yx'] = [height, width]
        new['crop_source'] = dict(path=str(path.resolve()), sha256=digest, xywh=[x, y, width, height])
        if header is not None:
            header = header.copy()
            for i, delta in ((1, x), (2, y)):
                for key in list(header):
                    if key.startswith(f'CRPIX{i}'):
                        header[key] = float(header[key])-delta
            for key in ('CHDRMETA', 'CHECKSUM', 'DATASUM'):
                if key in header:
                    del header[key]
            new['crop_fits_header'] = header.tostring(sep='\n', endcard=False, padding=False)
        cropped = np.asarray(array[y:y+height, x:x+width])
        temps, created = [], []
        try:
            for dest in destinations:
                with tempfile.NamedTemporaryFile(dir=dest.parent, prefix='.coronahdr-crop-', delete=False) as stream:
                    tmp = Path(stream.name)
                temps.append(tmp)
                if dest.suffix.lower() == '.fits':
                    out_header = header.copy() if header is not None else fits.Header()
                    out_header['CHDRMETA'] = json.dumps(new, allow_nan=False)
                    fits.PrimaryHDU(np.moveaxis(cropped, -1, 0) if cropped.ndim == 3 else cropped, out_header).writeto(tmp, overwrite=True)
                else:
                    tifffile.imwrite(tmp, cropped, photometric='rgb' if cropped.ndim == 3 else 'minisblack', metadata={'coronahdr': new})
            after = path.stat()
            if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity:
                raise ValueError('Source image changed while saving the crop')
            for tmp, dest in zip(temps, destinations):
                try:
                    os.link(tmp, dest)
                except OSError as error:
                    if error.errno not in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV):
                        raise
                    with tmp.open('rb') as src, dest.open('xb') as dst:
                        info = os.fstat(dst.fileno())
                        created.append((dest, info.st_dev, info.st_ino))
                        shutil.copyfileobj(src, dst, 1024*1024)
                else:
                    info = dest.stat()
                    created.append((dest, info.st_dev, info.st_ino))
        except BaseException:
            for dest, device, inode in created:
                try:
                    info = dest.stat(follow_symlinks=False)
                    if (info.st_dev, info.st_ino) == (device, inode):
                        dest.unlink()
                except FileNotFoundError:
                    pass
            raise
        finally:
            for tmp in temps:
                tmp.unlink(missing_ok=True)
    return [str(p) for p in destinations]

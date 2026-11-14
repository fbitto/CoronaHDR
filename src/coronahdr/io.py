"""FITS uses CYX RGB planes; TIFF uses YXS interleaved RGB. No scale changes."""
import hashlib
import json
from pathlib import Path
import numpy as np
from astropy.io import fits
import tifffile
from .model import Frame
from .sensor import SensorFrame
from .raw import RAW_EXTENSIONS, decode_raw


def read_frame(path, overrides=None):
    path = Path(path)
    metadata = {}
    conflicts_resolved = []
    overrides = overrides or {}
    raw_input = path.suffix.lower() in RAW_EXTENSIONS
    if path.suffix.lower() in ('.fits', '.fit', '.fts'):
        with fits.open(path) as hdus:
            hdu = next((h for h in hdus if h.data is not None and h.data.ndim >= 2), None)
            if hdu is None:
                raise ValueError(f'{path}: no image HDU')
            data = hdu.data.copy()
            metadata = json.loads(hdu.header.get('CHDRMETA', '{}'))
            for card, key in [('EXPTIME', 'exposure_s'), ('SATURATE', 'white_level'), ('BLACKLVL', 'black_level')]:
                if card in hdu.header:
                    if key in metadata and metadata[key] != hdu.header[card]:
                        if key not in overrides:
                            raise ValueError(f'{path}: conflicting FITS {card} and CHDRMETA {key}; provide an explicit override')
                        conflicts_resolved.append(dict(field=key, embedded=metadata[key], header=hdu.header[card]))
                    metadata[key] = hdu.header[card]
            if data.ndim == 3:
                if data.shape[0] != 3:
                    raise ValueError('FITS RGB must use three CYX planes')
                data = np.moveaxis(data, 0, -1)
            metadata['source_header'] = hdu.header.tostring(sep='\n', endcard=False, padding=False)
    elif path.suffix.lower() in ('.tif', '.tiff'):
        with tifffile.TiffFile(path) as tif:
            if len(tif.series) != 1 or len(tif.pages) != 1:
                raise ValueError('Only single-image TIFF is supported')
            data = tif.asarray()
            try:
                metadata = json.loads(tif.pages[0].description or '{}').get('coronahdr', {})
            except (ValueError, AttributeError):
                metadata = {}
            metadata['source_tiff_tags'] = {tag.name: str(tag.value) for tag in tif.pages[0].tags.values()
                                            if tag.name not in ('StripOffsets', 'StripByteCounts', 'TileOffsets', 'TileByteCounts', 'ImageDescription')}
            metadata['source_description'] = tif.pages[0].description
    elif raw_input:
        data, metadata = decode_raw(path)
    else:
        raise ValueError(f'{path}: accepts TIFF, FITS, and experimental Bayer RAW; XISF is pending')
    sidecar = path.with_suffix(path.suffix + '.json')
    if sidecar.exists():
        declared = json.loads(sidecar.read_text())
        conflicts = [k for k in declared if k in metadata and metadata[k] != declared[k]]
        unresolved = [k for k in conflicts if k not in overrides]
        if unresolved:
            raise ValueError(f'{path}: conflicting sidecar metadata: {conflicts}; use manifest overrides')
        conflicts_resolved.extend(dict(field=k, embedded=metadata[k], sidecar=declared[k]) for k in conflicts)
        metadata.update(declared)
    history = []
    for key, value in (overrides or {}).items():
        history.append(dict(field=key, previous=metadata.get(key), value=value))
        metadata[key] = value
    metadata['manual_overrides'] = history
    metadata['conflicts_resolved_by_overrides'] = conflicts_resolved
    metadata['source_dtype'] = str(data.dtype)
    with path.open('rb') as source:
        metadata['source_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
    if raw_input and metadata.get('state') != 'linear_cfa':
        raise ValueError('Experimental RAW ingestion requires an explicit state=linear_cfa declaration')
    detector = raw_input or metadata.get('state') == 'linear_cfa' or 'role' in metadata
    return SensorFrame(data, metadata, path) if detector else Frame(data, metadata, path)


def write_image(path, data, metadata=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    a = np.asarray(data, dtype=np.float32)
    if path.suffix == '.fits':
        header = fits.Header()
        header['CHDRMETA'] = json.dumps(metadata or {}, allow_nan=False)
        header['BUNIT'] = (metadata or {}).get('unit', 'relative response / s')
        fits.PrimaryHDU(np.moveaxis(a, -1, 0) if a.ndim == 3 else a, header).writeto(path)
    else:
        tifffile.imwrite(path, a, photometric='rgb' if a.ndim == 3 else 'minisblack',
                         metadata={'coronahdr': metadata or {}})

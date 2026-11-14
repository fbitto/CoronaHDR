"""Experimental Bayer decoding only: no LibRaw postprocess or rendered RGB."""
from pathlib import Path
from io import BytesIO
import struct
import numpy as np
import rawpy
import tifffile
from .sensor import BAYER_PATTERNS, PHASES, phase_tile

RAW_EXTENSIONS = {'.cr2', '.cr3', '.nef', '.nrw', '.arw', '.dng', '.raf', '.orf', '.rw2', '.pef', '.srw', '.raw'}


def cr3_exif(path):
    """Read Canon CMT1/CMT2 TIFF metadata, without unpacking image payloads."""
    canon_uuid = bytes.fromhex('85c0b687820f11e08111f4ce462b6a48')
    blocks = {}
    with Path(path).open('rb') as stream:
        def boxes(start, end, context=b''):
            while start < end:
                stream.seek(start)
                header = stream.read(8)
                if len(header) != 8:
                    raise ValueError('Truncated CR3 box header')
                size, kind = struct.unpack('>I4s', header)
                offset = start+8
                if size == 1:
                    extended = stream.read(8)
                    if len(extended) != 8:
                        raise ValueError('Truncated CR3 extended box size')
                    size = int.from_bytes(extended, 'big')
                    offset += 8
                elif size == 0:
                    size = end-start
                stop = start+size
                if stop > end or stop < offset:
                    raise ValueError('Invalid CR3 box size')
                if kind == b'moov' and not context:
                    boxes(offset, stop, b'moov')
                elif kind == b'uuid' and context == b'moov':
                    if stop-offset < 16:
                        raise ValueError('Truncated CR3 UUID')
                    if stream.read(16) == canon_uuid:
                        boxes(offset+16, stop, b'canon')
                elif kind in (b'CMT1', b'CMT2') and context == b'canon':
                    if kind in blocks or stop-offset > 16*1024*1024:
                        raise ValueError('Duplicate or oversized CR3 metadata block')
                    blocks[kind] = stream.read(stop-offset)
                start = stop
        stream.seek(0, 2)
        boxes(0, stream.tell())
    if set(blocks) != {b'CMT1', b'CMT2'}:
        raise ValueError('CR3 requires Canon CMT1 and CMT2 metadata blocks')
    tags = {}
    for payload in blocks.values():
        if payload[:4] not in (b'II*\x00', b'MM\x00*') or len(payload) < 8:
            raise ValueError('Invalid CR3 metadata TIFF header')
        byteorder = '<' if payload[:2] == b'II' else '>'
        with tifffile.FileHandle(BytesIO(payload)) as handle:
            handle.seek(struct.unpack(byteorder+'I', payload[4:8])[0])
            records = tifffile.tifffile.read_tags(handle, byteorder, 4, tifffile.TIFF.TAGS, maxifds=1)
        if not records:
            raise ValueError('Empty CR3 metadata TIFF directory')
        tags.update(records[0])
    return tags


def camera_exif(path):
    """Read standard EXIF from TIFF RAW and Canon CR3 using installed tifffile."""
    try:
        if Path(path).suffix.lower() == '.cr3':
            exif = cr3_exif(path)
        else:
            with tifffile.TiffFile(path) as tif:
                tags = tif.pages[0].tags
                exif = dict(tags['ExifTag'].value) if 'ExifTag' in tags else {}
                exif.update({key: tags[key].value for key in ('Make', 'Model') if key in tags})
        source = {key: exif[key] for key in ('Make', 'Model', 'ExposureTime', 'FNumber', 'ISOSpeedRatings',
            'BodySerialNumber', 'DateTimeOriginal', 'OffsetTimeOriginal', 'FocalLength') if key in exif}
        metadata = dict(source_exif=source, gain=None, gain_status='unknown; not a measured electron conversion gain')
        for key, field in [('ExposureTime', 'exposure_s'), ('FNumber', 'aperture'), ('FocalLength', 'focal_length_mm')]:
            if key in source:
                value = source[key]
                metadata[field] = float(value[0]/value[1] if isinstance(value, tuple) else value)
        if 'ISOSpeedRatings' in source:
            metadata['iso'] = float(source['ISOSpeedRatings'])
        if 'Model' in source and 'BodySerialNumber' in source:
            metadata['sensor_id'] = f"{source['Model']} serial {source['BodySerialNumber']}"
        return metadata
    except (OSError, ValueError, KeyError, ZeroDivisionError, struct.error) as error:
        return dict(gain=None, gain_status='unknown; not a measured electron conversion gain',
                    raw_exif_status=f'RAW EXIF unavailable: {error}; explicit metadata is required')


def decode_raw(path):
    try:
        with rawpy.imread(str(path)) as raw:
            if raw.raw_type != rawpy.RawType.Flat or raw.raw_image_visible.ndim != 2:
                raise ValueError('Only flat Bayer RAW mosaics are implemented; stacked/Foveon RAW is rejected')
            if raw.raw_pattern is None or raw.raw_pattern.shape != (2, 2):
                raise ValueError('Only 2x2 RGB Bayer RAW is implemented; X-Trans and other CFA layouts are rejected')
            data = raw.raw_image_visible.copy()
            sizes = raw.sizes
            if sizes.pixel_aspect != 1:
                raise ValueError('Non-square RAW sensor pixels require a validated geometry path')
            desc = raw.color_desc.decode('ascii')
            indices = raw.raw_colors[:2, :2].copy()
            pattern = ''.join(desc[int(i)] for i in indices.flat)
            if pattern not in BAYER_PATTERNS:
                raise ValueError(f'Unsupported RAW CFA colors: {pattern}')
            origin = [sizes.left_margin, sizes.top_margin]
            visible_pattern = ''.join(desc[int(i)] for i in raw.raw_colors_visible[:2, :2].flat)
            expected = ''.join(label[0] for label in phase_tile(pattern, origin).flat)
            if visible_pattern != expected:
                raise ValueError('RAW active-area CFA phase disagrees with sensor coordinates')
            blacks = raw.black_level_per_channel
            whites = raw.camera_white_level_per_channel
            if whites is None:
                whites = [raw.white_level]*len(blacks)
            labels = phase_tile(pattern, [0, 0])
            levels = {phase: int(indices[labels == phase][0]) for phase in PHASES}
            metadata = dict(decoded_layout='bayer_cfa', role='light', black_subtracted=False,
                black_level=[float(blacks[levels[p]]) for p in PHASES],
                white_level=[float(whites[levels[p]]) for p in PHASES],
                cfa_pattern=pattern, active_area_xywh=[*origin, data.shape[1], data.shape[0]],
                raw_sensor_shape_yx=[sizes.raw_height, sizes.raw_width],
                raw_decoder='rawpy '+rawpy.__version__, raw_color_desc=desc,
                raw_orientation_code=int(sizes.flip), orientation_applied=False,
                raw_validation='experimental decoder; real-data validation coverage is documented separately',
                corrections_applied_by_decoder='LibRaw unpack only; no postprocess, gamma, white balance, exposure or orientation transform')
            metadata.update(camera_exif(path))
            return data, metadata
    except rawpy.LibRawError as error:
        raise ValueError(f'{Path(path)}: LibRaw decoding failed: {error}') from error

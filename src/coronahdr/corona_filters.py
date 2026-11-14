"""AEL's display layers, computed from observed linear corona samples."""
import numpy as np

from ._ael_filters.filterbank import make_filter
from ._ael_filters.geometry import EclipseGeometry


def filter_layers(image, coverage, moon, sun, margin):
    """Calculate AEL layers without treating missing or lunar pixels as observations."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('A mono or RGB linear image is required')
    coverage = np.asarray(coverage)
    if coverage.shape != image.shape[:2] or not np.isfinite(coverage).all():
        raise ValueError('A matching finite geometric coverage map is required')
    if len(moon) != 3 or len(sun) != 3 or not np.isfinite([*moon, *sun, margin]).all():
        raise ValueError('Valid lunar and solar geometry is required')
    if moon[2] <= 0 or sun[2] <= 0 or margin < 0:
        raise ValueError('Radii must be positive and the lunar margin cannot be negative')
    geometry = EclipseGeometry(shape=image.shape[:2], sun_xy=tuple(sun[:2]), sun_radius_px=sun[2],
                               moon_xy=tuple(moon[:2]), moon_radius_px=moon[2])
    lum = (image[..., 0] + 2*image[..., 1] + image[..., 2])/4
    valid = np.isfinite(image).all(axis=-1) & (lum > 0) & (coverage >= .5) & ~geometry.moon_mask(margin)
    if not valid.any():
        raise ValueError('No observed corona remains outside the lunar exclusion')
    layers = []
    for name, opacity in (('nrgf', .39), ('wow_edge_aware', .20), ('achf_iso_04', .08)):
        layer, mode, _ = make_filter(name, lum, valid, geometry)
        layers.append((layer, mode, opacity))
    return valid, layers


def blend_layers(display, valid, layers, strength, transition=None):
    """Composite the AEL layers over a display image, leaving excluded pixels intact."""
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Filter strength must be between zero and one')
    original = np.asarray(display)
    result = np.array(display, dtype=np.float32, copy=True)
    if transition is not None:
        transition = np.asarray(transition, dtype=np.float32)
        if transition.shape != valid.shape or not np.isfinite(transition).all() or np.any((transition < 0) | (transition > 1)):
            raise ValueError('Filter transition must match the image and be between zero and one')
    for layer, mode, opacity in layers:
        active = valid & np.isfinite(layer)
        value = np.where(active, layer, 0)[..., None]
        if mode == 'multiply':
            blended = result * value
        elif mode == 'overlay':
            blended = np.where(result <= .5, 2*result*value, 1-2*(1-result)*(1-value))
        else:
            raise ValueError(f'Unsupported AEL layer blend: {mode}')
        weight = strength*opacity if transition is None else (strength*opacity*transition)[..., None]
        result = np.where(active[..., None], result + weight*(blended-result), result)
    unchanged = ~valid if transition is None else (~valid | (transition == 0))
    if strength == 0:
        unchanged = np.ones_like(valid, dtype=bool)
    return np.where(unchanged[..., None], np.clip(original, 0, 1), np.clip(result, 0, 1))


def lunar_blend_transition(context, shape, origin_yx=(0, 0)):
    """Fade in filtering outside the protected lunar zone in native image pixels."""
    height, width = shape
    y0, x0 = origin_yx
    mx, my, radius = context['moon']
    distance = np.hypot(np.arange(height, dtype=np.float64)[:, None] + y0 - my,
                        np.arange(width, dtype=np.float64)[None, :] + x0 - mx)
    edge = radius + context['margin']
    fade_width = max(8., .15*radius)
    t = np.clip((distance-edge)/fade_width, 0., 1.)
    return (t*t*(3.-2.*t)).astype(np.float32)


def job_filter_context(path, temporary=None, _seen=None):
    """Find a hash-bound job and its geometric coverage and measured lunar limb."""
    import hashlib
    import json
    from pathlib import Path
    from .lunar import fitted_disk
    from .rigid import apply_points

    path = Path(path)
    _seen = set() if _seen is None else _seen
    resolved = path.resolve()
    if resolved in _seen:
        raise ValueError('Cropped master provenance contains a cycle')
    _seen.add(resolved)
    from .crop_export import source_metadata, validate_crop
    from .display_export import source_image
    # The settings file retains hash-bound filter data after job cleanup.
    stem = path.stem
    for suffix in ('_aesthetic', '_scientific'):
        if not stem.endswith(suffix):
            continue
        settings = path.with_name(stem[:-len(suffix)] + '_settings.json')
        try:
            record = json.loads(settings.read_text()).get('filter_data', {}).get(path.name)
            if record:
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if digest == record['sha256']:
                    stat = path.stat()
                    saved = record['context']
                    return dict(coverage=None, count=1, coverage_bits=saved['coverage_bits'],
                                coverage_shape=saved['coverage_shape'], moon=saved['moon'], sun=saved['sun'],
                                margin=saved['margin'], source_identity=(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
        except (OSError, ValueError, KeyError, TypeError):
            pass
    try:
        metadata, _ = source_metadata(path)
        provenance = metadata.get('crop_source')
    except (OSError, ValueError, KeyError, TypeError):
        provenance = None
    if provenance:
        source = Path(provenance['path'])
        with source.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != provenance['sha256']:
                raise ValueError('The source master of this crop has changed')
        with source_image(source) as full, source_image(path) as cropped:
            x, y, width, height = validate_crop(tuple(provenance['xywh']), full.shape[:2])
            if cropped.shape[:2] != (height, width):
                raise ValueError('Cropped master dimensions do not match its provenance')
            for start in range(0, height, 128):
                if not np.array_equal(cropped[start:start+128], full[y+start:y+min(height, start+128), x:x+width], equal_nan=True):
                    raise ValueError('Cropped master samples do not match its source')
        context = job_filter_context(source, temporary, _seen)
        prior_x, prior_y = context.get('coverage_origin_xy', (0, 0))
        context['coverage_origin_xy'] = (prior_x+x, prior_y+y)
        for key in ('moon', 'sun'):
            cx, cy, radius = context[key]
            context[key] = (cx-x, cy-y, radius)
        stat = path.stat()
        context['source_identity'] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        return context
    candidates = []
    if path.name in ('master.tiff', 'master.fits') and (path.parent/'report.json').is_file():
        candidates.append(path.parent)
    if temporary:
        for job_file in (Path(temporary)/'.coronahdr-jobs').glob('*/job.json'):
            try:
                job = json.loads(job_file.read_text())
                expected = job.get('published_files', {}).get(path.name)
                if not expected or not Path(job['destination']).samefile(path.parent):
                    continue
                with path.open('rb') as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual == expected:
                    root = Path(job['result'])
                    candidates.append(root / 'aesthetic' if (root/'aesthetic'/'report.json').is_file() else root)
            except (OSError, ValueError, KeyError):
                continue
    for root in candidates:
        try:
            report = json.loads((root/'report.json').read_text())
            coverage = root/'coverage.fits'
            if not coverage.is_file():
                continue
            origin_y, origin_x = report['output']['origin_reference_yx']
            ref = report['reference']
            ref_disk = fitted_disk(report['inputs'][ref]['metadata'])
            if ref_disk is None:
                from .display_export import source_image
                from ._ael_filters.geometry import find_disc
                with source_image(path) as source:
                    step = max(1, int(np.ceil(max(source.shape[:2])/1200)))
                    disc = find_disc(np.asarray(source[::step, ::step]))
                mx, my = (disc['xy'][0]*step+origin_x, disc['xy'][1]*step+origin_y)
                mr = disc['r']*step
            else:
                mx, my, mr = ref_disk
            margin = 0.
            matrices = {row['index']: np.asarray(row['matrix_yx']) for row in report['transformations']}
            for index in report['accepted_indices']:
                disk = fitted_disk(report['inputs'][index]['metadata'])
                if disk is None or index not in matrices or ref_disk is None:
                    margin = None
                    break
                x, y, radius = disk
                transformed = apply_points(matrices[index], [[y, x]])[0]
                margin = max(margin, np.hypot(transformed[0]-my, transformed[1]-mx) + radius-mr)
            moon = (mx-origin_x, my-origin_y, mr)
            stat = path.stat()
            return dict(source_identity=(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns), coverage=str(coverage), count=len(report['accepted_indices']), moon=moon,
                        sun=moon, margin=float(max(0., margin)) if margin is not None else None)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
    raise ValueError('Matching job coverage and lunar measurements are unavailable. Keep the temporary job files to use corona filters.')


def preview_filter_context(path, temporary=None):
    """Use measured job geometry when available, otherwise estimate it for review."""
    try:
        return job_filter_context(path, temporary)
    except (OSError, ValueError):
        from pathlib import Path
        from .display_export import source_image
        from ._ael_filters.geometry import find_disc
        path = Path(path)
        with source_image(path) as image:
            step = max(1, int(np.ceil(max(image.shape[:2])/1200)))
            disc = find_disc(np.asarray(image[::step, ::step]))
        moon = (disc['xy'][0]*step, disc['xy'][1]*step, disc['r']*step)
        stat = path.stat()
        return dict(source_identity=(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns),
                    coverage=None, count=1, moon=moon, sun=moon, margin=None, approximate=True)


def image_filter_layers(image, context, max_side=None):
    """Calculate layers at native size, or a reduced size for responsive inspection."""
    import cv2
    from astropy.io import fits
    image = np.asarray(image)
    if context['count'] <= 0:
        raise ValueError('Coverage frame count must be positive')
    height, width = image.shape[:2]
    from contextlib import nullcontext
    source = fits.open(context['coverage'], memmap=True) if context.get('coverage') else nullcontext(None)
    with source as hdus:
        if 'coverage_bits' in context:
            import base64
            import zlib
            saved_height, saved_width = context['coverage_shape']
            ox, oy = context.get('coverage_origin_xy', (0, 0))
            if not (0 <= ox and 0 <= oy and height + oy <= saved_height and width + ox <= saved_width
                    and saved_height * saved_width <= 250_000_000):
                raise ValueError('Invalid saved coverage dimensions')
            expected = (saved_height * saved_width + 7)//8
            compressed = base64.b64decode(context['coverage_bits'], validate=True)
            decoder = zlib.decompressobj()
            packed = decoder.decompress(compressed, expected + 1)
            if len(packed) != expected or not decoder.eof or decoder.unused_data:
                raise ValueError('Saved coverage data does not match the image canvas')
            coverage = np.unpackbits(np.frombuffer(packed, dtype='u1'), count=saved_height*saved_width).reshape(saved_height, saved_width)
        else:
            coverage = np.asarray(hdus[0].data) if hdus is not None else np.ones((height, width), dtype='u1')
        if 'coverage_origin_xy' in context:
            ox, oy = context['coverage_origin_xy']
            coverage = coverage[oy:oy+height, ox:ox+width]
        if coverage.shape != (height, width):
            raise ValueError('Coverage map does not match the image canvas')
        factor = min(1., max_side/max(height, width)) if max_side else 1.
        if factor < 1:
            size = (max(1, round(width*factor)), max(1, round(height*factor)))
            # Nearest samples keep missing observations from being averaged into valid data.
            # OpenCV interprets FITS big-endian arrays as native bytes unless converted first.
            work = cv2.resize(np.asarray(image, dtype=np.float32), size, interpolation=cv2.INTER_NEAREST)
            native_coverage = coverage.astype(coverage.dtype.newbyteorder('='), copy=False)
            cov = cv2.resize(native_coverage, size, interpolation=cv2.INTER_NEAREST)
        else:
            work, cov = image, coverage
        geometry = []
        for disk in (context['moon'], context['sun']):
            geometry.append(((disk[0]+.5)*factor-.5, (disk[1]+.5)*factor-.5, disk[2]*factor))
        valid, layers = filter_layers(work, cov/context['count'], *geometry, context['margin']*factor)
        if factor < 1:
            valid = cv2.resize(valid.astype('u1'), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            y, x = np.ogrid[:height, :width]
            mx, my, radius = context['moon']
            valid &= (coverage >= .5*context['count']) & ((x-mx)**2+(y-my)**2 > (radius+context['margin'])**2)
            layers = [(cv2.resize(np.nan_to_num(layer, nan=.5), (width, height), interpolation=cv2.INTER_LINEAR), mode, opacity)
                      for layer, mode, opacity in layers]
        return valid, layers


def check_filter_source(path, context):
    """Reject a filter setup if its source master was replaced after review."""
    if context and context.get('source_identity'):
        stat = path.stat()
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != tuple(context['source_identity']):
            raise ValueError('HDR master changed after filter geometry was reviewed; enable filters again')

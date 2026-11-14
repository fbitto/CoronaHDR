"""Disk-backed detector preparation and tiled HDR; full-frame samples stay linear."""
from dataclasses import dataclass, field
import json
from pathlib import Path
import resource
import platform
import importlib.metadata
import numpy as np
from scipy.ndimage import map_coordinates
from .model import Frame
from .sensor import SensorFrame, phase_tile, PHASES
from .calibration import Master, build_master, subtract_detector, select_dark
from .prepare import prepare_light
from .io import write_image


@dataclass
class CachedFrame:
    data: np.ndarray
    metadata: dict
    path: Path
    auxiliary: dict = field(default_factory=dict)
    detector: bool = False

    @property
    def scale(self):
        return self.metadata.get('relative_scale', self.metadata['exposure_s'])


def bands(height, rows):
    start = 0
    while start < height:
        end = min(height, start+rows)
        if height-end < 32:
            end = height
        yield start, end
        start = end


def sliced_metadata(frame, start, end, x=0, width=None):
    m = dict(frame.metadata)
    width = frame.data.shape[1] if width is None else width
    ax, ay, *_ = m.get('active_area_xywh', [0, 0, frame.data.shape[1], frame.data.shape[0]])
    m['active_area_xywh'] = [ax+x, ay+start, width, end-start]
    if m.get('lunar_disk') is not None:
        cx, cy, radius = m['lunar_disk']
        m['lunar_disk'] = [cx-x, cy-start, radius]
    m['defective_pixels_xy'] = [[px-x, py-start] for px, py in m.get('defective_pixels_xy', [])
                               if x <= px < x+width and start <= py < end]
    for name in ('auto_defect_pixels_xy', 'repair_defects_xy'):
        if name in m:
            m[name] = [[px-x, py-start] for px, py in m[name] if x <= px < x+width and start <= py < end]
    if 'repair_defects_xyc' in m:
        m['repair_defects_xyc'] = [[px-x, py-start, c] for px, py, c in m['repair_defects_xyc']
                                   if x <= px < x+width and start <= py < end]
    return m


def slice_input(frame, start, end, x=0, width=None):
    width = frame.data.shape[1] if width is None else width
    data = frame.data[start:end, x:x+width]
    metadata = sliced_metadata(frame, start, end, x, width)
    if isinstance(frame, Master):
        return Master(data, metadata, frame.provenance)
    if frame.detector:
        return SensorFrame(data, metadata, frame.path)
    auxiliary = {key: value[start:end, x:x+width] for key, value in frame.auxiliary.items()}
    return Frame(data, metadata, frame.path, auxiliary)


def mapped(path, shape, dtype=np.float64):
    return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)


def cache_input(frame, directory):
    directory.mkdir()
    # Preserve the decoded source dtype exactly; calibration still computes float64.
    dtype = np.dtype(frame.metadata.get('source_dtype', frame.data.dtype))
    data = mapped(directory/'source.npy', frame.data.shape, dtype)
    data[:] = frame.data
    data.flush()
    return CachedFrame(data, frame.metadata, frame.path, detector=isinstance(frame, SensorFrame))


def tiled_master(inputs, role, directory, rows):
    if not inputs:
        raise ValueError(f'No {role} calibration inputs')
    data = mapped(directory/f'{role}.npy', inputs[0].data.shape)
    first = None
    missing = 0
    for start, end in bands(data.shape[0], rows):
        tile = build_master([slice_input(frame, start, end) for frame in inputs], role)
        if first is None:
            first = tile
        data[start:end] = tile.data
        for source in inputs:
            release_pages(source)
        missing += int((~np.isfinite(tile.data)).sum())
    metadata = dict(first.metadata, active_area_xywh=inputs[0].metadata['active_area_xywh'],
                    master_missing_samples=missing)
    data.flush()
    return Master(data, metadata, [dict(path=str(f.path), metadata=f.metadata) for f in inputs])


def tiled_flat(flat, bias, dark, directory, rows):
    corrected = mapped(directory/'corrected_flat.npy', flat.data.shape)
    steps = None
    for start, end in bands(flat.data.shape[0], rows):
        corrected[start:end], steps = subtract_detector(slice_input(flat, start, end),
            slice_input(bias, start, end) if bias else None,
            slice_input(dark, start, end) if dark else None)
    factors = {}
    state = flat.metadata['state']
    if state == 'linear_cfa':
        tile = phase_tile(flat.metadata['cfa_pattern'], flat.metadata['active_area_xywh'][:2])
        groups = [(phase, (slice(int(np.argwhere(tile == phase)[0, 0]), None, 2),
                          slice(int(np.argwhere(tile == phase)[0, 1]), None, 2))) for phase in PHASES]
    elif state == 'linear_rgb':
        groups = [(name, (slice(None), slice(None), i)) for i, name in enumerate(('R', 'G', 'B'))]
    else:
        groups = [('mono', (slice(None), slice(None)))]
    normalized = mapped(directory/'normalized_flat.npy', flat.data.shape)
    for name, selection in groups:
        values = corrected[selection]
        valid = np.isfinite(values) & (values > 0)
        if valid.sum() < 16:
            raise ValueError(f'Insufficient positive corrected flat samples for {name}')
        factors[name] = float(np.median(values[valid]))
        normalized[selection] = np.where(valid, values/factors[name], np.nan)
    normalized.flush()
    return Master(normalized, dict(flat.metadata, flat_normalization=factors,
        flat_normalization_method='one median per detector phase/channel after pedestal/dark-flat correction; no light normalization',
        flat_corrections=steps), flat.provenance)


def cache_prepared(frame, bias, dark, flat, directory, rows):
    directory.mkdir()
    shape = (*frame.data.shape[:2], 3) if frame.metadata['state'] == 'linear_cfa' else frame.data.shape
    data = mapped(directory/'data.npy', shape)
    auxiliaries = {key: mapped(directory/f'{key}.npy', shape[:2], np.float64 if key == 'soft_weight' else bool)
                   for key in ('coverage', 'lunar', 'saturation', 'low_signal', 'defective', 'soft_weight')}
    if frame.metadata.get('auto_defect_pixels_xy'):
        auxiliaries.update({key: mapped(directory/f'{key}.npy', shape[:2], bool)
            for key in ('detected_defects', 'repaired', 'repair_footprint')})
    first_metadata = None
    for start, end in bands(shape[0], rows):
        halo = 6 if frame.metadata.get('repair_defects_xy') else 2
        lo, hi = max(0, start-halo), min(shape[0], end+halo)
        tile = slice_input(frame, lo, hi)
        if frame.detector:
            prepared = prepare_light(tile, *[slice_input(master, lo, hi) if master else None for master in (bias, dark, flat)])
        else:
            prepared = tile
        if first_metadata is None:
            first_metadata = prepared.metadata
        section = slice(start-lo, end-lo)
        data[start:end] = prepared.data[section]
        masks = prepared.masks()
        maximum = prepared.data.max(axis=-1) if prepared.data.ndim == 3 else prepared.data
        high = np.clip((prepared.metadata['white_level']-maximum)/(.2*prepared.metadata['white_level']), 0, 1)
        for key, target in auxiliaries.items():
            if key == 'soft_weight':
                low = np.clip((prepared.mono-prepared.metadata['low_signal'])/(.05*(prepared.metadata['white_level']-prepared.metadata['low_signal'])), 0, 1)
                source = prepared.auxiliary.get('high_weight', high*high*(3-2*high))*(low*low*(3-2*low))
                source = np.nan_to_num(source)
            else:
                source = masks[key] if key in masks else prepared.auxiliary.get(key)
                if source is None:
                    source = np.zeros(prepared.data.shape[:2], bool)
            target[start:end] = source[section]
    metadata = dict(first_metadata, active_area_xywh=frame.metadata.get('active_area_xywh', [0, 0, shape[1], shape[0]]),
                    lunar_disk=frame.metadata['lunar_disk'], defective_pixels_xy=[])
    if frame.detector:
        metadata['source_detector_metadata'] = frame.metadata
        metadata['calibration_masters'] = {name: m.metadata for name, m in [('bias', bias), ('dark', dark), ('flat', flat)] if m}
    for name in ('auto_defect_pixels_xy', 'repair_defects_xy', 'repair_defects_xyc',
                 'repair_defects_prequalified', 'defect_policy'):
        if name in frame.metadata:
            metadata[name] = frame.metadata[name]
    if 'repaired' in auxiliaries:
        metadata['repaired_defects_xy'] = np.argwhere(auxiliaries['repaired'])[:, ::-1].tolist()
        metadata['repaired_native_count'] = int(auxiliaries['repaired'].sum())
    for array in [data, *auxiliaries.values()]:
        array.flush()
    (directory/'metadata.json').write_text(json.dumps(metadata, indent=2))
    return CachedFrame(data, metadata, frame.path, auxiliaries)


def registration_auxiliary(frame, factor, roi=None):
    height, width = frame.data.shape[:2]
    x, y, width, height = roi or [0, 0, width, height]
    if min(x, y) < 0 or min(width, height) < 32*factor or x+width > frame.data.shape[1] or y+height > frame.data.shape[0]:
        raise ValueError('Registration ROI must fit every frame and contain at least 32x32 auxiliary pixels')
    oh, ow = height//factor, width//factor
    image = np.zeros((oh, ow))
    valid = np.zeros((oh, ow), bool)
    for row in range(0, oh, 32):
        count = min(32, oh-row)
        source_row = y+row*factor
        size = max(count*factor, 32)
        lo = min(source_row, frame.data.shape[0]-size)
        tile = slice_input(frame, lo, lo+size, x, width)
        section = slice(source_row-lo, source_row-lo+count*factor)
        masks = tile.masks()
        good = masks['coverage'] & ~(masks['lunar'] | masks['saturation'] | masks['low_signal'] | masks['defective'])
        mono = tile.mono[section, :ow*factor]
        image[row:row+count] = np.nan_to_num(mono).reshape(count, factor, ow, factor).mean(axis=(1, 3))
        valid[row:row+count] = good[section, :ow*factor].reshape(count, factor, ow, factor).all(axis=(1, 3))
    metadata = dict(frame.metadata, state='linear_mono', lunar_disk=None, low_signal=-1e300,
                    white_level=1e300, active_area_xywh=[0, 0, ow, oh], defective_pixels_xy=[])
    return Frame(image, metadata, frame.path, dict(coverage=valid, defective=~valid,
                 lunar=np.zeros(valid.shape, bool), saturation=np.zeros(valid.shape, bool), low_signal=np.zeros(valid.shape, bool)))


def output_canvas(frames, shifts, policy):
    if policy not in ('reference', 'union', 'intersection'):
        raise ValueError('canvas must be explicitly reference, union or intersection')
    if policy == 'reference':
        return (0, 0), frames[0].data.shape[:2]
    if policy == 'intersection':
        if any(np.asarray(s).shape != (2,) for s in shifts):
            raise ValueError('Intersection canvas requires translation-only geometry')
        low = np.ceil(np.max(shifts, axis=0)).astype(int)
        high = np.floor(np.min([np.array(f.data.shape[:2])-1+s for f, s in zip(frames, shifts)], axis=0)).astype(int)+1
        if np.any(high <= low):
            raise ValueError('No shared rectangular field between the accepted frames')
        return tuple(int(v) for v in low), tuple(int(v) for v in high-low)
    if any(np.asarray(s).shape == (3,3) for s in shifts):
        from .rigid import apply_points
        bounds=[]
        for f,a in zip(frames,shifts):
            h,w=f.data.shape[:2]
            corners=np.array([[-.5,-.5],[-.5,w-.5],[h-.5,-.5],[h-.5,w-.5]])
            bounds.extend(apply_points(np.asarray(a),corners))
        low=np.floor(np.min(bounds,axis=0)+.5).astype(int)
        high=np.ceil(np.max(bounds,axis=0)+.5).astype(int)
        return tuple(int(v) for v in low),tuple(int(v) for v in high-low)
    low = np.floor(np.min(shifts, axis=0)).astype(int)
    high = np.ceil(np.max([np.array(f.data.shape[:2])+s for f, s in zip(frames, shifts)], axis=0)).astype(int)
    return tuple(int(v) for v in low), tuple(int(v) for v in high-low)


def sample_band(array, delta, start, end, width, order=1):
    if np.asarray(delta).shape == (3,3):
        inverse=np.linalg.inv(delta)
        y,x=np.meshgrid(np.arange(start,end),np.arange(width),indexing='ij')
        yy=inverse[0,0]*y+inverse[0,1]*x+inverse[0,2]
        xx=inverse[1,0]*y+inverse[1,1]*x+inverse[1,2]
    else:
        yy, xx = np.meshgrid(np.arange(start, end)-delta[0], np.arange(width)-delta[1], indexing='ij')
    y0, x0 = max(0, int(np.floor(yy.min()))-1), max(0, int(np.floor(xx.min()))-1)
    y1, x1 = min(array.shape[0], int(np.ceil(yy.max()))+2), min(array.shape[1], int(np.ceil(xx.max()))+2)
    shape = (end-start, width, *array.shape[2:])
    if y0 >= y1 or x0 >= x1:
        return np.zeros(shape)
    block = np.nan_to_num(np.asarray(array[y0:y1, x0:x1], dtype=np.float64))
    coordinates = [yy-y0, xx-x0]
    if array.ndim == 2:
        return map_coordinates(block, coordinates, output=np.float64, order=order, mode='constant', cval=0, prefilter=False)
    return np.stack([map_coordinates(block[..., c], coordinates, output=np.float64, order=order,
                     mode='constant', cval=0, prefilter=False) for c in range(array.shape[-1])], axis=-1)


def export_map(path, array, metadata, dtype='float32'):
    """Use FITS uint8 for masks and float32 for scientific samples/weights."""
    if dtype == 'float32':
        write_image(path, array, metadata)
    else:
        from astropy.io import fits
        header = fits.Header()
        header['CHDRMETA'] = json.dumps(metadata)
        header['BUNIT'] = metadata.get('unit', '1')
        fits.PrimaryHDU(np.asarray(array, dtype=dtype), header).writeto(path)


def accumulate_tiled(frames, shifts, origin, shape, output, work, rows, radiance_correction=None):
    channels = frames[0].data.shape[2:] if frames[0].data.ndim == 3 else ()
    master = mapped(work/'master.npy', (*shape, *channels), np.float32)
    names = ('coverage', 'lunar', 'saturation', 'low_signal', 'defective', 'contributions', 'support', 'unresolved_saturation', 'weight')
    maps = {key: mapped(work/f'aggregate_{key}.npy', shape, np.float64 if key == 'weight' else np.uint32) for key in names}
    for start, end in bands(shape[0], rows):
        numerator = np.zeros((end-start, shape[1], *channels))
        counts = {key: np.zeros((end-start, shape[1]), np.uint32) for key in names if key != 'weight'}
        weight = np.zeros((end-start, shape[1]))
        eligible = np.zeros_like(weight, np.uint32)
        unsaturated = np.zeros_like(eligible)
        for index, (frame, displacement) in enumerate(zip(frames, shifts)):
            if np.asarray(displacement).shape == (3,3):
                delta=np.array(displacement,copy=True);delta[:2,2]-=origin
            else:
                delta = np.asarray(displacement)-origin
            masks = {}
            for key in ('coverage', 'lunar', 'saturation', 'low_signal', 'defective'):
                sampled = sample_band(frame.auxiliary[key], delta, start, end, shape[1])
                masks[key] = sampled >= 1-1e-9 if key == 'coverage' else sampled > 1e-9
                counts[key] += masks[key]
            valid = masks['coverage'] & ~(masks['lunar'] | masks['saturation'] | masks['low_signal'] | masks['defective'])
            # Sample precomputed source reliability; never derive saturation from a transformed RGB value.
            w = sample_band(frame.auxiliary['soft_weight'], delta, start, end, shape[1])
            w = np.where(valid, w, 0)
            values = sample_band(frame.data, delta, start, end, shape[1])/frame.scale
            if radiance_correction is not None:
                values = radiance_correction(index, values, start, end)
                if values.shape != (end-start, shape[1], *channels) or not np.isfinite(values).all():
                    raise ValueError('Radiance correction returned invalid samples or shape')
            values = np.nan_to_num(values)
            numerator += values*w[..., None] if channels else values*w
            weight += w
            counts['contributions'] += w > 0
            observed = masks['coverage'] & ~masks['lunar'] & ~masks['defective']
            eligible += observed
            unsaturated += observed & ~masks['saturation']
            directory = work/f'aligned_{index:03d}'
            directory.mkdir(exist_ok=True)
            for key, array in {**masks, 'weight': w}.items():
                path = directory/f'{key}.npy'
                target = (np.lib.format.open_memmap(path, mode='r+') if path.exists()
                          else mapped(path, shape, np.float32 if key == 'weight' else np.uint8))
                target[start:end] = array
                target.flush()
                del target
            release_pages(frame)
        result = np.full(numerator.shape, np.nan)
        np.divide(numerator, weight[..., None] if channels else weight, out=result,
                  where=(weight > 0)[..., None] if channels else weight > 0)
        master[start:end] = result
        counts['support'] = weight > 0
        counts['unresolved_saturation'] = (eligible > 0) & (unsaturated == 0)
        for key in maps:
            maps[key][start:end] = weight if key == 'weight' else counts[key]
        print(f'HDR rows {start}:{end} of {shape[0]}', flush=True)
    for key, data in maps.items():
        dtype = 'float32' if key == 'weight' else 'uint8' if len(frames) <= 255 else 'int32'
        export_map(output/f'{key}.fits', data, dict(meaning=key, unit='1'), dtype)
        data.flush();data._mmap.close()
        (work/f'aggregate_{key}.npy').unlink()
    for index in range(len(frames)):
        directory = output/f'frame_{index:03d}'
        directory.mkdir()
        for key in ('coverage', 'lunar', 'saturation', 'low_signal', 'defective', 'weight'):
            data = np.load(work/f'aligned_{index:03d}'/f'{key}.npy', mmap_mode='r')
            export_map(directory/f'{key}.fits', data, dict(meaning=key, unit='1'),
                       'float32' if key == 'weight' else 'uint8')
            data._mmap.close()
            (work/f'aligned_{index:03d}'/f'{key}.npy').unlink()
    return master


def release_pages(frame):
    """Reopen clean memmaps so the process does not retain every visited image page."""
    def reopen(array):
        if not isinstance(array, np.memmap):
            return array
        path = array.filename
        array.flush()
        array._mmap.close()
        return np.load(path, mmap_mode='r')
    frame.data = reopen(frame.data)
    if isinstance(frame, CachedFrame):
        frame.auxiliary = {key: reopen(value) for key, value in frame.auxiliary.items()}


def prepare_disk_inputs(recipe, base, work, rows):
    from .pipeline import load_item
    calibration = recipe.get('calibration', {})
    if set(calibration)-{'bias', 'dark', 'flat', 'dark_flat'}:
        raise ValueError('Unknown calibration role')
    masters = {}
    for role, entries in calibration.items():
        inputs = []
        for index, item in enumerate(entries):
            source = load_item(item, base)
            if not isinstance(source, SensorFrame):
                raise ValueError(f'Calibration input needs explicit role={role} detector metadata')
            inputs.append(cache_input(source, work/f'input_{role}_{index:03d}'))
            del source
            release_pages(inputs[-1])
        groups = {}
        for frame in inputs:
            key = (frame.metadata['exposure_s'], frame.metadata.get('temperature_c'), frame.metadata.get('dark_contains_bias')) if role in ('dark', 'dark_flat') else None
            groups.setdefault(key, []).append(frame)
        if not groups:
            raise ValueError(f'Empty {role} calibration list')
        masters[role] = []
        for index, group in enumerate(groups.values()):
            folder = work/f'master_{role}_{index:03d}'
            folder.mkdir()
            masters[role].append(tiled_master(group, role, folder, rows))
            release_pages(masters[role][-1])
        print(f'Built {len(masters[role])} {role} master(s)', flush=True)
    bias = masters.get('bias', [None])[0]
    flat = masters.get('flat', [None])[0]
    if masters.get('dark_flat') and flat is None:
        raise ValueError('dark_flat supplied without flat')
    if flat:
        flat = tiled_flat(flat, bias, select_dark(flat, masters.get('dark_flat', [])), work, rows)
        masters['flat'] = [flat]
        release_pages(flat)
    frames = []
    original_inputs = []
    for index, item in enumerate(recipe['frames']):
        source = load_item(item, base)
        if calibration and not isinstance(source, SensorFrame):
            source = SensorFrame(source.data, dict(source.metadata, role='light'), source.path)
        original_inputs.append(dict(path=str(source.path), metadata=source.metadata))
        cached = cache_input(source, work/f'input_light_{index:03d}')
        del source
        if cached.detector:
            dark = select_dark(cached, masters.get('dark', []))
        else:
            dark = None
        prepared = cache_prepared(cached, bias, dark, flat, work/f'prepared_{index:03d}', rows)
        frames.append(prepared)
        release_pages(prepared)
        release_pages(cached)
        for group in masters.values():
            for master in group:
                release_pages(master)
        print(f'Prepared full field {index+1}/{len(recipe["frames"])}: {prepared.path.name}', flush=True)
    return frames, original_inputs, masters


def run_disk_manifest(recipe, base, output):
    from .registration_graph import solve_translations, qualify_radiometry
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    options = recipe['processing']
    from .lunar import apply_margin, validate_margin
    validate_margin(options.get('lunar_margin_px'))
    rows = options.get('tile_rows', 128)
    factor = options.get('registration_bin', 4)
    if not isinstance(rows, int) or rows < 32 or not isinstance(factor, int) or factor < 1:
        raise ValueError('tile_rows must be >=32 and registration_bin a positive integer')
    policy = options.get('canvas')
    if policy not in ('reference', 'union', 'intersection'):
        raise ValueError('Disk processing requires an explicit canvas=reference, union or intersection')
    rejection = options.get('rejection_policy', 'error')
    if rejection not in ('error', 'exclude'):
        raise ValueError('rejection_policy must be error or exclude')
    if not recipe.get('frames'):
        raise ValueError('No light frames')
    if 'reference' in recipe and (not isinstance(recipe['reference'], int)
            or isinstance(recipe['reference'], bool) or not 0 <= recipe['reference'] < len(recipe['frames'])):
        raise ValueError('reference must be a valid zero-based frame index')
    output.mkdir(parents=True)
    work = Path(options.get('work_directory', output/'work')).resolve()
    work.mkdir(parents=True)
    preparation_directory = work.resolve()
    if options.get('cache_from'):
        previous_report = json.loads((Path(options['cache_from'])/'report.json').read_text())
        if options.get('lunar_margin_px') is None and 'lunar_margin_px' in previous_report.get('output', {}):
            options['lunar_margin_px'] = previous_report['output']['lunar_margin_px']
            validate_margin(options['lunar_margin_px'])
        preparation_directory = Path(previous_report.get('preparation_directory', Path(options['cache_from'])/'work')).resolve()
        frames, original_inputs, masters = load_cached_run(Path(options['cache_from']), recipe)
    else:
        frames, original_inputs, masters = prepare_disk_inputs(recipe, base, work, rows)
    checkpoint = output/'preparation'
    checkpoint.mkdir()
    checkpoint_report = dict(recipe=recipe, inputs=original_inputs,
        preparation_directory=str(preparation_directory),
        calibration={role: [dict(metadata=m.metadata, sources=m.provenance) for m in group]
                     for role, group in masters.items()})
    if options.get('cache_from'):
        for key in ('prepared_overlays', 'lunar_margin_masks'):
            if key in previous_report:
                checkpoint_report[key] = previous_report[key]
    checkpoint_file = checkpoint/'report.new'
    checkpoint_file.write_text(json.dumps(checkpoint_report, indent=2, allow_nan=False))
    checkpoint_file.replace(checkpoint/'report.json')
    (output/'recipe.json').write_text(json.dumps(recipe, indent=2))
    frames, lunar_masks = apply_margin(frames, options.get('lunar_margin_px'), work)
    if not frames:
        raise ValueError('No light frames')
    reference = recipe.get('reference', max(range(len(frames)), key=lambda i: frames[i].scale))
    ref = frames[reference]
    for frame in frames:
        if frame.data.shape != ref.data.shape or frame.metadata['state'] != ref.metadata['state']:
            raise ValueError('Disk processing currently requires matching source shapes/channel states')
        for key in ('response_group', 'iso', 'gain', 'aperture'):
            if frame.metadata.get(key) != ref.metadata.get(key):
                raise ValueError(f'Incompatible acquisition response: {key}')
    requested = options.get('include_indices', list(range(len(frames))))
    if (not requested or any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(frames) for i in requested)
            or len(set(requested)) != len(requested) or reference not in requested):
        raise ValueError('include_indices must be unique valid input indices and contain the reference')
    auxiliaries = [None]*len(frames)
    for index in requested:
        auxiliaries[index] = registration_auxiliary(frames[index], factor, options.get('registration_roi_xywh'))
        release_pages(frames[index])
    rigid_mode=bool(options.get('rigid_review'))
    if rigid_mode:
        from .rigid import reviewed_rigid
        solved,metrics,rejected,attempts,rigid_audit=reviewed_rigid(options['rigid_review'],original_inputs,reference,requested)
        if rigid_audit['factor']!=factor or rigid_audit['roi_xywh']!=options.get('registration_roi_xywh'):
            raise ValueError('Rigid review auxiliary coordinate system mismatch')
    elif options.get('registration_review'):
        from .recovery import reviewed_translations
        solved, metrics, rejected, attempts = reviewed_translations(options['registration_review'], original_inputs, reference, requested)
    else:
        local_reference = requested.index(reference)
        local_solved, local_metrics, local_rejected, attempts = solve_translations([auxiliaries[i] for i in requested], local_reference, factor,
            recipe.get('max_shift', 24), options.get('bridges', True), options.get('cycle_tolerance_px', .5), options.get('geometry_photometry', True))
        from .decisions import ask
        if any('exceeds max_shift' in item['reason'] for item in local_rejected):
            wider = max(recipe.get('max_shift', 24), max(auxiliaries[reference].data.shape[:2])*factor)
            choice = ask('Some frames exceed the shift limit. Search across the registration field? '
                'Alignment quality checks remain enforced; the shared field may be smaller.',
                ['Search larger shifts', 'Keep current registrations'], 'Keep current registrations')
            if choice == 'Search larger shifts':
                local_solved, local_metrics, local_rejected, attempts = solve_translations(
                    [auxiliaries[i] for i in requested], local_reference, factor, wider,
                    options.get('bridges', True), options.get('cycle_tolerance_px', .5), options.get('geometry_photometry', True))
                recipe['max_shift'] = wider
                (output/'recipe.json').write_text(json.dumps(recipe, indent=2))
        solved = {requested[i]:shift for i,shift in local_solved.items()}
        metrics = {}
        for index, value in local_metrics.items():
            value = dict(value, path=[requested[i] for i in value['path']])
            if 'parent' in value:
                value['parent'] = requested[value['parent']]
            metrics[requested[index]] = value
        rejected = [dict(item, index=requested[item['index']]) for item in local_rejected]
        for edge in attempts:
            edge['parent'], edge['child'] = requested[edge['parent']], requested[edge['child']]
    geometrically_solved = dict(solved)
    if rigid_mode:
        from .rigid import rigid_radiometry,lower_matrix
        reference_checks={};photometric_rejections=[];qualified={reference:solved[reference]}
        for i,a in solved.items():
            if i==reference:continue
            check=rigid_radiometry(auxiliaries[reference],auxiliaries[i],lower_matrix(a,factor,options['registration_roi_xywh']))
            reference_checks[i]=check
            if check.get('samples',0)>=64 and check.get('raw_relative_error',1)<=.15 and max(check.get('raw_regional_relative_errors',[1]),default=1)<=.2:
                qualified[i]=a
            else:photometric_rejections.append(dict(index=i,reason='Rigid reference-level radiometric disagreement',diagnostic=check))
        solved=qualified
    else:
        solved, photometric_rejections, reference_checks = qualify_radiometry(auxiliaries, solved, metrics, reference, factor, attempts)
    rejected.extend(photometric_rejections)
    for item in rejected:
        print(f'WARNING: Frame {item["index"]} rejected: {item["reason"]}', flush=True)
    registration = dict(reference=reference, transformations={str(i): value for i, value in metrics.items() if i in solved},
        rejected_frames=rejected, edges=attempts, auxiliary_bin=factor,
        roi_xywh=options.get('registration_roi_xywh'), master_normalization_applied=False,
        geometry_photometry=options.get('geometry_photometry', True), reference_radiometry=reference_checks,
        geometrically_registered_indices=sorted(geometrically_solved), accepted_indices=sorted(solved))
    (output/'registration.json').write_text(json.dumps(registration, indent=2, allow_nan=False))
    if rejected and rejection == 'error':
        raise ValueError(f'{len(rejected)} frames lack reliable registration; inspect {output}/registration.json')
    accepted = sorted(solved)
    if len(accepted) < 2 and len(requested) > 1:
        raise ValueError('Fewer than two reliably registered frames; no HDR master exported')
    from .decisions import ask
    if rejected and rejection == 'exclude':
        ask(f'{len(rejected)} frames failed registration or radiometric checks. '
            f'Continue with {len(accepted)} accepted frames? Details are in registration.json. '
            'Cancel to stop and retain preparation files.', ['Continue with accepted frames'], 'Continue with accepted frames')
    selected = [frames[i] for i in accepted]
    shifts = [solved[i] for i in accepted]
    if all(np.asarray(s).shape == (2,) for s in shifts) and any(np.any(s) for s in shifts):
        try:
            _, crop_shape = output_canvas(selected, shifts, 'intersection')
        except ValueError:
            pass
        else:
            stage = options.get('decision_stage', 'Weighted HDR')
            choice = ask(f'{stage}: shared detector footprint is {crop_shape[1]} × {crop_shape[0]} pixels. '
                'Crop to this smaller field or keep the current canvas? Lunar and saturation gaps can remain.',
                ['Crop to shared field', 'Keep current canvas'], 'Keep current canvas')
            if choice == 'Crop to shared field':
                policy = options['canvas'] = 'intersection'
                (output/'recipe.json').write_text(json.dumps(recipe, indent=2))
    origin, shape = output_canvas(selected, shifts, policy)
    print(f'Accepted {len(accepted)}/{len(frames)} frames; {policy} canvas {shape}, origin {origin}', flush=True)
    from .performance import stage
    with stage('weighted integration and mask export'):
        master = accumulate_tiled(selected, shifts, origin, shape, output, work, rows)
    metadata = dict(state=ref.metadata['state'], scale='relative camera response per second; no normalization',
        missing='NaN; support=0', channel_order='RGB' if master.ndim == 3 else 'mono',
        canvas=policy, origin_reference_yx=list(origin), shape_yx=list(shape))
    with stage('weighted master export'):
        for extension in ('fits', 'tiff'):
            write_image(output/f'master.{extension}', master, metadata)
    cal_folder = output/'calibration'
    cal_folder.mkdir()
    for role, group in masters.items():
        for index, calibration_master in enumerate(group):
            write_image(cal_folder/f'{role}_{index:03d}.fits', calibration_master.data,
                        dict(calibration_master.metadata, unit='1' if role == 'flat' else 'detector response'))
    report = dict(preparation_directory=str(preparation_directory), recipe=recipe, inputs=original_inputs, reference=reference, requested_indices=requested, not_requested_indices=sorted(set(range(len(frames)))-set(requested)), accepted_indices=accepted,
        rejected_frames=rejected, transformations=[dict(index=i, **metrics[i]) for i in accepted],
        registration=registration, output=metadata, python=platform.python_version(),
        versions={name: importlib.metadata.version(name) for name in
                  ('coronahdr', 'numpy', 'scipy', 'scikit-image', 'astropy', 'tifffile', 'rawpy')},
        calibration={role: [dict(metadata=m.metadata, sources=m.provenance) for m in group] for role, group in masters.items()},
        corrections=[dict(index=i, steps=f.metadata.get('calibration_steps', ['input already prepared']),
                          demosaic=f.metadata.get('demosaic')) for i, f in enumerate(frames)],
        memory=dict(tile_rows=rows, cache_dtype='float64 prepared data and weights; original dtype detector cache',
            process_peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            rss_units='bytes on macOS; KiB on Linux', work_directory=str(work), full_rgb_ndarray_allocations=0),
        conventions=('source-to-reference homogeneous [y,x,1] matrices; ' if rigid_mode else 'source-to-reference [dy,dx]; ')+
            'output coordinate = reference coordinate - canvas origin; integer pixel centers; one geometric interpolation; no wrap',
        warnings=['Relative radiance only; no fitted diagnostic slope or offset was applied',
                  'RAW validation remains camera/dataset specific',
                  'Registration measurements use auxiliary debayerized images; auxiliary binning limits measured precision',
                  'Frames without reliable paths are explicitly excluded only under rejection_policy=exclude',
                  'Single-path transformations lack an independent cycle check; see per-frame consistency status',
                  'No thermal dark correction unless explicitly supplied; bias is pedestal correction only'])
    if options.get('lunar_margin_px') is not None:
        report['output']['lunar_margin_px'] = options['lunar_margin_px']
        report['lunar_margin_masks'] = lunar_masks
    if options.get('cache_from') and 'prepared_overlays' in previous_report:
        report['prepared_overlays'] = previous_report['prepared_overlays']
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    cleanup_scratch(work)
    return report


def load_cached_run(previous, recipe):
    """Reuse only a completed matching preparation; recheck every original hash."""
    import hashlib
    report = json.loads((previous/'report.json').read_text())
    for key in ('frames', 'calibration'):
        if recipe.get(key) != report['recipe'].get(key):
            raise ValueError(f'Cached preparation {key} recipe mismatch')
    sources = {item['path']: item['metadata'] for item in report['inputs']}
    for group in report['calibration'].values():
        for master in group:
            sources.update({item['path']: item['metadata'] for item in master['sources']})
    for name, metadata in sources.items():
        with Path(name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != metadata['source_sha256']:
                raise ValueError(f'Original hash changed; cached preparation is invalid: {name}')
    preparation = Path(report.get('preparation_directory', previous/'work'))
    frames = []
    for index, item in enumerate(report['inputs']):
        directory = Path(report.get('prepared_overlays', {}).get(str(index), preparation/f'prepared_{index:03d}'))
        metadata = json.loads((directory/'metadata.json').read_text())
        data = np.load(directory/'data.npy', mmap_mode='r')
        aux = {key: np.load(directory/f'{key}.npy', mmap_mode='r') for key in
               ('coverage', 'lunar', 'saturation', 'low_signal', 'defective', 'soft_weight')}
        if str(index) in report.get('lunar_margin_masks', {}):
            aux['lunar'] = np.load(report['lunar_margin_masks'][str(index)], mmap_mode='r')
        aux.update({key: np.load(directory/f'{key}.npy', mmap_mode='r') for key in
                    ('detected_defects', 'repaired', 'repair_footprint') if (directory/f'{key}.npy').exists()})
        if any(array.shape != data.shape[:2] for array in aux.values()) or data.dtype != np.float64:
            raise ValueError('Cached prepared shape/dtype mismatch')
        frames.append(CachedFrame(data, metadata, Path(item['path']), aux))
    masters = {}
    for role, group in report['calibration'].items():
        masters[role] = []
        for index, item in enumerate(group):
            path = preparation/('normalized_flat.npy' if role == 'flat' else f'master_{role}_{index:03d}/{role}.npy')
            masters[role].append(Master(np.load(path, mmap_mode='r'), item['metadata'], item['sources']))
    print(f'Reused matching preparation; verified {len(sources)} original hashes', flush=True)
    return frames, report['inputs'], masters


def cleanup_scratch(work):
    """Remove duplicate generated accumulation arrays; preserve reusable preparation."""
    import re
    for directory in work.iterdir():
        if directory.is_symlink() or not directory.is_dir():
            continue
        if re.fullmatch(r'aligned_[0-9]+', directory.name):
            names = ('coverage', 'lunar', 'saturation', 'low_signal', 'defective', 'weight')
        elif re.fullmatch(r'input_(?:light|bias|flat|dark|dark_flat)_[0-9]+', directory.name):
            names = ('source',)
        else:
            continue
        for name in names:
            (directory/f'{name}.npy').unlink(missing_ok=True)
        try:
            directory.rmdir()
        except OSError:
            pass
    for name in ('coverage', 'lunar', 'saturation', 'low_signal', 'defective', 'contributions', 'support', 'unresolved_saturation', 'weight'):
        (work/f'aggregate_{name}.npy').unlink(missing_ok=True)
    (work/'master.npy').unlink(missing_ok=True)

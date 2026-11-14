"""Optional linear stage TIFFs, written in bands with explicit validity masks."""
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile

from .calibration import calibrate, sample_defective, select_dark
from .pipeline import load_item
from .sensor import SensorFrame, level_map
from .streaming import CachedFrame, bands, load_cached_run, sample_band, slice_input


def write_stage(directory, shape, metadata, chunks):
    directory = Path(directory)
    directory.mkdir(parents=True)
    image = tifffile.memmap(directory/'image.tiff', shape=shape, dtype='float32',
        photometric='rgb' if len(shape) == 3 else 'minisblack', metadata={'coronahdr': metadata})
    masks = {}
    for start, end, values, validity in chunks:
        image[start:end] = values
        for name, array in validity.items():
            if name not in masks:
                masks[name] = tifffile.memmap(directory/f'{name}.tiff', shape=shape[:2],
                    dtype='float32' if name == 'soft_weight' else 'uint8', photometric='minisblack')
            masks[name][start:end] = array
    for array in [image, *masks.values()]:
        array.flush()
    (directory/'metadata.json').write_text(json.dumps(dict(metadata, mask_files={name: f'{name}.tiff' for name in masks}), indent=2)+'\n')


def calibrated_chunks(source, masters, rows):
    bias = masters.get('bias', [None])[0]
    dark = select_dark(source, masters.get('dark', []))
    flat = masters.get('flat', [None])[0]
    for start, end in bands(source.data.shape[0], rows):
        halo = 4 if source.metadata.get('repair_defects_xy') else 0
        lo, hi = max(0, start-halo), min(source.data.shape[0], end+halo)
        tile = slice_input(source, lo, hi)
        values, head = calibrate(tile, *[slice_input(master, lo, hi) if master else None for master in (bias, dark, flat)])
        finite = np.isfinite(values)
        defective = sample_defective(tile) | ~finite
        saturation = tile.data >= level_map(tile, 'white_level')
        if values.ndim == 3:
            defective, saturation = defective.any(axis=-1), saturation.any(axis=-1)
        y, x = np.indices(values.shape[:2])
        lunar = np.zeros(values.shape[:2], dtype=bool)
        if tile.metadata.get('lunar_disk') is not None:
            cx, cy, radius = tile.metadata['lunar_disk']
            lunar = (x-cx)**2+(y-cy)**2 <= (radius+tile.metadata.get('limb_margin_px', 4))**2
        mono = values.mean(axis=-1) if values.ndim == 3 else values
        masks = dict(coverage=np.ones(values.shape[:2], bool),
            lunar=lunar, saturation=saturation, defective=defective,
            low_signal=mono <= tile.metadata['low_signal'])
        if tile.metadata.get('auto_defect_pixels_xy'):
            for name, points in [('detected_defects', tile.metadata['auto_defect_pixels_xy']), ('repaired', head.get('repaired_defects_xy', []))]:
                mask = np.zeros(values.shape[:2], bool)
                for x, y in points:
                    mask[y, x] = True
                masks[name] = mask
        section = slice(start-lo, end-lo)
        yield start, end, values[section], {key: value[section] for key, value in masks.items()}


def registered_chunks(frame, matrix, origin, shape, rows):
    transform = np.eye(3)
    transform[:2, 2] = -np.asarray(origin)
    transform = transform@matrix
    for start, end in bands(shape[0], rows):
        values = sample_band(frame.data, transform, start, end, shape[1])
        masks = {}
        for name, array in frame.auxiliary.items():
            sampled = sample_band(array, transform, start, end, shape[1])
            masks[name] = sampled if name == 'soft_weight' else (sampled >= 1-1e-9 if name == 'coverage' else sampled > 0)
        values[~masks['coverage']] = np.nan
        yield start, end, values, masks


def export_intermediates(source, result, output, indices, stages, rows=96, warn=None):
    source, output = Path(source), Path(output)
    original = json.loads((source/'report.json').read_text())
    frames, _, masters = load_cached_run(source, original['recipe'])
    from dataclasses import replace
    for index, path in result.get('lunar_margin_masks', {}).items():
        i = int(index)
        frames[i] = replace(frames[i], auxiliary=dict(frames[i].auxiliary, lunar=np.load(path, mmap_mode='r')))
    if not stages or set(stages)-{'calibrated', 'debayered', 'registered'}:
        raise ValueError('Select valid intermediate image stages')
    estimated = 0
    for index in indices:
        pixels = int(np.prod(frames[index].data.shape[:2]))
        if 'calibrated' in stages:
            channels = 1 if original['inputs'][index]['metadata']['state'] == 'linear_cfa' else (3 if frames[index].data.ndim == 3 else 1)
            estimated += pixels*(4*channels+9)
        if 'debayered' in stages:
            estimated += pixels*(12 if frames[index].data.ndim == 3 else 4)+pixels*9
        if 'registered' in stages and index in result['accepted_indices']:
            estimated += int(np.prod(result['output']['shape_yx']))*((12 if frames[index].data.ndim == 3 else 4)+9)
    if estimated*1.02 > shutil.disk_usage(output.parent).free:
        raise OSError(f'Insufficient external disk space for intermediate exports: approximately {estimated/2**30:.2f} GiB required')
    message = f'Intermediate exports require approximately {estimated/2**30:.2f} GiB additional disk space'
    if warn:
        warn(message)
    else:
        print(message, flush=True)
    records = []
    transforms = {item['index']: item for item in result['transformations']}
    for index in indices:
        frame = frames[index]
        base = dict(frame.metadata, input_index=index, source_path=str(frame.path),
                    normalization='none', stretch='none', unit='linear detector response',
                    intermediate_note='Pre-HDR samples; no empirical gain/plane matching or exposure division')
        for stage in stages:
            directory = output/'intermediates'/stage/f'frame_{index:03d}'
            metadata = dict(base, stage=stage)
            if stage == 'calibrated':
                item = dict(original['recipe']['frames'][index], path=original['inputs'][index]['path'])
                detector = load_item(item, source)
                if result['output'].get('lunar_margin_px') is not None:
                    from .lunar import fitted_disk
                    detector.metadata = dict(detector.metadata, lunar_disk=fitted_disk(detector.metadata),
                                             limb_margin_px=result['output']['lunar_margin_px'])
                if isinstance(detector, SensorFrame):
                    policy = frame.metadata.get('source_detector_metadata', {})
                    detector.metadata = dict(detector.metadata, **{key: policy[key] for key in
                        ('auto_defect_pixels_xy', 'repair_defects_xy', 'repair_defects_xyc', 'repair_defects_prequalified', 'defective_pixels_xy', 'defect_policy') if key in policy})
                if not isinstance(detector, SensorFrame):
                    metadata = dict(metadata, stage_note='Input already prepared; no detector recalibration', **{'state': detector.metadata['state']})
                    chunks = ((a, b, frame.data[a:b], {k: v[a:b] for k, v in frame.auxiliary.items()})
                              for a, b in bands(frame.data.shape[0], rows))
                    shape = frame.data.shape
                else:
                    cached = CachedFrame(detector.data, detector.metadata, detector.path, detector=True)
                    tile = slice_input(cached, 0, min(rows, detector.data.shape[0]))
                    _, head = calibrate(tile, *[slice_input(master, 0, tile.data.shape[0]) if master else None
                        for master in (masters.get('bias', [None])[0], select_dark(detector, masters.get('dark', [])), masters.get('flat', [None])[0])])
                    metadata = dict(head, input_index=index, stage=stage, active_area_xywh=detector.metadata['active_area_xywh'],
                        lunar_disk=detector.metadata['lunar_disk'], source_path=str(detector.path), normalization='none', stretch='none')
                    for key in ('auto_defect_pixels_xy', 'repair_defects_xy', 'repair_defects_xyc', 'repair_defects_prequalified', 'defective_pixels_xy', 'defect_policy'):
                        if key in detector.metadata:
                            metadata[key] = detector.metadata[key]
                    if 'repaired_defects_xy' in frame.metadata:
                        metadata['repaired_defects_xy'] = frame.metadata['repaired_defects_xy']
                    shape = detector.data.shape
                    chunks = calibrated_chunks(cached, masters, rows)
            elif stage == 'debayered':
                shape = frame.data.shape
                metadata['stage_note'] = 'Linear camera RGB; mono inputs remain mono'
                chunks = ((a, b, frame.data[a:b], {k: v[a:b] for k, v in frame.auxiliary.items()})
                          for a, b in bands(shape[0], rows))
            elif stage == 'registered':
                if index not in result['accepted_indices']:
                    continue
                record = transforms[index]
                matrix = np.asarray(record.get('matrix_yx', np.eye(3)), dtype=float)
                if 'matrix_yx' not in record:
                    matrix[:2, 2] = record['shift_dy_dx']
                origin = result['output']['origin_reference_yx']
                shape = (*result['output']['shape_yx'], *frame.data.shape[2:])
                metadata.update(matrix_yx=matrix.tolist(), origin_reference_yx=origin,
                    active_area_xywh=[0, 0, shape[1], shape[0]],
                    lunar_disk=None, lunar_geometry='per-image transformed mask in lunar.tiff',
                    interpolation='one bilinear interpolation from prepared samples; outside coverage NaN')
                chunks = registered_chunks(frame, matrix, origin, shape, rows)
            else:
                raise ValueError(f'Unknown intermediate stage: {stage}')
            write_stage(directory, shape, metadata, chunks)
            records.append(dict(index=index, stage=stage, path=str(directory.relative_to(output)), shape=list(shape)))
            print(f'Saved {stage} intermediate {index}', flush=True)
    manifest = dict(format='float32 TIFF; RGB interleaved YXS; mono/CFA YX', estimated_bytes=estimated,
        mask_convention='uint8 0/1; soft_weight float32; no observation outside coverage is NaN',
        registered_selection='Only final HDR accepted frames; calibration/debayer exports include all requested frames',
        records=records)
    (output/'intermediates'/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest

"""Known-truth CFA spikes, moving scene features and selective repair controls."""
import json
from pathlib import Path
import numpy as np
import pytest

from coronahdr.defects import candidates, confirm, repair_cfa, build_profile, confirm_by_exposure
from coronahdr.prepare import prepare_light
from coronahdr.sensor import SensorFrame
from coronahdr.streaming import CachedFrame, cache_prepared, load_cached_run
from coronahdr.io import write_image, read_frame
from test_calibration import sensor


def sequence(moving=True):
    rng = np.random.default_rng(423)
    y, x = np.indices((96, 96))
    points = [[60, 50], [61, 50], [60, 51], [61, 51]]
    records, matrices, frames = [], [], []
    for i in range(4):
        shift = 12*i if moving else 0
        values = 100+.1*x+.03*y+rng.normal(0, 1, x.shape)
        values[25, 15+shift] += 3000  # A narrow moving scene feature must survive.
        for px, py in points:
            values[py, px] += 7000
        frames.append(values)
        records.append(candidates(values, np.ones(x.shape, bool)))
        matrix = np.eye(3)
        matrix[1, 2] = -shift
        matrices.append(matrix)
    return frames, records, matrices, points


def test_persistent_all_phase_spikes_confirm_but_moving_star_does_not():
    frames, records, matrices, points = sequence()
    result = confirm(records, matrices, frames[0].shape)
    assert sorted(result['pixels_xy']) == sorted(points)
    assert result['hits'] == [4]*4
    assert all(value >= 8 for value in result['motion_px'])


def test_no_motion_means_no_automatic_defect_claim_even_for_stationary_star():
    frames, records, matrices, points = sequence(moving=False)
    result = confirm(records, matrices, frames[0].shape)
    assert result['pixels_xy'] == []
    assert result['unconfirmed_motion_count'] >= len(points)


def test_detector_work_cannot_be_written_to_internal_project_storage():
    with pytest.raises(ValueError, match='external work directory'):
        build_profile('missing-source', 'missing-review', [], Path.cwd()/'must-not-write-defect-cache', 'aesthetic')


def test_detector_respects_selected_work_directory(tmp_path):
    work = tmp_path/'selected'
    work.mkdir()
    with pytest.raises(ValueError, match='selected work directory'):
        build_profile('missing-source', 'missing-review', [], tmp_path/'outside', 'aesthetic',
                      work_directory=work)
    assert not (tmp_path/'outside').exists()
    with pytest.raises(FileNotFoundError):
        build_profile('missing-source', 'missing-review', [], work/'defects', 'aesthetic',
                      work_directory=work)
    assert (work/'defects').is_dir()


def test_detector_accepts_equivalent_macos_unicode_paths(tmp_path):
    import unicodedata
    work = tmp_path/'sélected'
    work.mkdir()
    alternate = Path(unicodedata.normalize('NFD', str(work)))
    if not alternate.is_dir() or not alternate.samefile(work):
        pytest.skip('Filesystem does not equate Unicode-normalized folder names')
    destination = alternate/'defects'
    assert not destination.resolve().is_relative_to(work.resolve())
    with pytest.raises(FileNotFoundError):
        build_profile('missing-source', 'missing-review', [], destination, 'aesthetic',
                      work_directory=work)
    assert (work/'defects').is_dir()


def test_detector_rejects_symlink_escape_from_selected_work(tmp_path):
    work, outside = tmp_path/'selected', tmp_path/'outside'
    work.mkdir()
    outside.mkdir()
    (work/'escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='selected work directory'):
        build_profile('missing-source', 'missing-review', [], work/'escape/defects', 'aesthetic',
                      work_directory=work)
    assert not (outside/'defects').exists()


def test_invalid_neighborhood_and_two_views_are_not_confirmed():
    frames, records, matrices, points = sequence()
    assert confirm(records[:2], matrices[:2], frames[0].shape)['pixels_xy'] == []
    valid = np.ones(frames[0].shape, bool)
    valid[45:58, 55:68] = False
    detected, eligible = candidates(frames[0], valid)
    assert not detected[50:52, 60:62].any()


def test_repair_is_same_phase_local_and_preserves_negative_samples():
    y, x = np.indices((64, 64))
    data = -50.+.2*x+.3*y
    for py in range(2):
        for px in range(2):
            data[py::2, px::2] += 100*py+20*px
    truth = data.copy()
    data[30, 31] = 8000
    corrected, repaired = repair_cfa(data, [[31, 30]], np.zeros(data.shape, bool))
    assert repaired == [[31, 30]]
    assert corrected[30, 31] == pytest.approx(truth[30, 31])
    mask = np.ones(data.shape, bool)
    mask[30, 31] = False
    np.testing.assert_array_equal(corrected[mask], data[mask])
    assert corrected.min() < 0


@pytest.mark.parametrize('pattern', ['RGGB', 'BGGR', 'GRBG', 'GBRG'])
def test_scientific_excludes_and_aesthetic_repairs_before_demosaic_without_seams(tmp_path, pattern):
    data = np.full((96, 96), 300.)
    data[35, 40] = 8000
    points = [[40, 35]]
    raw = sensor(data, cfa_pattern=pattern, active_area_xywh=[1, 1, 96, 96],
                 auto_defect_pixels_xy=points, repair_defects_xy=points, defect_policy='aesthetic')
    prepared = prepare_light(raw)
    assert prepared.auxiliary['repaired'][35, 40]
    assert prepared.auxiliary['repair_footprint'][34:37, 39:42].all()
    np.testing.assert_allclose(prepared.data[34:37, 39:42], 200)
    cached = cache_prepared(CachedFrame(raw.data, raw.metadata, raw.path, detector=True),
                            None, None, None, tmp_path/'prepared', 35)
    np.testing.assert_allclose(cached.data, prepared.data, equal_nan=True)
    np.testing.assert_array_equal(cached.auxiliary['repaired'], prepared.auxiliary['repaired'])
    from coronahdr.intermediates import calibrated_chunks
    chunks = list(calibrated_chunks(CachedFrame(raw.data, raw.metadata, raw.path, detector=True), {}, 35))
    np.testing.assert_allclose(np.concatenate([part[2] for part in chunks]), 200)
    assert int(np.concatenate([part[3]['repaired'] for part in chunks]).sum()) == 1
    raw = sensor(data, auto_defect_pixels_xy=points, defective_pixels_xy=points, defect_policy='scientific')
    scientific = prepare_light(raw)
    assert scientific.masks()['defective'][34:37, 39:42].all()
    assert not scientific.auxiliary['repaired'].any()
    from coronahdr.pipeline import combine
    master, maps, *_ = combine([scientific], reference=0)
    assert not maps['support'][34:37, 39:42].any()
    assert np.isnan(master[34:37, 39:42]).all()


def test_saturated_occulted_and_insufficient_neighbors_are_never_invented():
    data = np.full((64, 64), 300.)
    data[20, 20] = 10000
    raw = sensor(data, repair_defects_xy=[[20, 20], [30, 30]], lunar_disk=[30, 30, 5])
    output = prepare_light(raw)
    assert output.metadata['repaired_defects_xy'] == []
    result, repaired = repair_cfa(data, [[2, 2]], np.ones(data.shape, bool))
    assert repaired == []
    np.testing.assert_array_equal(result, data)


@pytest.mark.parametrize('state', ['linear_cfa', 'linear_mono', 'linear_rgb'])
@pytest.mark.parametrize('mode', ['scientific', 'aesthetic'])
def test_derived_preparation_profile_is_reusable_and_original_is_immutable(tmp_path, mode, state, monkeypatch):
    arrays, _, matrices, points = sequence()
    if state != 'linear_cfa':
        for data in arrays:
            for x, y in points[1:]:
                data[y, x] -= 7000
        points = points[:1]
        if state == 'linear_rgb':
            arrays = [np.stack([data, np.full_like(data, 200), np.full_like(data, 300)], axis=-1) for data in arrays]
    source = tmp_path/'source'
    work = source/'work'
    work.mkdir(parents=True)
    inputs, items = [], []
    for index, data in enumerate(arrays):
        raw = sensor(data+100, active_area_xywh=[1, 1, 96, 96], state=state, defect_grid='native_unregistered')
        path = tmp_path/f'light_{index}.{"fits" if state == "linear_rgb" else "tiff"}'
        write_image(path, raw.data, raw.metadata)
        frame = read_frame(path)
        cache_prepared(CachedFrame(frame.data, frame.metadata, frame.path, detector=True),
                       None, None, None, work/f'prepared_{index:03d}', 35)
        inputs.append(dict(path=str(path), metadata=frame.metadata))
        items.append(dict(path=str(path)))
    recipe = dict(frames=items, calibration={}, reference=0)
    report = dict(recipe=recipe, inputs=inputs, calibration={}, preparation_directory=str(work))
    (source/'report.json').write_text(json.dumps(report))
    original = (source/'report.json').read_bytes()
    original_data = np.load(work/'prepared_000/data.npy').copy()
    review = tmp_path/'review.json'
    review.write_text(json.dumps(dict(matrices_yx={str(i): a.tolist() for i, a in enumerate(matrices)},
        input_source_sha256=[item['metadata']['source_sha256'] for item in inputs])))
    if mode == 'scientific' and state == 'linear_cfa':
        monkeypatch.setattr('coronahdr.performance.detector_workers', lambda *args: 1)
        _, serial = build_profile(source, review, list(range(4)), tmp_path/'serial-check', mode, apply_policy=False, work_directory=tmp_path)
        monkeypatch.setattr('coronahdr.performance.detector_workers', lambda *args: 2)
        _, parallel = build_profile(source, review, list(range(4)), tmp_path/'parallel-check', mode, apply_policy=False, work_directory=tmp_path)
        assert serial == parallel
    profile, summary = build_profile(source, review, list(range(4)), tmp_path/'derived', mode, work_directory=tmp_path)
    assert sorted(summary['pixels_xy']) == sorted(points)
    assert profile != source
    prepared, _, _ = load_cached_run(profile, recipe)
    if mode == 'scientific':
        assert prepared[0].auxiliary['defective'][50, 60]
    else:
        assert int(prepared[0].auxiliary['repaired'].sum()) == len(points)
        if state == 'linear_rgb':
            np.testing.assert_array_equal(prepared[0].data[..., 1:], original_data[..., 1:])
            assert prepared[0].auxiliary['repair_footprint'].sum() == 1
        assert np.max(prepared[0].data[50:52, 60:62]) < 500
    np.testing.assert_array_equal(np.load(work/'prepared_000/data.npy'), original_data)
    assert (source/'report.json').read_bytes() == original
    from coronahdr.intermediates import export_intermediates
    import tifffile
    result = dict(accepted_indices=[0], transformations=[dict(index=0, matrix_yx=np.eye(3).tolist())],
                  output=dict(origin_reference_yx=[0, 0], shape_yx=[96, 96]))
    export_intermediates(profile, result, tmp_path/'intermediates', [0], ['calibrated', 'debayered'], rows=35)
    base = tmp_path/'intermediates/intermediates/calibrated/frame_000'
    native = tifffile.imread(base/'image.tiff')
    mask = tifffile.imread(base/'repaired.tiff')
    metadata = json.loads((base/'metadata.json').read_text())
    assert metadata['auto_defect_pixels_xy'] == summary['pixels_xy']
    if mode == 'aesthetic':
        assert mask.sum() == len(points)
        assert native[50:52, 60:62].max() < 500
    else:
        assert not mask.any()
        assert np.isnan(native[50, 60]).all()
    skipped_profile, skipped = build_profile(source, review, [0, 1], tmp_path/'insufficient', mode, work_directory=tmp_path)
    assert skipped_profile == source and skipped['count'] == 0
    assert 'Fewer than three' in skipped['skipped_reason']
    if state != 'linear_cfa':
        report['recipe']['frames'][0]['overrides'] = {'defect_grid': 'registered'}
        (source/'report.json').write_text(json.dumps(report))
        unchanged, skipped = build_profile(source, review, list(range(4)), tmp_path/'not-native', mode, work_directory=tmp_path)
        assert unchanged == source and not skipped['applied']
        assert 'native_unregistered' in skipped['skipped_reason']



def test_rgb_confirmation_requires_recurrence_in_the_same_channel():
    records, matrices = [], []
    for i in range(4):
        data = np.full((64, 64, 3), 200.)
        data[30, 30, i % 3] = 7000
        records.append(candidates(data, np.ones(data.shape, bool), cfa=False))
        matrix = np.eye(3)
        matrix[1, 2] = i*12
        matrices.append(matrix)
    assert confirm(records, matrices, data.shape)['pixels_xy'] == []
    assert confirm(records, [np.eye(3)]*4, data.shape)['pixels_xy'] == []


def test_linear_repair_preserves_lunar_and_saturated_samples():
    data = np.full((64, 64, 3), 300.)
    data[20, 20, 0] = 10000
    raw = sensor(data, state='linear_rgb', repair_defects_xy=[[20, 20], [30, 30]],
                 lunar_disk=[30, 30, 5])
    output = prepare_light(raw)
    assert output.metadata['repaired_defects_xy'] == []


@pytest.mark.parametrize('rgb', [False, True])
def test_exposure_groups_recover_long_only_defects_and_keep_evidence(rgb):
    shape = (64, 64, 3) if rgb else (64, 64)
    records, poses = [], []
    for i in range(12):
        data = np.full(shape, 200.)
        if i >= 9:
            data[(30, 30, 0) if rgb else (30, 30)] = 7000
        records.append(candidates(data, np.ones(shape, bool), cfa=not rgb))
        matrix = np.eye(3)
        matrix[1, 2] = i*12
        poses.append(matrix)
    times = [.01]*9+[1.]*3
    result = confirm_by_exposure(records, poses, shape, times, list(range(12)))
    assert result['pixels_xy'] == [[30, 30]]
    assert result['pooled_comparison']['pixels_xy'] == []
    assert result['added_vs_pooled_xy'] == [[30, 30]]
    assert result['exposure_groups'][1]['analyzed_indices'] == [9, 10, 11]
    assert result['exposure_groups'][1]['hits'] == [3]
    assert result['exposure_groups'][1]['motion_px'] == [24.]
    if rgb:
        assert result['samples_xyc'] == [[30, 30, 0]]
    still = confirm_by_exposure(records, [np.eye(3)]*12, shape, times, list(range(12)))
    assert still['pixels_xy'] == []
    sparse = confirm_by_exposure(records, poses, shape, list(range(1, 13)), list(range(12)))
    assert sparse['pixels_xy'] == []
    assert all(g['skipped_reason'] for g in sparse['exposure_groups'])


@pytest.mark.parametrize('state', ['linear_cfa', 'linear_mono', 'linear_rgb'])
def test_automatic_repair_does_not_replace_normal_short_exposure_samples(state):
    data = np.full((64, 64), 300.)
    data[30, 30] = 301.
    if state == 'linear_rgb':
        data = np.repeat(data[..., None], 3, axis=-1)
    common = dict(state=state, auto_defect_pixels_xy=[[30, 30]], repair_defects_xy=[[30, 30]])
    normal = prepare_light(sensor(data, **common))
    unchanged = prepare_light(sensor(data, state=state))
    np.testing.assert_array_equal(normal.data, unchanged.data)
    assert normal.metadata['repaired_defects_xy'] == []
    hot = data.copy()
    hot[30, 30] = 7000
    repaired = prepare_light(sensor(hot, **common))
    assert repaired.metadata['repaired_defects_xy'] == [[30, 30]]


def test_exposure_confirmation_rejects_invalid_metadata():
    with pytest.raises(ValueError, match='one exposure'):
        confirm_by_exposure([], [], (64, 64), [1.], [])
    with pytest.raises(ValueError, match='positive'):
        confirm_by_exposure([(np.zeros((64, 64), bool),)*2], [np.eye(3)], (64, 64), [0.], [0])


def test_exposure_tiers_cannot_borrow_motion_from_other_tiers():
    data = np.full((64, 64), 200.)
    data[30, 30] = 7000
    record = candidates(data, np.ones(data.shape, bool))
    moved = np.eye(3)
    moved[1, 2] = 20
    result = confirm_by_exposure([record]*6, [np.eye(3)]*3+[moved]*3,
                                 data.shape, [.01]*3+[1.]*3, list(range(6)))
    assert result['pooled_comparison']['pixels_xy'] == [[30, 30]]
    assert result['pixels_xy'] == []
    assert result['pooled_only_xy'] == [[30, 30]]
    assert result['unconfirmed_motion_count'] == 2


@pytest.mark.parametrize('state', ['linear_cfa', 'linear_rgb'])
def test_profile_repairs_only_full_frame_candidates_in_each_sub(tmp_path, state):
    arrays, _, matrices, points = sequence()
    for x, y in points:
        arrays[0][y, x] -= 7000
    if state == 'linear_rgb':
        for array in arrays:
            for x, y in points[1:]:
                array[y, x] -= 7000
        points = points[:1]
        arrays = [np.stack([a, np.full_like(a, 200), np.full_like(a, 300)], axis=-1) for a in arrays]
    source = tmp_path/'source'
    work = source/'work'
    work.mkdir(parents=True)
    inputs, items = [], []
    for i, data in enumerate(arrays):
        raw = sensor(data+100, state=state, defect_grid='native_unregistered', exposure_s=.01 if i == 0 else 1.)
        path = tmp_path/f'light_{i}.tiff'
        write_image(path, raw.data, raw.metadata)
        frame = read_frame(path)
        cache_prepared(CachedFrame(frame.data, frame.metadata, frame.path, detector=True),
                       None, None, None, work/f'prepared_{i:03d}', 35)
        inputs.append(dict(path=str(path), metadata=frame.metadata))
        items.append(dict(path=str(path)))
    recipe = dict(frames=items, calibration={}, reference=0)
    (source/'report.json').write_text(json.dumps(dict(recipe=recipe, inputs=inputs, calibration={},
                                                      preparation_directory=str(work))))
    review = tmp_path/'review.json'
    review.write_text(json.dumps(dict(matrices_yx={str(i): m.tolist() for i, m in enumerate(matrices)},
        input_source_sha256=[item['metadata']['source_sha256'] for item in inputs])))
    profile, result = build_profile(source, review, list(range(4)), tmp_path/'derived', 'aesthetic', work_directory=tmp_path)
    assert sorted(result['pixels_xy']) == sorted(points)
    prepared, _, _ = load_cached_run(profile, recipe)
    assert prepared[0].metadata['repair_defects_xy'] == []
    assert not prepared[0].auxiliary['repaired'].any()
    np.testing.assert_array_equal(prepared[0].data, np.load(work/'prepared_000/data.npy'))
    assert prepared[1].auxiliary['repaired'].sum() == len(points)
    if state == 'linear_rgb':
        assert prepared[1].metadata['repair_defects_xyc'] == [[60, 50, 0]]
        np.testing.assert_array_equal(prepared[1].data[..., 1:],
                                      np.load(work/'prepared_001/data.npy')[..., 1:])

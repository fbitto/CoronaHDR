"""Disk/in-memory parity, detector tile seams, and canvas/bridge truth controls."""
import json
from pathlib import Path
import numpy as np
from astropy.io import fits
import pytest
from coronahdr.synthetic import make_scene, make_cfa_scene
from coronahdr.pipeline import run_manifest, load_item
from coronahdr.streaming import cache_input, cache_prepared, tiled_master, tiled_flat, sample_band, registration_auxiliary, output_canvas
from coronahdr.calibration import build_master, normalize_flat
from coronahdr.prepare import prepare_light
from coronahdr.registration import translate
from coronahdr.registration_graph import solve_translations, radiometry


def test_sample_band_matches_shift_and_boolean_footprints():
    rng = np.random.default_rng(42)
    for shape in ((79, 81), (79, 81, 3)):
        array = rng.normal(size=shape)
        for delta in ([0, 0], [4.28, -7.43], [-13.7, 12.4]):
            expected = translate(array, delta)
            actual = np.concatenate([sample_band(array, delta, 0, 40, 81), sample_band(array, delta, 40, 79, 81)])
            np.testing.assert_allclose(actual, expected, atol=1e-14)
    mask = np.ones((79, 81), bool)
    mask[30, 30] = False
    np.testing.assert_allclose(sample_band(mask, [.5, .5], 0, 79, 81), translate(mask, [.5, .5]))


def test_tiled_calibration_and_demosaic_have_no_internal_seams(tmp_path):
    manifest = make_cfa_scene(tmp_path/'inputs')
    recipe = json.loads(manifest.read_text())
    base = manifest.parent
    cached = {}
    masters = {}
    for role in ('bias', 'flat', 'dark_flat'):
        frames = [load_item(item, base) for item in recipe['calibration'][role]]
        cached[role] = [cache_input(f, tmp_path/f'{role}_{i}') for i, f in enumerate(frames)]
        masters[role] = tiled_master(cached[role], role, tmp_path, 35)
        np.testing.assert_array_equal(masters[role].data, build_master(frames, role).data)
    flat = tiled_flat(masters['flat'], masters['bias'], masters['dark_flat'], tmp_path, 35)
    expected_flat = normalize_flat(masters['flat'], masters['bias'], masters['dark_flat'])
    np.testing.assert_allclose(flat.data, expected_flat.data, equal_nan=True)
    source = load_item(recipe['frames'][0], base)
    cached_source = cache_input(source, tmp_path/'source')
    prepared = cache_prepared(cached_source, masters['bias'], None, flat, tmp_path/'prepared', 35)
    expected = prepare_light(source, masters['bias'], None, flat)
    np.testing.assert_allclose(prepared.data, expected.data, atol=1e-12, equal_nan=True)
    for key in ('coverage', 'lunar', 'saturation', 'low_signal', 'defective'):
        np.testing.assert_array_equal(prepared.auxiliary[key], expected.masks()[key])


def test_disk_pipeline_matches_memory_at_identity_and_subpixel(tmp_path):
    manifest = make_scene(tmp_path/'inputs')
    baseline = run_manifest(manifest, tmp_path/'memory')
    recipe = json.loads(manifest.read_text())
    recipe['processing'] = dict(mode='disk', tile_rows=35, registration_bin=1, canvas='reference', bridges=False, geometry_photometry=False)
    manifest.write_text(json.dumps(recipe))
    report = run_manifest(manifest, tmp_path/'disk')
    assert report['accepted_indices'] == [0, 1, 2, 3]
    np.testing.assert_allclose(fits.getdata(tmp_path/'disk/master.fits'), fits.getdata(tmp_path/'memory/master.fits'), atol=2e-4, rtol=2e-6, equal_nan=True)
    for key in ('support', 'contributions', 'coverage', 'lunar', 'saturation', 'unresolved_saturation'):
        np.testing.assert_array_equal(fits.getdata(tmp_path/f'disk/{key}.fits'), fits.getdata(tmp_path/f'memory/{key}.fits'))
    assert report['output']['origin_reference_yx'] == [0, 0]


def test_union_canvas_includes_translated_extents():
    from types import SimpleNamespace
    frames = [SimpleNamespace(data=np.zeros((64, 80))) for _ in range(3)]
    origin, shape = output_canvas(frames, [np.array([0, 0]), np.array([12.5, -20.2]), np.array([-8.2, 7.1])], 'union')
    assert origin == (-9, -21) and shape == (86, 109)
    with pytest.raises(ValueError):
        output_canvas(frames, [np.zeros(2)]*3, 'automatic')


def test_intersection_canvas_contains_only_shared_pixel_centers():
    from types import SimpleNamespace
    frames = [SimpleNamespace(data=np.zeros((64, 80))) for _ in range(3)]
    shifts = [np.array([0, 0]), np.array([12.5, -20.2]), np.array([-8.2, 7.1])]
    origin, shape = output_canvas(frames, shifts, 'intersection')
    assert origin == (13, 8) and shape == (42, 51)
    for displacement in shifts:
        assert np.all(np.array(origin)-displacement >= 0)
        assert np.all(np.array(origin)+shape-1-displacement <= [63, 79])
    with pytest.raises(ValueError, match='No shared'):
        output_canvas(frames[:2], [np.zeros(2), np.array([80, 0])], 'intersection')
    with pytest.raises(ValueError, match='translation-only'):
        output_canvas(frames[:1], [np.eye(3)], 'intersection')


def test_desktop_wider_shift_retry_and_crop_reuse_preparation(tmp_path, monkeypatch):
    manifest = make_scene(tmp_path/'inputs')
    recipe = json.loads(manifest.read_text())
    recipe['max_shift'] = 1
    recipe['processing'] = dict(mode='disk', tile_rows=35, registration_bin=1,
        canvas='union', bridges=False, geometry_photometry=False, rejection_policy='exclude')
    manifest.write_text(json.dumps(recipe))
    messages = []
    def choose(message, choices, default):
        messages.append(message)
        return choices[0]
    monkeypatch.setattr('coronahdr.decisions.ask', choose)
    report = run_manifest(manifest, tmp_path/'disk')
    assert report['accepted_indices'] == [0, 1, 2, 3]
    assert report['recipe']['max_shift'] > 1
    assert report['output']['canvas'] == 'intersection'
    assert all(v < 192 for v in report['output']['shape_yx'])
    assert any('shift limit' in message for message in messages)
    assert any(message.startswith('Weighted HDR: shared detector') for message in messages)


def test_completed_preparation_survives_interrupted_registration(tmp_path, monkeypatch):
    from coronahdr.streaming import load_cached_run
    manifest = make_scene(tmp_path/'inputs')
    recipe = json.loads(manifest.read_text())
    recipe['max_shift'] = 1
    recipe['processing'] = dict(mode='disk', tile_rows=35, registration_bin=1,
        canvas='union', bridges=False, geometry_photometry=False, rejection_policy='exclude')
    manifest.write_text(json.dumps(recipe))
    def cancel(*args):
        raise ValueError('User cancelled')
    monkeypatch.setattr('coronahdr.decisions.ask', cancel)
    with pytest.raises(ValueError, match='User cancelled'):
        run_manifest(manifest, tmp_path/'interrupted')
    checkpoint = tmp_path/'interrupted/preparation'
    assert not (tmp_path/'interrupted/report.json').exists()
    frames, inputs, masters = load_cached_run(checkpoint, recipe)
    assert len(frames) == len(recipe['frames'])
    before = frames[0].data.copy()
    saved = json.loads((checkpoint/'report.json').read_text())
    overlays = {'0': str(Path(saved['preparation_directory'])/'prepared_000')}
    saved['prepared_overlays'] = overlays
    (checkpoint/'report.json').write_text(json.dumps(saved))
    recipe['max_shift'] = 12
    recipe['processing']['cache_from'] = str(checkpoint)
    manifest.write_text(json.dumps(recipe))
    monkeypatch.setattr('coronahdr.decisions.ask', lambda message, choices, default: default)
    def no_preparation(*args):
        raise AssertionError('Completed detector preparation must be reused')
    monkeypatch.setattr('coronahdr.streaming.prepare_disk_inputs', no_preparation)
    report = run_manifest(manifest, tmp_path/'continued')
    assert report['accepted_indices'] == [0, 1, 2, 3]
    assert report['prepared_overlays'] == overlays
    np.testing.assert_array_equal(frames[0].data, before)
    Path(inputs[0]['path']).write_bytes(b'changed original')
    with pytest.raises(ValueError, match='Original hash changed'):
        load_cached_run(checkpoint, recipe)


def test_bridge_composition_and_inconsistent_paths_are_rejected(monkeypatch):
    import coronahdr.registration_graph as graph
    from types import SimpleNamespace
    frames = [SimpleNamespace(scale=v) for v in (1., 2., 4.)]
    truth = {0: np.array([0., 0.]), 1: np.array([10., -20.]), 2: np.array([100., -120.])}
    def fake(reference, moving, limit):
        parent, child = frames.index(reference), frames.index(moving)
        if {parent, child} == {0, 2}:
            raise ValueError('No common coronal structure')
        return (truth[child]-truth[parent])/2, dict(relative_residual=.01)
    monkeypatch.setattr(graph, 'register', fake)
    monkeypatch.setattr(graph, 'radiometry', lambda *args: {'applied': False})
    shifts, metrics, rejected, _ = solve_translations(frames, 0, 2, 200)
    assert not rejected and metrics[2]['path'] == [2, 1, 0]
    np.testing.assert_allclose(shifts[2], truth[2])
    shifts, _, rejected, _ = solve_translations(frames, 0, 2, 200, bridges=False)
    assert 2 not in shifts and rejected[0]['index'] == 2


def test_radiometric_diagnostics_do_not_modify_data(tmp_path):
    from coronahdr.model import Frame
    from test_stage1 import metadata
    y, x = np.indices((64, 64))
    original = 100+3*x+2*y
    reference = Frame(original, metadata(), Path('reference'))
    moving = Frame((original-17)/1.2, metadata(), Path('moving'))
    before = moving.data.copy()
    result = radiometry(reference, moving, [0, 0])
    assert not result['applied']
    assert abs(result['diagnostic_slope']-1.2) < 1e-5
    assert abs(result['diagnostic_offset_reference_units']-17) < 1e-4
    np.testing.assert_array_equal(before, moving.data)


def test_inconsistent_bridge_paths_do_not_enter_master(monkeypatch):
    import coronahdr.registration_graph as graph
    from types import SimpleNamespace
    frames = [SimpleNamespace(scale=v) for v in (1., 2., 4.)]
    def measured(reference, moving, limit):
        parent, child = frames.index(reference), frames.index(moving)
        shift = np.array([float(child-parent), 0.])
        if (parent, child) == (1, 2):
            shift += [3., 0.]
        return shift, dict(relative_residual=.01)
    monkeypatch.setattr(graph, 'register', measured)
    monkeypatch.setattr(graph, 'radiometry', lambda *args: {'applied': False})
    shifts, _, rejected, _ = solve_translations(frames, 0, 1, 24, cycle_tolerance=.5)
    assert 2 not in shifts and 'Inconsistent' in rejected[0]['reason']


def test_large_translations_with_independent_lunar_motion(tmp_path):
    from coronahdr.synthetic import corona
    from coronahdr.model import Frame
    from test_stage1 import metadata
    from coronahdr.registration import register
    y, x = np.indices((192, 192))
    offsets = [(0., 0.), (30.35, -45.42), (-24.28, 35.71)]
    frames = []
    for index, (dy, dx) in enumerate(offsets):
        data = corona(y-dy, x-dx, 192)
        disk = [96+dx+index*4, 96+dy-index*3, 20.]
        hidden = (x-disk[0])**2+(y-disk[1])**2 <= disk[2]**2
        data[hidden] = 0
        frames.append(Frame(data, metadata(white_level=20000, low_signal=1, lunar_disk=disk), Path(str(index))))
    for i in (1, 2):
        shift, _ = register(frames[0], frames[i], 80)
        np.testing.assert_allclose(shift, -np.array(offsets[i]), atol=.1)


def test_registration_auxiliary_binning_uses_debayerized_samples(tmp_path):
    from coronahdr.model import Frame
    from test_stage1 import metadata
    y, x = np.indices((192, 192))
    data = np.stack([100+x, 200+y, 300+x+y], axis=-1)
    frame = Frame(data, metadata(state='linear_rgb', white_level=10000), Path('prepared'))
    cached = cache_input(frame, tmp_path/'source')
    prepared = cache_prepared(cached, None, None, None, tmp_path/'prepared', 35)
    aux = registration_auxiliary(prepared, 4, [8, 8, 160, 160])
    truth = data[8:168, 8:168].mean(axis=-1).reshape(40, 4, 40, 4).mean(axis=(1, 3))
    np.testing.assert_allclose(aux.data, truth)
    assert aux.auxiliary['coverage'].all()


def test_sky_diagnosis_keeps_samples_unchanged(tmp_path):
    import runpy
    from test_stage1 import metadata
    if not (Path(__file__).parents[1]/'scripts/diagnose_radiometry.py').is_file():
        pytest.skip('Private research script is not included in the public checkout')
    function = runpy.run_path(str(Path(__file__).parents[1]/'scripts/diagnose_radiometry.py'))['survey']
    root = tmp_path/'run'
    (root/'work').mkdir(parents=True)
    y, x = np.indices((832, 832))
    original = 10+.01*x+.02*y
    for index, exposure in enumerate((1., 2.)):
        folder = root/'work'/f'prepared_{index:03d}'
        folder.mkdir()
        data = np.repeat((original*exposure+5)[..., None], 3, axis=-1)
        np.save(folder/'data.npy', data)
        (folder/'metadata.json').write_text(json.dumps(metadata(state='linear_rgb', white_level=10000,
            exposure_s=exposure, active_area_xywh=[0, 0, 832, 832])))
        for key in ('coverage', 'lunar', 'saturation', 'low_signal', 'defective'):
            np.save(folder/f'{key}.npy', np.full((832, 832), key == 'coverage', dtype=bool))
    (root/'registration.json').write_text(json.dumps(dict(reference=0, auxiliary_bin=4,
        roi_xywh=[0, 0, 832, 832], transformations={'0':dict(shift_dy_dx=[0, 0]), '1':dict(shift_dy_dx=[0, 0])})))
    result = function(root)
    assert not result['applied_corrections']
    diagnostic = result['reference_comparisons'][0]
    assert abs(diagnostic['diagnostic_offset_reference_units']-2.5) < 1e-3
    np.testing.assert_array_equal(np.load(root/'work/prepared_001/data.npy'),
                                 np.repeat((original*2+5)[..., None], 3, axis=-1))


def test_auxiliary_gain_offset_fit_removes_geometric_bias_without_changing_samples():
    from coronahdr.synthetic import corona
    from coronahdr.model import Frame
    from test_stage1 import metadata
    from coronahdr.registration import register
    y, x = np.indices((192, 192))
    reference = Frame(corona(y, x, 192), metadata(white_level=20000), Path('reference'))
    data = corona(y-12.35, x+17.42, 192)*1.2+200
    moving = Frame(data.copy(), metadata(white_level=20000), Path('moving'))
    measured, quality = register(reference, moving, 40, fit_photometry=True)
    np.testing.assert_allclose(measured, [-12.35, 17.42], atol=.1)
    assert not quality['auxiliary_photometry']['applied_to_master']
    np.testing.assert_array_equal(moving.data, data)


def test_reference_radiometric_gate_prevents_bridge_drift(monkeypatch):
    import coronahdr.registration_graph as graph
    from types import SimpleNamespace
    frames = [SimpleNamespace(name=i) for i in range(3)]
    def diagnose(reference, moving, delta):
        return dict(raw_relative_error=.1 if moving.name == 1 else .3, raw_regional_relative_errors=[.1])
    monkeypatch.setattr(graph, 'radiometry', diagnose)
    solved = {0: np.zeros(2), 1: np.ones(2), 2: np.full(2, 2.)}
    metrics = {0:dict(path=[0]), 1:dict(path=[1, 0]), 2:dict(path=[2, 1, 0])}
    accepted, rejected, checks = graph.qualify_radiometry(frames, solved, metrics, 0, 1, [])
    assert set(accepted) == {0, 1} and rejected[0]['index'] == 2


def test_cache_reuse_checks_recipe_and_original_hashes(tmp_path):
    from coronahdr.streaming import load_cached_run
    manifest = make_scene(tmp_path/'inputs')
    recipe = json.loads(manifest.read_text())
    recipe['processing'] = dict(mode='disk', tile_rows=35, registration_bin=1,
                                canvas='reference', geometry_photometry=False, bridges=False)
    manifest.write_text(json.dumps(recipe))
    report = run_manifest(manifest, tmp_path/'first')
    frames, inputs, masters = load_cached_run(tmp_path/'first', recipe)
    assert len(frames) == 4 and not masters
    assert inputs == report['inputs']
    changed = json.loads(json.dumps(recipe))
    changed['frames'][0]['overrides'] = {'exposure_s': 9.}
    with pytest.raises(ValueError, match='recipe mismatch'):
        load_cached_run(tmp_path/'first', changed)
    source = manifest.parent/recipe['frames'][0]['path']
    with source.open('ab') as stream:
        stream.write(b'changed original')
    with pytest.raises(ValueError, match='hash changed'):
        load_cached_run(tmp_path/'first', recipe)


def test_cleanup_preserves_prepared_cache_and_unrelated_files(tmp_path):
    from coronahdr.streaming import cleanup_scratch
    for name in ('prepared_000', 'aligned_000', 'input_light_000'):
        (tmp_path/name).mkdir()
    np.save(tmp_path/'prepared_000/data.npy', np.ones((32, 32)))
    np.save(tmp_path/'aligned_000/weight.npy', np.ones((32, 32)))
    np.save(tmp_path/'input_light_000/source.npy', np.ones((32, 32)))
    (tmp_path/'input_light_000/notes.txt').write_text('Retain any unrelated user notes')
    cleanup_scratch(tmp_path)
    assert (tmp_path/'prepared_000/data.npy').exists()
    assert not (tmp_path/'aligned_000').exists()
    assert (tmp_path/'input_light_000/notes.txt').exists()
    assert not (tmp_path/'input_light_000/source.npy').exists()


@pytest.mark.parametrize('processing,reference,message', [
    ({'mode':'typo'}, 0, 'processing.mode'),
    ({'mode':'memory','canvas':'union'}, 0, 'Union canvas'),
    ({'mode':'disk','canvas':'union'}, 10, 'zero-based'),
])
def test_invalid_processing_options_do_not_silently_fall_back(tmp_path, processing, reference, message):
    manifest = make_scene(tmp_path/'inputs')
    recipe = json.loads(manifest.read_text())
    recipe.update(processing=processing, reference=reference)
    manifest.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match=message):
        run_manifest(manifest, tmp_path/'output')
    assert not (tmp_path/'output').exists()

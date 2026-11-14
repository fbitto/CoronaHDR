"""Stage exports preserve calibration, CFA phase, samples and geometric validity."""
import json
import numpy as np
import tifffile
import pytest
from coronahdr.synthetic import make_cfa_scene
from coronahdr.pipeline import run_manifest, load_item
from coronahdr.calibration import calibrate, select_dark
from coronahdr.streaming import load_cached_run
from coronahdr.intermediates import export_intermediates, registered_chunks


def test_cfa_intermediate_stages_match_engine_and_masks(tmp_path, monkeypatch):
    manifest = make_cfa_scene(tmp_path/'inputs')
    recipe = json.loads(manifest.read_text())
    recipe['processing'] = dict(mode='disk', tile_rows=35, registration_bin=1,
        canvas='reference', bridges=False, geometry_photometry=False)
    manifest.write_text(json.dumps(recipe))
    source = tmp_path/'source'
    result = run_manifest(manifest, source)
    frames, _, masters = load_cached_run(source, recipe)
    detector = load_item(recipe['frames'][0], manifest.parent)
    expected, _ = calibrate(detector, masters['bias'][0], select_dark(detector, masters.get('dark', [])), masters['flat'][0])
    output = tmp_path/'export'
    record = export_intermediates(source, result, output, [0], ['calibrated', 'debayered', 'registered'], rows=35)
    assert len(record['records']) == 3
    base = output/'intermediates'
    calibrated = tifffile.imread(base/'calibrated/frame_000/image.tiff')
    np.testing.assert_allclose(calibrated, expected, rtol=1e-7, equal_nan=True)
    assert calibrated.dtype == np.float32 and calibrated.ndim == 2
    metadata = json.loads((base/'calibrated/frame_000/metadata.json').read_text())
    assert metadata['state'] == 'linear_cfa'
    assert metadata['cfa_pattern'] == detector.metadata['cfa_pattern']
    assert metadata['active_area_xywh'] == detector.metadata['active_area_xywh']
    debayered = tifffile.imread(base/'debayered/frame_000/image.tiff')
    np.testing.assert_allclose(debayered, frames[0].data, rtol=1e-7, equal_nan=True)
    registered = tifffile.imread(base/'registered/frame_000/image.tiff')
    coverage = tifffile.imread(base/'registered/frame_000/coverage.tiff').astype(bool)
    assert np.isnan(registered[~coverage]).all()
    np.testing.assert_allclose(registered[coverage], frames[0].data[coverage], rtol=1e-7, equal_nan=True)
    for stage in ('calibrated', 'debayered', 'registered'):
        for mask in ('coverage', 'lunar', 'saturation', 'low_signal', 'defective'):
            a = tifffile.imread(base/f'{stage}/frame_000/{mask}.tiff')
            assert a.shape == expected.shape and a.dtype == np.uint8
    from types import SimpleNamespace
    monkeypatch.setattr('coronahdr.intermediates.shutil.disk_usage', lambda path: SimpleNamespace(free=0))
    with pytest.raises(OSError, match='Insufficient external disk space'):
        export_intermediates(source, result, tmp_path/'no-space', [0], ['registered'])
    assert not (tmp_path/'no-space').exists()


def test_registered_intermediate_preserves_negative_values_and_has_no_wrap():
    from types import SimpleNamespace
    data = np.full((64, 64, 3), -5.)
    frame = SimpleNamespace(data=data, auxiliary=dict(coverage=np.ones((64, 64), bool), lunar=np.zeros((64, 64), bool)))
    matrix = np.eye(3)
    matrix[:2, 2] = [3, -2]
    chunks = list(registered_chunks(frame, matrix, [-1, -1], (64, 64), 32))
    image = np.concatenate([chunk[2] for chunk in chunks])
    coverage = np.concatenate([chunk[3]['coverage'] for chunk in chunks])
    assert np.all(image[coverage] == -5)
    assert np.isnan(image[~coverage]).all()
    assert not coverage[:4].any()

"""Inventory failures stay visible and never silently become prepared inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

if not (Path(__file__).parents[1]/'scripts/review_canon_series.py').is_file():
    pytest.skip('Private research script is not included in the public checkout', allow_module_level=True)


spec = importlib.util.spec_from_file_location('review_canon_series', Path(__file__).parents[1]/'scripts/review_canon_series.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_inventory_retains_failed_limb_and_checks_previous_source_bytes(tmp_path, monkeypatch):
    source = tmp_path/'source'; source.mkdir(); previous = tmp_path/'previous'; previous.mkdir()
    first = source/'a.CR3'; first.write_bytes(b'original-a')
    (source/'b.CR3').write_bytes(b'original-b')
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    (previous/'report.json').write_text(json.dumps(dict(inputs=[dict(path=str(first), metadata=dict(source_sha256=digest))])))
    calls = []
    def decode(path):
        calls.append(path.name)
        return np.full((128, 128), 100, dtype='uint16'), dict(sensor_id='Canon', iso=800., aperture=8.,
            focal_length_mm=250., exposure_s=.01, black_level=[10]*4, white_level=[1000]*4)
    def limb(data):
        if calls[-1] == 'b.CR3':
            raise ValueError('Incomplete lunar limb')
        return [64., 64., 30.], 1.
    monkeypatch.setattr(module, 'decode_raw', decode); monkeypatch.setattr(module, 'lunar_disk', limb)
    rows, recipe = module.inventory(source, tmp_path/'output', previous)
    assert rows[0]['previous_six_match'] and rows[0]['preparation_eligible']
    assert rows[1]['decoded'] and not rows[1]['preparation_eligible']
    assert rows[1]['rejection_reason'] == 'Incomplete lunar limb'
    assert len(recipe['frames']) == 1 and recipe['calibration'] == {}
    assert recipe['frames'][0]['overrides']['limb_margin_px'] == 8
    assert hashlib.sha256(first.read_bytes()).hexdigest() == digest


def test_product_link_uses_identical_source_and_preserves_rigid_direction():
    from coronahdr.rigid import matrix_yx
    spec = importlib.util.spec_from_file_location('compare_canon_series', Path(__file__).parents[1]/'scripts/compare_canon_series.py')
    comparison = importlib.util.module_from_spec(spec); spec.loader.exec_module(comparison)
    a = matrix_yx((128, 160), [2., 3.], .5)
    b = matrix_yx((128, 160), [-4., 5.], -.3)
    def report(matrix, digest):
        return dict(reference=0, inputs=[dict(metadata=dict(source_sha256=digest))],
                    transformations=[dict(index=0, matrix_yx=matrix.tolist())])
    link, witness = comparison.reference_link(report(a, 'same'), report(b, 'same'))
    np.testing.assert_allclose(link@a, b, atol=1e-12)
    assert witness['source_sha256'] == 'same'
    with pytest.raises(ValueError, match='no shared'):
        comparison.reference_link(report(a, 'one'), report(b, 'two'))
    source = np.ones((64, 64, 3)); y, x = np.mgrid[:64, :64]
    shifted = comparison.sample(source, matrix_yx((64, 64), [3, 0], 0), y, x)
    assert np.isnan(shifted[:3]).all() and np.all(shifted[3:] == 1)

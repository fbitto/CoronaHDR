from pathlib import Path
import json

import numpy as np
import pytest
from scipy.ndimage import maximum_filter

from coronahdr.lunar import apply_margin, fitted_disk, validate_margin
from coronahdr.streaming import CachedFrame


@pytest.mark.parametrize('provenance,radius', [
    ('dataset-specific radial limb fit for exclusion only; radius padded by 8px', 28),
    ('closed-limb bin4 radial-gradient fit; exclusion only; 8px padding', 28),
    ('closed-limb bin4 radial-gradient fit; exclusion only; unpadded radius; 8px limb_margin_px applied during preparation', 20),
    ('manual geometry', 20),
])
def test_margin_is_relative_to_unpadded_edge_and_keeps_footprint(tmp_path, provenance, radius):
    metadata = dict(lunar_disk=[40., 39.5, radius], lunar_mask_provenance=provenance,
                    limb_margin_px=16, demosaic='bilinear')
    old_mask = np.ones((80, 80), bool)
    frame = CachedFrame(np.ones((80, 80, 3)), metadata, Path('fixture'), {'lunar': old_mask})
    y, x = np.indices((80, 80))
    assert fitted_disk(metadata)[2] == 20
    previous = None
    for margin in range(-4, 5):
        frames, paths = apply_margin([frame], margin, tmp_path)
        expected = maximum_filter((x-40)**2+(y-39.5)**2 <= (20+margin)**2, size=3)
        actual = frames[0].auxiliary['lunar'].copy()
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.load(paths['0']), expected)
        if previous is not None:
            assert not (previous & ~actual).any()
        previous = actual
    assert frame.auxiliary['lunar'] is old_mask and old_mask.all()
    assert metadata['lunar_disk'][2] == radius


@pytest.mark.parametrize('margin', [-5, 5, .5, True, float('nan')])
def test_invalid_margin_is_rejected(margin):
    with pytest.raises(ValueError, match='-4 to'):
        validate_margin(margin)


def test_no_override_preserves_existing_masks(tmp_path):
    frames = []
    assert apply_margin(frames, None, tmp_path) == (frames, {})


def test_original_nikon_profile_without_provenance_has_embedded_padding():
    metadata = dict(lunar_disk=[30, 30, 28],
        response_group='Nikon Z50 II serial 123; ISO800; f8; 250mm; fixed response declaration')
    assert fitted_disk(metadata)[2] == 20
    metadata['response_group'] = 'Another Nikon acquisition'
    assert fitted_disk(metadata)[2] == 28


def test_cached_products_record_effective_margins_and_keep_independent_masks(tmp_path):
    from coronahdr.pipeline import run_manifest
    from coronahdr.synthetic import make_scene

    manifest = make_scene(tmp_path/'inputs')
    recipe = json.loads(manifest.read_text())
    recipe['processing'] = dict(mode='disk', tile_rows=35, registration_bin=1,
                                canvas='reference', geometry_photometry=False,
                                lunar_margin_px=-2)
    manifest.write_text(json.dumps(recipe))
    source = tmp_path/'source'
    run_manifest(manifest, source)
    source_report = (source/'report.json').read_bytes()
    masks = {}
    for name, override, expected in [('inherited', {}, -2),
                                     ('corona', {'lunar_margin_px': 0}, 0),
                                     ('prominence', {'lunar_margin_px': -2}, -2)]:
        recipe['processing'] = dict(mode='disk', tile_rows=35, registration_bin=1,
                                    canvas='reference', geometry_photometry=False,
                                    cache_from=str(source), work_directory=str(tmp_path/'temporary'/name), **override)
        manifest.write_text(json.dumps(recipe))
        output = tmp_path/name
        report = run_manifest(manifest, output)
        saved = json.loads((output/'recipe.json').read_text())
        assert saved == report['recipe']
        assert saved['processing']['lunar_margin_px'] == expected
        assert report['output']['lunar_margin_px'] == expected
        assert report['inputs'] == json.loads(source_report)['inputs']
        assert report['output']['origin_reference_yx'] == [0, 0]
        assert Path(report['memory']['work_directory']) == tmp_path/'temporary'/name
        assert not (output/'work').exists()
        masks[name] = np.load(report['lunar_margin_masks']['0']).copy()
        y, x = np.indices(masks[name].shape)
        cx, cy, radius = report['inputs'][0]['metadata']['lunar_disk']
        np.testing.assert_array_equal(masks[name], (x-cx)**2+(y-cy)**2 <= (radius+expected)**2)
    np.testing.assert_array_equal(masks['inherited'], masks['prominence'])
    assert (masks['corona'] & ~masks['prominence']).any()
    assert (source/'report.json').read_bytes() == source_report


def test_invalid_inherited_margin_is_rejected_before_cache_loading(tmp_path, monkeypatch):
    import coronahdr.streaming as streaming

    source = tmp_path/'source'
    source.mkdir()
    (source/'report.json').write_text(json.dumps(dict(output=dict(lunar_margin_px=5))))
    def unexpected_load(*args):
        pytest.fail('Invalid cached margin reached preparation loading')
    monkeypatch.setattr(streaming, 'load_cached_run', unexpected_load)
    recipe = dict(frames=[dict(path='unused')], processing=dict(canvas='reference', cache_from=str(source)))
    with pytest.raises(ValueError, match='-4 to'):
        streaming.run_disk_manifest(recipe, tmp_path, tmp_path/'invalid')

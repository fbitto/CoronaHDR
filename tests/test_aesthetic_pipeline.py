"""Reusable aesthetic engine selection, geometry and non-camera-specific export."""
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

import coronahdr.aesthetic_pipeline as engine
from coronahdr.streaming import CachedFrame


def test_selection_keeps_reference_without_assuming_a_fixed_index():
    matrices = {str(i): np.eye(3).tolist() for i in (0, 1, 3)}
    assert engine.select_frames(4, 1, matrices) == [0, 1, 3]
    assert engine.select_frames(4, 1, matrices, [3, 1]) == [3, 1]
    for indices in ([0], [1, 1], [1, 2], [1, 9]):
        with pytest.raises(ValueError):
            engine.select_frames(4, 1, matrices, indices)


def test_geometry_cannot_silently_introduce_scale_or_nonfinite_values():
    for value in (1.01, np.nan):
        bad = np.eye(3)
        bad[0, 0] = value
        with pytest.raises(ValueError, match='rigid'):
            engine.select_frames(2, 0, {'0': np.eye(3), '1': bad})


@pytest.mark.parametrize('crop', [False, True, 'visual_preparation'])
def test_reusable_export_rebuilds_auxiliaries_and_retains_injected_structure(tmp_path, monkeypatch, crop):
    source, output = tmp_path/'source', tmp_path/'result'
    source.mkdir()
    preparation = source/'work'
    preparation.mkdir()
    y, x = np.indices((1024, 1024))
    truth = np.repeat((1500+100*np.sin(x/13)+80*np.cos(y/17)+.5*x)[..., None], 3, axis=-1)
    truth[32, 32] = [-6, -4, -2]
    lunar = (y-511.5)**2+(x-511.5)**2 <= 204**2
    frames = []
    for i, data in enumerate(((truth-20)/1.05, truth)):
        directory = preparation/f'prepared_{i:03d}'
        directory.mkdir()
        (directory/'data.npy').touch()
        metadata = dict(state='linear_rgb', exposure_s=1., white_level=1e6, black_level=0.,
                        black_subtracted=True, response_group='synthetic', lunar_disk=[511.5, 511.5, 200.], low_signal=2.)
        coverage = np.ones(y.shape, bool)
        coverage[:8] = False
        auxiliary = {k: np.zeros(y.shape, bool) for k in ('saturation', 'low_signal', 'defective')}
        auxiliary.update(coverage=coverage, lunar=lunar, soft_weight=np.ones(y.shape))
        frames.append(CachedFrame(data, metadata, Path(str(i)), auxiliary))
    old = dict(inputs=[dict(path=str(i), metadata=dict(source_sha256=str(i))) for i in range(2)],
               preparation_directory=str(preparation), recipe=dict(reference=1, frames=[], calibration={}),
               output=dict(state='linear_rgb', origin_reference_yx=[0, 0], shape_yx=list(y.shape)),
               warnings=[], memory=dict(tile_rows=96), calibration={})
    if crop is True:
        old['output'].update(origin_reference_yx=[-16, -16], shape_yx=[1056, 1056])
        def choose_crop(message, choices, default):
            assert message.startswith('Aesthetic HDR:')
            return choices[0]
        monkeypatch.setattr('coronahdr.decisions.ask', choose_crop)
    if crop == 'visual_preparation':
        for key in ('output', 'warnings', 'memory'):
            old.pop(key)
    (source/'report.json').write_text(json.dumps(old))
    review = tmp_path/'review.json'
    review.write_text(json.dumps(dict(input_source_sha256=['0', '1'], matrices_yx={str(i): np.eye(3).tolist() for i in range(2)},
                                      alignment='ael_visual' if crop == 'visual_preparation' else 'reviewed')))
    monkeypatch.setattr(engine, 'load_cached_run', lambda *args: (frames, old['inputs'], {}))
    temporary = tmp_path/'temporary'/'work'
    result = engine.build_aesthetic_hdr(source, review, output, feather=8, weight_sigma=0, work_directory=temporary)
    actual = tifffile.imread(output/'master.tiff')
    valid = ~lunar
    valid[:8] = False
    np.testing.assert_allclose(actual[valid], truth[valid], rtol=1e-6, atol=.001)
    assert np.isnan(actual[lunar]).all() and np.isnan(actual[:8]).all()
    assert (actual[32, 32] < 0).all()
    assert result['reference'] == 1 and result['accepted_indices'] == [0, 1]
    assert result['output']['origin_reference_yx'] == [0, 0]
    assert result['output']['shape_yx'] == [1024, 1024]
    assert result['output']['canvas'] == ('union' if crop == 'visual_preparation' else 'intersection' if crop else 'fixed_source_run')
    assert 'Rebuilt' in result['auxiliary_provenance']
    assert not (output/'work/master.npy').exists()
    assert Path(result['memory']['work_directory']) == temporary
    assert temporary.is_dir() and not (output/'work').exists()
    with pytest.raises(FileExistsError):
        engine.build_aesthetic_hdr(source, review, output)


def test_cli_routes_aesthetic_to_the_shared_engine(tmp_path, monkeypatch, capsys):
    from coronahdr.cli import main
    import sys
    def run(source, review, output, **kwargs):
        assert kwargs == dict(indices=[0, 1], feather=192, weight_sigma=8, lunar_margin=-2)
        return dict(reference=1, accepted_indices=[0, 1], rejected_frames=[], output=dict(qualification='empirical'))
    monkeypatch.setattr(engine, 'build_aesthetic_hdr', run)
    monkeypatch.setattr(sys, 'argv', ['coronahdr', 'aesthetic', str(tmp_path), '--geometry-review', 'review.json',
                                     '--output', str(tmp_path/'out'), '--indices', '0', '1', '--lunar-margin', '-2'])
    main()
    assert json.loads(capsys.readouterr().out)['accepted'] == 2

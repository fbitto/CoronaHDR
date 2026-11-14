"""Independent constant/affine and linearity controls for Bayer interpolation."""
import numpy as np
import pytest
from coronahdr.sensor import phase_map
from coronahdr.prepare import demosaic_linear, prepare_light
from coronahdr.calibration import build_master, normalize_flat
from test_calibration import sensor


@pytest.mark.parametrize('pattern', ['RGGB', 'BGGR', 'GRBG', 'GBRG'])
@pytest.mark.parametrize('origin', [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_affine_rgb_and_negative_channels(pattern, origin):
    y, x = np.indices((64, 64))
    truth = np.stack([100+2*x+3*y, -200+.2*x+.1*y, 600-.3*x+2*y], axis=-1)
    raw = sensor(np.zeros((64, 64)), cfa_pattern=pattern, black_subtracted=True,
                 active_area_xywh=[*origin, 64, 64])
    labels = phase_map(raw.data.shape, raw.metadata)
    for i, color in enumerate(('R', 'G', 'B')):
        selected = np.char.startswith(labels, color)
        raw.data[selected] = truth[..., i][selected]
    output = prepare_light(raw)
    np.testing.assert_allclose(output.data[1:-1, 1:-1], truth[1:-1, 1:-1], atol=1e-12)
    assert output.data[..., 1].min() < 0
    assert not output.masks()['coverage'][0].any()
    assert output.metadata['cfa_tile_at_active_origin'] == labels[:2, :2].tolist()


def test_demosaic_is_linear():
    rng = np.random.default_rng(321)
    first, second = rng.normal(size=(2, 64, 64))
    metadata = sensor(first).metadata
    np.testing.assert_allclose(demosaic_linear(2*first-3*second, metadata),
        2*demosaic_linear(first, metadata)-3*demosaic_linear(second, metadata), atol=1e-14)


def test_raw_saturation_survives_dark_and_flat_division():
    raw = sensor(np.full((64, 64), 300.))
    raw.data[15, 20] = 10000.
    raw.data[25, 30] = np.nan
    raw.metadata['lunar_disk'] = [32., 32., 7.]
    output = prepare_light(raw)
    masks = output.masks()
    assert masks['saturation'][14:17, 19:22].all()
    assert masks['defective'][24:27, 29:32].all()
    assert masks['lunar'][32, 40]
    assert output.auxiliary['high_weight'][15, 20] == 0


def test_calibration_precedes_demosaic_and_does_not_balance_color():
    raw = sensor(np.zeros((64, 64)), black_level=[100, 110, 130, 150])
    labels = phase_map(raw.data.shape, raw.metadata)
    phases = dict(R=900., G1=600., G2=600., B=300.)
    from coronahdr.sensor import level_map
    response = np.ones((64, 64))
    response[:, 16:40] = .5
    for phase, value in phases.items():
        raw.data[labels == phase] = (value*response)[labels == phase]
    raw.data += level_map(raw, 'black_level')
    flat_samples = sensor(response*1000+level_map(raw, 'black_level'), role='flat', black_level=[100, 110, 130, 150])
    flat = normalize_flat(build_master([flat_samples], 'flat'))
    output = prepare_light(raw, flat=flat)
    np.testing.assert_allclose(output.data[2:-2, 2:-2], np.broadcast_to([900., 600., 300.], (60, 60, 3)))
    assert output.metadata['calibration_steps'][-1].startswith('divide')

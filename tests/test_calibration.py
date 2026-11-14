"""Analytic detector tests verify pedestal algebra and all four CFA phases."""
from pathlib import Path
import numpy as np
import pytest
from coronahdr.sensor import SensorFrame, phase_map, level_map, phase_tile, PHASES
from coronahdr.calibration import build_master, calibrate, normalize_flat, compatible, select_dark


def sensor(data, role='light', **changes):
    metadata = dict(state='linear_cfa', role=role, sensor_id='synthetic sensor', exposure_s=2.,
        iso=100., gain=1., white_level=10000., black_level=100., black_subtracted=False,
        active_area_xywh=[0, 0, data.shape[1], data.shape[0]], cfa_pattern='RGGB',
        temperature_c=12., response_group='synthetic fixed response', lunar_disk=None, low_signal=-100.)
    if role in ('dark', 'dark_flat'):
        metadata['dark_contains_bias'] = True
    metadata.update(changes)
    return SensorFrame(data, metadata, Path(role+'.fits'))


def master(data, role, **changes):
    return build_master([sensor(data, role, **changes)], role)


def test_bias_dark_flat_recovery_and_no_double_bias():
    y, x = np.indices((64, 64))
    bias = 100 + .03*x + .05*y
    dark = 2 + .01*y
    response = .7 + .6*x/63
    light_truth = 200 + .7*y - .3*x
    raw = sensor(light_truth*response+bias+dark)
    bias_master = master(bias, 'bias', exposure_s=0.)
    full_dark = master(bias+dark, 'dark')
    dark_only = master(dark, 'dark', black_subtracted=True, dark_contains_bias=False)
    flat_frame = sensor(2000*response+bias+dark, 'flat')
    dark_flat = master(bias+dark, 'dark_flat')
    flat = normalize_flat(build_master([flat_frame], 'flat'), bias_master, dark_flat)
    labels = phase_map(raw.data.shape, raw.metadata)
    phase_medians = np.zeros(raw.data.shape)
    for phase in PHASES:
        phase_medians[labels == phase] = np.median(response[labels == phase])
    for selected in (full_dark, dark_only):
        result, metadata = calibrate(raw, bias_master, selected, flat)
        np.testing.assert_allclose(result, light_truth*phase_medians, rtol=1e-13, atol=1e-12)
        if selected is full_dark:
            assert any('no separate bias' in step for step in metadata['calibration_steps'])
    assert set(flat.metadata['flat_normalization']) == set(PHASES)


def test_per_phase_black_levels_and_negative_samples():
    raw = sensor(np.zeros((64, 64)), black_level=[100, 110, 120, 130])
    raw.data = level_map(raw, 'black_level') - 7
    result, metadata = calibrate(raw)
    assert np.all(result == -7.)
    assert metadata['calibration_steps'] == ['subtract declared black-level map only; no measured bias master',
                                           'no dark correction', 'no flat correction']


@pytest.mark.parametrize('pattern', ['RGGB', 'BGGR', 'GRBG', 'GBRG'])
@pytest.mark.parametrize('origin', [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_crop_retains_full_sensor_cfa_phase(pattern, origin):
    source = sensor(np.arange(80*80).reshape(80, 80), cfa_pattern=pattern,
                    active_area_xywh=[origin[0], origin[1], 80, 80], lunar_disk=[40., 40., 10.])
    cropped = source.crop(3, 5, 64, 64)
    np.testing.assert_array_equal(phase_map(source.data.shape, source.metadata)[5:69, 3:67],
                                  phase_map(cropped.data.shape, cropped.metadata))
    np.testing.assert_array_equal(source.data[5:69, 3:67], cropped.data)
    assert cropped.metadata['lunar_disk'] == [37., 35., 10.]


def test_incompatible_masters_are_not_scaled_or_rephased():
    source = sensor(np.full((64, 64), 200.))
    for changes, message in [({'exposure_s': 1.}, 'exposure'), ({'temperature_c': 13.}, 'temperature'),
        ({'iso': 200.}, 'iso'), ({'cfa_pattern': 'BGGR'}, 'CFA'),
        ({'active_area_xywh': [1, 0, 64, 64]}, 'active_area')]:
        with pytest.raises(ValueError, match=message):
            calibrate(source, dark=master(np.full((64, 64), 100.), 'dark', **changes))
    with pytest.raises(ValueError, match='temperature'):
        unknown = sensor(np.full((64, 64), 200.))
        unknown.metadata.pop('temperature_c')
        calibrate(unknown, dark=master(np.full((64, 64), 100.), 'dark'))
    with pytest.raises(ValueError, match='sensor_id'):
        calibrate(sensor(np.ones((64, 64, 3)), state='linear_rgb'),
                  bias=master(np.ones((64, 64, 3)), 'bias', state='linear_rgb', exposure_s=0., sensor_id='other'))


def test_pedestal_contradictions_and_dark_selection():
    raw = sensor(np.full((64, 64), 200.), black_subtracted=True)
    with pytest.raises(ValueError, match='twice'):
        calibrate(raw, dark=master(np.full((64, 64), 100.), 'dark'))
    with pytest.raises(ValueError, match='twice'):
        calibrate(raw, bias=master(np.full((64, 64), 100.), 'bias', exposure_s=0.))
    with pytest.raises(ValueError, match='contradicts'):
        sensor(np.ones((64, 64)), role='dark', dark_contains_bias=False)
    dark = master(np.ones((64, 64)), 'dark')
    assert select_dark(sensor(np.ones((64, 64))), [dark]) is dark
    with pytest.raises(ValueError, match='found 0'):
        select_dark(sensor(np.ones((64, 64)), exposure_s=3.), [dark])
    with pytest.raises(ValueError, match='found 2'):
        select_dark(sensor(np.ones((64, 64))), [dark, dark])


def test_flat_invalid_samples_are_not_filled():
    data = np.full((64, 64), 2100.)
    data[10, 10] = 100.
    data[12, 12] = 10000.
    flat = normalize_flat(master(data, 'flat'))
    assert np.isnan(flat.data[10, 10]) and np.isnan(flat.data[12, 12])
    calibrated, _ = calibrate(sensor(np.full((64, 64), 300.)), flat=flat)
    assert np.isnan(calibrated[10, 10]) and np.isnan(calibrated[12, 12])
    assert calibrated[20, 20] == 200.


def test_matching_mono_and_rgb_routes():
    for state, shape in [('linear_mono', (64, 64)), ('linear_rgb', (64, 64, 3))]:
        source = sensor(np.full(shape, 300.), state=state)
        bias = master(np.full(shape, 100.), 'bias', state=state, exposure_s=0.)
        flat = normalize_flat(master(np.full(shape, 2100.), 'flat', state=state), bias)
        result, _ = calibrate(source, bias=bias, flat=flat)
        np.testing.assert_allclose(result, 200.)
    with pytest.raises(ValueError, match='state|shape'):
        calibrate(sensor(np.ones((64, 64)), state='linear_mono'), bias=master(np.ones((64, 64)), 'bias', exposure_s=0.))


def test_flat_optical_configuration_is_checked_without_restricting_bias():
    source = sensor(np.full((64, 64), 300.), aperture=8., focal_length_mm=250.)
    flat = normalize_flat(master(np.full((64, 64), 2100.), 'flat', aperture=5.6, focal_length_mm=250.))
    with pytest.raises(ValueError, match='aperture'):
        calibrate(source, flat=flat)
    bias = master(np.full((64, 64), 100.), 'bias', exposure_s=0., aperture=3.5, focal_length_mm=16.)
    result, _ = calibrate(source, bias=bias)
    assert np.all(result == 200.)

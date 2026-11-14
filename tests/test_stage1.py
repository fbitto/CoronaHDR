"""Acceptance tests use analytic truth and independent spatial regions."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from astropy.io import fits
from coronahdr.io import read_frame, write_image
from coronahdr.model import Frame
from coronahdr.pipeline import combine, run_manifest
from coronahdr.registration import register
from coronahdr.synthetic import make_scene, demo


def metadata(**changes):
    result = dict(state='linear_mono', exposure_s=1., white_level=1000., low_signal=-100.,
                  black_level=64., black_subtracted=True, response_group='fixed', lunar_disk=None)
    result.update(changes)
    return result


def frame(data, **changes):
    return Frame(np.asarray(data), metadata(**changes), Path('memory'))


@pytest.mark.parametrize('extension', ['fits', 'tiff'])
@pytest.mark.parametrize('rgb', [False, True])
def test_float32_roundtrip(tmp_path, extension, rgb):
    rng = np.random.default_rng(19)
    shape = (45, 51, 3) if rgb else (45, 51)
    data = rng.uniform(-123., 20000., shape).astype(np.float32)
    data[0, 0] = np.nan
    m = metadata(state='linear_rgb' if rgb else 'linear_mono')
    path = tmp_path/f'image.{extension}'
    write_image(path, data, m)
    result = read_frame(path)
    np.testing.assert_array_equal(result.data, data)
    if extension == 'fits':
        raw = fits.getdata(path)
        assert raw.dtype.kind == 'f' and raw.dtype.itemsize == 4
        assert raw.shape == ((3, 45, 51) if rgb else (45, 51))
    else:
        import tifffile
        raw = tifffile.imread(path)
        assert raw.dtype == np.float32 and raw.shape == shape
    with pytest.raises(FileExistsError):
        write_image(path, data, m)


def test_missing_and_nonlinear_metadata(tmp_path):
    path = tmp_path/'undeclared.tiff'
    write_image(path, np.ones((40, 40)))
    with pytest.raises(ValueError, match='explicit metadata required'):
        read_frame(path)
    for state in ['nonlinear', 'CFA', 'unknown']:
        with pytest.raises(ValueError, match='linear_mono or linear_rgb'):
            frame(np.ones((40, 40)), state=state)
    with pytest.raises(ValueError, match='black-subtracted'):
        frame(np.ones((40, 40)), black_subtracted=False)


@pytest.mark.parametrize('changes', [dict(exposure_s=0), dict(exposure_s=float('nan')),
    dict(white_level=-1), dict(low_signal=1000), dict(lunar_disk=[1, 2, -1]),
    dict(response_group=''), dict(black_level=float('inf'))])
def test_invalid_metadata(changes):
    with pytest.raises(ValueError):
        frame(np.ones((40, 40)), **changes)


def test_contradiction_and_explicit_override(tmp_path):
    path = tmp_path/'input.fits'
    write_image(path, np.ones((40, 40)), metadata())
    with fits.open(path, mode='update') as hdus:
        hdus[0].header['EXPTIME'] = 2.
    with pytest.raises(ValueError, match='conflicting FITS'):
        read_frame(path)
    resolved = read_frame(path, {'exposure_s': 3.})
    assert resolved.metadata['exposure_s'] == 3.
    assert resolved.metadata['conflicts_resolved_by_overrides'][0]['header'] == 2.
    path = tmp_path/'input.tiff'
    write_image(path, np.ones((40, 40)), metadata())
    corrected = read_frame(path, {'exposure_s': 2.})
    assert corrected.metadata['manual_overrides'] == [dict(field='exposure_s', previous=1., value=2.)]
    path.with_suffix('.tiff.json').write_text(json.dumps({'exposure_s': 3.}))
    with pytest.raises(ValueError, match='conflicting sidecar'):
        read_frame(path)
    resolved = read_frame(path, {'exposure_s': 4.})
    assert resolved.metadata['exposure_s'] == 4.
    assert resolved.metadata['conflicts_resolved_by_overrides'][0]['sidecar'] == 3.


def test_negative_values_and_linear_scale():
    a = np.full((40, 40, 3), [200., -8., 40.])
    master, maps, *_ = combine([frame(a, state='linear_rgb', exposure_s=2.)], 0)
    np.testing.assert_allclose(master, a/2)
    assert master[..., 1].min() == -4
    assert maps['support'].all()


def test_masks_no_hidden_or_defective_contributions():
    a = np.full((64, 64), 100.)
    a[5, 5] = np.nan
    a[8:10, 8:10] = 1000
    a[12, 12] = -200
    master, maps, _, per_frame, _ = combine([frame(a, lunar_disk=[32, 32, 9],
                                defective_pixels_xy=[[4, 4]])])
    invalid = per_frame[0]['lunar'] | per_frame[0]['saturation'] | per_frame[0]['defective'] | per_frame[0]['low_signal']
    assert np.isnan(master[invalid]).all()
    assert not maps['contributions'][invalid].any()
    assert maps['unresolved_saturation'][8:10, 8:10].all()
    assert not maps['unresolved_saturation'][32, 32]
    assert not maps['support'][5, 5] and not maps['support'][4, 4]


def test_bad_registration_and_acquisition_rejected():
    constant = frame(np.full((64, 64), 100.))
    with pytest.raises(ValueError, match='contrast|Unreliable'):
        register(constant, constant)
    with pytest.raises(ValueError, match='response groups'):
        combine([constant, frame(constant.data, response_group='other')])
    with pytest.raises(ValueError, match='Changed iso'):
        combine([constant, frame(constant.data, iso=200)])
    rng = np.random.default_rng(47)
    with pytest.raises(ValueError, match='Unreliable|exceeds'):
        register(frame(rng.uniform(10, 500, (64, 64))), frame(rng.uniform(10, 500, (64, 64))))


def test_clean_subpixel_and_moving_moon(tmp_path):
    shifts = []
    for motion in [False, True]:
        manifest = make_scene(tmp_path/str(motion), noise=0, moon_motion=motion)
        recipe = json.loads(manifest.read_text())
        frames = [read_frame(manifest.parent/e['path']) for e in recipe['frames']]
        _, _, metrics, *_ = combine(frames, 0)
        measured = np.array([e['shift_dy_dx'] for e in metrics])
        truth = np.array(json.loads((manifest.parent/'truth.json').read_text())['expected_shifts_dy_dx'])
        assert np.linalg.norm(measured-truth, axis=1).max() < .1
        shifts.append(measured)
    assert np.linalg.norm(shifts[0]-shifts[1], axis=1).max() < .02


def test_demo_radiance_borders_masks_and_exports(tmp_path):
    result = demo(tmp_path/'demo')
    assert result['max_shift_error_px'] < .1
    assert result['relative_radiance_rmse'] < .02
    assert result['unresolved_saturation_pixels'] > 0
    folder = tmp_path/'demo/output'
    master = fits.getdata(folder/'master.fits')
    support = fits.getdata(folder/'support.fits') > 0
    assert np.array_equal(np.isfinite(master), support)
    sat = fits.getdata(folder/'unresolved_saturation.fits') > 0
    assert np.isnan(master[sat]).all()
    for i in range(4):
        prefix = folder/f'frame_{i:03d}'
        w = fits.getdata(prefix/'weight.fits')
        excluded = fits.getdata(prefix/'coverage.fits') == 0
        for key in ['lunar', 'saturation', 'low_signal', 'defective']:
            excluded |= fits.getdata(prefix/f'{key}.fits') > 0
        assert not w[excluded].any()
    # Moving frames lose reference-canvas borders instead of wrapping data.
    coverage = fits.getdata(folder/'frame_001/coverage.fits')
    assert not coverage[-1].any() and not coverage[:, 0].any()
    import tifffile
    np.testing.assert_array_equal(master, tifffile.imread(folder/'master.tiff'))
    report = json.loads((folder/'report.json').read_text())
    assert len(report['inputs']) == 4 and report['python'] == '3.14.8'
    with pytest.raises(FileExistsError):
        run_manifest(tmp_path/'demo/inputs/manifest.json', folder)


def test_cli_inspect_and_failure(tmp_path):
    path = tmp_path/'input.tiff'
    write_image(path, np.full((40, 40), 7.), metadata())
    result = subprocess.run([sys.executable, '-c', 'from coronahdr import main; main()', 'inspect', str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['shape'] == [40, 40]
    result = subprocess.run([sys.executable, '-c', 'from coronahdr import main; main()', 'inspect', str(tmp_path/'absent.fits')], capture_output=True, text=True)
    assert result.returncode == 2 and 'error:' in result.stderr


def test_fits_history_and_calibrated_scale(tmp_path):
    path = tmp_path/'history.fits'
    write_image(path, np.full((40, 40), 80.), metadata())
    with fits.open(path, mode='update') as hdus:
        hdus[0].header.add_history('Original calibration provenance')
        hdus[0].header.add_history('Second provenance record')
    item = read_frame(path)
    assert 'Second provenance record' in item.metadata['source_header']
    json.dumps(item.metadata, allow_nan=False)
    with pytest.raises(ValueError, match='scale_provenance'):
        frame(np.ones((40, 40)), relative_scale=2.)
    calibrated = frame(np.full((40, 40), 80.), relative_scale=2., scale_provenance='Measured effective exposure in common response seconds')
    master, *_ = combine([calibrated])
    assert np.all(master == 40.)

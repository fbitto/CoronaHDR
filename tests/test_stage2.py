"""Pipeline acceptance and mocked decoder contracts; mocks do not validate RAW files."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import rawpy
from astropy.io import fits
from coronahdr.io import read_frame
from coronahdr.raw import decode_raw
from coronahdr.sensor import SensorFrame
from coronahdr.pipeline import combine, run_manifest
from coronahdr.synthetic import demo, make_cfa_scene
from test_calibration import sensor


class BayerDecoder:
    def __init__(self):
        self.raw_type = rawpy.RawType.Flat
        self.raw_pattern = np.array([[0, 1], [3, 2]])
        self.raw_colors = np.tile(self.raw_pattern, (36, 36))
        self.raw_colors_visible = self.raw_colors[5:69, 3:67]
        self.raw_image_visible = np.full((64, 64), 500, np.uint16)
        self.sizes = SimpleNamespace(pixel_aspect=1, left_margin=3, top_margin=5,
                                     raw_height=72, raw_width=72, flip=6)
        self.color_desc = b'RGBG'
        self.black_level_per_channel = [100, 110, 130, 120]
        self.camera_white_level_per_channel = [4000, 4010, 4030, 4020]
        self.white_level = 4095
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def postprocess(self, *args, **kwargs):
        raise AssertionError('Rendered RAW postprocess must never be called')


def test_experimental_decoder_keeps_phase_levels_and_orientation(monkeypatch, tmp_path):
    decoder = BayerDecoder()
    monkeypatch.setattr(rawpy, 'imread', lambda path: decoder)
    data, metadata = decode_raw('camera.dng')
    assert data.dtype == np.uint16
    assert metadata['black_level'] == [100., 110., 120., 130.]
    assert metadata['white_level'] == [4000., 4010., 4020., 4030.]
    assert metadata['active_area_xywh'] == [3, 5, 64, 64]
    assert metadata['raw_orientation_code'] == 6 and not metadata['orientation_applied']
    assert 'state' not in metadata and decoder.closed
    path = tmp_path/'camera.dng'
    path.write_bytes(b'mocked Bayer decoder contract, not a real RAW file')
    with pytest.raises(ValueError, match='explicit state=linear_cfa'):
        read_frame(path)
    corrections = dict(state='linear_cfa', sensor_id='declared camera', exposure_s=1., iso=100., gain=1.,
                       response_group='fixed', lunar_disk=None, low_signal=1.)
    frame = read_frame(path, corrections)
    assert isinstance(frame, SensorFrame)
    assert 'experimental' in frame.metadata['raw_validation']
    assert len(frame.metadata['manual_overrides']) == len(corrections)
    with pytest.raises(ValueError, match='before HDR'):
        combine([frame])


@pytest.mark.parametrize('variant', ['xtrans', 'colors', 'aspect', 'phase', 'stacked'])
def test_unsupported_raw_layouts_fail(monkeypatch, variant):
    decoder = BayerDecoder()
    if variant == 'xtrans':
        decoder.raw_pattern = np.ones((6, 6), int)
    elif variant == 'colors':
        decoder.color_desc = b'RGBE'
    elif variant == 'aspect':
        decoder.sizes.pixel_aspect = 2
    elif variant == 'phase':
        decoder.raw_colors_visible = np.zeros((64, 64), int)
    else:
        decoder.raw_image_visible = np.zeros((64, 64, 3))
    monkeypatch.setattr(rawpy, 'imread', lambda path: decoder)
    with pytest.raises(ValueError, match='Only|Unsupported|Non-square|phase'):
        decode_raw('camera.dng')
    assert decoder.closed


def test_raw_scalar_white_and_decoder_error(monkeypatch, tmp_path):
    decoder = BayerDecoder()
    decoder.camera_white_level_per_channel = None
    monkeypatch.setattr(rawpy, 'imread', lambda path: decoder)
    _, metadata = decode_raw('camera.dng')
    assert metadata['white_level'] == [4095.]*4
    monkeypatch.undo()
    bad = tmp_path/'invalid.dng'
    bad.write_bytes(b'not a RAW file')
    with pytest.raises(ValueError, match='LibRaw decoding failed'):
        read_frame(bad)


def test_cfa_end_to_end_keeps_masks_distinct_and_relative_color(tmp_path):
    result = demo(tmp_path/'cfa', cfa=True)
    assert result['max_shift_error_px'] < .1 and result['relative_radiance_rmse'] < .02
    assert result['unresolved_saturation_pixels'] > 0
    output = tmp_path/'cfa/output'
    master = np.moveaxis(fits.getdata(output/'master.fits'), 0, -1)
    support = fits.getdata(output/'support.fits') > 0
    unresolved = fits.getdata(output/'unresolved_saturation.fits') > 0
    assert np.isnan(master[unresolved]).all()
    assert np.array_equal(np.isfinite(master).all(axis=-1), support)
    # Detector saturation must not become a detector defect through demosaicing.
    for index in range(4):
        folder = output/f'frame_{index:03d}'
        sat = fits.getdata(folder/'saturation.fits') > 0
        defect = fits.getdata(folder/'defective.fits') > 0
        assert not (sat & defect).any()
        weight = fits.getdata(folder/'weight.fits')
        excluded = fits.getdata(folder/'coverage.fits') == 0
        for key in ('lunar', 'saturation', 'low_signal', 'defective'):
            excluded |= fits.getdata(folder/f'{key}.fits') > 0
        assert not weight[excluded].any()
    report = json.loads((output/'report.json').read_text())
    assert len(report['calibration']['dark']) == 4
    assert len(report['calibration']['bias'][0]['sources']) == 3
    flat = report['calibration']['flat'][0]['metadata']
    assert set(flat['flat_normalization']) == {'R', 'G1', 'G2', 'B'}
    assert all(item['demosaic'] for item in report['corrections'])
    assert (output/'calibration/flat_000.fits').exists()
    with fits.open(output/'calibration/flat_000.fits') as hdus:
        assert hdus[0].header['BUNIT'] == '1'
    import tifffile
    np.testing.assert_array_equal(master, tifffile.imread(output/'master.tiff'))


def test_incomplete_dark_set_is_an_error_not_partial_calibration(tmp_path):
    manifest = make_cfa_scene(tmp_path/'inputs')
    recipe = json.loads(manifest.read_text())
    recipe['calibration']['dark'] = recipe['calibration']['dark'][:2]
    manifest.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='found 0'):
        run_manifest(manifest, tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_detector_crop_is_recorded_and_masters_must_match(tmp_path):
    from coronahdr.io import write_image
    from coronahdr.pipeline import load_item
    frame = sensor(np.full((80, 80), 500.))
    path = tmp_path/'detector.fits'
    write_image(path, frame.data, frame.metadata)
    cropped = load_item(dict(path=path.name, crop_xywh=[1, 3, 64, 64]), tmp_path)
    assert cropped.metadata['active_area_xywh'] == [1, 3, 64, 64]
    assert cropped.metadata['crop_history'] == [[1, 3, 64, 64]]


def test_standard_raw_exif_without_external_dependency(monkeypatch):
    import coronahdr.raw as raw_module
    class Container:
        def __enter__(self):
            self.pages = [SimpleNamespace(tags={
                'Make': SimpleNamespace(value='NIKON CORPORATION'),
                'Model': SimpleNamespace(value='NIKON Z50_2'),
                'ExifTag': SimpleNamespace(value=dict(ExposureTime=(10, 600), FNumber=(800, 100),
                    FocalLength=(2500, 10), ISOSpeedRatings=800, BodySerialNumber='fixture'))})]
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(raw_module.tifffile, 'TiffFile', lambda path: Container())
    metadata = raw_module.camera_exif('camera.NEF')
    assert metadata['exposure_s'] == 1/60
    assert metadata['iso'] == 800 and metadata['gain'] is None
    assert metadata['aperture'] == 8 and metadata['focal_length_mm'] == 250
    assert metadata['sensor_id'] == 'NIKON Z50_2 serial fixture'
    declared = sensor(np.full((64, 64), 200.), gain=None)
    assert declared.metadata['gain'] is None


def test_dataset_lunar_exclusion_fit_is_only_mask_geometry():
    import runpy
    if not (Path(__file__).parents[1]/'scripts/validate_real_raw.py').is_file():
        pytest.skip('Private research script is not included in the public checkout')
    function = runpy.run_path(str(Path(__file__).parents[1]/'scripts/validate_real_raw.py'))['lunar_exclusion']
    y, x = np.indices((1536, 1536))
    radius = np.hypot(x-810., y-745.)
    image = np.where(radius < 275., 1008., 1008.+1000*np.exp(-(radius-275)/120))
    disk, residual = function(image, [0, 0, 1536, 1536])
    np.testing.assert_allclose(disk[:2], [810., 745.], atol=1.)
    assert 280 < disk[2] < 286 and residual < 2.

"""Known-truth controls for the opt-in recovery experiment."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from coronahdr.empirical import warp_rigid
from coronahdr.model import Frame
from coronahdr.rigid import control_points, matrix_distance, matrix_yx

if not (Path(__file__).parents[1]/'scripts/benchmark_alignment.py').is_file():
    pytest.skip('Private research script is not included in the public checkout', allow_module_level=True)


spec = importlib.util.spec_from_file_location('benchmark_alignment', Path(__file__).parents[1]/'scripts/benchmark_alignment.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def frame(a, mask):
    return Frame(a, dict(state='linear_mono', exposure_s=1., white_level=1e6, black_level=0.,
        black_subtracted=True, response_group='synthetic', lunar_disk=None, low_signal=-1e6),
        Path('synthetic'), dict(coverage=mask))


def scene(size=256):
    rng = np.random.default_rng(92)
    y, x = np.indices((size, size), dtype=float)
    radius = np.hypot(y-size/2, x-size/2)
    a = 200*np.exp(-radius/65)+10*gaussian_filter(rng.normal(size=(size, size)), 2)
    return a, y, x


def test_fft_translation_sign_fractional_exposure_and_no_wrap():
    a, y, x = scene()
    mask = np.hypot(y-124, x-123) > 30
    r, rm = warp_rigid(a, mask, [3.35, -5.2], 0)
    r = r*1.7+8
    rm &= np.hypot(y-134, x-140) > 30
    shift, history = benchmark.fft_seed(r, a, rm, mask, [3, -5])
    np.testing.assert_allclose(shift, [3.35, -5.2], atol=.15)
    assert history


@pytest.mark.parametrize('method', benchmark.METHODS)
def test_proposals_require_unfiltered_validation_with_clipping_noise_and_moving_moon(method):
    a, y, x = scene()
    mask = (np.hypot(y-124, x-123) > 30) & (a < 130)
    r, rm = warp_rigid(a, mask, [1.35, -2.2], .65)
    rm &= (np.hypot(y-134, x-140) > 30) & (r < 125)
    r = 1.1*r+2+np.random.default_rng(5).normal(0, .01, a.shape)
    ref, mov = frame(r, rm), frame(a, mask)
    proposal, _ = benchmark.propose(ref, mov, [1, -2], [128, 128], [128, 128], method)
    measured, quality = benchmark.validate_proposal(ref, mov, proposal, 1, [0, 0, 256, 256])
    assert matrix_distance(measured, matrix_yx(a.shape, [1.35, -2.2], .65), control_points(a.shape)) < .25
    assert quality['split_control_error_px'] <= 1


@pytest.mark.parametrize('moon_only', [False, True])
def test_flat_and_moon_dominated_images_do_not_establish_coronal_geometry(moon_only):
    a = np.full((128, 128), 20.)
    y, x = np.indices(a.shape)
    disk = np.hypot(y-64, x-64) <= 30
    if moon_only:
        a[disk] = 0
    mask = ~disk
    for method in benchmark.METHODS[1:]:
        with pytest.raises(ValueError, match='contrast'):
            benchmark.propose(frame(a, mask), frame(a, mask), [0, 0], [64, 64], [64, 64], method)
    with pytest.raises(ValueError):
        benchmark.validate_proposal(frame(a, mask), frame(a, mask), np.eye(3), 1, [0, 0, 128, 128])


def test_native_controls_and_unchanged_photometric_boundaries():
    expected = control_points((64, 64))*4+[901.5, 1801.5]
    np.testing.assert_allclose(benchmark.native_controls((64, 64), 4, [1800, 900, 256, 256]), expected)
    assert benchmark.photometry_pass(dict(samples=64, raw_relative_error=.15, raw_regional_relative_errors=[.2]))
    assert not benchmark.photometry_pass(dict(samples=64, raw_relative_error=.150001, raw_regional_relative_errors=[.2]))
    assert not benchmark.photometry_pass(dict(samples=64, raw_relative_error=.1, raw_regional_relative_errors=[.200001]))


@pytest.mark.parametrize('angle', [.65, -.65])
def test_logpolar_initializer_detects_rotation_without_changing_scale(angle):
    a, _, _ = scene()
    mask = np.ones(a.shape, bool)
    r, rm = warp_rigid(a, mask, [1.35, -2.2], angle)
    estimate = benchmark.logpolar_angle(r, a, rm, mask, [1.35, -2.2])
    assert abs(estimate-angle) < .15


def test_prominence_patch_shift_and_incomplete_support():
    a = np.zeros((128, 128)); y, x = np.indices(a.shape)
    a += np.exp(-((y-64)**2+(x-87)**2)/4)
    m, mask = warp_rigid(a, np.ones(a.shape, bool), [-2, 3], 0)
    rows = benchmark.prominence_matches(a, m, np.ones(a.shape, bool), mask, [2, -3], [64, 64], 20)
    assert rows and max(r['correlation'] for r in rows) > .99
    np.testing.assert_allclose(rows[0]['shift_yx'], [2, -3])
    assert not benchmark.prominence_matches(a, m, np.zeros(a.shape, bool), mask, [2, -3], [64, 64], 20)


def test_native_details_keep_observed_samples_separate_from_hdr_eligibility(tmp_path):
    import json
    import tifffile
    from PIL import Image
    a = np.full((800, 800, 3), 10.)
    metadata = dict(state='linear_rgb', exposure_s=1., white_level=100., black_level=0.,
        black_subtracted=True, response_group='synthetic', lunar_disk=[400, 400, 100],
        low_signal=0., source_sha256='synthetic')
    masks = {k: np.zeros(a.shape[:2], bool) for k in ('lunar', 'saturation', 'low_signal', 'defective')}
    masks['coverage'] = np.ones(a.shape[:2], bool)
    masks['lunar'][400, 400] = True
    f = Frame(a, metadata, Path('synthetic'), masks)
    baseline = tmp_path/'baseline'; baseline.mkdir(); output = tmp_path/'details'; output.mkdir()
    tifffile.imwrite(baseline/'master.tiff', a.astype('float32'), photometric='rgb')
    (baseline/'report.json').write_text(json.dumps(dict(inputs=[dict(metadata=metadata)], reference=0,
                                                     output=dict(origin_reference_yx=[0, 0]))))
    counts, display = benchmark.native_details([f], [0], baseline, output)
    assert counts[0]['native_usable_pixels'] == 800*800-1 and display['native_pixels']
    with Image.open(output/'native-details.png') as masked, Image.open(output/'observed-short-details.png') as observed:
        assert masked.size == observed.size == (1536, 826)
        assert masked.getpixel((768+384, 58+384)) == (48, 48, 48)
        assert observed.getpixel((768+384, 58+384)) != (48, 48, 48)
    np.testing.assert_array_equal(f.data, a)
    empty = tmp_path/'empty-recovery'; empty.mkdir()
    _, display = benchmark.native_details([f], [], baseline, empty, half=128)
    assert len(display['files']) == 2 and display['half_width_native_px'] == 128
    with Image.open(empty/'native-details.png') as image:
        assert image.size == (256, 314)


def test_exposure_controls_span_three_shortest_tiers_and_time():
    from types import SimpleNamespace
    frames = [SimpleNamespace(scale=t, metadata=dict(source_exif=dict(DateTimeOriginal=f'{i:03d}')))
              for i, t in enumerate([.004, .01, .025, .1]*9)]
    controls = benchmark.select_controls(frames, list(range(36)), True)
    assert controls == [0, 16, 32, 1, 17, 33, 2, 18, 34]
    assert benchmark.select_controls(frames, list(range(36))) == list(range(0, 36, 4))
    assert benchmark.select_controls(frames, [], True) == []


def test_rgb_auxiliary_handles_non_divisible_roi_and_matches_mono():
    from coronahdr.streaming import CachedFrame, registration_auxiliary
    rng = np.random.default_rng(5)
    a = rng.normal(20, 1, (160, 180, 3))
    metadata = dict(state='linear_rgb', exposure_s=.1, white_level=100., black_level=0.,
        black_subtracted=True, response_group='synthetic', lunar_disk=None, low_signal=0.)
    f = CachedFrame(a, metadata, Path('synthetic'))
    rgb = benchmark.rgb_auxiliary(f, 4, [3, 5, 133, 135])
    mono = registration_auxiliary(f, 4, [3, 5, 133, 135])
    assert rgb.shape == (33, 33, 3)
    np.testing.assert_allclose(rgb.mean(axis=-1), mono.mono/f.scale)


def test_native_crop_preserves_pixels_at_edges_and_variable_radius():
    a = np.arange(100).reshape(10, 10).astype(float)
    crop = benchmark.native_crop(a, [1, 2], 4)
    assert crop.shape == (8, 8) and np.isnan(crop[:3]).all()
    np.testing.assert_array_equal(crop[3:, 2:], a[:5, :6])
    assert benchmark.native_crop(a, [5, 5], 2).shape == (4, 4)

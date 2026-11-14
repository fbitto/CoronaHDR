import numpy as np
import pytest

from coronahdr.corona_filters import blend_layers, filter_layers


def test_layers_preserve_missing_lunar_and_uncovered_pixels():
    y, x = np.indices((192, 192))
    radius = np.hypot(x-96, y-96)
    image = np.repeat(((1+.1*np.sin(x/7)*np.cos(y/11))/(1+radius))[..., None], 3, axis=-1).astype('f4')
    image[0, 0] = np.nan
    coverage = np.ones((192, 192), dtype='f4')
    coverage[:4] = 0
    valid, layers = filter_layers(image, coverage, (96, 96, 25), (96, 96, 25), 4)
    assert [opacity for _, _, opacity in layers] == [.39, .20, .08]
    assert not valid[96, 96] and not valid[0, 0] and not valid[3, 80]
    base = np.full((192, 192, 3), .5, dtype='f4')
    filtered = blend_layers(base, valid, layers, 1)
    assert np.array_equal(filtered[~valid], base[~valid])
    assert np.array_equal(blend_layers(base, valid, layers, 0), base)
    assert np.isfinite(filtered).all()


def test_rejects_invalid_coverage_and_geometry():
    image = np.ones((8, 8, 3), dtype='f4')
    with pytest.raises(ValueError, match='coverage'):
        filter_layers(image, np.ones((4, 4)), (4, 4, 2), (4, 4, 2), 1)
    with pytest.raises(ValueError, match='margin'):
        filter_layers(image, np.ones((8, 8)), (4, 4, 2), (4, 4, 2), -1)


def test_filter_preview_and_full_export_match_and_keep_source(tmp_path):
    import hashlib
    import tifffile
    from astropy.io import fits
    from coronahdr.desktop_previews import render_preview
    from coronahdr.display_export import analyze, export
    y, x = np.indices((192, 192))
    radius = np.hypot(x-96, y-96)
    data = np.repeat(((1+.1*np.sin(x/7)*np.cos(y/11))/(1+radius))[..., None], 3, axis=-1).astype('f4')
    data[radius < 25] = np.nan
    source = tmp_path/'master.tiff'
    tifffile.imwrite(source, data, photometric='rgb')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    coverage = tmp_path/'coverage.fits'
    fits.PrimaryHDU(np.ones((192, 192), dtype='u1')).writeto(coverage)
    options = dict(coverage=str(coverage), count=1, moon=(96, 96, 25), sun=(96, 96, 25), margin=4, strength=.8)
    levels = dict(exposure=0., black=0., white=1., midtones=.5, saturation=1.)
    preview, _, _, _ = render_preview(source, levels=levels, filter_config=options)
    receipt = analyze(source)
    target = tmp_path/'display.tiff'
    export(source, target, levels, None, False, receipt['identity'], options)
    exported = tifffile.imread(target)
    for y0, x0 in ((5, 5), (30, 95), (150, 120)):
        shown = preview.pixelColor(x0, y0)
        assert np.max(np.abs(np.array([shown.red(), shown.green(), shown.blue()])-np.rint(exported[y0, x0].astype(float)*255/65535))) <= 1
    assert np.all(exported[96, 96] == 0)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def test_context_requires_verified_master_and_lunar_coverage(tmp_path):
    import hashlib
    import json
    import tifffile
    from astropy.io import fits
    from coronahdr.corona_filters import job_filter_context, check_filter_source
    destination = tmp_path/'outputs'
    destination.mkdir()
    master = destination/'test_aesthetic.tiff'
    tifffile.imwrite(master, np.ones((40, 40), dtype='f4'))
    root = tmp_path/'work'/'.coronahdr-jobs'/'test'
    result = root/'result'
    result.mkdir(parents=True)
    fits.PrimaryHDU(np.ones((40, 40), dtype='u1')).writeto(result/'coverage.fits')
    report = dict(reference=0, inputs=[dict(metadata=dict(lunar_disk=[20, 20, 5]))],
                  output=dict(origin_reference_yx=[0, 0]), accepted_indices=[0],
                  transformations=[dict(index=0, matrix_yx=np.eye(3).tolist())])
    (result/'report.json').write_text(json.dumps(report))
    job = dict(destination=str(destination), result=str(result),
               published_files={master.name: hashlib.sha256(master.read_bytes()).hexdigest()})
    (root/'job.json').write_text(json.dumps(job))
    context = job_filter_context(master, tmp_path/'work')
    assert context['moon'] == (20, 20, 5) and context['margin'] == 0
    check_filter_source(master, context)
    master.write_bytes(b'replaced')
    with pytest.raises(ValueError, match='changed'):
        check_filter_source(master, context)
    with pytest.raises(ValueError, match='unavailable'):
        job_filter_context(master, tmp_path/'work')


def test_published_filter_data_survives_job_cleanup(tmp_path):
    import json
    import shutil
    from astropy.io import fits
    from coronahdr.corona_filters import job_filter_context, image_filter_layers
    from coronahdr.desktop_jobs import publish_final_files
    result, destination = tmp_path/'result', tmp_path/'output'
    result.mkdir()
    destination.mkdir()
    y, x = np.indices((192, 192))
    image = ((1 + .1*np.sin(x/7))/(1 + np.hypot(x-96, y-96))).astype('f4')
    fits.PrimaryHDU(image).writeto(result/'master.fits')
    coverage = np.ones((192, 192), dtype='u1')
    coverage[:5] = 0
    fits.PrimaryHDU(coverage).writeto(result/'coverage.fits')
    report = dict(reference=0, inputs=[dict(metadata=dict(lunar_disk=[96, 96, 25]))],
                  output=dict(origin_reference_yx=[0, 0]), accepted_indices=[0],
                  transformations=[dict(index=0, matrix_yx=np.eye(3).tolist())])
    (result/'report.json').write_text(json.dumps(report))
    settings = destination/'test_settings.json'
    settings.write_text('{}')
    jobpath = tmp_path/'job.json'
    job = dict(result=str(result), destination=str(destination), hdr_mode='aesthetic',
               output_base_name='test', output_format='fits', settings_path=str(settings))
    publish_final_files(jobpath, job)
    shutil.rmtree(result)
    master = destination/'test_aesthetic.fits'
    context = job_filter_context(master)
    valid, _ = image_filter_layers(image, context, max_side=96)
    assert not valid[2, 80] and valid[20, 80] and not valid[96, 96]
    master.write_bytes(b'changed')
    with pytest.raises(ValueError):
        job_filter_context(master)


def test_approximate_context_requires_margin(tmp_path, monkeypatch):
    from astropy.io import fits
    from coronahdr.corona_filters import preview_filter_context
    from coronahdr._ael_filters import geometry
    image = tmp_path/'old.fits'
    fits.PrimaryHDU(np.ones((64, 64), dtype='f4')).writeto(image)
    monkeypatch.setattr(geometry, 'find_disc', lambda image: dict(xy=(32, 32), r=10))
    context = preview_filter_context(image)
    assert context['approximate'] and context['coverage'] is None and context['margin'] is None


def test_reduced_filter_converts_fits_byte_order_before_resizing(tmp_path):
    from astropy.io import fits
    from coronahdr.corona_filters import image_filter_layers
    y, x = np.indices((384, 384))
    radius = np.hypot(x-192, y-192)
    values = ((1+.1*np.sin(x/7)*np.cos(y/11))/(1+radius)*100000).astype('f4')
    image = np.repeat(values[..., None], 3, axis=-1)
    image[0, 0] = np.nan
    coverage = tmp_path/'coverage.fits'
    fits.PrimaryHDU(np.ones((384, 384), dtype='i2')).writeto(coverage)
    context = dict(coverage=str(coverage), count=1, moon=(192, 192, 50), sun=(192, 192, 50), margin=4)
    with np.errstate(over='raise', invalid='raise'):
        native_valid, native_layers = image_filter_layers(image, context, max_side=192)
        fits_valid, fits_layers = image_filter_layers(image.astype('>f4'), context, max_side=192)
    assert np.array_equal(native_valid, fits_valid)
    for (native, mode, _), (converted, fits_mode, _) in zip(native_layers, fits_layers):
        assert mode == fits_mode
        np.testing.assert_allclose(native, converted, equal_nan=True)


def test_lunar_transition_protects_edge_and_is_crop_consistent():
    from coronahdr.corona_filters import lunar_blend_transition
    context = dict(moon=(50., 50., 20.), margin=5.)
    whole = lunar_blend_transition(context, (110, 110))
    assert whole[50, 75] == 0
    assert 0 < whole[50, 79] < 1
    assert whole[50, 83] == 1
    np.testing.assert_array_equal(whole[30:60, 60:95], lunar_blend_transition(context, (30, 35), (30, 60)))
    base = np.ones((110, 110, 3), dtype='f8')*.800000031
    valid = np.ones((110, 110), dtype=bool)
    layers = [(np.full((110, 110), .25, dtype='f4'), 'multiply', .5)]
    filtered = blend_layers(base, valid, layers, 1., whole)
    assert np.array_equal(filtered[50, 75], base[50, 75])
    assert filtered[50, 75, 0] > filtered[50, 79, 0] > filtered[50, 83, 0]
    assert filtered[50, 83, 0] == pytest.approx(.5)


def test_filtered_cropped_export_matches_full_export_region(tmp_path):
    import tifffile
    from astropy.io import fits
    from coronahdr.display_export import analyze, export
    y, x = np.indices((192, 192))
    radius = np.hypot(x-96, y-96)
    image = np.repeat(((1+.1*np.sin(x/7)*np.cos(y/11))/(1+radius))[..., None], 3, axis=-1).astype('f4')
    image[radius < 25] = np.nan
    source = tmp_path/'source.tiff'
    tifffile.imwrite(source, image, photometric='rgb')
    coverage = tmp_path/'coverage.fits'
    fits.PrimaryHDU(np.ones((192, 192), dtype='u1')).writeto(coverage)
    options = dict(coverage=str(coverage), count=1, moon=(96, 96, 25), sun=(96, 96, 25), margin=4, strength=1.)
    levels = dict(exposure=0., black=0., white=1., midtones=.5, saturation=1.)
    full, cropped = tmp_path/'full.tiff', tmp_path/'cropped.tiff'
    crop = (50, 55, 90, 80)
    export(source, full, levels, None, False, analyze(source)['identity'], options)
    export(source, cropped, levels, None, False, analyze(source, crop)['identity'], options, crop)
    np.testing.assert_array_equal(tifffile.imread(cropped), tifffile.imread(full)[55:135, 50:140])

import hashlib
import json

import numpy as np
import pytest
import tifffile
from astropy.io import fits
from PySide6.QtWidgets import QApplication

from coronahdr.crop_export import save_crop
from coronahdr.desktop_previews import HDRPreview, render_preview
from coronahdr.display_export import analyze, export


def source_master(tmp_path):
    data = np.arange(20*30*3, dtype='f4').reshape(20, 30, 3)
    data[6, 5] = np.nan
    metadata = dict(origin_reference_yx=[100, 200], shape_yx=[20, 30], reference=4, state='linear_rgb', channel_order='RGB')
    header = fits.Header()
    header['CHDRMETA'] = json.dumps(metadata)
    header['CRPIX1'] = 12.
    header['CRPIX2'] = 10.
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    path = tmp_path/'master.fits'
    fits.PrimaryHDU(np.moveaxis(data, -1, 0), header).writeto(path)
    return path, data


def test_linear_crop_formats_coordinates_and_no_overwrite(tmp_path):
    path, data = source_master(tmp_path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    targets = [tmp_path/'crop.fits', tmp_path/'crop.tiff']
    save_crop(path, targets, (3, 4, 12, 10))
    with fits.open(targets[0]) as hdus:
        actual = np.moveaxis(hdus[0].data, 0, -1)
        meta = json.loads(hdus[0].header['CHDRMETA'])
        assert hdus[0].header['CRPIX1'] == 9 and hdus[0].header['CRPIX2'] == 6
    with tifffile.TiffFile(targets[1]) as tif:
        tiff = tif.asarray()
        tiffmeta = json.loads(tif.pages[0].description)['coronahdr']
    assert np.array_equal(actual, data[4:14, 3:15], equal_nan=True)
    assert np.array_equal(tiff, actual, equal_nan=True)
    assert meta == tiffmeta and meta['origin_reference_yx'] == [104, 203] and meta['shape_yx'] == [10, 12]
    assert meta['crop_source']['sha256'] == before
    with pytest.raises(FileExistsError):
        save_crop(path, targets, (3, 4, 12, 10))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_crop_invalid_or_missing_metadata(tmp_path):
    path, _ = source_master(tmp_path)
    with pytest.raises(ValueError, match='inside'):
        save_crop(path, [tmp_path/'bad.fits'], (25, 0, 10, 3))
    plain = tmp_path/'plain.tiff'
    tifffile.imwrite(plain, np.zeros((10, 10), 'f4'))
    with pytest.raises(ValueError, match='metadata'):
        save_crop(plain, [tmp_path/'bad.tiff'], (0, 0, 3, 3))


def test_cropped_stretched_export_and_preview_controls(tmp_path):
    app = QApplication.instance() or QApplication([])
    path, _ = source_master(tmp_path)
    crop = (3, 4, 12, 10)
    levels = dict(exposure=0., black=0., white=1., midtones=.5, saturation=1.)
    receipt = analyze(path, crop)
    assert receipt['missing'] == 1
    output = tmp_path/'stretched.tiff'
    export(path, output, levels, None, False, receipt['identity'], None, crop)
    assert tifffile.imread(output).shape == (10, 12, 3)
    image, _, _, _ = render_preview(path, levels=levels)
    widget = HDRPreview()
    widget.images['corona'] = image
    widget.begin_crop()
    widget.crop_drawn(crop)
    widget.apply_crop()
    assert widget.crop == crop and widget.labels['corona'].image.size().width() == 12
    widget.reset_crop()
    assert widget.crop is None and widget.labels['corona'].image.width() == 30
    widget.close()
    app.processEvents()


def test_crop_mouse_selection_and_corner_resize():
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest
    app = QApplication.instance() or QApplication([])
    widget = HDRPreview()
    widget.resize(1100, 700)
    image = QImage(200, 100, QImage.Format.Format_RGB888)
    image.fill(0)
    widget.images['corona'] = image
    widget.show()
    widget.begin_crop()
    app.processEvents()
    canvas = widget.labels['corona']
    start = QPoint(canvas.width()//4, canvas.height()//4)
    end = QPoint(canvas.width()//2, canvas.height()//2)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end)
    assert abs(widget.crop_fields['x'].value()-50) <= 1
    assert abs(widget.crop_fields['width'].value()-50) <= 1
    corner = QPoint(round(100*canvas.width()/200), round(50*canvas.height()/100))
    end = QPoint(round(150*canvas.width()/200), round(75*canvas.height()/100))
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=corner)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end)
    assert abs(widget.crop_fields['width'].value()-100) <= 2
    widget.close()


def test_reopened_crop_finds_verified_source_coverage(tmp_path):
    from coronahdr.corona_filters import job_filter_context
    path, _ = source_master(tmp_path)
    work = tmp_path/'work'
    jobroot = work/'.coronahdr-jobs'/'job'
    result = jobroot/'result'
    result.mkdir(parents=True)
    fits.PrimaryHDU(np.ones((20, 30), dtype='u1')).writeto(result/'coverage.fits')
    report = dict(reference=0, inputs=[dict(metadata=dict(lunar_disk=[210, 110, 5]))],
                  output=dict(origin_reference_yx=[100, 200]), accepted_indices=[0],
                  transformations=[dict(index=0, matrix_yx=np.eye(3).tolist())])
    (result/'report.json').write_text(json.dumps(report))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    job = dict(destination=str(tmp_path), result=str(result), published_files={path.name: digest})
    (jobroot/'job.json').write_text(json.dumps(job))
    target = tmp_path/'crop.tiff'
    save_crop(path, [target], (3, 4, 12, 10))
    context = job_filter_context(target, work)
    assert context['coverage_origin_xy'] == (3, 4)
    assert context['moon'] == (7, 6, 5)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        job_filter_context(target, work)


def test_crop_refuses_unhandled_coordinate_metadata(tmp_path):
    path, _ = source_master(tmp_path)
    with fits.open(path, mode='update') as hdus:
        hdus[0].header['LTV1'] = 3
    with pytest.raises(ValueError, match='distortions'):
        save_crop(path, [tmp_path/'cropped.fits'], (0, 0, 10, 10))
    assert not (tmp_path/'cropped.fits').exists()


def test_crop_a_cropped_tiff_retains_coordinate_history(tmp_path):
    path, data = source_master(tmp_path)
    first, second = tmp_path/'first.tiff', tmp_path/'second.fits'
    save_crop(path, [first], (3, 4, 12, 10))
    save_crop(first, [second], (2, 1, 5, 4))
    with fits.open(second) as hdus:
        meta = json.loads(hdus[0].header['CHDRMETA'])
        assert hdus[0].header['CRPIX1'] == 7 and hdus[0].header['CRPIX2'] == 5
        assert meta['origin_reference_yx'] == [105, 205]
        assert np.array_equal(np.moveaxis(hdus[0].data, 0, -1), data[5:9, 5:10], equal_nan=True)


def test_crop_rolls_back_if_second_publication_fails(tmp_path, monkeypatch):
    import coronahdr.crop_export as module
    path, _ = source_master(tmp_path)
    targets = [tmp_path/'crop.fits', tmp_path/'crop.tiff']
    original = module.os.link
    def link(source, dest):
        if dest == targets[1]:
            raise FileExistsError('collision')
        return original(source, dest)
    monkeypatch.setattr(module.os, 'link', link)
    with pytest.raises(FileExistsError):
        save_crop(path, targets, (0, 0, 5, 5))
    assert not targets[0].exists() and not targets[1].exists()

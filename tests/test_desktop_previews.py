import json

import numpy as np
import pytest
import tifffile
from astropy.io import fits
from PySide6.QtCore import QThread
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from coronahdr.desktop_previews import HDRPreview, hdr_display, render_preview


def test_filter_geometry_dialog_explains_every_control(monkeypatch, tmp_path):
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QDoubleSpinBox
    app = QApplication.instance() or QApplication([])
    widget = HDRPreview()
    path = tmp_path/'image.fits'
    widget.paths['corona'] = path
    widget.filter_checkbox.blockSignals(True)
    widget.filter_checkbox.setChecked(True)
    widget.filter_checkbox.blockSignals(False)
    task = type('Task', (), {})()
    task.source_path = path
    task.setup_generation = widget.filter_setup_generation
    def inspect(dialog):
        form = dialog.layout()
        fields = dialog.findChildren(QDoubleSpinBox)
        assert len(fields) == 7
        assert all(field.toolTip() and form.labelForField(field).toolTip() for field in fields)
        assert fields[-1].value() == 0
        buttons = dialog.findChild(QDialogButtonBox)
        assert all(button.toolTip() for button in buttons.buttons())
        return QDialog.DialogCode.Rejected
    monkeypatch.setattr(QDialog, 'exec', inspect)
    context = dict(moon=(20, 20, 10), sun=(20, 20, 10), margin=None, approximate=True)
    widget.filter_context_ready(task, context, None)
    widget.close()


def test_native_fits_and_tiff_preview_preserve_source_and_missing(tmp_path):
    image = np.ones((40, 50, 3), np.float32)
    image[..., 0] = np.arange(50) + 1
    image[..., 1] = np.arange(40)[:, None] + 1
    image[..., 2] = image[..., 0] + image[..., 1]
    image[0, 0] = np.nan
    metadata = dict(origin_reference_yx=[2, 3], reference=1)
    tif = tmp_path/'image.tiff'
    tifffile.imwrite(tif, image, photometric='rgb', metadata={'coronahdr': metadata})
    header = fits.Header()
    header['CHDRMETA'] = json.dumps(metadata)
    fit = tmp_path/'image.fits'
    fits.PrimaryHDU(np.moveaxis(image, -1, 0), header).writeto(fit)
    before = tif.read_bytes()
    a, shape, geometry, gains = render_preview(tif)
    b, _, _, _ = render_preview(fit)
    assert a.size() == b.size() and shape == (40, 50) and geometry == ((2, 3), 1)
    assert a.pixelColor(0, 0).red() == 48 and gains is not None
    assert tif.read_bytes() == before
    neutral, _, _, _ = render_preview(fit, 10, gains)
    assert neutral.size() == a.size()
    assert neutral.pixelColor(15, 15) != a.pixelColor(15, 15)


def test_monochrome_preview_disables_color_gains(tmp_path):
    path = tmp_path/'mono.fits'
    fits.PrimaryHDU(np.arange(2500, dtype=np.float32).reshape(50, 50)).writeto(path)
    image, shape, geometry, gains = render_preview(path)
    assert image.size().width() == 50 and shape == (50, 50)
    assert geometry is None and gains is None


def test_hdr_levels_adjust_exposure_and_preserve_invalid_pixels():
    source = np.array([[[4., 4., 4.], [4., 1., .5], [np.nan, np.nan, np.nan]]])
    original = source.copy()
    levels = dict(exposure=0., black=0., white=1., midtones=.5, saturation=1.)
    full, valid = hdr_display(source, 4., [1., 1., 1.], levels)
    dim, _ = hdr_display(source, 4., [1., 1., 1.], dict(levels, exposure=-2.))
    assert full[0, 1, 0] > dim[0, 1, 0] > 0
    assert not valid[0, 2] and np.array_equal(source, original, equal_nan=True)


def test_single_preview_has_adjustable_levels(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = HDRPreview()
    assert list(widget.views) == ['corona']
    assert set(widget.level_controls) == {'black', 'white', 'midtones', 'saturation'}
    image = np.ones((80, 80, 3), np.float32)
    image[..., 0] = 4
    image[30:50, 30:50] = np.nan
    path = tmp_path/'prominence.tiff'
    tifffile.imwrite(path, image, photometric='rgb')
    levels = dict(exposure=0., black=0., white=1., midtones=.5, saturation=1.)
    unmasked, _, _, _ = render_preview(path, 0, None, levels)
    dim, _, _, _ = render_preview(path, 0, None, dict(levels, exposure=-2.))
    assert unmasked.pixelColor(10, 10).red() > dim.pixelColor(10, 10).red()
    assert unmasked.pixelColor(35, 35).red() == 48
    widget.close()


def test_single_preview_zoom(monkeypatch):
    app = QApplication.instance() or QApplication([])
    widget = HDRPreview()
    widget.resize(500, 500)
    widget.show()
    image = QImage(1200, 900, QImage.Format.Format_RGB888)
    image.fill(0)
    widget.images = dict(corona=image)
    widget.geometry = dict(corona=((900, 1200), ((0, 0), 0)))
    widget.set_zoom(1.)
    app.processEvents()
    a = widget.views['corona']
    a.horizontalScrollBar().setValue(300)
    a.verticalScrollBar().setValue(200)
    widget.set_zoom(.5)
    assert widget.labels['corona'].width() == round(600/widget.devicePixelRatioF())
    widget.set_zoom('fit')
    app.processEvents()
    assert widget.labels['corona'].width() <= a.viewport().width()
    widget.close()


def test_render_tasks_discard_stale_results(monkeypatch, tmp_path):
    import coronahdr.desktop_previews as module
    app = QApplication.instance() or QApplication([])
    widget = HDRPreview()
    threads = []
    def render(path, stretch, gains, levels=None):
        threads.append(QThread.currentThread())
        image = QImage(20, 20, QImage.Format.Format_RGB888)
        image.fill(0)
        return image, (20, 20), ((0, 0), 0), [1, 1, 1]
    monkeypatch.setattr(module, 'render_preview', render)
    widget.set_path('corona', tmp_path/'a.tiff')
    widget.set_path('corona', tmp_path/'b.tiff')
    for _ in range(100):
        app.processEvents()
        if not widget.tasks:
            break
        QTest.qWait(10)
    assert threads and all(thread != widget.thread() for thread in threads)
    assert not widget.tasks and 'b.tiff' in widget.titles['corona'].toolTip()
    widget.close()


def test_export_prompts_before_interpolating(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    import cv2
    import coronahdr.desktop_previews as module
    app = QApplication.instance() or QApplication([])
    data = np.ones((40, 50, 3), dtype=np.float32)
    data[3, 3] = np.nan
    source, destination = tmp_path/'source.tiff', tmp_path/'display.png'
    tifffile.imwrite(source, data, photometric='rgb')
    widget = HDRPreview()
    widget.set_path('corona', source)
    for _ in range(200):
        app.processEvents()
        if not widget.tasks:
            break
        QTest.qWait(10)
    asked = []
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args, **kwargs: (str(destination), 'PNG image (*.png)'))
    monkeypatch.setattr(module, 'ask_choice', lambda *args: (asked.append(args[2]) or 'Interpolate small gaps', True))
    widget.choose_export()
    for _ in range(300):
        app.processEvents()
        if not widget.export_tasks:
            break
        QTest.qWait(10)
    assert asked and destination.is_file()
    assert cv2.imread(str(tmp_path/'display_validity.png'), cv2.IMREAD_UNCHANGED)[3, 3] == 127
    widget.close()


def test_filter_controls_config_and_native_zoom():
    app = QApplication.instance() or QApplication([])
    widget = HDRPreview()
    widget.filter_checkbox.blockSignals(True)
    widget.filter_checkbox.setChecked(True)
    widget.filter_checkbox.blockSignals(False)
    widget.filter_context = dict(coverage='coverage.fits', count=1, moon=(10, 10, 5),
                                 sun=(10, 10, 5), margin=2)
    widget.filter_strength.setValue(35)
    assert widget.filter_options()['strength'] == .35
    widget.set_zoom(1.)
    assert 'filters are hidden' in widget.notice.text()
    widget.filter_checkbox.setChecked(False)
    assert widget.filter_options() is None
    widget.close()


def test_open_standalone_tiff_and_unsigned_fits_without_settings(tmp_path):
    from coronahdr.display_export import analyze, export
    data = np.arange(20*30*3, dtype=np.uint16).reshape(20, 30, 3)
    data[0, 0] = 0
    tiff = tmp_path/'standalone.tiff'
    tifffile.imwrite(tiff, data, photometric='rgb', compression='deflate', description='Created by external software', metadata=None)
    fit = tmp_path/'standalone.fits'
    fits.PrimaryHDU(np.moveaxis(data, -1, 0)).writeto(fit)
    levels = dict(exposure=0., black=0., white=1., midtones=.5, saturation=1.)
    for source in (tiff, fit):
        before = source.read_bytes()
        image, shape, geometry, _ = render_preview(source, levels=levels)
        assert image.width() == 30 and shape == (20, 30) and geometry is None
        receipt = analyze(source)
        output = tmp_path/(source.stem+'_'+source.suffix[1:]+'.tiff')
        export(source, output, levels, None, False, receipt['identity'])
        assert tifffile.imread(output).shape == (20, 30, 3)
        assert source.read_bytes() == before


def test_preview_sections_can_be_expanded():
    app = QApplication.instance() or QApplication([])
    widget = HDRPreview()
    widget.show()
    app.processEvents()
    assert all(not toggle.isChecked() for toggle in widget.preview_sections.values())
    widget.preview_sections['crop'].click()
    assert widget.preview_sections['crop'].isChecked()
    assert widget.select_crop_button.isVisible()
    widget.preview_sections['crop'].click()
    assert not widget.select_crop_button.isVisible()
    widget.close()


def test_background_preview_can_finish_after_window_is_closed(monkeypatch, tmp_path):
    import shiboken6
    import coronahdr.desktop_previews as module
    app = QApplication.instance() or QApplication([])
    task = module.PreviewTask('corona', tmp_path/'image.tiff', 0, None, 1)
    monkeypatch.setattr(module, 'render_preview', lambda *args: ('image', (1, 1), None, None))
    shiboken6.delete(task.signals)
    task.run()
    setup = module.ExportTask(lambda: 'done')
    shiboken6.delete(setup.signals)
    setup.run()
    app.processEvents()


@pytest.mark.parametrize('zoom', [.5, 1., 2.])
def test_drag_pans_zoomed_preview_and_respects_edges(zoom):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    app = QApplication.instance() or QApplication([])
    widget = HDRPreview()
    widget.resize(500, 450)
    widget.show()
    image = QImage(2400, 1800, QImage.Format.Format_RGB888)
    image.fill(0)
    widget.images['corona'] = image
    widget.set_zoom(zoom)
    app.processEvents()
    view, canvas = widget.views['corona'], widget.labels['corona']
    horizontal, vertical = view.horizontalScrollBar(), view.verticalScrollBar()
    horizontal.setValue(min(200, horizontal.maximum()//2))
    vertical.setValue(min(200, vertical.maximum()//2))
    before = (horizontal.value(), vertical.value())
    start = QPoint(horizontal.value()+100, vertical.value()+100)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(canvas, start-QPoint(40, 30))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=start-QPoint(40, 30))
    assert horizontal.value() == min(horizontal.maximum(), before[0]+40)
    assert vertical.value() == min(vertical.maximum(), before[1]+30)
    assert canvas.pan_start is None
    widget.close()


def test_filters_hidden_at_100_and_200_but_present_at_50(monkeypatch, tmp_path):
    import coronahdr.desktop_previews as module
    app = QApplication.instance() or QApplication([])
    widget = HDRPreview()
    widget.paths['corona'] = tmp_path/'image.fits'
    widget.filter_checkbox.blockSignals(True)
    widget.filter_checkbox.setChecked(True)
    widget.filter_checkbox.blockSignals(False)
    widget.filter_context = dict(coverage='coverage.fits', count=1, moon=(10, 10, 5), sun=(10, 10, 5), margin=2)
    tasks = []
    class Pool:
        def start(self, task):
            tasks.append(task)
    monkeypatch.setattr(module.QThreadPool, 'globalInstance', lambda: Pool())
    widget.set_zoom(2.)
    assert tasks[-1].filter_config is None and '100% and 200%' in widget.notice.text()
    widget.tasks.clear()
    widget.set_zoom(1.)
    assert len(tasks) == 1
    widget.set_zoom(.5)
    assert tasks[-1].filter_config is not None
    widget.tasks.clear()
    widget.close()


def test_zoom_selection_persists_after_focus_moves():
    from coronahdr.desktop_style import STYLE, apply_theme
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    widget = HDRPreview()
    widget.show()
    app.processEvents()
    assert widget.zoom_buttons['fit'].isChecked()
    widget.zoom_buttons[2.].click()
    widget.preview_sections['stretch'].click()
    app.processEvents()
    assert widget.zoom == 2. and widget.zoom_buttons[2.].isChecked()
    assert sum(button.isChecked() for button in widget.zoom_buttons.values()) == 1
    assert 'border: 1px solid #e5a00d' in STYLE
    widget.set_zoom(.5)
    assert widget.zoom_buttons[.5].isChecked() and not widget.zoom_buttons[2.].isChecked()
    widget.close()

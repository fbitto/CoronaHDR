"""Inspection-only display and nonlinear export of an HDR master."""
import json
from pathlib import Path

import numpy as np
import tifffile
from astropy.io import fits
from PySide6.QtCore import QObject, QPoint, QRect, QRunnable, QThreadPool, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QFileDialog, QSlider, QVBoxLayout, QWidget, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QSpinBox, QToolButton, QSizePolicy)
from .desktop_style import label_tooltips, scroll_tooltips
from .desktop_paths import ask_choice


def hdr_display(data, high, gains, levels):
    """Apply the same inspection-only tone curve to preview and export samples."""
    valid = np.isfinite(data).all(axis=-1)
    values = np.where(valid[..., None], np.maximum(data, 0.), 0.)
    exposure = 2.**levels['exposure']
    values = np.arcsinh(values*np.asarray(gains)*exposure/(high/100))/np.arcsinh(100.)
    values = np.clip((values-levels['black'])/(levels['white']-levels['black']), 0., 1.)
    values = values**(np.log(.5)/np.log(levels['midtones']))
    luminance = values @ np.array([.299, .587, .114])
    values = luminance[..., None] + levels['saturation']*(values-luminance[..., None])
    return np.clip(values, 0., 1.), valid


def render_preview(path, stretch=0, gains=None, levels=None, filter_config=None):
    """Render native samples without modifying the FITS/TIFF source."""
    path = Path(path)
    if filter_config:
        from .corona_filters import check_filter_source
        check_filter_source(path, filter_config)
    hdus = None
    try:
        if path.suffix.lower() in ('.fits', '.fit', '.fts'):
            hdus = fits.open(path, memmap=True)
            try:
                hdu = next(h for h in hdus if h.data is not None and h.data.ndim >= 2)
            except ValueError as error:
                if 'memory-mapped' not in str(error):
                    raise
                hdus.close()
                hdus = fits.open(path, memmap=False)
                hdu = next(h for h in hdus if h.data is not None and h.data.ndim >= 2)
            array = hdu.data
            try:
                metadata = json.loads(hdu.header.get('CHDRMETA', '{}'))
            except (TypeError, ValueError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            if array.ndim == 3 and array.shape[0] == 3:
                array = np.moveaxis(array, 0, -1)
        else:
            with tifffile.TiffFile(path) as tif:
                try:
                    description = json.loads(tif.pages[0].description or '{}')
                    metadata = description.get('coronahdr', {}) if isinstance(description, dict) else {}
                    if not isinstance(metadata, dict):
                        metadata = {}
                except (TypeError, ValueError):
                    metadata = {}
            try:
                array = tifffile.memmap(path)
            except ValueError:
                array = tifffile.imread(path)
        if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[-1] != 3):
            raise ValueError('Preview requires mono or RGB image data')
        height, width = array.shape[:2]
        sample = np.asarray(array[::10, ::10], dtype=np.float64)
        if sample.ndim == 2:
            sample = np.repeat(sample[..., None], 3, axis=-1)
        good = np.isfinite(sample).all(axis=-1)
        if not good.any():
            raise ValueError('Image has no finite sampled observations')
        derived = None
        if array.ndim == 3:
            maximum = -np.inf
            for start in range(0, height, 128):
                block = np.asarray(array[start:start+128], dtype=np.float64)
                finite = np.isfinite(block).all(axis=-1)
                maximum = max(maximum, float(np.max(block, where=finite[..., None], initial=-np.inf)))
            white = maximum * (1 + 1e-6)
            selected = sample[good & (sample > 0).all(axis=-1) & (sample < white).all(axis=-1)] if white > 0 else np.empty((0, 3))
            if len(selected) >= 2:
                span = np.diff(np.percentile(selected, [1, 99], axis=0), axis=0)[0]
                if np.all(span > np.finfo(float).eps):
                    derived = (span.max()/span).tolist()
        effective = np.asarray(gains if gains is not None else (1., 1., 1.))
        sampled = sample[good]*effective
        high = max(float(np.percentile(sampled, 99.8)), 1e-6)
        soft = high/100 * 2**(-stretch/10)
        filter_data = None
        if filter_config:
            from .corona_filters import image_filter_layers
            filter_data = image_filter_layers(array, filter_config, max_side=1000)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        for start in range(0, height, 128):
            data = np.asarray(array[start:start+128], dtype=np.float64)
            if data.ndim == 2:
                data = np.repeat(data[..., None], 3, axis=-1)
            valid = np.isfinite(data).all(axis=-1)
            if levels is not None:
                display, valid = hdr_display(data, high, effective, levels)
            else:
                display = np.arcsinh(np.maximum(np.nan_to_num(data), 0)*effective/soft)/np.arcsinh(high/soft)
            if filter_data:
                from .corona_filters import blend_layers, lunar_blend_transition
                mask, layers = filter_data
                rows = slice(start, start+len(data))
                transition = lunar_blend_transition(filter_config, valid.shape, (start, 0))
                display = blend_layers(display, mask[rows] & valid, [(a[rows], mode, opacity) for a, mode, opacity in layers], filter_config['strength'], transition)
            rgb[start:start+len(data)] = np.rint(np.clip(display, 0, 1)*255).astype(np.uint8)
            rgb[start:start+len(data)][~valid] = 48
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()
        geometry = (tuple(metadata['origin_reference_yx']), metadata['reference']) if 'origin_reference_yx' in metadata and 'reference' in metadata else None
        return image, (height, width), geometry, derived
    finally:
        if hdus is not None:
            hdus.close()


class PreviewSignals(QObject):
    done = Signal(object, object, object)


class PreviewTask(QRunnable):
    def __init__(self, role, path, stretch, gains, generation, levels=None, filter_config=None):
        super().__init__()
        self.setAutoDelete(False)
        self.role, self.path, self.stretch, self.gains, self.generation = role, path, stretch, gains, generation
        self.levels = levels
        self.filter_config = filter_config
        try:
            stat = Path(path).stat()
            self.file_identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        except OSError:
            self.file_identity = None
        self.signals = PreviewSignals()

    def run(self):
        try:
            result = render_preview(self.path, self.stretch, self.gains, self.levels, self.filter_config) if self.filter_config else render_preview(self.path, self.stretch, self.gains, self.levels)
            error = None
        except Exception as exception:
            result, error = None, str(exception)
        try:
            self.signals.done.emit(self, result, error)
        except RuntimeError:
            # The preview may finish after its window has closed.
            pass


class ExportTask(QRunnable):
    def __init__(self, function, *args):
        super().__init__()
        self.setAutoDelete(False)
        self.function, self.args = function, args
        self.signals = PreviewSignals()

    def run(self):
        try:
            result, error = self.function(*self.args), None
        except Exception as exception:
            result, error = None, str(exception)
        try:
            self.signals.done.emit(self, result, error)
        except RuntimeError:
            # Filter setup can outlive a closed window.
            pass


class ImageCanvas(QLabel):
    cropSelected = Signal(object)

    def __init__(self, scroll_area=None):
        super().__init__('Waiting for image')
        self.scroll_area = scroll_area
        self.pan_enabled = False
        self.pan_start = None
        self.image = None
        self.selecting_crop = False
        self.drag_start = None
        self.drag_end = None
        self.selection_rect = None

    def set_image(self, image, zoom):
        self.image = image
        self.resize(max(1, round(image.width()*zoom)), max(1, round(image.height()*zoom)))
        self.update()

    def clear_image(self, text):
        self.image = None
        self.setText(text)
        self.update()

    def paintEvent(self, event):
        if self.image is None:
            return super().paintEvent(event)
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image)
        if self.selecting_crop:
            painter.setPen(QPen(QColor('#ffb000'), 2, Qt.PenStyle.DashLine))
            if self.drag_start is not None and self.drag_end is not None:
                painter.drawRect(QRect(self.drag_start, self.drag_end).normalized())
            elif self.selection_rect:
                x, y, width, height = self.selection_rect
                painter.drawRect(QRect(round(x*self.width()/self.image.width()), round(y*self.height()/self.image.height()),
                                       round(width*self.width()/self.image.width()), round(height*self.height()/self.image.height())))

    def mousePressEvent(self, event):
        if self.selecting_crop and self.image is not None and event.button() == Qt.MouseButton.LeftButton:
            point = event.position().toPoint()
            self.drag_start = point
            if self.selection_rect:
                x, y, width, height = self.selection_rect
                left, top = round(x*self.width()/self.image.width()), round(y*self.height()/self.image.height())
                right, bottom = round((x+width)*self.width()/self.image.width()), round((y+height)*self.height()/self.image.height())
                for corner, opposite in ((QPoint(left, top), QPoint(right, bottom)), (QPoint(right, top), QPoint(left, bottom)),
                                         (QPoint(left, bottom), QPoint(right, top)), (QPoint(right, bottom), QPoint(left, top))):
                    if (point-corner).manhattanLength() <= 14:
                        self.drag_start = opposite
                        break
            self.drag_end = point
            self.update()
        elif (self.pan_enabled and self.image is not None and self.scroll_area is not None
              and event.button() == Qt.MouseButton.LeftButton):
            self.pan_start = (event.globalPosition().toPoint(),
                              self.scroll_area.horizontalScrollBar().value(),
                              self.scroll_area.verticalScrollBar().value())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.selecting_crop and self.drag_start is not None:
            self.drag_end = event.position().toPoint()
            self.update()
        elif self.pan_start is not None and self.scroll_area is not None:
            origin, horizontal, vertical = self.pan_start
            delta = event.globalPosition().toPoint() - origin
            self.scroll_area.horizontalScrollBar().setValue(horizontal-delta.x())
            self.scroll_area.verticalScrollBar().setValue(vertical-delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.selecting_crop and self.drag_start is not None and event.button() == Qt.MouseButton.LeftButton:
            rect = QRect(self.drag_start, event.position().toPoint()).normalized()
            x0 = max(0, min(self.image.width()-1, int(rect.left()*self.image.width()/self.width())))
            y0 = max(0, min(self.image.height()-1, int(rect.top()*self.image.height()/self.height())))
            x1 = max(x0+1, min(self.image.width(), round((rect.right()+1)*self.image.width()/self.width())))
            y1 = max(y0+1, min(self.image.height(), round((rect.bottom()+1)*self.image.height()/self.height())))
            self.cropSelected.emit((x0, y0, x1-x0, y1-y0))
            self.drag_end = event.position().toPoint()
            self.drag_start = None
            self.update()
        elif self.pan_start is not None and event.button() == Qt.MouseButton.LeftButton:
            self.pan_start = None
            self.unsetCursor()
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class HDRPreview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.paths = {'corona': None}
        self.images, self.geometry, self.tasks = {}, {}, set()
        self.generations = {'corona': 0}
        self.gains = None
        self.neutral_error = None
        self.zoom = 'fit'
        self.rendered_zoom = None
        self.syncing = False
        self.export_tasks = set()
        self.export_directory = None
        self.export_base = 'CoronaHDR'
        self.filter_context = None
        self.filter_temporary = None
        self.filter_setup_tasks = set()
        self.filter_setup_generation = 0
        self.crop = None
        self.preview_identity = None
        layout = QVBoxLayout(self)
        def section(title):
            toggle = QToolButton()
            toggle.setText(title.upper())
            toggle.setProperty('sectionHeader', True)
            toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            toggle.setToolTip(f'Show or hide {title.lower()} controls.')
            toggle.setCheckable(True)
            toggle.setArrowType(Qt.ArrowType.RightArrow)
            toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            body = QWidget()
            body_layout = QVBoxLayout(body)
            body_layout.setContentsMargins(4, 0, 4, 4)
            body_layout.setSpacing(4)
            body.hide()
            toggle.toggled.connect(body.setVisible)
            toggle.toggled.connect(lambda checked: toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow))
            return toggle, body, body_layout
        stretch_toggle, stretch_body, stretch_layout = section('Stretch')
        color_toggle, color_body, color_layout = section('Color')
        filter_toggle, filter_body, filter_layout = section('AEL corona filters')
        crop_toggle, crop_body, crop_layout = section('Crop')
        self.preview_sections = dict(stretch=stretch_toggle, color=color_toggle, filters=filter_toggle, crop=crop_toggle)
        controls = QHBoxLayout()
        view_label = QLabel('View:')
        view_label.setToolTip('Preview magnification: Fit, 50%, 100% or 200% of native pixels.')
        controls.addWidget(view_label)
        self.zoom_group = QButtonGroup(self)
        self.zoom_group.setExclusive(True)
        self.zoom_buttons = {}
        for label, mode in [('Fit', 'fit'), ('50%', .5), ('100%', 1.), ('200%', 2.)]:
            button = QPushButton(label)
            button.setCheckable(True)
            button.setProperty('zoomOption', True)
            self.zoom_group.addButton(button)
            self.zoom_buttons[mode] = button
            button.setChecked(mode == 'fit')
            button.setToolTip(f'Show the HDR preview at {label} zoom.')
            button.clicked.connect(lambda checked=False, value=mode: self.set_zoom(value))
            button.setMaximumWidth(75)
            controls.addWidget(button)
        self.neutral = QCheckBox('Color neutralization')
        self.neutral.setToolTip('Apply corona-derived channel gains to the preview and nonlinear export. This can change prominence color and is not photometric calibration; the linear master is unchanged.')
        self.neutral.toggled.connect(self.refresh)
        controls.addStretch()
        layout.addLayout(controls)
        color_layout.addWidget(self.neutral)
        label_width = max(QLabel(text).fontMetrics().horizontalAdvance(text) for text in ('Exposure (EV)', 'Black point', 'White point', 'Midtones', 'Saturation', 'AEL corona filters')) + 24
        value_width = QLabel('-10.00').fontMetrics().horizontalAdvance('-10.00') + 8
        self.adjustment_labels = {}
        filter_row = QHBoxLayout()
        self.filter_checkbox = QCheckBox('AEL corona filters')
        self.filter_checkbox.setToolTip('Quick AEL-based approximation at Fit and 50%. Use Eclipse Image Processor or AEL for refined processing.')
        self.filter_checkbox.toggled.connect(self.filter_toggled)
        self.filter_checkbox.setFixedWidth(label_width)
        filter_row.addWidget(self.filter_checkbox)
        self.filter_strength = QSlider(Qt.Orientation.Horizontal)
        self.filter_strength.setRange(0, 100)
        self.filter_strength.setValue(100)
        self.filter_strength.setToolTip('Strength of the combined AEL corona filters in the preview and nonlinear export.')
        self.filter_strength.valueChanged.connect(lambda _: self.refresh())
        filter_row.addWidget(self.filter_strength, 1)
        self.filter_value = QLabel('100%')
        self.filter_value.setFixedWidth(value_width)
        self.filter_value.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.filter_value.setToolTip('Current corona filter strength.')
        self.filter_strength.valueChanged.connect(lambda value: self.filter_value.setText(f'{value}%'))
        filter_row.addWidget(self.filter_value)
        filter_layout.addLayout(filter_row)
        self.views, self.labels, self.sliders, self.titles = {}, {}, {}, {}
        for role, title in [('corona', 'HDR')]:
            row = QHBoxLayout()
            heading = QLabel(f'{title} — not available')
            controls.addWidget(heading)
            stretch_label = QLabel('Exposure (EV)')
            stretch_label.setToolTip('Exposure in stops for the preview and nonlinear export. The linear master is unchanged.')
            stretch_label.setFixedWidth(label_width)
            stretch_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.adjustment_labels['exposure'] = stretch_label
            row.addWidget(stretch_label)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-50, 50)
            slider.setValue(0)
            slider.setToolTip(f'Change {title.lower()} exposure for the preview and nonlinear export; the linear master is unchanged.')
            slider.valueChanged.connect(lambda value, name=role: self.render(name))
            row.addWidget(slider, 1)
            exposure_value = QLabel('0.0')
            exposure_value.setFixedWidth(value_width)
            exposure_value.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            exposure_value.setToolTip('Current HDR preview exposure in stops.')
            slider.valueChanged.connect(lambda value: exposure_value.setText(f'{value/10:.1f}'))
            row.addWidget(exposure_value)
            stretch_layout.addLayout(row)
            self.level_controls = {}
            for key, title_text, minimum, maximum, value in (
                ('black', 'Black point', 0, 99, 0), ('white', 'White point', 1, 200, 100),
                ('midtones', 'Midtones', 10, 90, 50), ('saturation', 'Saturation', 0, 200, 100)):
                level_row = QHBoxLayout()
                level_label = QLabel(title_text)
                level_label.setFixedWidth(label_width)
                level_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self.adjustment_labels[key] = level_label
                level_row.addWidget(level_label)
                level = QSlider(Qt.Orientation.Horizontal)
                level.setRange(minimum, maximum)
                level.setValue(value)
                detail = {'black': 'Values below this normalized level become black.',
                          'white': 'Values above this normalized level become white.',
                          'midtones': 'Maps this normalized input level to middle gray.',
                          'saturation': 'Zero is monochrome; 1.0 preserves the current color strength.'}[key]
                level.setToolTip(f'{detail} Affects the preview and nonlinear export; the linear master stays unchanged.')
                number = QLabel(f'{value/100:.2f}')
                number.setFixedWidth(value_width)
                number.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                number.setToolTip(f'Current {title_text.lower()} value.')
                level.valueChanged.connect(lambda v, label=number: label.setText(f'{v/100:.2f}'))
                level.valueChanged.connect(lambda _: self.level_changed())
                level_row.addWidget(level, 1)
                level_row.addWidget(number)
                (color_layout if key == 'saturation' else stretch_layout).addLayout(level_row)
                self.level_controls[key] = level
            view = QScrollArea()
            view.setObjectName('hdrImageView')
            view.viewport().setStyleSheet('background: #111111;')
            view.setToolTip('Scroll or drag the zoomed HDR preview to inspect different image areas. In crop mode, dragging selects the crop.')
            scroll_tooltips(view, f'the {title.lower()} preview')
            view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            view.setWidgetResizable(False)
            view.setMinimumSize(150, 100)
            label = ImageCanvas(view)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setObjectName('inspectionCanvas')
            view.setWidget(label)
            label.cropSelected.connect(self.crop_drawn)
            layout.addWidget(view, 1)
            self.views[role], self.labels[role], self.sliders[role], self.titles[role] = view, label, slider, heading
        crop_row = QHBoxLayout()
        crop_fields_row = QGridLayout()
        self.select_crop_button = QPushButton('Select crop')
        self.select_crop_button.setToolTip('Draw a crop rectangle on the full image and adjust its pixel coordinates.')
        self.select_crop_button.setEnabled(False)
        self.select_crop_button.clicked.connect(self.begin_crop)
        crop_row.addWidget(self.select_crop_button)
        self.crop_fields = {}
        for index, (key, title) in enumerate((('x', 'X'), ('y', 'Y'), ('width', 'Width'), ('height', 'Height'))):
            row, column = divmod(index, 2)
            crop_fields_row.addWidget(QLabel(title), row, column*2)
            field = QSpinBox()
            field.setRange(0 if key in ('x', 'y') else 1, 1000000)
            field.setEnabled(False)
            field.setToolTip(f'{title} of the crop in native image pixels.')
            field.lineEdit().setToolTip(field.toolTip())
            crop_fields_row.addWidget(field, row, column*2+1)
            self.crop_fields[key] = field
            field.valueChanged.connect(self.crop_fields_changed)
        self.apply_crop_button = QPushButton('Apply crop')
        self.apply_crop_button.setToolTip('Show the selected area; this does not change the source master.')
        self.apply_crop_button.setEnabled(False)
        self.apply_crop_button.clicked.connect(self.apply_crop)
        crop_row.addWidget(self.apply_crop_button)
        self.reset_crop_button = QPushButton('Reset crop')
        self.reset_crop_button.setToolTip('Return the preview and stretched export to the full image.')
        self.reset_crop_button.setEnabled(False)
        self.reset_crop_button.clicked.connect(self.reset_crop)
        crop_row.addWidget(self.reset_crop_button)
        crop_layout.addLayout(crop_row)
        crop_layout.addLayout(crop_fields_row)
        self.save_crop_button = QPushButton('Save cropped HDR…')
        self.save_crop_button.setToolTip('Save the selected crop as a separate linear FITS and/or TIFF master.')
        self.save_crop_button.clicked.connect(self.choose_crop_export)
        self.save_crop_button.setEnabled(False)
        crop_layout.addWidget(self.save_crop_button)
        self.export_button = QPushButton('Save stretched image…')
        self.export_button.setToolTip('Save a 16-bit nonlinear version of the displayed HDR, without changing the linear master.')
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.choose_export)
        section_view = QScrollArea()
        section_view.setWidgetResizable(True)
        section_view.setMinimumHeight(150)
        section_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        section_view.setToolTip('Scroll through preview adjustment and crop controls.')
        scroll_tooltips(section_view, 'the preview adjustment controls')
        section_content = QWidget()
        section_layout = QVBoxLayout(section_content)
        section_layout.setContentsMargins(0, 0, 0, 0)
        section_layout.setSpacing(3)
        for toggle, body in ((stretch_toggle, stretch_body), (color_toggle, color_body),
                             (filter_toggle, filter_body), (crop_toggle, crop_body)):
            section_layout.addWidget(toggle)
            section_layout.addWidget(body)
        section_layout.addStretch()
        section_view.setWidget(section_content)
        self.adjustment_panel = QWidget()
        self.adjustment_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        panel_layout = QVBoxLayout(self.adjustment_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(6)
        panel_layout.addWidget(section_view, 1)
        panel_layout.addWidget(self.export_button)
        self.notice = QLabel('Preview and nonlinear export only; missing pixels are gray in the preview.')
        self.notice.setWordWrap(True)
        panel_layout.addWidget(self.notice)
        layout.addWidget(self.adjustment_panel)
        label_tooltips(self)

    def set_path(self, role, path):
        if role != 'corona':
            return
        self.filter_checkbox.setChecked(False)
        self.filter_context = None
        self.crop = None
        self.preview_identity = None
        self.labels[role].selecting_crop = False
        self.save_crop_button.setEnabled(False)
        self.select_crop_button.setEnabled(False)
        self.apply_crop_button.setEnabled(False)
        self.reset_crop_button.setEnabled(False)
        for field in self.crop_fields.values():
            field.setEnabled(False)
        self.paths[role] = Path(path) if path else None
        self.generations[role] += 1
        self.images.pop(role, None)
        self.export_button.setEnabled(False)
        self.geometry.pop(role, None)
        if role == 'corona':
            self.gains = None
            self.neutral_error = None
        if not path:
            self.labels[role].clear_image('No image available')
            self.titles[role].setText(role.title()+' — not available')
            return
        self.labels[role].clear_image('Loading preview…')
        self.titles[role].setText(role.title()+' — loading…')
        self.render(role)

    def render(self, role):
        path = self.paths.get(role)
        if not path:
            return
        self.export_button.setEnabled(False)
        self.generations[role] += 1
        gains = self.gains if self.neutral.isChecked() else None
        if any(task.role == role for task in self.tasks):
            return
        task = PreviewTask(role, path, self.sliders[role].value(), gains, self.generations[role], levels=self.levels(), filter_config=self.filter_options() if self.zoom not in (1., 2.) else None)
        self.tasks.add(task)
        task.signals.done.connect(self.finished)
        QThreadPool.globalInstance().start(task)

    def filter_options(self):
        if not self.filter_checkbox.isChecked() or self.filter_context is None:
            return None
        return dict(self.filter_context, strength=self.filter_strength.value()/100)

    def filter_toggled(self, enabled):
        self.filter_setup_generation += 1
        if not enabled:
            self.export_button.setEnabled(bool(self.images.get('corona')) and not self.export_tasks)
            self.refresh()
            return
        if not self.paths.get('corona'):
            self.filter_checkbox.setChecked(False)
            self.notice.setText('Load an HDR image before enabling AEL corona filters.')
            return
        from .corona_filters import preview_filter_context
        self.export_button.setEnabled(False)
        self.notice.setText('Finding the matching coverage and lunar measurements…')
        task = ExportTask(preview_filter_context, self.paths['corona'], self.filter_temporary)
        task.source_path = self.paths['corona']
        task.setup_generation = self.filter_setup_generation
        self.filter_setup_tasks.add(task)
        task.signals.done.connect(self.filter_context_ready)
        QThreadPool.globalInstance().start(task)

    def filter_context_ready(self, task, context, error):
        self.filter_setup_tasks.discard(task)
        if (task.source_path != self.paths.get('corona') or task.setup_generation != self.filter_setup_generation
                or not self.filter_checkbox.isChecked()):
            return
        if error:
            self.filter_checkbox.setChecked(False)
            self.notice.setText(f'AEL corona filters unavailable: {error}')
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Review approximate filter geometry')
        form = QFormLayout(dialog)
        warning = ('The original coverage and lunar motion are unavailable. This approximation treats finite pixels as usable; geometric coverage is unverified. Review the estimated lunar edge and motion margin. ' if context.get('approximate') else '')
        description = QLabel(warning + 'The solar position is approximated from the Moon. These filters are for visual inspection; use Eclipse Image Processor or AEL for refined processing.')
        description.setWordWrap(True)
        form.addRow(description)
        boxes = {}
        values = [('moon_x', 'Moon center X', context['moon'][0]), ('moon_y', 'Moon center Y', context['moon'][1]),
                  ('moon_r', 'Moon radius', context['moon'][2]), ('sun_x', 'Sun center X (approximate)', context['sun'][0]),
                  ('sun_y', 'Sun center Y (approximate)', context['sun'][1]), ('sun_r', 'Sun radius (approximate)', context['sun'][2]),
                  ('margin', 'Lunar motion margin (pixels)', context['margin'] if context['margin'] is not None else 0.)]
        tips = {
            'moon_x': 'Horizontal position of the lunar center in native image pixels. X increases to the right; the first pixel center is 0.',
            'moon_y': 'Vertical position of the lunar center in native image pixels. Y increases downward; the first pixel center is 0.',
            'moon_r': 'Radius of the measured or estimated lunar edge in native image pixels. Review it against the image before applying the filters.',
            'sun_x': 'Approximate horizontal solar center in native image pixels, initially taken from the Moon. X increases to the right; this is not a measured solar center.',
            'sun_y': 'Approximate vertical solar center in native image pixels, initially taken from the Moon. Y increases downward; this is not a measured solar center.',
            'sun_r': 'Approximate solar radius in native image pixels, initially taken from the lunar radius. This is not a measured solar radius.',
            'margin': 'Extra protection outside the lunar edge, in native pixels, for possible lunar motion between frames. Zero adds no extra margin; increase it if the Moon moved during the sequence. Review the edge and result carefully when motion is unknown.',
        }
        for key, title, value in values:
            box = QDoubleSpinBox()
            box.setDecimals(2)
            box.setRange(.01 if key.endswith('_r') else (0. if key == 'margin' else -100000.), 100000.)
            box.setToolTip(tips[key])
            box.setValue(value)
            form.addRow(title, box)
            form.labelForField(box).setToolTip(tips[key])
            boxes[key] = box
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setToolTip('Use these values for the filter preview and nonlinear export. The linear master is unchanged.')
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setToolTip('Close without enabling the AEL corona filters.')
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if not dialog.exec():
            self.filter_checkbox.setChecked(False)
            return
        context['moon'] = tuple(boxes['moon_'+v].value() for v in ('x', 'y', 'r'))
        context['sun'] = tuple(boxes['sun_'+v].value() for v in ('x', 'y', 'r'))
        context['margin'] = boxes['margin'].value()
        self.filter_context = context
        self.refresh()

    def refresh(self):
        self.render('corona')

    def levels(self):
        return dict(exposure=self.sliders['corona'].value()/10,
                    **{name: slider.value()/100 for name, slider in self.level_controls.items()})

    def level_changed(self):
        black, white = self.level_controls['black'], self.level_controls['white']
        if black.value() >= white.value():
            if self.sender() is black:
                white.setValue(black.value()+1)
            else:
                black.setValue(white.value()-1)
        self.render('corona')

    def begin_crop(self):
        if 'corona' not in self.images:
            return
        self.crop = None
        self.save_crop_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.labels['corona'].selecting_crop = True
        self.apply_crop_button.setEnabled(True)
        self.reset_crop_button.setEnabled(True)
        for field in self.crop_fields.values():
            field.setEnabled(True)
        self.labels['corona'].selection_rect = None
        self.update_views()
        self.notice.setText('Drag a rectangle on the image, adjust its pixel coordinates if needed, then select Apply crop.')

    def crop_drawn(self, crop):
        for name, value in zip(('x', 'y', 'width', 'height'), crop):
            self.crop_fields[name].setValue(value)
        self.notice.setText('Adjust the crop coordinates if needed, then select Apply crop.')

    def crop_fields_changed(self):
        if len(self.crop_fields) == 4 and self.labels['corona'].selecting_crop:
            self.labels['corona'].selection_rect = tuple(self.crop_fields[name].value() for name in ('x', 'y', 'width', 'height'))
            self.labels['corona'].update()

    def apply_crop(self):
        if 'corona' not in self.images:
            return
        from .crop_export import validate_crop
        crop = tuple(self.crop_fields[name].value() for name in ('x', 'y', 'width', 'height'))
        image = self.images['corona']
        try:
            validate_crop(crop, (image.height(), image.width()))
        except ValueError as error:
            self.notice.setText(str(error))
            return
        self.crop = crop
        self.labels['corona'].selecting_crop = False
        self.apply_crop_button.setEnabled(False)
        self.reset_crop_button.setEnabled(True)
        for field in self.crop_fields.values():
            field.setEnabled(False)
        self.save_crop_button.setEnabled(not self.export_tasks)
        self.export_button.setEnabled(not self.export_tasks and not (self.filter_checkbox.isChecked() and self.filter_context is None))
        self.update_views()

    def reset_crop(self):
        self.crop = None
        self.labels['corona'].selecting_crop = False
        self.apply_crop_button.setEnabled(False)
        self.reset_crop_button.setEnabled(False)
        for field in self.crop_fields.values():
            field.setEnabled(False)
        self.save_crop_button.setEnabled(False)
        self.export_button.setEnabled(bool(self.images.get('corona')) and not self.export_tasks and not (self.filter_checkbox.isChecked() and self.filter_context is None))
        self.update_views()

    def choose_crop_export(self):
        path = self.paths.get('corona')
        if not path or self.crop is None or self.export_tasks:
            return
        directory = self.export_directory or path.parent
        name, selected = QFileDialog.getSaveFileName(self, 'Save cropped linear HDR',
            str(Path(directory)/f'{self.export_base}_cropped'),
            'FITS and TIFF (*.fits *.tiff);;FITS image (*.fits);;TIFF image (*.tiff)',
            options=QFileDialog.Option.DontConfirmOverwrite)
        if not name:
            return
        base = Path(name)
        if base.suffix.lower() in ('.fits', '.fit', '.fts', '.tif', '.tiff'):
            base = base.with_suffix('')
        extensions = ('.fits', '.tiff') if selected.startswith('FITS and TIFF') else ('.tiff',) if selected.startswith('TIFF') else ('.fits',)
        destinations = [Path(str(base)+extension) for extension in extensions]
        if any(dest.exists() or dest.is_symlink() for dest in destinations):
            self.notice.setText('A cropped output already exists; choose a different name.')
            return
        from .crop_export import save_crop
        self.save_crop_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.notice.setText('Saving cropped linear HDR…')
        task = ExportTask(save_crop, path, destinations, self.crop, self.preview_identity)
        self.export_tasks.add(task)
        task.signals.done.connect(self.crop_export_finished)
        QThreadPool.globalInstance().start(task)

    def crop_export_finished(self, task, result, error):
        self.export_tasks.discard(task)
        self.save_crop_button.setEnabled(self.crop is not None and bool(self.images.get('corona')))
        self.export_button.setEnabled(bool(self.images.get('corona')))
        self.notice.setText(f'Crop save failed: {error}' if error else 'Saved cropped HDR: '+', '.join(result))

    def choose_export(self):
        path = self.paths.get('corona')
        if path is None or self.export_tasks or (self.filter_checkbox.isChecked() and self.filter_context is None):
            return
        directory = self.export_directory or path.parent
        initial = str(Path(directory)/f'{self.export_base}_prominence_stretched')
        name, selected = QFileDialog.getSaveFileName(self, 'Save nonlinear 16-bit image', initial,
            'TIFF image (*.tiff *.tif);;PNG image (*.png);;FITS image (*.fits *.fit *.fts)',
            options=QFileDialog.Option.DontConfirmOverwrite)
        if not name:
            return
        destination = Path(name)
        if not destination.suffix:
            extension = '.png' if selected.startswith('PNG') else '.fits' if selected.startswith('FITS') else '.tiff'
            destination = destination.with_suffix(extension)
        mask = destination.with_name(destination.stem+'_validity.png')
        if destination.exists() or destination.is_symlink() or mask.exists() or mask.is_symlink():
            self.notice.setText('Output image or validity mask already exists; choose a different name.')
            return
        from .display_export import analyze
        self.export_button.setEnabled(False)
        self.notice.setText('Checking missing pixels before export…')
        task = ExportTask(analyze, path, self.crop)
        task.destination, task.source, task.levels = destination, path, self.levels()
        task.crop = self.crop
        task.filter_config = self.filter_options()
        task.gains = list(self.gains) if self.neutral.isChecked() and self.gains is not None else None
        self.export_tasks.add(task)
        task.signals.done.connect(self.export_checked)
        QThreadPool.globalInstance().start(task)

    def export_checked(self, task, result, error):
        self.export_tasks.discard(task)
        if error:
            self.notice.setText(f'Export check failed: {error}')
            self.export_button.setEnabled(bool(self.images.get('corona')))
            return
        repair = False
        if result['eligible']:
            answer, accepted = ask_choice(self, 'Interpolate missing pixels?',
                f"The image has {result['missing']} missing pixels. {result['eligible']} are in small enclosed gaps and can be estimated from nearby samples. Larger gaps, including the lunar disk, remain missing. Interpolated pixels are identified in the validity mask.",
                ['Leave gaps black', 'Interpolate small gaps'])
            if not accepted:
                self.notice.setText('Export cancelled.')
                self.export_button.setEnabled(bool(self.images.get('corona')))
                return
            repair = answer == 'Interpolate small gaps'
        elif result['missing']:
            answer, accepted = ask_choice(self, 'Missing pixels in export',
                f"The image has {result['missing']} missing pixels, but none are small enclosed gaps. They will remain black and be marked missing in the validity mask; the lunar disk is not interpolated.",
                ['Export with missing pixels black', 'Cancel'])
            if not accepted or answer == 'Cancel':
                self.notice.setText('Export cancelled.')
                self.export_button.setEnabled(bool(self.images.get('corona')))
                return
        from .display_export import export
        self.notice.setText('Saving nonlinear image and validity mask…')
        worker = ExportTask(export, task.source, task.destination, task.levels, task.gains, repair, result['identity'], task.filter_config, task.crop)
        self.export_tasks.add(worker)
        worker.signals.done.connect(self.export_finished)
        QThreadPool.globalInstance().start(worker)

    def export_finished(self, task, result, error):
        self.export_tasks.discard(task)
        self.export_button.setEnabled(bool(self.images.get('corona')))
        if error:
            self.notice.setText(f'Export failed: {error}')
        else:
            self.notice.setText(f"Saved {result['image']} and validity mask; {result['interpolated']} interpolated, {result['missing']} missing pixels.")

    def finished(self, task, result, error):
        self.tasks.discard(task)
        role = task.role
        if task.generation != self.generations[role]:
            if self.paths[role]:
                self.render(role)
            return
        if error:
            self.labels[role].clear_image(f'Preview unavailable: {error}')
            self.images.pop(role, None)
            return
        image, shape, geometry, gains = result
        self.preview_identity = task.file_identity
        self.images[role] = image
        self.select_crop_button.setEnabled(True)
        self.export_button.setEnabled(not self.export_tasks and not self.labels[role].selecting_crop and not (self.filter_checkbox.isChecked() and self.filter_context is None))
        self.geometry[role] = (shape, geometry)
        self.titles[role].setText('HDR')
        self.titles[role].setToolTip(str(task.path))
        if role == 'corona':
            previous = self.gains
            self.gains = gains
            if self.neutral.isChecked() and gains is None:
                self.neutral.setChecked(False)
                self.neutral_error = 'Color neutralization unavailable: insufficient valid RGB variation.'
            elif self.neutral.isChecked() and gains != previous:
                if task.gains is None:
                    self.render('corona')
        self.update_views()

    def set_zoom(self, value):
        old = self.zoom
        self.zoom = value
        self.zoom_buttons[value].setChecked(True)
        self.labels['corona'].pan_enabled = value != 'fit'
        self.update_views()
        if self.filter_checkbox.isChecked() and (old in (1., 2.)) != (value in (1., 2.)):
            self.refresh()

    def update_views(self):
        if self.neutral_error:
            self.notice.setText(self.neutral_error)
        else:
            self.notice.setText('At 100% and 200%, filters are hidden to show native detail. Nonlinear export includes enabled filters.' if self.filter_checkbox.isChecked() and self.zoom in (1., 2.) else 'Quick inspection only; for refined results use Eclipse Image Processor or Agentic Eclipse Library. Missing pixels are gray.')
        if not self.images:
            return
        image = next(iter(self.images.values()))
        if self.crop is not None:
            image = image.copy(*self.crop)
        center = None
        if self.rendered_zoom and 'corona' in self.images and self.labels['corona'].image is not None:
            view, label = self.views['corona'], self.labels['corona']
            center = tuple((bar.value() + span/2 - max(0, (span-size)/2))/self.rendered_zoom
                           for bar, span, size in ((view.horizontalScrollBar(), view.viewport().width(), label.width()),
                                                   (view.verticalScrollBar(), view.viewport().height(), label.height())))
        zoom = self.zoom
        if zoom == 'fit':
            zoom = min(min(self.views[role].viewport().width()/image.width(), self.views[role].viewport().height()/image.height()) for role in self.images)
        else:
            zoom /= self.devicePixelRatioF()
        self.syncing = True
        for role, original in self.images.items():
            image = original.copy(*self.crop) if self.crop is not None else original
            self.labels[role].set_image(image, zoom)
        self.rendered_zoom = zoom
        if center is not None:
            for role in self.images:
                view, label = self.views[role], self.labels[role]
                for bar, span, size, coordinate in ((view.horizontalScrollBar(), view.viewport().width(), label.width(), center[0]),
                                                     (view.verticalScrollBar(), view.viewport().height(), label.height(), center[1])):
                    bar.setValue(round(coordinate*zoom + max(0, (span-size)/2) - span/2))
        self.syncing = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.zoom == 'fit':
            self.update_views()

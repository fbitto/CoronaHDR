"""Desktop front end for existing prepared CoronaHDR runs, using the CLI engine."""
import json
import codecs
from html import escape
from importlib import metadata
import math
from pathlib import Path
import re
import sys

import numpy as np
import tifffile
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QRunnable, QThreadPool, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFontDatabase, QImage, QTextCursor
from PySide6.QtWidgets import (QAbstractButton, QApplication, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPlainTextEdit, QProgressBar, QPushButton,
    QSizePolicy, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QTextBrowser, QMenu, QVBoxLayout, QWidget)

from .desktop_paths import ask_choice, ask_text, choose_directory, choose_output_folder
from .desktop_jobs import create_job, set_job_status, is_published
from .desktop_cleanup import CleanupDialog, remember_location
from .desktop_previews import HDRPreview
from .desktop_style import apply_theme, heading, label_tooltips, path_tooltip, scroll_tooltips, ScrollSafeComboBox, ScrollSafeSpinBox
from .work_paths import default_work_directory


def review_alignment_frames(parent, rows, *, conservative=False):
    dialog = QDialog(parent)
    dialog.setWindowTitle('Review frame alignment')
    dialog.resize(1000, 550)
    layout = QVBoxLayout(dialog)
    note = QLabel('Review preparation results and choose qualified frames. The reference is required. '
        'Rejected frames cannot be selected. Scientific stacking may apply additional radiometric checks.' if conservative else
        'Review alignment results. Candidate frames are excluded unless checked. '
        'Including them relaxes the displayed alignment checks for those frames only. '
        'Misalignment can blur or double coronal detail and affect hot-pixel detection. '
        'These fits have no independent graph validation and are for aesthetic use only. '
        'Failed fits cannot be included.')
    note.setWordWrap(True)
    layout.addWidget(note)
    table = QTableWidget(len(rows), 8)
    table.setHorizontalHeaderLabels(['Include' if conservative else 'Include candidate', 'File', 'Status', 'Exposure', 'Correlation',
        'Median error' if conservative else 'Rotation (°)', 'Regional error' if conservative else 'Fit difference (px)', 'Criteria or reason'])
    for column, tip in {3: 'Exposure time from image metadata.',
                        4: 'Held-out coronal correlation; the visual check requires at least 0.650.',
                        5: 'Estimated rotation in degrees; the visual check requires an absolute value no greater than 0.500°.',
                        6: 'Maximum disagreement with the spatially split rigid fit in native pixels; the limit is 1 pixel. An unavailable result means the check did not produce a valid fit.',
                        7: 'Alignment outcome or reason for flagging or rejecting this frame.'}.items():
        table.horizontalHeaderItem(column).setToolTip(tip)
    if conservative:
        table.horizontalHeaderItem(4).setToolTip('Correlation from the registration fit when available; this is not the visual held-out correlation.')
        table.horizontalHeaderItem(5).setToolTip('Reference-level median radiometric disagreement; the scientific gate is 15%.')
        table.horizontalHeaderItem(6).setToolTip('Largest reference-level regional disagreement; the scientific gate is 20%.')
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    for row_number, row in enumerate(rows):
        item = QTableWidgetItem()
        selectable = row['status'] == ('accepted' if conservative else 'candidate')
        item.setFlags((Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                      if selectable and not row.get('reference') else Qt.ItemFlag.NoItemFlags)
        item.setCheckState(Qt.CheckState.Checked if conservative and selectable else Qt.CheckState.Unchecked)
        table.setItem(row_number, 0, item)
        measurements = row.get('measurements', {})
        check = measurements.get('independent_check', {})
        def value(key, source):
            number = source.get(key)
            return f'{number:.3f}' if isinstance(number, (int, float)) and math.isfinite(number) else '—'
        correlation = 'Reference' if row.get('reference') else value('correlation' if conservative else 'heldout_correlation', measurements)
        rotation = 'Reference' if row.get('reference') else value('median_error' if conservative else 'angle_deg', measurements)
        difference = 'Reference' if row.get('reference') else value('regional_error', measurements) if conservative else value('disagreement_native_px', check)
        if conservative and not row.get('reference'):
            rotation = f'{100*measurements["median_error"]:.1f}%' if rotation != '—' else rotation
            difference = f'{100*measurements["regional_error"]:.1f}%' if difference != '—' else difference
        values = (row['filename'], row['status'].capitalize(), exposure_labels(row.get('exposure_s'))[1],
                  correlation, rotation, difference, row['reason'])
        for column, value in enumerate(values, 1):
            table.setItem(row_number, column, QTableWidgetItem(value))
            table.item(row_number, column).setToolTip(table.horizontalHeaderItem(column).toolTip() or value)
    table.resizeColumnsToContents()
    table.horizontalHeader().setStretchLastSection(True)
    layout.addWidget(table)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return [row['index'] for i, row in enumerate(rows)
            if row['status'] == ('accepted' if conservative else 'candidate') and table.item(i, 0).checkState() == Qt.CheckState.Checked]


def exposure_labels(value):
    """Measured seconds plus the nearest nominal third-stop shutter speed."""
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return 'Unknown', 'Unknown'
    if not math.isfinite(seconds) or seconds <= 0:
        return 'Unknown', 'Unknown'
    denominators = (1, 2, 3, 4, 5, 6, 8, 10, 13, 15, 20, 25, 30, 40, 50,
                    60, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800,
                    1000, 1250, 1600, 2000, 2500, 3200, 4000, 5000, 6400,
                    8000, 10000, 12800, 16000, 20000, 25600, 32000)
    speeds = [(1 / n, f'1/{n} s') for n in denominators]
    speeds.extend((n, f'{n:g} s') for n in
                  (1.3, 1.6, 2, 2.5, 3.2, 4, 5, 6, 8, 10, 13, 15, 20, 25, 30))
    # Outside the nominal range, retain measured seconds rather than invent a shutter label.
    nominal = min(speeds, key=lambda speed: abs(math.log2(seconds) - math.log2(speed[0])))[1]
    if seconds < 1 / denominators[-1] or seconds > 30:
        nominal = f'{seconds:.5f} s'
    return f'{seconds:.5f}', nominal


def inspection_preview(path, high=None):
    image = tifffile.memmap(path)
    step = max(1, int(np.ceil(max(image.shape[:2])/1000)))
    a = np.array(image[::step, ::step], dtype=float)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=-1)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError('Inspection preview requires mono or RGB TIFF')
    valid = np.isfinite(a).all(axis=-1)
    if not valid.any():
        raise ValueError('Image has no finite observations')
    high = max(float(np.percentile(a[valid], 99.8)), 1e-6) if high is None else high
    soft = high/100
    display = np.arcsinh(np.maximum(np.nan_to_num(a), 0)/soft)/np.arcsinh(high/soft)
    rgb = np.ascontiguousarray(np.rint(np.clip(display, 0, 1)*255).astype('uint8'))
    rgb[~valid] = 48
    qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
    return qimage, high


class PublicationSignals(QObject):
    finished = Signal(object, object, object)


class SavedLogTask(QRunnable):
    def __init__(self, path, generation):
        super().__init__()
        self.setAutoDelete(False)
        self.path, self.generation = path, generation
        self.signals = PublicationSignals()

    def run(self):
        try:
            self.signals.finished.emit(self, self.path.read_text(encoding='utf-8', errors='replace'), None)
        except OSError as error:
            self.signals.finished.emit(self, None, str(error))


class PublicationTask(QRunnable):
    def __init__(self, job):
        super().__init__()
        self.setAutoDelete(False)
        self.job = job
        self.signals = PublicationSignals()

    def run(self):
        try:
            self.signals.finished.emit(self, is_published(self.job), None)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.signals.finished.emit(self, False, str(error))


def about_text():
    try:
        version = metadata.version('coronahdr')
    except metadata.PackageNotFoundError:
        version = 'unknown'
    dependencies = []
    for name in ('numpy', 'scipy', 'scikit-image', 'astropy', 'tifffile', 'rawpy', 'PySide6-Essentials', 'opencv-python-headless'):
        try:
            info = metadata.metadata(name)
            license_name = info.get('License-Expression') or info.get('License', '')
            if not license_name or '\n' in license_name or len(license_name) > 100:
                license_name = next((c.rsplit(' :: ', 1)[-1] for c in info.get_all('Classifier', [])
                                     if c.startswith('License ::')), 'See package metadata')
            dependencies.append(f'<li>{escape(name)}: {escape(license_name)}</li>')
        except metadata.PackageNotFoundError:
            dependencies.append(f'<li>{escape(name)}: package metadata unavailable</li>')
    return f'''<h2>CoronaHDR {escape(version)}</h2>
        <p>Creates linear HDR masters from total solar eclipse images.</p>
        <p>Author: Francisco Bitto</p>
        <p>CoronaHDR was developed using Agentic Eclipse Library by Pere Guerra Serra. Experimental registration and corona filters include AEL code under the MIT license:<br/>
        <a href="https://github.com/PeterWar/Agentic-Eclipse-Library">https://github.com/PeterWar/Agentic-Eclipse-Library</a></p>
        <p>After stacking, you can continue processing your image with Francisco Bitto’s
        Eclipse Image Processor:<br/>
        <a href="https://github.com/fbitto/Eclipse-Image-Processor">https://github.com/fbitto/Eclipse-Image-Processor</a></p>
        <p>Preview color neutralization adapts the gain-only P1/P99 method from Eclipse Image Processor and eclipsetools, copyright © 2025 Samuli Vuorinen, under the MIT license. See the included third-party notices.</p>
        <h3>Research and inspiration</h3>
        <p>Research sources include <a href="https://github.com/naavis/eclipsetools">eclipsetools</a>,
        <a href="https://github.com/naugustin77/EclipseForgeHDR">EclipseForgeHDR</a>,
        <a href="https://github.com/LudoRey/umbra">Umbra</a>,
        <a href="https://github.com/zdenyhraz/IPC">IPC</a>,
        <a href="https://github.com/LaurentVoillot/Eclipse_HDR">Eclipse_HDR</a>,
        <a href="https://matheo.uliege.be/handle/2268.2/4511">Gava’s 2018 thesis</a>,
        <a href="https://www.cloudynights.com/forums/topic/888126-solar-eclipse-image-processing-achf-fnrgf/">Cloudy Nights discussions</a>,
        and <a href="https://siril.readthedocs.io/en/stable/processing/background.html">Siril’s background documentation</a>.
        These sources informed the work; their implementations were not copied into CoronaHDR.</p>
        <h3>Software dependencies</h3>
        <p>License information is taken from installed package metadata; consult each package for its complete notices.</p>
        <ul>{''.join(dependencies)}</ul>'''


class CoronaWindow(QMainWindow):
    def __init__(self, source=None, review=None):
        super().__init__()
        self.setProperty('coronahdr_tooltips', True)
        self.setWindowTitle('CoronaHDR — Linear HDR Master')
        self.resize(1360, 850)
        self.setMinimumSize(960, 600)
        self.report = None
        self.folder_workflow = source is None
        self.process = None
        self.job_path = None
        self.loaded_log_path = None
        self.saved_log_generation = 0
        self.saved_log_tasks = set()
        self.cancel_requested = False
        self.publication_tasks = set()
        self.buffer = ''
        self.log_decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.active_hdr_mode = None
        self.stage_names = []
        self.current_stage = None
        self.log_write_failed = False
        self.source = QLineEdit(str(source or ''))
        self.review = QLineEdit(str(review or ''))
        path_tooltip(self.source, 'Prepared linear images selected by the app for this run.')
        path_tooltip(self.review, 'Hash-bound registration geometry selected by the app for this run.')
        self.destination = QLineEdit()
        path_tooltip(self.destination, 'Folder for the HDR master and diagnostic files. Existing files are never overwritten.')
        self.temporary = QLineEdit(str(default_work_directory()))
        path_tooltip(self.temporary, 'Prepared caches and diagnostic masks are retained here for reuse. Keep this folder available after processing.')
        self.method = ScrollSafeComboBox()
        self.method.setToolTip('Aesthetic blends exposures with empirical brightness matching. Radiometric keeps strict response-consistency rejection.')
        self.method.addItem('Aesthetic — smooth HDR blend', 'aesthetic')
        self.method.addItem('Radiometric — weighted, strict rejection', 'weighted')
        self.method.setItemData(0, 'Blend exposures with empirical brightness matching for visual output; not calibrated radiance.', Qt.ItemDataRole.ToolTipRole)
        self.method.setItemData(1, 'Combine exposures using strict relative radiometric consistency checks; incompatible frames can be rejected.', Qt.ItemDataRole.ToolTipRole)
        scroll_tooltips(self.method.view(), 'the HDR method choices')
        self.reference = QLabel('Reference: load a prepared run')
        self.selection = QLabel('')
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(['Use', 'Index', 'File', 'Exposure\n(s)', 'Shutter\n(nominal)', 'Input\nstate', 'Geometry'])
        self.table.horizontalHeaderItem(4).setToolTip('Nearest nominal photographic shutter speed; measured seconds are shown in the preceding column.')
        for column, tip in enumerate((
                'Include this frame. The reference is required; unavailable geometry cannot be selected.',
                'Zero-based index of the original frame in the acquisition.',
                'Original image filename. Hover over a filename to see its full path.',
                'Measured exposure time in seconds, displayed to five decimal places.',
                'Nearest nominal photographic shutter speed; display rounding does not change processing.',
                'Declared source data state, such as linear CFA or linear RGB.',
                'Whether registration is reviewed, unavailable, or the fixed reference.')):
            self.table.horizontalHeaderItem(column).setToolTip(tip)
        self.table.setToolTip('Review the input frames and choose which reliable frames contribute to the HDR master.')
        scroll_tooltips(self.table, 'the frame table')
        scroll_tooltips(self.table.horizontalHeader(), 'the frame column headings')
        scroll_tooltips(self.table.verticalHeader(), 'the frame row headings')
        corner = self.table.findChild(QAbstractButton, 'qt_tableview_cornerbutton')
        if corner:
            corner.setToolTip('Select table cells; this does not change the frame inclusion checkboxes.')
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(150)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemChanged.connect(self.update_selection)
        form = QFormLayout()
        self.source.setReadOnly(True)
        self.review.setReadOnly(True)
        form.addRow('Output folder', self.path_row(self.destination, self.browse_destination))
        form.addRow('Temporary files folder', self.path_row(self.temporary, self.browse_temporary))
        form.addRow('HDR method', self.method)
        self.lunar_margin = ScrollSafeSpinBox()
        self.lunar_margin.setRange(-4, 4)
        self.lunar_margin.setSuffix(' px')
        self.lunar_margin.setToolTip('Margin relative to the fitted lunar edge. Negative retains more limb; positive excludes more.')
        self.lunar_margin.lineEdit().setToolTip(self.lunar_margin.toolTip())
        form.addRow('Lunar margin', self.lunar_margin)
        self.form_widget = QWidget()
        self.form_widget.setLayout(form)
        controls = QWidget()
        controls.setMinimumWidth(360)
        controls.setMaximumWidth(460)
        left = QVBoxLayout(controls)
        left.setContentsMargins(6, 12, 12, 12)
        left.setSpacing(6)
        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        self.preparation_heading = heading('Preparation', 'Select originals or resume a saved job before building an HDR master.')
        settings_layout.addWidget(self.preparation_heading)
        actions_bar = QHBoxLayout()
        self.import_button = QPushButton('New stack…')
        self.import_button.setProperty('primary', True)
        self.import_button.clicked.connect(self.import_data)
        self.open_image_button = QPushButton('Open HDR…')
        self.open_image_button.setToolTip('Open a completed FITS or TIFF image for inspection without a settings file.')
        self.open_image_button.clicked.connect(self.open_hdr_image)
        self.resume_button = QPushButton('Resume / revise…')
        self.resume_button.setToolTip('Find saved folder jobs and reuse verified completed stages.')
        self.resume_button.clicked.connect(self.resume_work)
        self.cleanup_button = QPushButton('Clean up temporary data…')
        self.cleanup_button.setToolTip('Review previous jobs and remove selected temporary data while protecting final images and logs.')
        self.cleanup_button.clicked.connect(lambda: CleanupDialog(self, self.temporary.text()).exec())
        self.about_button = QPushButton('About CoronaHDR')
        self.about_button.setToolTip('Learn about CoronaHDR, its author, acknowledgements and related software.')
        self.about_button.clicked.connect(self.show_about)
        for button in (self.resume_button, self.cleanup_button, self.about_button):
            button.setParent(settings)
            button.hide()
        for button in (self.import_button, self.open_image_button):
            button.setMinimumHeight(38)
            actions_bar.addWidget(button, 1)
        self.more_button = QPushButton('More')
        self.more_button.setMinimumHeight(38)
        self.more_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.more_button.setToolTip('Resume work, clean up temporary data, or learn about CoronaHDR.')
        menu = QMenu(self.more_button)
        menu_items = []
        for button in (self.resume_button, self.cleanup_button, self.about_button):
            action = menu.addAction(button.text())
            action.triggered.connect(button.click)
            menu_items.append((action, button))
        menu.aboutToShow.connect(lambda: [action.setEnabled(button.isEnabled()) for action, button in menu_items])
        self.more_button.setMenu(menu)
        actions_bar.addWidget(self.more_button, 1)
        settings_layout.addLayout(actions_bar)
        if not source:
            intro = QLabel('Create an HDR stack or open an existing master for inspection.')
            intro.setWordWrap(True)
            intro.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            settings_layout.addWidget(intro)
        settings_layout.addWidget(self.form_widget)
        self.review_heading = heading('Frame review', 'Review registration availability and select which reliable frames to include.')
        settings_layout.addWidget(self.review_heading)
        settings_layout.addWidget(self.reference)
        settings_layout.addWidget(self.selection)
        settings_layout.addWidget(self.table, 1)
        if not source:
            settings_layout.addStretch(1)
        self.controls_scroll = QScrollArea()
        scroll_tooltips(self.controls_scroll, 'the preparation settings and frame review')
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setWidget(settings)
        if not source:
            self.controls_scroll.setMaximumHeight(90)
        left.addWidget(self.controls_scroll, 0)
        self.output_heading = heading('Output and processing', 'Start or stop processing, inspect progress, and open completed outputs.')
        left.addWidget(self.output_heading)
        self.start_button = QPushButton('Build HDR master')
        self.start_button.setProperty('primary', True)
        self.start_button.clicked.connect(self.start_job)
        self.cancel_button = QPushButton('Cancel processing')
        self.cancel_button.clicked.connect(self.cancel_job)
        self.cancel_button.setEnabled(False)
        self.open_button = QPushButton('Open result folder')
        self.open_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.job['destination'])))
        self.open_button.setEnabled(False)
        self.diagnostics_button = QPushButton('Open processing files')
        self.diagnostics_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.job['result'])))
        self.diagnostics_button.hide()
        actions = QHBoxLayout()
        for widget in (self.start_button, self.open_button, self.diagnostics_button):
            actions.addWidget(widget)
        left.addLayout(actions)
        self.progress = QProgressBar()
        self.status = QLabel('Load a completed disk-backed run and its reviewed geometry.')
        self.status.setWordWrap(False)
        self.status.setProperty('role', 'status')
        self.warnings = []
        self.warning_panel = QLabel()
        self.warning_panel.setWordWrap(True)
        self.warning_panel.setTextFormat(Qt.TextFormat.PlainText)
        self.warning_panel.setProperty('role', 'warning')
        self.warning_panel.hide()
        self.log_header = QWidget()
        log_header = QHBoxLayout(self.log_header)
        log_header.setContentsMargins(0, 0, 0, 0)
        log_header.addWidget(heading('Job log', 'Messages from the current or most recent job in this app session.'))
        log_header.addStretch()
        self.copy_log_button = QPushButton('Copy log')
        self.copy_log_button.setEnabled(False)
        self.copy_log_button.clicked.connect(self.copy_job_log)
        log_header.addWidget(self.copy_log_button)
        self.view_log_button = QPushButton('View log')
        self.view_log_button.setEnabled(False)
        self.view_log_button.clicked.connect(self.view_job_log)
        log_header.addWidget(self.view_log_button)
        self.copy_log_path_button = QPushButton('Copy log path')
        self.copy_log_path_button.setEnabled(False)
        self.copy_log_path_button.clicked.connect(self.copy_job_log_path)
        log_header.addWidget(self.copy_log_path_button)
        self.log_toggle = QPushButton('Show log')
        self.log_toggle.setCheckable(True)
        self.log_toggle.setToolTip('Show or hide recent job messages in the main window.')
        log_header.addWidget(self.log_toggle)
        self.log_dialog = None
        self.log = QPlainTextEdit()
        log_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        families = QFontDatabase.families()
        for family in ('Menlo', 'Consolas', 'DejaVu Sans Mono', 'Courier New'):
            if family in families and QFontDatabase.isFixedPitch(family):
                log_font.setFamily(family)
                break
        log_font.setPointSizeF(max(1., self.log.font().pointSizeF() - 1.))
        self.log.setFont(log_font)
        scroll_tooltips(self.log, 'the job log')
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMinimumHeight(100)
        self.log.hide()
        self.log_toggle.toggled.connect(self.log.setVisible)
        self.log_toggle.toggled.connect(lambda checked: self.log_toggle.setText('Hide log' if checked else 'Show log'))
        view = QWidget()
        right = QVBoxLayout(view)
        right.setContentsMargins(12, 12, 6, 12)
        right.setSpacing(6)
        self.hdr_preview = HDRPreview()
        self.preview = self.hdr_preview.labels['corona']
        self.preview_title = self.hdr_preview.titles['corona']
        left.insertWidget(1, self.hdr_preview.adjustment_panel, 1)
        right.addWidget(self.hdr_preview, 2)
        source_preview = QPushButton('Preview source master')
        source_preview.clicked.connect(lambda: self.show_preview(Path(self.source.text())/'master.tiff'))
        right.addWidget(source_preview)
        right.addWidget(self.warning_panel)
        right.addWidget(self.log_header)
        right.addWidget(self.log, 1)
        splitter = QSplitter()
        splitter.addWidget(view)
        splitter.addWidget(controls)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([960, 400])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setToolTip('Drag the divider to resize the controls and image preview.')
        splitter.handle(1).setToolTip('Drag to resize the controls and image preview.')
        self.setCentralWidget(splitter)
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_button)
        self.progress.setMaximumWidth(220)
        self.cancel_button.hide()
        for widget, tip in (
                (self.import_button, 'Select original image and calibration folders, processing options and storage locations.'),
                (self.start_button, 'Build the HDR master from the selected frames using the chosen method.'),
                (self.cancel_button, 'Stop the worker safely. Incomplete files remain in temporary storage; originals are unchanged.'),
                (self.open_button, 'Open the completed result folder in the system file browser.'),
                (self.diagnostics_button, 'Open reports, masks and intermediate products retained in the temporary job folder.'),
                (self.reference, 'The reference frame defines the registration coordinates and cannot be deselected.'),
                (self.selection, 'Number of checked frames compared with the total input count; later radiometric rejection may reduce it.'),
                (self.progress, 'Progress for the current activity only. An animated bar means no count is available.'),
                (self.status, 'Current processing state or the most recent error; full details appear in the log.'),
                (self.warning_panel, 'Recent processing warnings. Full messages remain in the log and report.'),
                (self.log, 'Read-only worker messages and errors. Select text to copy it; the job also retains process.log.'),
                (self.copy_log_button, 'Copy the complete log from the current or most recent job, including lines no longer visible here. Available after the worker produces log output.'),
                (self.view_log_button, 'Open the complete current or most recent job log in a separate window. It updates while the job runs.'),
                (self.copy_log_path_button, 'Copy the job log file path. After a successful job, use the saved output log when available; otherwise use the temporary log.'),
                (self.preview, 'Inspect the HDR with stretch and optional color neutralization. Missing pixels are gray. The linear master is unchanged; nonlinear exports use the displayed adjustments.'),
                (self.preview_title, 'Identifies the displayed product and its inspection display conventions.'),
                (source_preview, 'Display the original prepared master using the shared inspection stretch.'),
                (self.controls_scroll, 'Scroll to reach preparation settings and frame review without moving the processing actions.')):
            widget.setToolTip(tip)
        label_tooltips(self)
        self.start_button.setEnabled(False)
        if not source:
            for widget in (self.form_widget, self.reference, self.selection, self.table,
                           self.start_button, source_preview, self.preparation_heading, self.review_heading,
                           self.cancel_button, self.open_button, self.progress, self.log_header, self.log, self.output_heading):
                widget.hide()
            self.status.setText('Choose New stack to process images or Open HDR to inspect a master.')
        if source:
            self.load_run()

    def show_about(self):
        dialog = QDialog(self)
        dialog.setProperty('coronahdr_tooltips', True)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle('About CoronaHDR')
        dialog.resize(620, 550)
        layout = QVBoxLayout(dialog)
        content = QTextBrowser()
        content.setOpenExternalLinks(True)
        content.setToolTip('Information about CoronaHDR, its author, research sources and software dependencies.')
        content.document().setDefaultStyleSheet('a { color: #9cdcfe; }')
        content.setHtml(about_text())
        layout.addWidget(content)
        close = QPushButton('Close')
        close.setToolTip('Close the About CoronaHDR window.')
        close.clicked.connect(dialog.close)
        layout.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)
        dialog.show()

    def path_row(self, edit, callback, extra_text=None, extra_callback=None):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        row.setToolTip(edit.toolTip())
        layout.addWidget(edit)
        browse = QPushButton('Choose…')
        browse.setToolTip(edit.toolTip())
        edit.textChanged.connect(lambda _text: (browse.setToolTip(edit.toolTip()), row.setToolTip(edit.toolTip())))
        browse.clicked.connect(callback)
        layout.addWidget(browse)
        if extra_text:
            button = QPushButton(extra_text)
            button.setToolTip(edit.toolTip())
            button.clicked.connect(extra_callback)
            layout.addWidget(button)
        return row

    def browse_destination(self):
        path = choose_output_folder(self, self.destination.text())
        if path:
            self.destination.setText(path)

    def browse_temporary(self):
        path = choose_directory(self, 'Choose temporary files folder', self.temporary.text())
        if path:
            self.temporary.setText(path)

    def load_run(self):
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            return
        try:
            root = Path(self.source.text()).resolve()
            report = json.loads((root/'report.json').read_text())
            geometry = json.loads(Path(self.review.text()).read_text()) if self.review.text() else None
            if geometry and geometry['input_source_sha256'] != [a['metadata']['source_sha256'] for a in report['inputs']]:
                raise ValueError('Geometry review does not match the source inputs')
            self.report = report
            margin = report['output'].get('lunar_margin_px', 0) if 'output' in report else 0
            from .lunar import fitted_disk
            if any(i['metadata'].get('lunar_disk') and fitted_disk(i['metadata'])[2] < i['metadata']['lunar_disk'][2]
                   and 'Nikon' in i['metadata'].get('response_group', '') for i in report['inputs']):
                margin = report.get('output', {}).get('lunar_margin_px', -2)
            self.lunar_margin.setValue(margin)
            self.loaded_source = root
            self.loaded_review = Path(self.review.text()).resolve() if geometry else None
            reference = report.get('reference', report['recipe'].get('reference'))
            self.reference.setText(f'Reference: frame {reference} · fixed source canvas')
            self.table.blockSignals(True)
            self.table.setRowCount(len(report['inputs']))
            reasons = {entry['index']: entry.get('reason', 'No reliable reviewed geometry')
                       for entry in (geometry or {}).get('rejected_frames', [])}
            for i, item in enumerate(report['inputs']):
                reliable = geometry is not None and str(i) in geometry['matrices_yx']
                overridden = reliable and geometry.get('metrics', {}).get(str(i), {}).get('user_decision') == 'Include experimental frame'
                checkbox = QTableWidgetItem()
                checkbox.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                checkbox.setCheckState(Qt.CheckState.Checked if reliable else Qt.CheckState.Unchecked)
                if not reliable or i == reference:
                    checkbox.setFlags(Qt.ItemFlag.NoItemFlags)
                checkbox.setToolTip(f'Excluded: {reasons.get(i, "No reliable reviewed geometry")}. This frame cannot be included without an accepted geometry review.' if not reliable else
                                    'Included by user despite the visual alignment criteria; inspect this result carefully.' if overridden else
                                    'The reference frame is required and cannot be deselected.' if i == reference else
                                    self.table.horizontalHeaderItem(0).toolTip())
                self.table.setItem(i, 0, checkbox)
                exposure, shutter = exposure_labels(item['metadata'].get('exposure_s'))
                values = [str(i), Path(item['path']).name, exposure, shutter,
                          item['metadata'].get('state', 'Unknown'), 'Reference' if reliable and i == reference else ('Experimental' if overridden else 'Reviewed' if reliable else 'Excluded')]
                for col, value in enumerate(values, 1):
                    self.table.setItem(i, col, QTableWidgetItem(value))
                    self.table.item(i, col).setToolTip(self.table.horizontalHeaderItem(col).toolTip())
                self.table.item(i, 2).setToolTip(item['path'])
                self.table.item(i, 6).setToolTip(checkbox.toolTip())
            self.table.blockSignals(False)
            self.table.resizeColumnsToContents()
            self.table.setColumnWidth(2, min(self.table.columnWidth(2), 145))
            self.destination.setText(str(root.parent/'CoronaHDR-desktop-output'))
            self.open_button.setEnabled(False)
            self.update_selection()
            self.show_preview(root/'master.tiff')
            self.status.setText('Ready. Select frames and a new output folder. Source data remain unchanged.')
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.table.blockSignals(False)
            self.report = None
            self.start_button.setEnabled(False)
            self.status.setText(str(error))

    def selected_indices(self):
        return [i for i in range(self.table.rowCount()) if self.table.item(i, 0).checkState() == Qt.CheckState.Checked]

    def update_selection(self, *args):
        count = len(self.selected_indices())
        self.selection.setText(f'{count} selected / {self.table.rowCount()} inputs')
        idle = not self.process or self.process.state() == QProcess.ProcessState.NotRunning
        has_reference = bool(self.report and self.report.get('reference', self.report['recipe'].get('reference')) in self.selected_indices())
        self.start_button.setEnabled(bool(has_reference and self.loaded_review and count and idle))
        self.start_button.setToolTip('Build the HDR master from the selected frames using the chosen method.' if self.start_button.isEnabled() else
            'Select reviewed frames including the reference, and wait for any active job to finish before building the HDR master.')

    def show_preview(self, path):
        self.hdr_preview.set_path('corona', path)

    def open_hdr_image(self):
        name, _ = QFileDialog.getOpenFileName(self, 'Open HDR image', '',
            'FITS and TIFF images (*.fits *.fit *.fts *.tif *.tiff)')
        if not name:
            return
        path = Path(name)
        self.hdr_preview.filter_temporary = self.temporary.text() or None
        self.hdr_preview.export_directory = path.parent
        self.hdr_preview.export_base = path.stem
        self.show_preview(path)
        self.status.setText(f'Opening {path.name} for inspection. The job log, if shown, belongs to the last job.')

    def start_job(self):
        try:
            if Path(self.source.text()).resolve() != self.loaded_source or Path(self.review.text()).resolve() != self.loaded_review:
                raise ValueError('Load the changed source/review before starting')
            self.job_path, self.job = create_job(self.loaded_source, self.loaded_review, self.destination.text(), self.selected_indices(), self.method.currentData(), lunar_margin=self.lunar_margin.value(), temporary=self.temporary.text())
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.status.setText(str(error))
            return
        self.launch_job()

    def import_data(self):
        from .desktop_folders import FolderDialog
        dialog = FolderDialog(self)
        dialog.settings_loaded.connect(self.show_saved_settings)
        if dialog.exec():
            self.job_path, self.job = dialog.job_path, dialog.job
            self.temporary.setText(str(self.job_path.parent.parent.parent))
            self.launch_job()

    def show_saved_settings(self, settings_file, options):
        """Inspect only named output files beside the selected settings file."""
        from .desktop_jobs import final_log_name, output_base_name
        directory = Path(settings_file).parent
        base = output_base_name(options['base_name'] or 'CoronaHDR')
        self.hdr_preview.filter_temporary = options.get('temporary')
        self.hdr_preview.export_directory = directory
        self.hdr_preview.export_base = base
        mode = options['mode']
        display_mode = 'aesthetic' if mode == 'both' else mode
        extensions = ('tiff', 'fits') if options['output_format'] == 'both' else (options['output_format'],)
        def image_for(kind):
            return next((directory/f'{base}_{kind}.{ext}' for ext in extensions
                         if (directory/f'{base}_{kind}.{ext}').is_file()), None)
        corona = image_for(display_mode)
        self.hdr_preview.set_path('corona', corona)
        if not corona:
            self.hdr_preview.labels['corona'].setText('Saved corona image not found beside settings.')
        self.saved_log_generation += 1
        self.loaded_log_path = directory/final_log_name(base, mode)
        self.log.clear()
        self.log_header.show()
        self.log.setVisible(self.log_toggle.isChecked())
        if self.folder_workflow:
            self.controls_scroll.setMaximumHeight(165)
        for button in (self.copy_log_button, self.copy_log_path_button, self.view_log_button):
            button.setEnabled(False)
        if self.log_dialog is not None:
            self.log_dialog.close()
            self.log_dialog = None
        self.status.setText('Loading saved output preview and log…')
        task = SavedLogTask(self.loaded_log_path, self.saved_log_generation)
        self.saved_log_tasks.add(task)
        task.signals.finished.connect(self.saved_log_finished)
        QThreadPool.globalInstance().start(task)

    def saved_log_finished(self, task, text, error):
        self.saved_log_tasks.discard(task)
        if task.generation != self.saved_log_generation:
            return
        if error:
            self.loaded_log_path = None
            self.status.setText(f'Saved log unavailable: {error}')
            return
        self.log.setPlainText(text)
        for button in (self.copy_log_button, self.copy_log_path_button, self.view_log_button):
            button.setEnabled(True)
        self.status.setText('Saved settings, available images and log loaded for inspection.')

    def resume_work(self):
        from .desktop_jobs import unfinished_jobs, resume_folder_job, OutputNameConflict
        try:
            jobs = unfinished_jobs(self.temporary.text(), include_completed=True)
            if not jobs:
                folder = choose_directory(self, 'Choose the temporary files folder used for the earlier run', self.temporary.text())
                if not folder:
                    return
                self.temporary.setText(folder)
                jobs = unfinished_jobs(folder, include_completed=True)
                if not jobs:
                    self.status.setText('No unfinished folder jobs found in this temporary files folder.')
                    return
            labels = [f'{Path(job["folders"]["light"]).name} → {job["destination"]} [{job.get("status", "unknown")}; {path.parent.name}]'
                      for path, job in jobs]
            choice, accepted = ask_choice(self, 'Resume unfinished work',
                'Choose saved work. Completed jobs require a new output folder; preparation is verified before reuse.', labels)
            if not accepted:
                return
            path, job = jobs[labels.index(choice)]
            options = {}
            if (job.get('status') == 'completed' or is_published(job)) and job.get('publication') == 'standalone_final_files_v1':
                name, accepted = ask_text(self, 'Choose a new output base name',
                    'The completed files are kept. Enter a different base name for the new copies:',
                    text=job.get('output_base_name', ''))
                if not accepted:
                    return
                options.update(destination=job['destination'], base_name=name)
            elif job.get('status') == 'completed' or is_published(job):
                target = Path(job['destination'])
                destination = choose_output_folder(self, str(target.with_name(target.name+'-revised')))
                if not destination:
                    return
                options['destination'] = destination
            while True:
                try:
                    self.job_path, self.job = resume_folder_job(path, **options)
                    break
                except OutputNameConflict as error:
                    name, accepted = ask_text(self, 'Output file already exists',
                        f'{error}\nEnter a different output base name:', text=options.get('base_name', job.get('output_base_name', '')))
                    if not accepted:
                        return
                    options['base_name'] = name
            self.launch_job()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.status.setText(str(error))

    def launch_job(self):
        self.saved_log_generation += 1
        self.loaded_log_path = None
        self.hdr_preview.export_directory = Path(self.job['destination'])
        self.hdr_preview.export_base = self.job.get('output_base_name') or 'CoronaHDR'
        remember_location(Path(self.job_path).parent.parent.parent)
        self.warnings = []
        self.warning_panel.clear()
        self.warning_panel.hide()
        if 'source' not in self.job:
            self.hdr_preview.set_path('corona', None)
        self.cancel_requested = False
        self.buffer = ''
        self.log_decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.active_hdr_mode = None
        mode = self.job.get('hdr_mode', 'aesthetic')
        if self.job.get('kind') == 'folders':
            modes = ['scientific', 'aesthetic'] if mode == 'both' else [mode]
            self.stage_names = [('prepare', 'Prepare and align images')] + [(m, f'Build {m} HDR') for m in modes] + [('publish', 'Save outputs')]
        else:
            self.stage_names = [('build', 'Build HDR'), ('publish', 'Save outputs')]
        self.current_stage = None
        self.log_write_failed = False
        self.log.clear()
        self.copy_log_button.setEnabled(False)
        self.view_log_button.setEnabled(False)
        self.copy_log_path_button.setEnabled(False)
        if self.log_dialog is not None:
            self.log_dialog.close()
            self.log_dialog = None
        self.progress.setValue(0)
        self.progress.setRange(0, 0)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONUNBUFFERED', '1')
        environment.insert('CORONAHDR_INTERACTIVE', '1')
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self.read_process)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.start_button.setEnabled(False)
        self.form_widget.setEnabled(False)
        self.import_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.table.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.show()
        self.progress.show()
        self.output_heading.show()
        self.log_header.show()
        self.log.setVisible(self.log_toggle.isChecked())
        if self.folder_workflow:
            self.controls_scroll.setMaximumHeight(165)
        self.open_button.setEnabled(False)
        self.diagnostics_button.hide()
        self.status.setText('Processing in the background. You may cancel safely.')
        self.process.start(sys.executable, ['-u', '-m', 'coronahdr.desktop_jobs', str(self.job_path)])

    def show_activity(self, text, completed=None, total=None, unit=None):
        prefix = ''
        if self.current_stage is not None:
            names = [name for _, name in self.stage_names]
            prefix = f'Step {self.current_stage+1} of {len(names)} — {names[self.current_stage]}: '
        self.status.setText(prefix + text)
        if completed is None or total is None or total <= 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, total)
            self.progress.setValue(min(completed, total))
            self.progress.setFormat(f'%v / %m {unit or "items"}')

    def set_stage(self, key):
        for index, (stage_key, _) in enumerate(self.stage_names):
            if stage_key == key:
                self.current_stage = index
                self.show_activity('Starting…')
                return

    def read_process(self, final=False):
        text = self.log_decoder.decode(bytes(self.process.readAllStandardOutput()), final=final)
        try:
            with (self.job_path.parent/'process.log').open('a', encoding='utf-8') as stream:
                stream.write(text)
            if text:
                self.copy_log_button.setEnabled(True)
                self.view_log_button.setEnabled(True)
                self.copy_log_path_button.setEnabled(True)
                if self.log_dialog is not None:
                    viewer = self.log_dialog.findChild(QPlainTextEdit)
                    viewer.moveCursor(QTextCursor.MoveOperation.End)
                    viewer.insertPlainText(text)
        except OSError as error:
            self.log_write_failed = True
            self.log.appendPlainText(f'Cannot write process log: {error}')
        self.buffer += text
        lines = self.buffer.split('\n')
        self.buffer = lines.pop()
        if final and self.buffer:
            lines.append(self.buffer)
            self.buffer = ''
        for line in lines:
            if line.startswith('CORONAHDR_STAGE '):
                self.set_stage(line.removeprefix('CORONAHDR_STAGE ').strip())
                continue
            if line.startswith('CORONAHDR_PREVIEW_READY '):
                event = json.loads(line.removeprefix('CORONAHDR_PREVIEW_READY '))
                if event['role'] == 'corona':
                    self.hdr_preview.filter_temporary = self.job_path.parent.parent.parent if self.job_path else None
                    self.hdr_preview.set_path(event['role'], event.get('path'))
                    if not event.get('path') and event.get('reason'):
                        self.hdr_preview.labels[event['role']].setText(event['reason'])
                continue
            if line.startswith('CORONAHDR_SELECTION_REVIEW '):
                rows = json.loads(line.removeprefix('CORONAHDR_SELECTION_REVIEW '))
                self.show_activity('Waiting for your frame selection…')
                selected = review_alignment_frames(self, rows, conservative=True)
                self.process.write((json.dumps(selected)+'\n').encode('utf-8'))
                self.show_activity('Continuing processing…' if selected is not None else 'Selection cancelled…')
                continue
            if line.startswith('CORONAHDR_FRAME_REVIEW '):
                rows = json.loads(line.removeprefix('CORONAHDR_FRAME_REVIEW '))
                self.show_activity('Waiting for your frame review…')
                selected = review_alignment_frames(self, rows)
                self.process.write((json.dumps(selected)+'\n').encode('utf-8'))
                self.show_activity('Continuing processing…' if selected is not None else 'Selection cancelled…')
                continue
            if line.startswith('CORONAHDR_DECISION '):
                request = json.loads(line.removeprefix('CORONAHDR_DECISION '))
                self.show_activity('Waiting for your processing decision…')
                answer, accepted = ask_choice(self, 'Review processing decision',
                    request['message'], request['choices'])
                self.process.write(((answer if accepted else '')+'\n').encode('utf-8'))
                self.show_activity('Continuing processing…' if accepted else 'Decision cancelled…')
                continue
            self.log.appendPlainText(line)
            if line.startswith('WARNING: '):
                self.add_warning(line.removeprefix('WARNING: '))
            if line in ('Starting scientific HDR', 'Starting aesthetic HDR'):
                self.active_hdr_mode = line.removeprefix('Starting ').removesuffix(' HDR')
                self.show_activity(f'Preparing {self.active_hdr_mode} processing…')
            frame_match = re.search(r'Prepared full field (\d+)/(\d+)', line)
            row_match = re.search(r'HDR rows \d+:(\d+) of (\d+)', line)
            if frame_match:
                done, total = int(frame_match[1]), int(frame_match[2])
                if done == total:
                    self.show_activity('Preparing alignment…')
                else:
                    self.show_activity('Preparing images', done, total, 'frames')
            elif row_match:
                done, total = int(row_match[1]), int(row_match[2])
                if done == total:
                    self.show_activity('Writing diagnostics and master files…')
                else:
                    self.show_activity('Integrating aligned frames', done, total, 'rows')
            elif line.startswith('Prepared matching auxiliary'):
                self.show_activity('Preparing response-matching measurements…')
            elif line.startswith('Empirical match'):
                self.show_activity('Matching exposures and backgrounds…')
            elif line.startswith('Saved ') and ' intermediate ' in line:
                self.show_activity('Saving linear intermediate images and masks…')
            elif line.startswith('Defect candidates frame'):
                self.show_activity('Checking persistent detector defects…')
            elif line.startswith('Prepared defect-handled frame'):
                self.show_activity('Preparing detector-corrected images…')

    def view_job_log(self):
        if self.log_dialog is not None:
            self.log_dialog.raise_()
            self.log_dialog.activateWindow()
            return
        try:
            path = self.loaded_log_path if self.loaded_log_path is not None else self.job_path.parent/'process.log'
            text = path.read_text(encoding='utf-8', errors='replace')
        except (AttributeError, OSError) as error:
            self.status.setText(f'Cannot view job log: {error}')
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Job log')
        dialog.resize(850, 400)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        layout = QVBoxLayout(dialog)
        viewer = QPlainTextEdit()
        viewer.setFont(self.log.font())
        viewer.setReadOnly(True)
        viewer.setPlainText(text)
        viewer.setToolTip('Complete log for this job. New messages are added while processing continues.')
        layout.addWidget(viewer)
        close = QPushButton('Close')
        close.setToolTip('Close the log window; processing continues.')
        close.clicked.connect(dialog.close)
        layout.addWidget(close)
        self.log_dialog = dialog
        dialog.destroyed.connect(lambda: setattr(self, 'log_dialog', None) if self.log_dialog is dialog else None)
        dialog.show()

    def copy_job_log_path(self):
        try:
            path = self.loaded_log_path if self.loaded_log_path is not None else self.job_path.parent/'process.log'
            job = getattr(self, 'job', None) or {}
            if self.loaded_log_path is None and job.get('status') == 'completed' and job.get('publication') == 'standalone_final_files_v1':
                from .desktop_jobs import final_log_name
                saved = Path(job['destination'])/final_log_name(job.get('output_base_name', ''), job.get('hdr_mode', 'aesthetic'))
                if saved.is_file():
                    path = saved
            if not path.is_file():
                raise FileNotFoundError(path)
        except (AttributeError, KeyError, OSError, ValueError) as error:
            self.status.setText(f'Cannot copy job log path: {error}')
            return
        QApplication.clipboard().setText(str(path.absolute()))
        self.copy_log_path_button.setText('Copied')
        QTimer.singleShot(2000, self.copy_log_path_button, lambda: self.copy_log_path_button.setText('Copy log path'))

    def copy_job_log(self):
        try:
            path = self.loaded_log_path if self.loaded_log_path is not None else self.job_path.parent/'process.log'
            text = path.read_text(encoding='utf-8', errors='replace')
        except (AttributeError, OSError) as error:
            self.status.setText(f'Cannot copy job log: {error}')
            return
        QApplication.clipboard().setText(text)
        self.copy_log_button.setText('Copied')
        QTimer.singleShot(2000, self.copy_log_button, lambda: self.copy_log_button.setText('Copy log'))

    def add_warning(self, message):
        if message not in self.warnings:
            self.warnings.append(message)
        self.warning_panel.setText(f'{len(self.warnings)} warning(s) — full details in the log and report\n'
                                   + '\n'.join(self.warnings[-4:]))
        self.warning_panel.show()

    def cancel_job(self):
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.cancel_requested = True
            self.cancel_button.setEnabled(False)
            self.status.setText('Cancelling… partial job files are retained; no result will be published.')
            process = self.process
            process.terminate()
            QTimer.singleShot(5000, process, process.kill)

    def finished(self, exit_code, exit_status):
        self.read_process(final=True)
        self.progress.setRange(0, 100)
        self.cancel_button.setEnabled(False)
        self.cancel_button.hide()
        self.form_widget.setEnabled(True)
        self.import_button.setEnabled(True)
        self.resume_button.setEnabled(True)
        self.table.setEnabled(True)
        job = json.loads(self.job_path.read_text())
        self.job = job
        for message in job.get('warnings', []):
            self.add_warning(message)
        if (job.get('status') != 'completed' and job.get('publication') == 'standalone_final_files_v1'
                and job.get('published_files')):
            self.import_button.setEnabled(False)
            self.resume_button.setEnabled(False)
            self.status.setText('Checking published output files…')
            task = PublicationTask(job)
            self.publication_tasks.add(task)
            task.signals.finished.connect(self.publication_checked)
            QThreadPool.globalInstance().start(task)
            return
        if job.get('status') != 'completed' and is_published(job):
            set_job_status(self.job_path, 'completed')
            job['status'] = 'completed'
        self.finish_job(job, exit_code)

    def publication_checked(self, task, published, error):
        self.publication_tasks.discard(task)
        self.import_button.setEnabled(True)
        self.resume_button.setEnabled(True)
        if error:
            self.add_warning(f'Could not verify published outputs: {error}')
        if published:
            set_job_status(self.job_path, 'completed')
            task.job['status'] = 'completed'
        self.finish_job(task.job, self.process.exitCode())

    def finish_job(self, job, exit_code):
        if job['status'] == 'completed':
            self.progress.setValue(100)
            self.progress.setFormat('Complete')
            log_saved = False
            try:
                if self.log_write_failed:
                    raise OSError('The temporary log could not be fully recorded')
                from .desktop_jobs import save_completed_log
                save_completed_log(self.job_path, job)
                log_saved = True
            except (OSError, ValueError) as error:
                self.add_warning(f'Could not save the job log in the output folder: {error}. The temporary log may still be available.')
            formats = {'both': 'FITS and TIFF', 'fits': 'FITS', 'tiff': 'TIFF'}[job.get('output_format', 'both')]
            message = (f'Completed. {formats} and job log are in the output folder.' if log_saved else
                       f'Completed. {formats} is ready, but the job log could not be saved in the output folder.')
            if job.get('publication') == 'standalone_final_files_v1':
                message += ' Reports and masks are in the temporary job folder.'
            self.status.setText(f'{message} {len(self.warnings)} warning(s).')
            self.open_button.setEnabled(True)
            self.open_button.show()
            if job.get('publication') == 'standalone_final_files_v1':
                self.diagnostics_button.show()
            preview = Path(job['destination'])
            if job.get('publication') == 'standalone_final_files_v1':
                mode = 'aesthetic' if job.get('hdr_mode') == 'both' else job.get('hdr_mode', 'aesthetic')
                ext = 'fits' if job.get('output_format') == 'fits' else 'tiff'
                preview = Path(job['destination'])/f'{job.get("output_base_name") or "CoronaHDR"}_{mode}.{ext}'
            elif job.get('hdr_mode') == 'both':
                preview /= 'aesthetic'
                preview /= 'master.tiff'
            else:
                preview /= 'master.tiff'
            self.show_preview(preview)
        elif self.cancel_requested:
            self.progress.setFormat('Cancelled')
            self.progress.setValue(0)
            set_job_status(self.job_path, 'cancelled', 'Cancelled by the user; incomplete files retained in the job folder')
            self.status.setText('Cancelled. No completed output was published; source data are intact.')
        else:
            self.progress.setFormat('Failed')
            self.progress.setValue(0)
            self.status.setText(f'Processing failed (exit {exit_code}). See the log; no completed output was published.')
        self.update_selection()

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.progress.setRange(0, 100)
            self.progress.setFormat('Failed')
            self.progress.setValue(0)
            set_job_status(self.job_path, 'failed', self.process.errorString())
            self.status.setText(self.process.errorString())
            self.cancel_button.setEnabled(False)
            self.form_widget.setEnabled(True)
            self.import_button.setEnabled(True)
            self.resume_button.setEnabled(True)
            self.table.setEnabled(True)
            self.update_selection()

    def closeEvent(self, event):
        if self.hdr_preview.export_tasks:
            self.status.setText('Please wait for the image export to finish before closing.')
            event.ignore()
        elif self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.cancel_job()
            event.ignore()
        else:
            event.accept()


def launch(source=None, review=None):
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName('CoronaHDR')
    apply_theme(app)
    window = CoronaWindow(source, review)
    window.show()
    return app.exec()

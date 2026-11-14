"""Folder selection without technical acquisition controls."""
from pathlib import Path
import sys
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget
from .desktop_style import heading, label_tooltips, path_tooltip, scroll_tooltips, ScrollSafeComboBox, ScrollSafeSpinBox
from .desktop_paths import ask_choice, ask_text, choose_directory, choose_output_folder
from .desktop_jobs import create_folder_job, OutputNameConflict
from .folder_import import ROLES, related_folders
from .work_paths import default_work_directory


class FolderDialog(QDialog):
    settings_loaded = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty('coronahdr_tooltips', True)
        self.setWindowTitle('Process image folders')
        self.resize(820, 640)
        self.job_path = self.job = None
        self.folders = {}
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll_tooltips(scroll, 'the folder and processing settings')
        scroll.setToolTip('Scroll to reach image folders, HDR settings and optional exports.')
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)
        note = QLabel('Select the original image folders. Supported RAW data get a new acquisition recipe '
            'with measured lunar exclusions and coronal registration. '
            'Select matching calibration folders when available. Choose temporary and output locations on any drive.')
        note.setWordWrap(True)
        layout.addWidget(note)
        load = QPushButton('Load settings…')
        load.setToolTip('Load a saved job settings JSON into this form for review and editing.')
        load.clicked.connect(self.load_settings)
        layout.addWidget(load)
        layout.addWidget(heading('Image folders and output', 'Choose original image folders and locations for processing files and final results.'))
        form = QFormLayout()
        for role in ROLES:
            edit = QLineEdit()
            edit.setReadOnly(True)
            path_tooltip(edit, {
                'light': 'Original eclipse exposures to align and blend; source files are never modified.',
                'bias': 'Bias exposures for detector pedestal correction; they do not measure thermal dark current.',
                'dark': 'Matched dark exposures for thermal signal and pedestal correction.',
                'flat': 'Matched flat-field exposures for pixel response and optical illumination correction.',
                'dark_flat': 'Dark exposures matched to the flats, for correcting their pedestal and thermal signal.'}[role])
            self.folders[role] = edit
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            line.addWidget(edit)
            choose = QPushButton('Choose…')
            choose.setToolTip(edit.toolTip())
            row.setToolTip(edit.toolTip())
            edit.textChanged.connect(lambda _text, field=edit, button=choose, container=row:
                                     (button.setToolTip(field.toolTip()), container.setToolTip(field.toolTip())))
            choose.clicked.connect(lambda checked=False, role=role: self.choose(role))
            line.addWidget(choose)
            clear = QPushButton('Clear')
            clear.setToolTip('Clear this folder selection without deleting any files.')
            clear.clicked.connect(edit.clear)
            clear.setVisible(False)
            edit.textChanged.connect(lambda text, button=clear: button.setVisible(bool(text)))
            line.addWidget(clear)
            form.addRow('Lights' if role == 'light' else role.replace('_', ' ').title(), row)
        layout.addLayout(form)
        self.temporary = QLineEdit(str(default_work_directory()))
        path_tooltip(self.temporary, 'Prepared caches, reports, diagnostic masks and optional intermediates stay here; processing may need substantial free space.')
        self.destination = QLineEdit()
        self.destination.setPlaceholderText('Defaults to HDR_output beside the Lights folder')
        path_tooltip(self.destination, 'The selected final FITS and/or TIFF images, settings JSON and job log are saved here. Defaults to HDR_output beside Lights; existing files are never overwritten.')
        self._automatic_destination = True
        self.destination.textEdited.connect(lambda _: setattr(self, '_automatic_destination', False))
        self.folders['light'].textChanged.connect(self.update_default_destination)
        for label, edit, temporary in [('Temporary files folder', self.temporary, True),
                                        ('Output folder', self.destination, False)]:
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            line.addWidget(edit)
            choose = QPushButton('Choose…')
            choose.setToolTip(edit.toolTip())
            row.setToolTip(edit.toolTip())
            edit.textChanged.connect(lambda _text, field=edit, button=choose, container=row:
                                     (button.setToolTip(field.toolTip()), container.setToolTip(field.toolTip())))
            choose.clicked.connect(lambda checked=False, edit=edit, temporary=temporary: self.choose_storage(edit, temporary))
            line.addWidget(choose)
            form.addRow(label, row)
        self.base_name = QLineEdit()
        self.base_name.setPlaceholderText('Optional — for example, Eclipse 2026')
        self.base_name.setToolTip('Optional output base, for example Eclipse_scientific.tiff and Eclipse_scientific.log. A blank name uses CoronaHDR; existing files are never overwritten.')
        form.addRow('Output file base name', self.base_name)
        self.output_format = ScrollSafeComboBox()
        self.output_format.addItem('FITS and TIFF', 'both')
        self.output_format.addItem('FITS only', 'fits')
        self.output_format.addItem('TIFF only', 'tiff')
        self.output_format.setToolTip('Choose which final image files to save in the output folder. Temporary processing may still create both formats.')
        scroll_tooltips(self.output_format.view(), 'the final image format choices')
        form.addRow('Final image format', self.output_format)
        storage_note = QLabel('Temporary processing can require substantial disk space, potentially tens or hundreds of GB. Consider an external drive with enough free space.')
        storage_note.setWordWrap(True)
        layout.addWidget(storage_note)
        self.lunar_margin = ScrollSafeSpinBox()
        self.lunar_margin.setRange(-4, 4)
        self.lunar_margin.setSuffix(' px')
        self.lunar_margin.setToolTip('Relative to the detected lunar edge. Negative values retain more limb; positive values exclude more.')
        self.lunar_margin.lineEdit().setToolTip(self.lunar_margin.toolTip())
        form.addRow('Lunar margin', self.lunar_margin)
        self.hdr_mode = ScrollSafeComboBox()
        self.hdr_mode.addItem('Aesthetic — smooth HDR, empirical matching', 'aesthetic')
        self.hdr_mode.addItem('Scientific — weighted HDR, strict rejection', 'scientific')
        self.hdr_mode.addItem('Both — scientific and aesthetic', 'both')
        self.hdr_mode.setToolTip('Aesthetic uses empirical brightness matching and smooth blending. Scientific applies strict radiometric rejection; both creates separate products. Neither is absolutely calibrated radiance.')
        for index, tip in enumerate((
                'Create a visual HDR with empirical matching and smooth exposure transitions; the result is not calibrated radiance.',
                'Create a relative-linear HDR with strict radiometric rejection; incompatible exposures can be excluded.',
                'Create separate scientific and aesthetic FITS/TIFF masters, reports and masks. This requires more time and temporary space.')):
            self.hdr_mode.setItemData(index, tip, Qt.ItemDataRole.ToolTipRole)
        scroll_tooltips(self.hdr_mode.view(), 'the HDR mode choices')
        layout.addWidget(heading('Alignment', 'Choose how frames are aligned before HDR blending.'))
        self.alignment = ScrollSafeComboBox()
        self.alignment.addItem('Conservative — existing shift-only checks', 'conservative')
        self.alignment.addItem('AEL-style visual — rotation, experimental', 'ael_visual')
        self.alignment.setToolTip('Conservative uses the existing shift-only checks. AEL-style visual estimates rotation for aesthetic output; it is experimental and does not apply independent graph or lens-distortion checks. When AEL-style visual and Both outputs are selected, scientific uses conservative alignment and aesthetic uses AEL-style visual alignment.')
        self.alignment.setItemData(0, 'Use existing conservative shift-only registration; field rotation is not corrected.', Qt.ItemDataRole.ToolTipRole)
        self.alignment.setItemData(1, 'Use masked AEL-style ECC rotation for an aesthetic stack. This is experimental and is not qualified as scientific geometry. Frames can be excluded; inspect coronal detail and the lunar limb for unsupported gaps.', Qt.ItemDataRole.ToolTipRole)
        scroll_tooltips(self.alignment.view(), 'the alignment choices')
        layout.addWidget(self.alignment)
        layout.addWidget(heading('Hot pixels', 'Optionally detect persistent detector defects after measuring alignment.'))
        self.detect_hot_pixels = QCheckBox('Detect persistent hot pixels — exclude in scientific mode, selectively repair in aesthetic mode')
        self.detect_hot_pixels.setToolTip('Find recurring isolated sensor outliers by exposure tier: at least three detections, 60% recurrence and 8 pixels of measured scene motion are required. Scientific excludes them; aesthetic repairs eligible samples. A new cache may require substantial space.')
        layout.addWidget(self.detect_hot_pixels)
        layout.addWidget(heading('HDR build', 'Choose scientific, aesthetic or both outputs.'))
        layout.addWidget(self.hdr_mode)
        layout.addWidget(heading('Optional intermediate exports', 'Save selected pre-HDR image stages and their masks in the temporary job folder.'))
        self.save_intermediates = {}
        stages = QHBoxLayout()
        for name in ('calibrated', 'debayered', 'registered'):
            checkbox = QCheckBox(f'Save {name}')
            checkbox.setToolTip({
                'calibrated': 'Save detector samples after bias, dark and flat correction, before demosaicing.',
                'debayered': 'Save prepared linear camera RGB at native geometry; monochrome inputs remain monochrome.',
                'registered': 'Save accepted frames aligned to the HDR canvas, before HDR exposure division and empirical matching.'}[name] + ' Writes float32 TIFFs and masks to the temporary job folder; can require substantial disk space.')
            self.save_intermediates[name] = checkbox
            stages.addWidget(checkbox)
        layout.addLayout(stages)
        layout.addWidget(QLabel('Intermediate float32 TIFFs and masks use additional space in the temporary folder.'))
        self.bias_fallback = QCheckBox('Use bias instead of missing darks and dark-flats (pedestal only)')
        self.bias_fallback.setToolTip('Available when a bias folder is selected and both dark and dark-flat folders are empty. Bias corrects the pedestal only, not thermal dark signal.')
        self.bias_fallback.setEnabled(False)
        layout.addWidget(self.bias_fallback)
        self.calibration_warning = QLabel()
        self.calibration_warning.setToolTip('Explains missing calibration and the limits of any pedestal-only fallback; processing may continue with these limits recorded.')
        self.calibration_warning.setWordWrap(True)
        self.calibration_warning.setProperty('role', 'warning')
        outer.addWidget(self.calibration_warning)
        self.bias_fallback.toggled.connect(self.update_calibration_warning)
        for edit in self.folders.values():
            edit.textChanged.connect(self.update_calibration_warning)
        self.update_calibration_warning()
        self.status = QLabel('Preparation and geometry are created automatically inside the temporary files folder.')
        self.status.setWordWrap(True)
        self.status.setProperty('role', 'status')
        outer.addWidget(self.status)
        self.status.setToolTip('Current folder-processing guidance or the most recent validation error.')
        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        if sys.platform == 'darwin':
            actions.setStyleSheet('QDialogButtonBox { button-layout: 1; }')
        actions.rejected.connect(self.reject)
        actions.button(QDialogButtonBox.StandardButton.Cancel).setToolTip('Close without starting a job.')
        start = actions.addButton('Process folders', QDialogButtonBox.ButtonRole.AcceptRole)
        start.setDefault(True)
        start.setProperty('primary', True)
        start.setToolTip('Validate the selected originals and storage paths, then prepare, register and build the HDR in a background worker.')
        start.clicked.connect(self.submit)
        outer.addWidget(actions)
        start.setFocus()
        label_tooltips(self)

    def load_settings(self):
        from .job_settings import read_settings
        path, _ = QFileDialog.getOpenFileName(self, 'Load job settings', '', 'JSON files (*.json)')
        if not path:
            return
        try:
            folders, options = read_settings(path)
        except ValueError as error:
            self.status.setText(str(error))
            return
        self._automatic_destination = False
        for role, field in self.folders.items():
            field.setText(folders.get(role, ''))
        self.temporary.setText(options['temporary'])
        self.destination.setText(options['destination'])
        self.base_name.setText(options['base_name'])
        for control, key in ((self.hdr_mode, 'mode'), (self.alignment, 'alignment'), (self.output_format, 'output_format')):
            control.setCurrentIndex(control.findData(options[key]))
        self.lunar_margin.setValue(options['lunar_margin'] if options['lunar_margin'] is not None else 0)
        self.detect_hot_pixels.setChecked(options['detect_hot_pixels'])
        self.bias_fallback.setChecked(options['bias_fallback'])
        for name, box in self.save_intermediates.items():
            box.setChecked(name in options['save_intermediates'])
        missing = [path for path in [*folders.values(), options['temporary']] if path and not Path(path).is_dir()]
        self.status.setText('Settings loaded for review.' + (' Unavailable folders: ' + ', '.join(missing) if missing else ''))
        self.settings_loaded.emit(path, options)

    def choose(self, role):
        path = choose_directory(self, f'Select {role} folder', self.folders[role].text())
        if path:
            self.folders[role].setText(path)
            if role == 'light':
                from pathlib import Path
                self.lunar_margin.setValue(-2 if any(p.suffix.lower() == '.nef' for p in Path(path).iterdir()) else 0)
                self.offer_related(path)

    def choose_storage(self, edit, temporary):
        if temporary:
            path = choose_directory(self, 'Choose temporary files folder', edit.text())
        else:
            path = choose_output_folder(self, edit.text())
        if path:
            edit.setText(path)
            if not temporary:
                self._automatic_destination = False

    def update_default_destination(self, lights):
        if self._automatic_destination:
            self.destination.setText(str(Path(lights).absolute().parent/'HDR_output') if lights else '')

    def offer_related(self, path):
        candidates = related_folders(path)
        if not any(candidates.values()):
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Nearby calibration folders')
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel('Use calibration folders found in the same nearby tree?'))
        choices = {}
        for role, paths in candidates.items():
            if not paths:
                continue
            layout.addWidget(QLabel(role.replace('_', ' ').title()))
            choice = ScrollSafeComboBox()
            choice.setToolTip('Choose a nearby calibration folder or keep the current selection. No files are moved or changed.')
            scroll_tooltips(choice.view(), 'the calibration folder choices')
            choice.addItem('Keep current selection', None)
            choice.setItemData(0, 'Leave the existing selection unchanged.', Qt.ItemDataRole.ToolTipRole)
            for candidate in paths:
                choice.addItem(str(candidate), str(candidate))
                choice.setItemData(choice.count()-1, f'Select calibration folder: {candidate}', Qt.ItemDataRole.ToolTipRole)
            if len(paths) == 1 and not self.folders[role].text():
                choice.setCurrentIndex(1)
            choices[role] = choice
            layout.addWidget(choice)
        apply = QPushButton('Use selected folders')
        apply.setToolTip('Apply the selected calibration folders; existing choices are kept unless explicitly replaced.')
        apply.clicked.connect(dialog.accept)
        layout.addWidget(apply)
        skip = QPushButton('Skip')
        skip.setToolTip('Ignore nearby suggestions and keep all existing selections.')
        skip.clicked.connect(dialog.reject)
        label_tooltips(dialog)
        layout.addWidget(skip)
        if dialog.exec():
            for role, choice in choices.items():
                if choice.currentData():
                    self.folders[role].setText(choice.currentData())

    def submit(self):
        try:
            from .desktop_jobs import unfinished_jobs, resume_folder_job
            selected = {role: str(Path(edit.text()).resolve()) for role, edit in self.folders.items() if edit.text()}
            matches = [(path, job) for path, job in unfinished_jobs(self.temporary.text())
                       if job.get('folders') == selected and job.get('hdr_mode', 'aesthetic') == self.hdr_mode.currentData()
                       and job.get('alignment', 'conservative') == self.alignment.currentData()
                       and job.get('output_format', 'both') == self.output_format.currentData()
                       and job.get('detect_hot_pixels', False) == self.detect_hot_pixels.isChecked()
                       and not job.get('prominence', False)
                       and job.get('lunar_margin_px') == self.lunar_margin.value()
                       and job.get('output_base_name', '') == self.base_name.text().strip()
                       and job.get('save_intermediates', []) == [name for name, box in self.save_intermediates.items() if box.isChecked()]
                       and job.get('bias_fallback', False) == self.bias_fallback.isChecked()]
            if matches:
                path, job = matches[0]
                choice, accepted = ask_choice(self, 'Saved work found',
                    f'A matching unfinished job exists for {job["destination"]}. Resume its saved settings and verified work?',
                    ['Resume saved work', 'Start a new job'])
                if not accepted:
                    return
                if choice == 'Resume saved work':
                    self.job_path, self.job = resume_folder_job(path)
                    self.accept()
                    return
            self.job_path, self.job = create_folder_job({role: edit.text() for role, edit in self.folders.items()},
                mode=self.hdr_mode.currentData(), bias_fallback=self.bias_fallback.isChecked(),
                save_intermediates=[name for name, checkbox in self.save_intermediates.items() if checkbox.isChecked()],
                detect_hot_pixels=self.detect_hot_pixels.isChecked(), lunar_margin=self.lunar_margin.value(),
                temporary=self.temporary.text(), destination=self.destination.text(), base_name=self.base_name.text(),
                alignment=self.alignment.currentData(), output_format=self.output_format.currentData())
            self.accept()
        except OutputNameConflict as error:
            name, accepted = ask_text(self, 'Output file already exists',
                f'{error}\nEnter a different output base name:', text=self.base_name.text())
            if accepted:
                self.base_name.setText(name)
                self.submit()
        except (ValueError, OSError) as error:
            self.status.setText(str(error))

    def update_calibration_warning(self, *args):
        available = bool(self.folders['bias'].text() and not self.folders['dark'].text()
                         and not self.folders['dark_flat'].text())
        self.bias_fallback.setEnabled(available)
        if not available:
            self.bias_fallback.setChecked(False)
        if self.bias_fallback.isChecked():
            text = 'Warning: bias corrects the pedestal only. Thermal current, hot pixels and exposure-dependent dark signal are not corrected by bias alone. You may continue.'
        elif not self.folders['dark'].text() or not self.folders['dark_flat'].text():
            text = 'Warning: dark or dark-flat correction is missing. Processing may continue; residual dark signal can affect the result.'
        else:
            text = 'Calibration compatibility will be checked. Validated bias files in dark slots are accepted with a pedestal-only warning; no thermal correction is claimed.'
        self.calibration_warning.setText(text)

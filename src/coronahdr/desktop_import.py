"""Explicit acquisition recipes for the desktop import workflow."""
import json
from pathlib import Path

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout)

from .desktop_style import heading, label_tooltips, path_tooltip, scroll_tooltips
from .desktop_paths import choose_directory, choose_output_folder
from .desktop_jobs import create_recipe_job
from .raw import RAW_EXTENSIONS


class ImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty('coronahdr_tooltips', True)
        self.setWindowTitle('Import and calibrate — explicit acquisition recipe')
        self.resize(900, 700)
        self.base = Path.cwd()
        self.job_path = self.job = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)
        description = QLabel('Load an existing recipe or add files by role. Review metadata declarations below. '
            'Set reference to a zero-based light index, or null for automatic selection. '
            'Unknown linearity, exposure and detector metadata must be declared explicitly. '
            'Bias frames belong under bias; thermal dark correction is never inferred.')
        description.setWordWrap(True)
        layout.addWidget(description)
        layout.addWidget(heading('Acquisition recipe', 'Add input files and review their detector metadata and processing settings.'))
        files = QHBoxLayout()
        load = QPushButton('Load recipe…')
        load.setToolTip('Load an acquisition JSON recipe. Relative image paths resolve against its directory.')
        load.clicked.connect(self.load_recipe)
        files.addWidget(load)
        for role in ('light', 'bias', 'dark', 'flat', 'dark_flat'):
            button = QPushButton(f'Add {role}…')
            button.setToolTip(f'Add original {role.replace("_", " ")} files to the recipe without moving or modifying them. Required metadata must still be declared.')
            button.clicked.connect(lambda checked=False, role=role: self.add_files(role))
            files.addWidget(button)
        layout.addLayout(files)
        declare = QPushButton('Declare listed RAW files as linear Bayer CFA')
        declare.setToolTip('Explicitly set state=linear_cfa on listed RAW lights and calibrations. TIFF/FITS remain unchanged.')
        declare.clicked.connect(self.declare_raw_cfa)
        layout.addWidget(declare)
        response = QHBoxLayout()
        response_label = QLabel('Common light response group')
        response.addWidget(response_label)
        self.response_group = QLineEdit()
        self.response_group.setPlaceholderText('e.g. camera-ISO800-f8')
        self.response_group.setToolTip('Declare a common camera, ISO/gain, aperture and response. Exposure times may differ. Do not group incompatible acquisitions.')
        response_label.setBuddy(self.response_group)
        response.addWidget(self.response_group)
        apply_response = QPushButton('Declare for listed lights')
        apply_response.setToolTip('Apply the declared response group to all currently listed lights; conflicting declarations are rejected.')
        apply_response.clicked.connect(self.declare_response_group)
        response.addWidget(apply_response)
        layout.addLayout(response)
        self.editor = QPlainTextEdit()
        scroll_tooltips(self.editor, 'the acquisition recipe')
        self.editor.setToolTip('Editable acquisition JSON: review paths, detector declarations, calibration, reference and processing settings before starting.')
        self.editor.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.editor.setPlainText(json.dumps(dict(method='weighted', reference=None, max_shift=256,
            frames=[], calibration={}, processing=dict(mode='disk', tile_rows=96,
                registration_bin=1, canvas='reference', rejection_policy='error')), indent=2))
        layout.addWidget(self.editor)
        layout.addWidget(heading('Output', 'Choose where the result and temporary preparation files will be stored.'))
        output = QHBoxLayout()
        output_label = QLabel('Output folder')
        output.addWidget(output_label)
        from .work_paths import default_work_directory
        self.destination = QLineEdit(str(default_work_directory()/'imported-run'))
        path_tooltip(self.destination, 'Folder for the HDR master and diagnostic files. Existing files are never overwritten.')
        output_label.setBuddy(self.destination)
        output.addWidget(self.destination)
        choose = QPushButton('Choose…')
        choose.setToolTip('Select the existing folder where HDR products will be published.')
        choose.clicked.connect(self.choose_output)
        output.addWidget(choose)
        layout.addLayout(output)
        temporary = QHBoxLayout()
        temporary_label = QLabel('Temporary files folder')
        temporary.addWidget(temporary_label)
        self.temporary = QLineEdit(str(default_work_directory()))
        path_tooltip(self.temporary, 'Prepared caches and diagnostic masks are retained here for reuse. Keep this folder available after processing.')
        temporary_label.setBuddy(self.temporary)
        temporary.addWidget(self.temporary)
        choose_temporary = QPushButton('Choose…')
        choose_temporary.setToolTip(self.temporary.toolTip())
        self.temporary.textChanged.connect(lambda _text: choose_temporary.setToolTip(self.temporary.toolTip()))
        choose_temporary.clicked.connect(self.choose_temporary)
        temporary.addWidget(choose_temporary)
        layout.addLayout(temporary)
        self.status = QLabel('No source file will be changed. Missing metadata will stop processing with an error.')
        self.status.setToolTip('Current recipe guidance or the most recent metadata or storage validation error.')
        self.status.setWordWrap(True)
        self.status.setProperty('role', 'status')
        layout.addWidget(self.status)
        actions = QHBoxLayout()
        start = QPushButton('Calibrate, register and export')
        start.setToolTip('Validate the recipe and launch detector calibration, registration and weighted HDR export in the background.')
        start.clicked.connect(self.submit)
        close = QPushButton('Close')
        close.setToolTip('Close this dialog without starting processing or modifying originals.')
        close.clicked.connect(self.reject)
        actions.addWidget(start)
        actions.addWidget(close)
        layout.addLayout(actions)
        label_tooltips(self)

    def load_recipe(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load acquisition recipe', str(self.base), 'JSON (*.json)')
        if path:
            try:
                recipe = json.loads(Path(path).read_text())
                if not isinstance(recipe, dict):
                    raise ValueError('Recipe must be a JSON object')
                self.editor.setPlainText(json.dumps(recipe, indent=2))
                self.base = Path(path).resolve().parent
                self.status.setText(f'Relative paths use {self.base}')
            except (OSError, ValueError) as error:
                self.status.setText(str(error))

    def add_paths(self, role, paths):
        recipe = json.loads(self.editor.toPlainText())
        entries = recipe.setdefault('frames', []) if role == 'light' else recipe.setdefault('calibration', {}).setdefault(role, [])
        known = {(self.base/item['path']).resolve() for item in entries}
        for path in paths:
            path = Path(path).resolve()
            if path not in known:
                entry = dict(path=str(path))
                if role != 'light':
                    entry['overrides'] = dict(role=role)
                entries.append(entry)
                known.add(path)
        self.editor.setPlainText(json.dumps(recipe, indent=2))

    def add_files(self, role):
        paths, _ = QFileDialog.getOpenFileNames(self, f'Add {role} frames', str(self.base),
            'Image files (*.fits *.fit *.fts *.tif *.tiff *.NEF *.nef *.CR2 *.cr2 *.CR3 *.cr3 *.ARW *.arw *.DNG *.dng);;All files (*)')
        try:
            self.add_paths(role, paths)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            self.status.setText(f'Correct the recipe before adding files: {error}')

    def declare_raw_cfa(self):
        try:
            recipe = json.loads(self.editor.toPlainText())
            raw_items = [item for entries in [recipe.get('frames', []), *recipe.get('calibration', {}).values()]
                         for item in entries if Path(item['path']).suffix.lower() in RAW_EXTENSIONS]
            for item in raw_items:
                state = item.get('overrides', {}).get('state')
                if state is not None and state != 'linear_cfa':
                    raise ValueError(f"Conflicting state={state} for {item['path']}; review this declaration first")
            for item in raw_items:
                item.setdefault('overrides', {})['state'] = 'linear_cfa'
            self.editor.setPlainText(json.dumps(recipe, indent=2))
            self.status.setText(f'Explicit linear_cfa declaration applied to {len(raw_items)} RAW files. '
                                'Other metadata and TIFF/FITS declarations are unchanged.')
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            self.status.setText(f'Cannot declare RAW state: {error}')

    def declare_response_group(self):
        try:
            group = self.response_group.text().strip()
            if not group:
                raise ValueError('Enter a response group identifying common camera, ISO/gain, aperture and response')
            recipe = json.loads(self.editor.toPlainText())
            lights = recipe.get('frames', [])
            if not lights:
                raise ValueError('Add light frames first')
            for item in lights:
                previous = item.get('overrides', {}).get('response_group')
                if previous is not None and previous != group:
                    raise ValueError(f"Conflicting response_group={previous} for {item['path']}; review this declaration first")
            for item in lights:
                item.setdefault('overrides', {})['response_group'] = group
            self.editor.setPlainText(json.dumps(recipe, indent=2))
            self.status.setText(f'Explicit response group applied to {len(lights)} lights. '
                'Exposure times are preserved. Lunar masks and low-signal thresholds still require valid metadata.')
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            self.status.setText(f'Cannot declare response group: {error}')

    def choose_output(self):
        path = choose_output_folder(self, self.destination.text())
        if path:
            self.destination.setText(path)

    def choose_temporary(self):
        path = choose_directory(self, 'Choose temporary files folder', self.temporary.text())
        if path:
            self.temporary.setText(path)

    def submit(self):
        try:
            recipe = json.loads(self.editor.toPlainText())
            if not isinstance(recipe, dict):
                raise ValueError('Recipe must be a JSON object')
            self.job_path, self.job = create_recipe_job(recipe, self.base, self.destination.text(), temporary=self.temporary.text())
            self.accept()
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.status.setText(str(error))

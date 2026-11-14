"""Review and clean application-owned temporary jobs."""
from datetime import datetime
from pathlib import Path
import sys

from PySide6.QtCore import QObject, QRunnable, QSettings, QThreadPool, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QTextOption
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget,
    QMenu, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from .cleanup import clean_jobs, scan_jobs
from .desktop_paths import choose_directory


def locations():
    value = QSettings('CoronaHDR', 'CoronaHDR').value('temporary_locations', [])
    return [value] if isinstance(value, str) else list(value or [])


def format_size(size):
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if size < 1024 or unit == 'TiB':
            return f'{size:.0f} {unit}' if unit == 'B' else f'{size:.2f} {unit}'
        size /= 1024


def remember_location(path):
    if not path:
        return
    paths = locations()
    path = str(Path(path).expanduser().absolute())
    if path not in paths:
        paths.append(path)
        QSettings('CoronaHDR', 'CoronaHDR').setValue('temporary_locations', paths)


def message_dialog(parent, title, text, *, confirm=False):
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    message = QPlainTextEdit()
    message.setReadOnly(True)
    message.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    message.setPlainText(text)
    layout.addWidget(message)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No if confirm
                               else QDialogButtonBox.StandardButton.Ok)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    if confirm:
        buttons.button(QDialogButtonBox.StandardButton.No).setDefault(True)
    layout.addWidget(buttons)
    screen = parent.screen() if parent else QApplication.primaryScreen()
    available = screen.availableGeometry()
    dialog.resize(min(700, available.width() * 3 // 4), min(450, available.height() * 3 // 4))
    return dialog.exec() == QDialog.DialogCode.Accepted


class CleanupSignals(QObject):
    progress = Signal(str)
    finished = Signal(object, object)


class CleanupTask(QRunnable):
    def __init__(self, paths, selected=None):
        super().__init__()
        self.setAutoDelete(False)
        self.paths, self.selected = paths, selected
        self.signals = CleanupSignals()

    def run(self):
        try:
            result = (scan_jobs(self.paths, self.signals.progress.emit) if self.selected is None else
                      clean_jobs(self.paths, self.selected, self.signals.progress.emit))
            self.signals.finished.emit(result, None)
        except Exception as error:
            self.signals.finished.emit(None, str(error))


class SizeItem(QTableWidgetItem):
    def __lt__(self, other):
        return self.data(Qt.ItemDataRole.UserRole) < other.data(Qt.ItemDataRole.UserRole)


class CleanupDialog(QDialog):
    def __init__(self, parent, initial):
        super().__init__(parent)
        self.setWindowTitle('Clean up temporary data')
        self.resize(1200, 680)
        self.task = None
        self.jobs = []
        remember_location(initial)
        layout = QVBoxLayout(self)
        note = QLabel('Select inactive CoronaHDR jobs to permanently remove temporary data. '
            'Final images and logs are protected. Removing preparation, masks and reports prevents resuming or inspecting that work. '
            'Dependencies in disconnected or unlisted locations cannot be detected. Sizes are estimates, not guaranteed free space.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.locations = QListWidget()
        self.locations.addItems(locations())
        self.locations.setMaximumHeight(90)
        self.locations.setToolTip('Temporary folders searched for application job records; disconnected folders remain listed.')
        self.locations.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.locations.customContextMenuRequested.connect(self.location_menu)
        layout.addWidget(self.locations)
        actions = QHBoxLayout()
        self.add_button = QPushButton('Add temporary folder…')
        self.add_button.setToolTip('Add a temporary folder on any connected drive to the search list.')
        self.add_button.clicked.connect(self.add_location)
        self.remove_button = QPushButton('Remove folder from list')
        self.remove_button.setToolTip('Stop searching the selected folder. This does not delete its contents.')
        self.remove_button.clicked.connect(self.remove_location)
        self.refresh_button = QPushButton('Refresh')
        self.refresh_button.setToolTip('Search the listed locations again and update job sizes and status.')
        self.refresh_button.clicked.connect(self.scan)
        for button in (self.add_button, self.remove_button, self.refresh_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.select_all = QCheckBox('Select all inactive jobs')
        self.select_all.setToolTip('Select or clear all inactive jobs. Locked jobs cannot be selected.')
        self.select_all.clicked.connect(self.select_all_jobs)
        layout.addWidget(self.select_all)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(['Select', 'Folder size (estimate)', 'Date and time', 'Status', 'Lights', 'Temporary folder', 'Output folder'])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setToolTip('Check each inactive job whose temporary data you want to delete.')
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.job_menu)
        self.table.itemChanged.connect(self.update_total)
        layout.addWidget(self.table)
        self.status = QLabel('Scanning…')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        bottom = QHBoxLayout()
        self.delete_button = QPushButton('Delete selected data…')
        self.delete_button.setToolTip('Review and confirm permanent deletion of selected job data.')
        self.delete_button.clicked.connect(self.confirm_delete)
        self.close_button = QPushButton('Close')
        self.close_button.setToolTip('Close the cleanup window.')
        self.close_button.clicked.connect(self.close)
        bottom.addWidget(self.delete_button)
        bottom.addWidget(self.close_button)
        layout.addLayout(bottom)
        self.scan()

    def paths(self):
        return [self.locations.item(i).text() for i in range(self.locations.count())]

    def open_folder(self, path):
        if not path or not Path(path).is_dir():
            message_dialog(self, 'Folder unavailable', f'This folder is missing or unavailable:\n{path}')
        elif not QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).absolute()))):
            message_dialog(self, 'Cannot open folder', f'The system could not open this folder:\n{path}')

    def folder_menu(self, widget, position, path, job_path=None, available=True):
        name = 'Finder' if sys.platform == 'darwin' else 'Explorer' if sys.platform == 'win32' else 'file manager'
        menu = QMenu(self)
        action = menu.addAction(f'Open in {name}' if available else 'Folder unavailable', lambda: self.open_folder(path))
        action.setEnabled(available)
        if job_path and job_path != path:
            menu.addAction(f'Open job folder in {name}', lambda: self.open_folder(job_path))
        menu.exec(widget.viewport().mapToGlobal(position))

    def location_menu(self, position):
        item = self.locations.itemAt(position)
        if item:
            self.folder_menu(self.locations, position, item.text())

    def job_menu(self, position):
        index = self.table.indexAt(position)
        if not index.isValid() or index.column() not in (4, 5, 6):
            return
        row = self.table.item(index.row(), 0).data(Qt.ItemDataRole.UserRole)
        job = row['job']
        path = (job.get('folders', {}).get('light', job.get('source', '')) if index.column() == 4 else
                row['location'] if index.column() == 5 else job.get('destination', ''))
        if path:
            self.folder_menu(self.table, position, path, row['path'],
                             row['output_available'] if index.column() == 6 else True)

    def add_location(self):
        path = choose_directory(self, 'Choose a temporary files folder', '')
        if path:
            remember_location(path)
            self.locations.clear()
            self.locations.addItems(locations())
            self.scan()

    def remove_location(self):
        row = self.locations.currentRow()
        if row >= 0:
            self.locations.takeItem(row)
            QSettings('CoronaHDR', 'CoronaHDR').setValue('temporary_locations', self.paths())
            self.scan()

    def set_busy(self, busy):
        for item in (self.add_button, self.remove_button, self.refresh_button, self.delete_button, self.close_button, self.select_all, self.table):
            item.setEnabled(not busy)

    def run_task(self, selected=None):
        self.set_busy(True)
        self.task = CleanupTask(self.paths(), selected)
        self.task.signals.progress.connect(self.status.setText)
        self.task.signals.finished.connect(self.completed if selected is not None else self.scanned)
        QThreadPool.globalInstance().start(self.task)

    def scan(self):
        self.run_task()

    def scanned(self, result, error):
        self.task = None
        self.set_busy(False)
        if error:
            self.status.setText(f'Scan failed: {error}')
            return
        self.jobs = sorted(result['jobs'], key=lambda row: row['date'], reverse=True)
        self.table.setSortingEnabled(False)
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.jobs))
        for i, row in enumerate(self.jobs):
            box = QTableWidgetItem()
            box.setData(Qt.ItemDataRole.UserRole, row)
            box.setFlags(Qt.ItemFlag.NoItemFlags if row['active'] else Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            box.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(i, 0, box)
            try:
                date = datetime.fromisoformat(row['date']).astimezone().strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                date = row['date']
            job = row['job']
            values = [format_size(row['bytes']), f'{date} ({row["date_source"]})', 'Active / locked' if row['active'] else job.get('status', 'unknown'),
                      Path(job.get('folders', {}).get('light', job.get('source', ''))).name,
                      row['location'], job.get('destination', '') + ('' if row['output_available'] else ' (missing or unavailable)')]
            for column, value in enumerate(values, 1):
                item = SizeItem(str(value)) if column == 1 else QTableWidgetItem(str(value))
                if column == 1:
                    item.setData(Qt.ItemDataRole.UserRole, row['bytes'])
                item.setToolTip(str(value))
                self.table.setItem(i, column, item)
        self.table.blockSignals(False)
        self.table.setSortingEnabled(True)
        self.table.sortItems(2, Qt.SortOrder.DescendingOrder)
        self.table.resizeColumnsToContents()
        self.unavailable = result['unavailable']
        self.unsafe = result['unsafe']
        self.update_total()

    def select_all_jobs(self, checked):
        self.table.blockSignals(True)
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.table.blockSignals(False)
        self.update_total()

    def selected_jobs(self):
        return [item.data(Qt.ItemDataRole.UserRole) for i in range(self.table.rowCount())
                if (item := self.table.item(i, 0)) and item.checkState() == Qt.CheckState.Checked]

    def update_total(self, *args):
        selected = self.selected_jobs()
        total = sum(row['bytes'] for row in selected)
        eligible = sum(not row['active'] for row in self.jobs)
        self.select_all.blockSignals(True)
        self.select_all.setCheckState(Qt.CheckState.Checked if eligible and len(selected) == eligible else
                                      Qt.CheckState.PartiallyChecked if selected else Qt.CheckState.Unchecked)
        self.select_all.blockSignals(False)
        unavailable = f' Unavailable: {", ".join(self.unavailable)}.' if getattr(self, 'unavailable', []) else ''
        unsafe = f' Cannot inspect: {"; ".join(self.unsafe)}.' if getattr(self, 'unsafe', []) else ''
        self.status.setText(f'{len(self.jobs)} jobs found; selected estimate {format_size(total)}.{unavailable}{unsafe}')
        self.delete_button.setEnabled(bool(selected) and self.task is None and not getattr(self, 'unsafe', []))

    def confirm_delete(self):
        selected = self.selected_jobs()
        if not selected:
            return
        paths = '\n'.join(f'{row["path"]} ({format_size(row["bytes"])})' for row in selected)
        answer = message_dialog(self, 'Permanently delete temporary data?',
            f'The following jobs will be checked and their temporary data permanently deleted:\n\n{paths}\n\nEstimated total: {format_size(sum(row["bytes"] for row in selected))}.\n\n'
            'Logs will be copied to the output folder. Protected or dependent jobs will be skipped. '
            'Deleted work cannot be resumed. Unlisted or disconnected dependencies cannot be detected.', confirm=True)
        if answer:
            self.run_task([row['path'] for row in selected])

    def completed(self, result, error):
        self.task = None
        self.set_busy(False)
        if error:
            message_dialog(self, 'Cleanup failed', error)
        else:
            text = '\n'.join(f'{path}: {item["status"]}' + (f' — {item["reason"]}' if item.get('reason') else '')
                             for path, item in result.items())
            text += f'\n\nEstimated removed: {format_size(sum(item.get("estimated_bytes", 0) for item in result.values()))}.'
            message_dialog(self, 'Cleanup results', text)
        self.scan()

    def closeEvent(self, event):
        if self.task is not None:
            event.ignore()
        else:
            super().closeEvent(event)

    def reject(self):
        if self.task is None:
            super().reject()

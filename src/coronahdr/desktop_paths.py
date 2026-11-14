"""Portable directory pickers for desktop inputs and outputs."""
from pathlib import Path

from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QLabel, QLineEdit, QVBoxLayout


def _prompt(parent, title, message, choices=None, text=''):
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    label = QLabel(message)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    layout.addWidget(label)
    if choices is None:
        control = QLineEdit(text)
    else:
        control = QComboBox()
        control.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        control.setMinimumContentsLength(20)
        control.addItems(choices)
    layout.addWidget(control)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    screen = parent.screen() if parent else QApplication.primaryScreen()
    dialog.setFixedWidth(min(680, screen.availableGeometry().width() * 4 // 5))
    accepted = dialog.exec() == QDialog.DialogCode.Accepted
    return (control.text() if choices is None else control.currentText()), accepted


def ask_choice(parent, title, message, choices):
    return _prompt(parent, title, message, choices=choices)


def ask_text(parent, title, message, text=''):
    return _prompt(parent, title, message, text=text)


def choose_directory(parent, title, current):
    return QFileDialog.getExistingDirectory(
        parent, title, current,
        QFileDialog.Option.ShowDirsOnly) or None


def choose_output_folder(parent, current):
    current = Path(current) if current.strip() else Path.cwd()
    return choose_directory(
        parent, 'Choose output folder',
        str(current if current.is_dir() else current.parent)) or None

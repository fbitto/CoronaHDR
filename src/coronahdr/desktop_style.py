"""Local photographic-workspace styling and scroll-safe parameter controls."""
from html import escape
from math import log
import re

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, Qt
from PySide6.QtGui import QColor, QPalette, QTextDocument
from PySide6.QtWidgets import QApplication, QAbstractItemView, QComboBox, QHeaderView, QLabel, QSpinBox, QToolTip, QWidget


STYLE = '''
QWidget { color: #dddddd; }
QMainWindow, QDialog { background: #2b2b2b; }
QLabel { background: transparent; }
QLabel[role="heading"] { color: #aaaaaa; font-weight: 600;
    letter-spacing: 0.8px; padding-top: 6px; padding-bottom: 4px; }
QLabel[role="status"] { color: #9cdcfe; padding: 6px; }
QLabel[role="warning"] { background: #3b3325; color: #f0ce8b;
    border: 1px solid #806332; border-radius: 3px; padding: 6px; }
QLabel#inspectionCanvas { background: #1c1c1c; color: #aaaaaa; border: 1px solid #444444; }
QPushButton, QToolButton { background: #555555; border: 1px solid #666666;
    border-radius: 3px; padding: 4px 6px; font-weight: 500; }
QPushButton:hover, QToolButton:hover { background: #777777; border-color: #888888; }
QPushButton:pressed, QPushButton:checked, QToolButton:pressed { background: #575757; border-color: #999999; }
QPushButton:disabled { background: #303030; color: #777777; border-color: #444444; }
QPushButton[primary="true"] { background: #666666; border-color: #777777; color: #ffffff; }
QPushButton[primary="true"]:hover { background: #888888; }
QPushButton[primary="true"]:disabled { background: #303030; color: #777777; border-color: #444444; }
QLineEdit, QSpinBox, QComboBox, QPlainTextEdit { background: #303030;
    border: 1px solid #555555; border-radius: 2px; padding: 4px; selection-background-color: #575757; }
QLineEdit:read-only, QPlainTextEdit:read-only { background: #252525; }
QComboBox { combobox-popup: 0; }
QComboBox QAbstractItemView { background: #303030; color: #dddddd;
    selection-background-color: #575757; selection-color: #ffffff; }
QPushButton::menu-indicator { subcontrol-origin: padding; subcontrol-position: right center; right: 6px; }
QPushButton:focus, QLineEdit:focus, QSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTableWidget:focus { border: 1px solid #9cdcfe; }
QPushButton[zoomOption="true"]:checked, QPushButton[zoomOption="true"]:checked:focus {
    border: 1px solid #e5a00d; }
QWidget:disabled { color: #777777; }
QTableWidget { background: #303030; alternate-background-color: #383838;
    gridline-color: #444444; border: 1px solid #555555;
    selection-background-color: #575757; selection-color: #ffffff; }
QHeaderView::section { background: #383838; color: #bbbbbb; padding: 3px;
    border: none; border-right: 1px solid #555555; border-bottom: 1px solid #555555; }
QTableCornerButton::section { background: #383838; border: 1px solid #555555; }
QCheckBox { spacing: 5px; color: #bbbbbb; }
QCheckBox:focus { color: #9cdcfe; }
QCheckBox::indicator, QTableWidget::indicator { width: 13px; height: 13px; }
QCheckBox::indicator:unchecked, QTableWidget::indicator:unchecked {
    background: #303030; border: 1px solid #999999; border-radius: 2px; }
QCheckBox::indicator:checked, QTableWidget::indicator:checked {
    background: #e5a00d; border: 1px solid #f5b025; border-radius: 2px; }
QProgressBar { background: #252525; border: 1px solid #555555;
    border-radius: 3px; text-align: center; min-height: 16px; }
QProgressBar::chunk { background: #666666; }
QScrollArea { border: none; background: #2b2b2b; }
QScrollArea#hdrImageView { background: #111111; border: 1px solid #3c3c3c; }
QStatusBar { background: #202327; border-top: 1px solid #454a50; }
QToolButton { padding: 5px 8px; text-align: left; }
QToolButton[sectionHeader="true"] { background: #303030; color: #dddddd;
    border: none; border-bottom: 1px solid #202020; border-radius: 0;
    text-align: left; padding: 6px 8px; font-weight: 500; }
QToolButton[sectionHeader="true"]:hover { background: #383838; }
QSlider::groove:horizontal { background: #444444; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal { background: #777777; width: 12px; margin: -4px 0;
    border: 1px solid #888888; border-radius: 6px; }
QSlider::handle:horizontal:hover { background: #999999; }
QSlider::sub-page:horizontal { background: #777777; border-radius: 2px; }
QSplitter::handle { background: #444444; }
QToolTip, QLabel#coronahdrTooltip { background: #151a21; color: #f5f7fa;
    border: 1px solid #8796a8; border-radius: 3px; padding: 7px; }
'''


def tooltip_paragraph(text):
    """Preserve plain text while permitting long paths to wrap at narrow widths."""
    text = re.sub(r'(\S{28})(?=\S)', lambda match: match.group(1)+'\u200b', text)
    body = escape(text).replace('\n', '<br/>')
    document = QTextDocument()
    document.setDefaultFont(QToolTip.font())
    document.setDocumentMargin(0)
    document.setHtml(body)
    sizes = []
    for width in range(160, 421, 20):
        document.setTextWidth(width)
        height = document.size().height()
        sizes.append((abs(log(((width+16)/(height+16))/(16/9))), width))
    width = min(sizes)[1]
    return f'<qt><table width="{width}" cellspacing="0" cellpadding="0"><tr><td>{body}</td></tr></table></qt>'


class TooltipParagraphs(QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.tip = QLabel(None, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.tip.setObjectName('coronahdrTooltip')
        self.tip.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.tip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.tip.setTextFormat(Qt.TextFormat.RichText)
        self.tip.setWordWrap(True)
        self.tip.setMaximumWidth(450)
        self.tip.setFont(QToolTip.font())
        self.owner = None
        self.owner_rect = QRect()

    def eventFilter(self, widget, event):
        if widget is self.owner and event.type() in (QEvent.Type.Leave, QEvent.Type.Hide, QEvent.Type.MouseButtonPress, QEvent.Type.KeyPress):
            self.tip.hide()
            self.owner = None
        elif widget is self.owner and event.type() == QEvent.Type.MouseMove and not self.owner_rect.isNull() and not self.owner_rect.contains(event.pos()):
            self.tip.hide()
            self.owner = None
        if event.type() != QEvent.Type.ToolTip or not isinstance(widget, QWidget) or widget is self.tip:
            return False
        ancestor = widget
        view = None
        owned = False
        while ancestor is not None:
            if view is None and isinstance(ancestor, QAbstractItemView):
                view = ancestor
            owned |= bool(ancestor.property('coronahdr_tooltips'))
            ancestor = ancestor.parentWidget()
        if not owned:
            return False
        text = None
        tip_rect = QRect()
        if view is not None:
            position = view.viewport().mapFromGlobal(widget.mapToGlobal(event.pos()))
            if isinstance(view, QHeaderView):
                section = view.logicalIndexAt(position)
                if section >= 0:
                    text = view.model().headerData(section, view.orientation(), Qt.ItemDataRole.ToolTipRole)
                    start, size = view.sectionViewportPosition(section), view.sectionSize(section)
                    tip_rect = (QRect(start, 0, size, view.viewport().height()) if view.orientation() == Qt.Orientation.Horizontal else
                                QRect(0, start, view.viewport().width(), size))
            else:
                index = view.indexAt(position)
                if index.isValid():
                    text = index.data(Qt.ItemDataRole.ToolTipRole)
                    tip_rect = view.visualRect(index)
            if text and widget is not view.viewport():
                tip_rect.moveTopLeft(widget.mapFromGlobal(view.viewport().mapToGlobal(tip_rect.topLeft())))
        ancestor = widget
        while not text and ancestor is not None:
            text = ancestor.toolTip()
            ancestor = ancestor.parentWidget()
        if not isinstance(text, str) or not text:
            return False
        self.owner = widget
        self.owner_rect = tip_rect
        self.tip.setText(tooltip_paragraph(text))
        self.tip.adjustSize()
        point = event.globalPos() + QPoint(12, 18)
        screen = QApplication.screenAt(event.globalPos()) or QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            point.setX(max(area.left(), min(point.x(), area.right()-self.tip.width()+1)))
            point.setY(max(area.top(), min(point.y(), area.bottom()-self.tip.height()+1)))
        self.tip.move(point)
        self.tip.show()
        return True


def apply_theme(app):
    app.setStyle('Fusion')
    palette = QPalette()
    for role, color in ((QPalette.ColorRole.Window, '#2b2b2b'),
                        (QPalette.ColorRole.WindowText, '#dddddd'),
                        (QPalette.ColorRole.Base, '#303030'),
                        (QPalette.ColorRole.AlternateBase, '#383838'),
                        (QPalette.ColorRole.Text, '#dddddd'),
                        (QPalette.ColorRole.Button, '#3c3c3c'),
                        (QPalette.ColorRole.ButtonText, '#dddddd'),
                        (QPalette.ColorRole.Highlight, '#575757'),
                        (QPalette.ColorRole.HighlightedText, '#ffffff')):
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet(STYLE)
    tooltip_font = app.font('QPushButton')
    tooltip_font.setPointSizeF(max(1.0, tooltip_font.pointSizeF() - 1.0))
    QToolTip.setFont(tooltip_font)
    if not hasattr(app, '_coronahdr_tooltip_paragraphs'):
        app._coronahdr_tooltip_paragraphs = TooltipParagraphs(app)
        app.installEventFilter(app._coronahdr_tooltip_paragraphs)


def heading(text, tip=None):
    label = QLabel(text.upper())
    label.setProperty('role', 'heading')
    if tip:
        label.setToolTip(tip)
    return label


class ScrollSafeComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class ScrollSafeSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        event.ignore()


def label_tooltips(widget):
    """Give form captions their control's help and static labels their full text."""
    for label in widget.findChildren(QLabel):
        if not label.toolTip():
            buddy = label.buddy()
            label.setToolTip((buddy.toolTip().split('\n', 1)[0] if buddy else '') or label.text())


def path_tooltip(edit, description):
    """Keep a truncated path readable without hiding the control's purpose."""
    def update(text):
        edit.setToolTip(f'{description}\nPath: {text}' if text.strip() else f'{description}\nNo path selected.')
    edit.textChanged.connect(update)
    update(edit.text())


def scroll_tooltips(area, content):
    area.verticalScrollBar().setToolTip(f'Scroll vertically through {content}.')
    area.horizontalScrollBar().setToolTip(f'Scroll horizontally through {content}.')

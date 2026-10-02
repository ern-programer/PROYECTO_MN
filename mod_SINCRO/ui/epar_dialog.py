"""Dialogos con cabecera propia para los flujos de configuracion y carga."""
from __future__ import annotations

from PyQt6.QtCore import QByteArray, QEvent, Qt
from PyQt6.QtGui import QColor, QGuiApplication, QPalette
from PyQt6.QtWidgets import (
    QAbstractButton, QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QStyle,
    QToolButton, QVBoxLayout,
)


def button_accent(button, fallback: str = "#7898a1") -> str:
    if isinstance(button, QAbstractButton):
        button.ensurePolished()
        return button.palette().color(QPalette.ColorRole.Button).name()
    return fallback


class EParDialog(QDialog):
    def __init__(
        self, parent=None, *, title: str, accent: str = "#7898a1",
        settings=None, geometry_key: str = "",
    ):
        super().__init__(parent, Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle(title)
        self.setSizeGripEnabled(True)
        self._drag_offset = None
        self._settings = settings
        self._geometry_key = geometry_key
        self._geometry_restored = False
        self._accent = QColor(accent)
        if not self._accent.isValid():
            self._accent = QColor("#7898a1")
        self._accent_text = "#172028" if self._accent.lightness() >= 128 else "#e7edef"
        self.setProperty("eparDialog", True)
        self.setStyleSheet("QDialog[eparDialog='true'] { border:1px solid " + self._accent.name() + "; }")
        self.content_layout = QVBoxLayout(self)
        self.content_layout.setContentsMargins(8, 8, 8, 24)
        self.content_layout.setSpacing(10)

        self._header = QFrame(self)
        self._header.setObjectName("eparDialogHeader")
        self._header.setFixedHeight(44)
        self._header.setStyleSheet("""
        QFrame#eparDialogHeader { background:ACCENT; border-left:10px solid EDGE;
          border-top-left-radius:12px; border-bottom-left-radius:4px; }
        QLabel#eparDialogTitle { color:FOREGROUND; background:transparent; font-weight:600; }
        QToolButton { background:ACCENT; border:0; border-radius:4px; }
        QToolButton:hover { background:HOVER; }
        """.replace("ACCENT", self._accent.name()).replace("EDGE", self._accent.darker(135).name())
            .replace("FOREGROUND", self._accent_text).replace("HOVER", self._accent.lighter(115).name()))
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(18, 6, 6, 6)
        self._title_label = QLabel(title)
        self._title_label.setObjectName("eparDialogTitle")
        header_layout.addWidget(self._title_label, 1)
        self._close_button = QToolButton()
        self._close_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton))
        self._close_button.setToolTip("Cerrar")
        self._close_button.setAccessibleName("Cerrar")
        self._close_button.setFixedSize(28, 28)
        self._close_button.clicked.connect(self.reject)
        header_layout.addWidget(self._close_button)
        self.content_layout.addWidget(self._header)
        for widget in (self._header, self._title_label):
            widget.setCursor(Qt.CursorShape.SizeAllCursor)
            widget.installEventFilter(self)

    def apply_compact_style(self):
        self.setProperty("eparCompact", True)
        self.setStyleSheet(self.styleSheet() + """
        QDialog[eparCompact='true'] { background:#171d25; color:#dce3e3; }
        QDialog[eparCompact='true'] QLabel { color:#bac7cb; background:transparent; }
        QDialog[eparCompact='true'] QLineEdit { background:#25323d; color:#e0e6e6;
          border:1px solid #51616e; border-radius:4px; padding:6px; }
        QDialog[eparCompact='true'] QPushButton { background:#303d49; color:#dce3e3;
          border:1px solid #51616e; border-radius:6px; padding:9px 12px; }
        QDialog[eparCompact='true'] QPushButton:hover { background:#435462; }
        QDialog[eparCompact='true'] QPushButton:focus { border:1px solid ACCENT; }
        """.replace("ACCENT", self._accent.name()))

    def showEvent(self, event):
        super().showEvent(event)
        if not self._geometry_restored:
            self.content_layout.activate()
            if self._settings is not None and self._geometry_key:
                geometry = self._settings.value(self._geometry_key)
                if isinstance(geometry, QByteArray) and not geometry.isEmpty():
                    self.restoreGeometry(geometry)
            self._geometry_restored = True
        self._ensure_on_screen()

    def _ensure_on_screen(self):
        screens = QGuiApplication.screens()
        screen = next((screen for screen in screens if screen.availableGeometry().intersects(self.frameGeometry())), self.screen())
        if screen is None:
            return
        area = screen.availableGeometry()
        self.resize(min(self.width(), area.width()), min(self.height(), area.height()))
        self.move(max(area.left(), min(self.x(), area.right() - self.width() + 1)),
                  max(area.top(), min(self.y(), area.bottom() - self.height() + 1)))

    def done(self, result):
        if self._settings is not None and self._geometry_key:
            self._settings.setValue(self._geometry_key, self.saveGeometry())
            self._settings.sync()
        super().done(result)

    def eventFilter(self, watched, event):
        if watched in (self._header, self._title_label):
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                return True
            if event.type() == QEvent.Type.MouseMove and self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
                self.move(event.globalPosition().toPoint() - self._drag_offset)
                return True
            if event.type() == QEvent.Type.MouseButtonRelease:
                self._drag_offset = None
                return True
        return super().eventFilter(watched, event)


class StudyLoadDialog(EParDialog):
    def __init__(self, parent=None, *, accent: str = "#7898a1", settings=None):
        super().__init__(parent, title="Cargar estudio", accent=accent, settings=settings,
                         geometry_key="dialogs/load_study/geometry")
        self.setObjectName("eparStudyLoadDialog")
        self.choice = ""
        self.setMinimumWidth(640)
        self.apply_compact_style()
        self.setStyleSheet(self.styleSheet() + """
        QDialog#eparStudyLoadDialog QPushButton#loadOption { text-align:left; font-weight:600;
          border-left:5px solid ACCENT; }
        QDialog#eparStudyLoadDialog QPushButton[loadChoice='smart'] { background:PRIMARY;
          border-left:5px solid ACCENT; }
        QDialog#eparStudyLoadDialog QPushButton#loadMapping { background:ACCENT; color:FOREGROUND; }
        """.replace("ACCENT", self._accent.name()).replace("PRIMARY", self._accent.darker(140).name())
            .replace("FOREGROUND", self._accent_text))
        self._add_choice("smart", "Carpeta inteligente", "Esfuerzo y Reposo\nEM / ATT / CT / Scatter", QStyle.StandardPixmap.SP_DirOpenIcon)
        self._add_choice("sa", "Cortes SA reconstruidos", "Gated SPECT Short Axis\nAnálisis de fase / FEVI", QStyle.StandardPixmap.SP_FileDialogDetailedView)
        self._add_choice("files", "Elegir archivos", "Uno o dos estudios\nEsfuerzo / Reposo", QStyle.StandardPixmap.SP_FileIcon)
        footer = QHBoxLayout()
        mapping = QPushButton("Mapeo de series")
        mapping.setObjectName("loadMapping")
        self._mapping_button = mapping
        mapping.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView))
        mapping.setAutoDefault(False)
        mapping.clicked.connect(lambda: self._choose("mapping"))
        footer.addWidget(mapping)
        footer.addStretch(1)
        cancel = QPushButton("Cancelar")
        cancel.setAutoDefault(False)
        cancel.clicked.connect(self.reject)
        footer.addWidget(cancel)
        self.content_layout.addSpacing(4)
        self.content_layout.addLayout(footer)
        self.content_layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)

    def mapping_accent(self) -> str:
        return button_accent(self._mapping_button, self._accent.name())

    def _add_choice(self, key: str, text: str, description: str, icon):
        row = QHBoxLayout()
        row.setSpacing(16)
        button = QPushButton(text)
        button.setObjectName("loadOption")
        button.setProperty("loadChoice", key)
        button.setIcon(self.style().standardIcon(icon))
        button.setFixedWidth(235)
        button.setMinimumHeight(48)
        button.setAutoDefault(False)
        button.setDefault(key == "smart")
        button.clicked.connect(lambda: self._choose(key))
        row.addWidget(button)
        label = QLabel(description)
        label.setObjectName("loadDescription")
        label.setWordWrap(True)
        row.addWidget(label, 1)
        self.content_layout.addLayout(row)

    def _choose(self, choice: str):
        self.choice = choice
        self.accept()
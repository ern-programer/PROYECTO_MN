"""Consola flotante EPar + que aloja el sidebar real de GammaSync."""
from __future__ import annotations

from PyQt6.QtCore import QByteArray, QEvent, QPoint, QTimer, Qt, pyqtSignal
from PyQt6 import sip
from PyQt6.QtGui import QCloseEvent, QGuiApplication, QWindow
from PyQt6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QSizeGrip, QVBoxLayout, QWidget


class EParPlusConsole(QWidget):
    """Ventana independiente que mueve, sin copiar, los controles laterales."""

    dockRequested = pyqtSignal()

    def __init__(self, owner):
        super().__init__(None, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self._owner = owner
        self._closing_for_app = False
        self._drag_offset = None
        self.setObjectName("eparPlusConsole")
        self.setWindowTitle("GammaSync — EPar +")
        self.setMinimumSize(620, 520)
        self.resize(760, 850)
        self.setStyleSheet("""
        QWidget#eparPlusConsole { background: #171a25; color: #e7e1cf; }
        QFrame#eparPlusPill { background: #202535; border-left: 18px solid #d9a05f;
          border-right: 5px solid #79a9b8; border-top-right-radius: 24px;
          border-bottom-right-radius: 8px; }
        QLabel#eparPlusStudy { background: #7fa6ad; color: #111827; border-radius: 13px;
          padding: 6px 12px; font-weight: 700; }
        QLabel#eparPlusStatus { background: #d6a35f; color: #171a25; border-radius: 13px;
          padding: 6px 12px; font-size: 11pt; font-weight: 800; }
        QFrame#eparPlusPill QPushButton { background: #aeb2d8; color: #161a28;
          border: none; border-radius: 12px; padding: 7px 10px; font-weight: 700; }
        QFrame#eparPlusPill QPushButton:hover { background: #d4b9dc; }
        QFrame#eparPlusPill QPushButton[eparPrimary="true"] { background: #22a6bd; }
        QWidget#eparPlusSidebarHost { background: #e7eaf0; border: 1px solid #596276;
          border-radius: 8px; }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(5, 5, 5, 5)
        root.setSpacing(5)

        pill = QFrame()
        pill.setObjectName("eparPlusPill")
        self._header = pill
        pill_layout = QVBoxLayout(pill)
        pill_layout.setContentsMargins(12, 8, 8, 8)
        pill_layout.setSpacing(5)

        top = QHBoxLayout()
        self._study_label = QLabel("SIN ESTUDIO")
        self._study_label.setObjectName("eparPlusStudy")
        top.addWidget(self._study_label, 1)
        self._status_label = QLabel("LISTO")
        self._status_label.setObjectName("eparPlusStatus")
        top.addWidget(self._status_label)
        pill_layout.addLayout(top)

        actions = QHBoxLayout()
        actions.setSpacing(5)
        self._add_action(actions, "CARGAR", owner.load_one_or_two_studies)
        self._add_action(actions, "PROCESAR", owner.process_current, primary=True)
        self._add_action(actions, "PDF", owner.open_pdf)
        self._add_action(actions, "CONFIG", owner.open_ui_preferences_dialog)
        images_btn = self._add_action(actions, "VER IMÁGENES", self._show_images)
        images_btn.setToolTip("Maximiza y trae al frente la ventana principal con los visores.")
        dock_btn = self._add_action(actions, "ACOPLAR", self.dockRequested.emit)
        dock_btn.setToolTip("Devuelve los controles al lateral de la ventana principal.")
        pill_layout.addLayout(actions)
        root.addWidget(pill)

        self._sidebar_host = QWidget()
        self._sidebar_host.setObjectName("eparPlusSidebarHost")
        self._sidebar_layout = QVBoxLayout(self._sidebar_host)
        self._sidebar_layout.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._sidebar_host, 1)
        root.addWidget(QSizeGrip(self), 0, Qt.AlignmentFlag.AlignRight)
        QApplication.instance().installEventFilter(self)

    def _add_action(self, layout, text: str, callback, primary: bool = False) -> QPushButton:
        btn = QPushButton(text)
        btn.setProperty("eparPrimary", primary)
        btn.clicked.connect(callback)
        layout.addWidget(btn)
        return btn

    def eventFilter(self, watched, event):
        if watched is self._header or (isinstance(watched, QLabel) and self._header.isAncestorOf(watched)):
            if event.type() == event.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                return True
            if event.type() == event.Type.MouseMove and self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
                self.move(event.globalPosition().toPoint() - self._drag_offset)
                return True
            if event.type() == event.Type.MouseButtonRelease:
                self._drag_offset = None
                return True
        if event.type() == QEvent.Type.Show and self.isVisible():
            if isinstance(watched, QWindow) and watched.transientParent() is self._owner.windowHandle():
                watched.setTransientParent(self.windowHandle())
            elif isinstance(watched, QWidget) and watched.isWindow() and watched.parentWidget() is not None and watched.parentWidget().window() is self._owner:
                watched.winId()
                watched.windowHandle().setTransientParent(self.windowHandle())
                QTimer.singleShot(0, lambda w=watched: self._place_dialog_above(w))
        return super().eventFilter(watched, event)

    def _place_dialog_above(self, window: QWidget) -> None:
        if sip.isdeleted(self) or sip.isdeleted(window) or not self.isVisible() or not window.isVisible():
            return
        if window.windowHandle() is not None and self.windowHandle() is not None:
            window.windowHandle().setTransientParent(self.windowHandle())
        window.raise_()
        window.activateWindow()

    def _show_images(self) -> None:
        """Trae al frente la ventana principal y maximiza el área de imágenes."""
        self._owner.showMaximized()
        self._owner.raise_()
        self._owner.activateWindow()

    def take_sidebar(self, sidebar: QWidget) -> None:
        sidebar.setParent(self._sidebar_host)
        self._sidebar_layout.addWidget(sidebar)
        sidebar.show()

    def release_sidebar(self) -> QWidget | None:
        item = self._sidebar_layout.takeAt(0)
        sidebar = item.widget() if item is not None else None
        if sidebar is not None:
            sidebar.setParent(None)
        return sidebar

    def set_study_text(self, text: str) -> None:
        self._study_label.setText(text or "SIN ESTUDIO")

    def set_status_text(self, text: str) -> None:
        self._status_label.setText(text or "LISTO")

    def bring_to_front(self) -> None:
        """Restaura y activa la consola si quedó minimizada o detrás."""
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        self.ensure_on_screen()
        self.raise_()
        self.activateWindow()

    def restore_safe_geometry(self, geometry) -> None:
        if isinstance(geometry, QByteArray) and not geometry.isEmpty():
            self.restoreGeometry(geometry)
        self.ensure_on_screen()

    def ensure_on_screen(self) -> None:
        rect = self.frameGeometry()
        screens = QGuiApplication.screens()
        if any(screen.availableGeometry().intersects(rect) for screen in screens):
            return
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        self.resize(min(self.width(), area.width()), min(self.height(), area.height()))
        self.move(QPoint(area.left() + 24, area.top() + 24))

    def close_for_app(self) -> None:
        self._closing_for_app = True
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._closing_for_app:
            event.accept()
            return
        event.ignore()
        self.dockRequested.emit()

"""EPar+ Modern: consola LCARS compacta, independiente de EPar +."""
from __future__ import annotations

from PyQt6.QtCore import QByteArray, QEvent, QPoint, QTime, QTimer, Qt, pyqtSignal
from PyQt6 import sip
from PyQt6.QtGui import QCloseEvent, QFont, QGuiApplication, QTextDocument, QWindow
from PyQt6.QtWidgets import QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizeGrip, QSizePolicy, QVBoxLayout, QWidget
from version import __version__


class EParModernConsole(QWidget):
    """Consola LCARS horizontal con panel detallado ocultable."""

    dockRequested = pyqtSignal()

    def __init__(self, owner):
        super().__init__(None, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self._owner = owner
        self._closing_for_app = False
        self._drag_offset = None
        self._clinical_labels = []
        self._clinical_font_sizes = {}
        self._clinical_resize_timer = QTimer(self)
        self._clinical_resize_timer.setSingleShot(True)
        self._clinical_resize_timer.timeout.connect(self._fit_clinical_text)
        self.setObjectName("eparModernConsole")
        self.setWindowTitle("GammaSync — EPar+ Modern")
        self.setMinimumWidth(960)
        self.resize(1180, 760)
        self.setStyleSheet(self._stylesheet())

        root = QVBoxLayout(self)
        root.setContentsMargins(7, 1, 7, 7)
        root.setSpacing(5)

        drag_row = QHBoxLayout()
        drag_row.setContentsMargins(0, 0, 0, 0)
        drag_row.addStretch(1)
        self._drag_handle = QLabel("...")
        self._drag_handle.setObjectName("modernDragHandle")
        self._drag_handle.setFixedSize(72, 10)
        self._drag_handle.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)
        self._drag_handle.setCursor(Qt.CursorShape.OpenHandCursor)
        self._drag_handle.setToolTip("Arrastrar para mover la ventana.")
        self._drag_handle.setAccessibleName("Mover ventana")
        self._drag_handle.setStyleSheet("color:#7fa0a5; background:transparent; font-size:20px; font-weight:700;")
        drag_row.addWidget(self._drag_handle)
        drag_row.addStretch(1)
        root.addLayout(drag_row)

        pill = QFrame()
        pill.setObjectName("modernPill")
        self._header = pill
        pill_grid = QGridLayout(pill)
        pill_grid.setContentsMargins(8, 8, 8, 8)
        pill_grid.setHorizontalSpacing(5)
        pill_grid.setVerticalSpacing(5)

        nav = self._button("CONTROLES", self.toggle_controls, "blue")
        nav.setObjectName("navCap")
        nav.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        nav.setToolTip("Mostrar u ocultar los controles de procesamiento.")
        pill_grid.addWidget(nav, 0, 0, 1, 2)

        self._load_btn = self._button("CARGAR", owner.load_one_or_two_studies, "blue")
        pill_grid.addWidget(self._load_btn, 1, 0, 1, 2)
        self._process_btn = self._button("PROCESAR", owner.process_current, "amber")
        pill_grid.addWidget(self._process_btn, 2, 0)
        self._dock_btn = self._button("ACOPLAR", self.dockRequested.emit, "blue")
        pill_grid.addWidget(self._dock_btn, 2, 1)

        study_panel = QFrame()
        study_panel.setObjectName("studyPanel")
        study_panel.setMinimumHeight(180)
        study_layout = QVBoxLayout(study_panel)
        study_layout.setContentsMargins(10, 5, 10, 5)
        title_row = QHBoxLayout()
        title = QLabel("DATOS DEL PACIENTE")
        title.setObjectName("microTitle")
        title_row.addWidget(title)
        title_row.addStretch(1)
        self._stage_label = QLabel("SINCRO")
        self._stage_label.setObjectName("microValue")
        title_row.addWidget(self._stage_label)
        study_layout.addLayout(title_row)
        self._study_label = QLabel("SIN ESTUDIO CARGADO")
        self._study_label.setObjectName("studyText")
        self._study_label.setTextFormat(Qt.TextFormat.RichText)
        self._study_label.setWordWrap(True)
        self._study_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        study_layout.addWidget(self._study_label, 1)
        pill_grid.addWidget(study_panel, 0, 2, 2, 3)

        ops = self._button("CONFIGURACIÓN", owner.open_ui_preferences_dialog, "amber")
        ops.setObjectName("opsCap")
        ops.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        ops.setToolTip("Abrir la configuración de GammaSync.")
        pill_grid.addWidget(ops, 0, 5, 2, 1)

        status_panel = QFrame()
        status_panel.setObjectName("statusPanel")
        status_panel.setMinimumHeight(180)
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(8, 5, 8, 5)
        status_title = QLabel("RESULTADOS EN VIVO")
        status_title.setObjectName("statusTitle")
        status_layout.addWidget(status_title)
        self._status_label = QLabel("SIN RESULTADOS: PROCESÁ EL ESTUDIO")
        self._status_label.setObjectName("statusText")
        self._status_label.setTextFormat(Qt.TextFormat.RichText)
        self._status_label.setWordWrap(True)
        self._status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        status_layout.addWidget(self._status_label, 1)
        self._progress_label = QLabel("SISTEMA LISTO")
        self._progress_label.setObjectName("progressText")
        status_layout.addWidget(self._progress_label)
        pill_grid.addWidget(status_panel, 0, 6, 2, 3)

        slim = QFrame()
        slim.setObjectName("slimControls")
        slim_l = QVBoxLayout(slim)
        slim_l.setContentsMargins(3, 3, 3, 3)
        slim_l.setSpacing(3)
        slim_l.addWidget(self._button("−", self.showMinimized, "blue"))
        slim_l.addWidget(self._button("□", self._show_images, "blue"))
        slim_l.addWidget(self._button("×", self.dockRequested.emit, "red"))
        pill_grid.addWidget(slim, 0, 9, 2, 1)

        time_panel = QFrame()
        time_panel.setObjectName("timePanel")
        time_l = QVBoxLayout(time_panel)
        time_l.setContentsMargins(8, 2, 8, 2)
        self._clock_label = QLabel("00:00:00")
        self._clock_label.setObjectName("clockText")
        self._clock_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        time_l.addWidget(self._clock_label)
        self._date_label = QLabel("")
        self._date_label.setObjectName("dateText")
        self._date_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        time_l.addWidget(self._date_label)
        pill_grid.addWidget(time_panel, 0, 10, 2, 2)

        bottom_actions = [
            ("VER IMÁGENES", self._show_images, "blue"),
            ("PDF", owner.open_pdf, "teal"),
            ("HTML", owner.open_html_report, "cyan"),
        ]
        for col, (text, callback, role) in enumerate(bottom_actions, start=2):
            pill_grid.addWidget(self._button(text, callback, role), 2, col)

        self._info_label = QLabel(f"GAMMASYNC {__version__.rsplit('.', 1)[0]}\nEPAR+ MODERN")
        self._info_label.setObjectName("brandText")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pill_grid.addWidget(self._info_label, 2, 6, 1, 3)

        pill_grid.setColumnStretch(2, 2)
        pill_grid.setColumnStretch(3, 2)
        pill_grid.setColumnStretch(4, 2)
        pill_grid.setColumnStretch(6, 2)
        pill_grid.setColumnStretch(7, 2)
        pill_grid.setColumnStretch(8, 2)
        pill_grid.setRowStretch(0, 1)
        root.addWidget(pill, 1)

        self._sidebar_host = QWidget()
        self._sidebar_host.setObjectName("modernSidebarHost")
        self._sidebar_host.hide()
        self._sidebar_layout = QVBoxLayout(self._sidebar_host)
        self._sidebar_layout.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._sidebar_host, 1)
        resize_row = QHBoxLayout()
        resize_row.setContentsMargins(0, 0, 0, 0)
        left_grip = QSizeGrip(self)
        left_grip.setToolTip("Redimensionar desde la esquina inferior izquierda.")
        resize_row.addWidget(left_grip)
        resize_row.addStretch(1)
        right_grip = QSizeGrip(self)
        right_grip.setToolTip("Redimensionar desde la esquina inferior derecha.")
        resize_row.addWidget(right_grip)
        root.addLayout(resize_row)

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_clock()
        self._clinical_labels = [(self._study_label, 10), (self._status_label, 9)]
        self._clinical_resize_timer.start(0)
        QApplication.instance().installEventFilter(self)

    def _button(self, text: str, callback, role: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setProperty("lcarsRole", role)
        btn.clicked.connect(callback)
        return btn

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Resize and any(watched is label for label, _ in self._clinical_labels):
            self._clinical_resize_timer.start(0)
        if watched is self._drag_handle or watched is self._header or (isinstance(watched, QLabel) and self._header.isAncestorOf(watched)):
            if event.type() == event.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                if watched is self._drag_handle:
                    self._drag_handle.setCursor(Qt.CursorShape.ClosedHandCursor)
                return True
            if event.type() == event.Type.MouseMove and self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
                self.move(event.globalPosition().toPoint() - self._drag_offset)
                return True
            if event.type() == event.Type.MouseButtonRelease:
                self._drag_offset = None
                self._drag_handle.setCursor(Qt.CursorShape.OpenHandCursor)
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._clinical_resize_timer.start(0)

    def _fit_clinical_text(self):
        for label, minimum in self._clinical_labels:
            area = label.contentsRect().adjusted(2, 2, -2, -2)
            if area.width() <= 0 or area.height() <= 0:
                continue
            label.ensurePolished()
            document = QTextDocument()
            document.setDocumentMargin(0)
            font = QFont(label.font())
            size = minimum
            is_placeholder = label.text().strip().casefold().startswith(("sin estudio", "sin resultados"))
            candidates = (12,) if is_placeholder else range(26, minimum - 1, -1)
            for candidate in candidates:
                font.setPixelSize(candidate)
                document.setDefaultFont(font)
                document.setHtml(label.text())
                document.setTextWidth(area.width())
                if document.size().height() <= area.height() and document.idealWidth() <= area.width():
                    size = candidate
                    break
            if self._clinical_font_sizes.get(label) != size:
                self._clinical_font_sizes[label] = size
                label.setStyleSheet(f"font-size:{size}px;")
            policy = label.sizePolicy()
            policy.setHeightForWidth(False)
            label.setSizePolicy(policy)

    def _stylesheet(self) -> str:
        return """
        QWidget#eparModernConsole { background:#0a0c12; color:#c9d4d8; font-family:'Arial Narrow','Bahnschrift Condensed','Segoe UI'; }
        QFrame#modernPill { background:#121722; border:1px solid #343945; border-radius:10px; }
        QFrame#modernPill QPushButton#navCap { background:#b8c2ad; color:#2a2b2d; border-radius:0; border-top-left-radius:20px; border-bottom-left-radius:6px; padding:8px; font-weight:800; }
        QFrame#modernPill QPushButton#opsCap { background:#d8bb78; color:#2a2b2d; border-radius:0; padding:8px; font-weight:800; }
        QFrame#modernPill QPushButton#navCap:hover { background:#c9d2bf; }
        QFrame#modernPill QPushButton#opsCap:hover { background:#e4ca8c; }
        QFrame#studyPanel { background:#85a4a8; border-top-left-radius:18px; border-bottom-left-radius:5px; }
        QLabel#microTitle, QLabel#microValue { color:#263136; background:transparent; font-weight:800; font-size:9px; }
        QLabel#studyText { color:#102127; background:transparent; font-weight:600; font-size:10px; }
        QFrame#statusPanel { background:#07171f; border:2px solid #2a7383; }
        QLabel#statusTitle { color:#d9b86f; background:transparent; font-size:9px; font-weight:800; }
        QLabel#statusText { color:#81c7ce; background:transparent; font-size:9px; }
        QLabel#progressText { color:#d9b86f; background:transparent; border-top:1px solid #204a56; padding-top:2px; font-size:8px; font-weight:700; }
        QFrame#slimControls { background:#7d9da5; }
        QFrame#timePanel { background:#18212a; border-right:12px solid #758e92; border-top-right-radius:18px; }
        QLabel#clockText { color:#7fa0a5; background:transparent; font-size:28px; font-weight:700; }
        QLabel#dateText { color:#b6a77d; background:transparent; font-size:9px; }
        QLabel#brandText { color:#958861; background:transparent; font-size:10px; font-weight:700; }
        QFrame#modernPill QPushButton { border:none; border-radius:13px; padding:7px 10px; min-height:20px; color:#172028; font-weight:800; }
        QFrame#modernPill QPushButton[lcarsRole='blue'] { background:#7898a1; }
        QFrame#modernPill QPushButton[lcarsRole='teal'] { background:#5aa69f; }
        QFrame#modernPill QPushButton[lcarsRole='cyan'] { background:#1692aa; }
        QFrame#modernPill QPushButton[lcarsRole='amber'] { background:#cfaa67; }
        QFrame#modernPill QPushButton[lcarsRole='red'] { background:#a7615e; }
        QFrame#modernPill QPushButton:hover { background:#d4c18d; }
        QWidget#modernSidebarHost { background:#e7eaf0; border:1px solid #394454; border-radius:7px; }
        """

    def _update_clock(self) -> None:
        now = QTime.currentTime()
        self._clock_label.setText(now.toString("HH:mm:ss"))
        from datetime import date
        self._date_label.setText(date.today().strftime("%d-%m-%Y"))

    def toggle_controls(self) -> None:
        self._sidebar_host.setVisible(not self._sidebar_host.isVisible())
        self.adjustSize()

    def _show_images(self) -> None:
        self._owner.showMaximized()
        self._owner.raise_()
        self._owner.activateWindow()

    def take_sidebar(self, sidebar: QWidget) -> None:
        self._sidebar_host.hide()
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
        self._study_label.setText(text or "SIN ESTUDIO CARGADO")
        self._clinical_resize_timer.start(0)

    def set_status_text(self, text: str) -> None:
        self._progress_label.setText((text or "SISTEMA LISTO").upper())

    def set_patient_html(self, html: str) -> None:
        self._study_label.setText(html or "SIN ESTUDIO CARGADO")
        self._clinical_resize_timer.start(0)

    def set_results_html(self, html: str) -> None:
        self._status_label.setText(html or "SIN RESULTADOS: PROCESÁ EL ESTUDIO")
        self._clinical_resize_timer.start(0)

    def bring_to_front(self) -> None:
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

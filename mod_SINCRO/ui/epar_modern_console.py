"""EPar+ Modern: consola LCARS compacta, independiente de EPar +."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QAbstractAnimation, QByteArray, QEasingCurve, QEvent, QPoint, QPropertyAnimation, QSize, QTime, QTimer, Qt, pyqtSignal
from PyQt6 import sip
from PyQt6.QtGui import QCloseEvent, QColor, QCursor, QFont, QGuiApplication, QIcon, QPainter, QPixmap, QTextDocument, QWindow
from PyQt6.QtWidgets import QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizeGrip, QSizePolicy, QStackedWidget, QToolButton, QVBoxLayout, QWidget
from version import __version__


class EParModernConsole(QWidget):
    """Consola LCARS horizontal con panel detallado ocultable."""

    dockRequested = pyqtSignal()

    def __init__(self, owner):
        super().__init__(None, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self._owner = owner
        self._closing_for_app = False
        self._drag_offset = None
        self._top_mode = False
        self._top_collapsed = False
        self._top_geometry = None
        self._top_screen = None
        self._top_slide = QPropertyAnimation(self, b"pos", self)
        self._top_slide.setDuration(180)
        self._top_slide.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._top_tab = QWidget(self, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowDoesNotAcceptFocus)
        self._top_tab.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._top_tab.setFixedSize(180, 6)
        self._top_tab.setCursor(Qt.CursorShape.PointingHandCursor)
        self._top_tab.setToolTip("Desplegar EPar+ Modern")
        self._top_tab.setStyleSheet("background:#85a4a8; border-bottom:2px solid #d8bb78;")
        self._top_open_timer = QTimer(self)
        self._top_open_timer.setSingleShot(True)
        self._top_open_timer.setInterval(250)
        self._top_open_timer.timeout.connect(self._expand_top_console)
        self._top_hide_timer = QTimer(self)
        self._top_hide_timer.setSingleShot(True)
        self._top_hide_timer.setInterval(600)
        self._top_hide_timer.timeout.connect(self._collapse_top_console)
        self._top_watch_timer = QTimer(self)
        self._top_watch_timer.setInterval(100)
        self._top_watch_timer.timeout.connect(self._watch_top_console)
        self._raw_preview_frames = []
        self._raw_preview_scaled_frames = []
        self._raw_preview_index = 0
        self._raw_preview_direction = 1
        self._raw_preview_timer = QTimer(self)
        self._raw_preview_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._raw_preview_timer.setInterval(35)
        self._raw_preview_timer.timeout.connect(self._advance_raw_preview)
        self._clinical_labels = []
        self._clinical_font_sizes = {}
        self._asynchrony_panel = None
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
        pill_grid.setContentsMargins(8, 6, 8, 8)
        pill_grid.setHorizontalSpacing(5)
        pill_grid.setVerticalSpacing(3)

        nav_host = QWidget()
        nav_host.setObjectName("navHost")
        nav_layout = QVBoxLayout(nav_host)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)
        nav = self._button("CONTROLES", self.toggle_controls, "blue")
        self._controls_btn = nav
        nav.setObjectName("navCap")
        nav.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        nav.setFixedHeight(42)
        nav.setToolTip("Mostrar u ocultar los controles de procesamiento.")
        self._controls_slot = QWidget(nav_host)
        self._controls_slot.setFixedHeight(42)
        nav_layout.addWidget(self._controls_slot, 0)
        nav.setParent(pill)
        self._asynchrony_btn = self._button("ASINCRONÍA", owner.toggle_modern_asynchrony, "blue")
        self._asynchrony_btn.setObjectName("asyncCap")
        self._asynchrony_btn.setToolTip("Abrir el panel lateral de asincronía.")
        self._asynchrony_slot = QWidget(nav_host)
        self._asynchrony_slot.setMinimumHeight(124)
        nav_layout.addWidget(self._asynchrony_slot, 1)
        self._asynchrony_btn.setParent(pill)
        nav_layout.setStretch(0, 0)
        nav_layout.setStretch(1, 1)
        pill_grid.addWidget(nav_host, 0, 0, 1, 2)

        self._restart_btn = QPushButton(pill)
        self._restart_btn.setObjectName("modernRestartButton")
        self._restart_btn.setIcon(QIcon(str(Path(__file__).parent / "icons" / "power.svg")))
        self._restart_btn.setIconSize(QSize(20, 20))
        self._restart_btn.setFixedSize(42, 42)
        self._restart_btn.setToolTip("Reiniciar sesión")
        self._restart_btn.setAccessibleName("Reiniciar sesión")
        self._restart_btn.clicked.connect(owner.restart_workspace_state)

        self._load_btn = self._button("CARGAR", owner.load_modern_studies, "blue")
        self._load_btn.setObjectName("loadCap")
        pill_grid.addWidget(self._load_btn, 1, 0, 1, 2)
        self._position_navigation_buttons()
        self._process_btn = self._button("PROCESAR", owner.open_modern_processing, "amber")
        self._process_btn.setObjectName("processCap")
        pill_grid.addWidget(self._process_btn, 2, 0)
        self._dock_btn = self._button("ACOPLAR", self.dockRequested.emit, "blue")
        self._dock_btn.setObjectName("dockCap")
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
        self._results_stack = QStackedWidget()
        self._results_stack.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._results_stack.addWidget(self._status_label)
        self._raw_preview_label = QLabel()
        self._raw_preview_label.setObjectName("modernRawPreview")
        self._raw_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._raw_preview_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._raw_preview_label.setStyleSheet("background:black; border:none;")
        self._results_stack.addWidget(self._raw_preview_label)
        status_layout.addWidget(self._results_stack, 1)
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
        self._top_mode_btn = QToolButton()
        self._top_mode_btn.setIcon(QIcon(str(Path(__file__).parent / "icons" / "panel-top.svg")))
        self._top_mode_btn.setCheckable(True)
        self._top_mode_btn.setToolTip("Modo autoocultable en el borde superior")
        self._top_mode_btn.setAccessibleName("Modo autoocultable")
        self._top_mode_btn.toggled.connect(self.set_top_mode)
        self._top_pin_btn = QToolButton()
        self._top_pin_btn.setIcon(QIcon(str(Path(__file__).parent / "icons" / "anchor.svg")))
        self._top_pin_btn.setCheckable(True)
        self._top_pin_btn.setEnabled(False)
        self._top_pin_btn.setToolTip("Anclar desplegada")
        self._top_pin_btn.setAccessibleName("Anclar desplegada")
        self._top_pin_btn.toggled.connect(self._set_top_pinned)
        for button in (self._top_mode_btn, self._top_pin_btn):
            button.setFixedSize(28, 24)
            button.setIconSize(QSize(18, 18))
            button.setStyleSheet("QToolButton { background:#7898a1; border:0; border-radius:4px; } QToolButton:checked { background:#d8bb78; } QToolButton:disabled { background:#52666b; }")
            slim_l.addWidget(button, 0, Qt.AlignmentFlag.AlignHCenter)
        slim_l.addWidget(self._button("□", self._show_images, "blue"))
        close_button = self._button("×", self.close, "red")
        close_button.setToolTip("Cerrar GammaSync")
        slim_l.addWidget(close_button)
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

    def _position_navigation_buttons(self):
        position = self._controls_slot.mapTo(self._header, QPoint(0, 0))
        self._controls_btn.setGeometry(position.x(), position.y(), self._controls_slot.width(), 42)
        self._controls_btn.raise_()
        self._restart_btn.move(position - QPoint(2, 4))
        self._restart_btn.raise_()
        async_position = self._asynchrony_slot.mapTo(self._header, QPoint(0, 0)) + QPoint(0, 3)
        async_height = self._load_btn.y() - async_position.y() - 3
        if async_height > 0:
            self._asynchrony_btn.setGeometry(async_position.x(), async_position.y(), self._asynchrony_slot.width(), async_height)
            self._asynchrony_btn.raise_()

    def set_top_mode(self, enabled: bool) -> None:
        if enabled == self._top_mode:
            return
        self._top_mode = enabled
        self._top_mode_btn.setChecked(enabled)
        self._top_pin_btn.setEnabled(enabled)
        if enabled:
            self._top_geometry = self.geometry()
            self._top_screen = self.screen()
            self._top_collapsed = False
            self._position_top_console()
            self._top_watch_timer.start()
        else:
            self._top_slide.stop()
            self._top_open_timer.stop()
            self._top_hide_timer.stop()
            self._top_watch_timer.stop()
            self._top_tab.hide()
            was_collapsed = self._top_collapsed
            self._top_collapsed = False
            self._top_pin_btn.setChecked(False)
            if self._top_geometry is not None:
                self.setGeometry(self._top_geometry)
                self.ensure_on_screen()
            if was_collapsed:
                self.show()

    def _position_top_console(self) -> None:
        if not self._top_mode:
            return
        screen = self._top_screen
        if screen not in QGuiApplication.screens():
            screen = QGuiApplication.primaryScreen()
            self._top_screen = screen
        if screen is None:
            return
        area = screen.availableGeometry()
        self.move(area.left() + (area.width() - self.width()) // 2, area.top())
        self._top_tab.move(area.left() + (area.width() - self._top_tab.width()) // 2, area.top())

    def _set_top_pinned(self, pinned: bool) -> None:
        self._top_pin_btn.setToolTip("Desanclar y permitir autoocultado" if pinned else "Anclar desplegada")
        was_visible = self.isVisible()
        geometry = self.geometry()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, pinned)
        self.setGeometry(geometry)
        if was_visible:
            self.show()
        if pinned:
            self._top_hide_timer.stop()
            self._expand_top_console()

    def _top_interaction_active(self) -> bool:
        app = QApplication.instance()
        return (self.frameGeometry().contains(QCursor.pos()) or self._drag_offset is not None
                or app.activeModalWidget() is not None or app.activePopupWidget() is not None
                or any(window is not self._top_tab and window is not self and window.isVisible()
                       and (window.parentWidget() is self or (window.parentWidget() is not None
                           and self.isAncestorOf(window.parentWidget()))) for window in app.topLevelWidgets()))

    def _watch_top_console(self) -> None:
        if not self._top_mode:
            return
        if self._top_slide.state() == QAbstractAnimation.State.Running:
            return
        self._position_top_console()
        if self._top_collapsed:
            self._keep_top_tab_above()
            if self._top_tab.frameGeometry().contains(QCursor.pos()):
                if not self._top_open_timer.isActive():
                    self._top_open_timer.start()
            else:
                self._top_open_timer.stop()
        elif self.isVisible() and not self.isMinimized():
            if self._top_pin_btn.isChecked() or self._top_interaction_active():
                self._top_hide_timer.stop()
                if self._top_pin_btn.isChecked() and not self._top_owned_window_active():
                    self.raise_()
            elif not self._top_hide_timer.isActive():
                self._top_hide_timer.start()

    def _top_owned_window_active(self) -> bool:
        app = QApplication.instance()
        return (app.activeModalWidget() is not None or app.activePopupWidget() is not None
                or any(window is not self and window is not self._top_tab and window.isVisible()
                      and window.parentWidget() is not None
                      and (window.parentWidget() is self or self.isAncestorOf(window.parentWidget()))
                       for window in app.topLevelWidgets()))

    def _expand_top_console(self) -> None:
        if not self._top_mode:
            return
        self._top_open_timer.stop()
        was_collapsed = self._top_collapsed
        self._top_collapsed = False
        self._top_tab.hide()
        self.show()
        self._position_top_console()
        self.raise_()
        self.activateWindow()
        if was_collapsed:
            target = self.pos()
            self._top_slide.setStartValue(target - QPoint(0, self.height() - 6))
            self._top_slide.setEndValue(target)
            self._top_slide.start()

    def collapse_for_processing(self) -> None:
        self._collapse_top_console(force=True)

    def _keep_top_tab_above(self) -> None:
        if self._top_mode and self._top_collapsed:
            self._top_tab.show()
            self._top_tab.raise_()

    def _collapse_top_console(self, force: bool = False) -> None:
        if not self._top_mode or self.isMinimized():
            return
        if not force and (self._top_pin_btn.isChecked() or self._top_interaction_active()):
            return
        self._top_open_timer.stop()
        self._top_slide.stop()
        self._top_collapsed = True
        self.hide()
        self._position_top_console()
        self._keep_top_tab_above()

    def eventFilter(self, watched, event):
        if self._top_mode and self._top_collapsed and watched is not self._top_tab and event.type() in (QEvent.Type.Show, QEvent.Type.WindowActivate):
            if isinstance(watched, QWindow) or (isinstance(watched, QWidget) and watched.isWindow()):
                QTimer.singleShot(0, self._keep_top_tab_above)
        if watched is self._top_tab and self._top_mode:
            if event.type() == QEvent.Type.Enter:
                self._top_open_timer.start()
            elif event.type() == QEvent.Type.Leave:
                self._top_open_timer.stop()
            elif event.type() == QEvent.Type.MouseButtonPress:
                self._expand_top_console()
        if watched in (self._controls_slot, self._controls_slot.parentWidget(), self._asynchrony_slot, self._load_btn, self._header) and event.type() in (QEvent.Type.Move, QEvent.Type.Resize, QEvent.Type.Show):
            self._position_navigation_buttons()
        if watched is self._raw_preview_label and event.type() == QEvent.Type.Resize:
            self._scale_raw_preview()
        if event.type() == QEvent.Type.Resize and any(watched is label for label, _ in self._clinical_labels):
            self._clinical_resize_timer.start(0)
        if watched is self._drag_handle or watched is self._header or (isinstance(watched, QLabel) and self._header.isAncestorOf(watched)):
            if event.type() == event.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                if self._top_mode:
                    return True
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
        QWidget#navHost { background:transparent; }
        QFrame#modernPill QPushButton#navCap { background:#bac2ad; color:#1f2328; border-radius:0; border-top-left-radius:18px; border-top-right-radius:10px; padding:8px; font-weight:800; }
        QFrame#modernPill QPushButton#asyncCap { background:#c6e8bf; color:#1f2328; border-radius:0; border-bottom-left-radius:10px; border-bottom-right-radius:10px; font-weight:800; }
        QFrame#modernPill QPushButton#opsCap { background:#d8bb78; color:#2a2b2d; border-radius:0; border-top-right-radius:8px; border-bottom-right-radius:8px; padding:8px; font-weight:800; }
        QFrame#modernPill QPushButton#navCap:hover { background:#c9d2bf; }
        QFrame#modernPill QPushButton#asyncCap:hover { background:#d3efcd; }
        QFrame#modernPill QPushButton#opsCap:hover { background:#e4ca8c; }
        QFrame#modernPill QPushButton#modernRestartButton { background:#e51d20; color:white; border-radius:0; border-top-left-radius:20px; border-right:3px solid #121722; border-bottom:3px solid #121722; padding:0; min-height:39px; max-height:39px; font-family:'Segoe UI Symbol'; font-size:22px; font-weight:400; }
        QFrame#modernPill QPushButton#modernRestartButton:hover { background:#ff3538; }
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
        QFrame#modernPill QPushButton#loadCap { border-radius:0; border-top-left-radius:12px; border-bottom-left-radius:12px; background:#86a8b3; }
        QFrame#modernPill QPushButton#processCap { border-top-left-radius:14px; border-top-right-radius:14px; }
        QFrame#modernPill QPushButton#dockCap { border-top-left-radius:14px; border-top-right-radius:14px; }
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

    def toggle_asynchrony(self) -> None:
        if self._asynchrony_panel is None:
            from ui.epar_modern_asynchrony import EParModernAsynchrony
            self._asynchrony_panel = EParModernAsynchrony(self._owner, self)
        if self._asynchrony_panel.isVisible():
            self._asynchrony_panel.hide()
        else:
            self._asynchrony_panel.open_near_console()

    def release_asynchrony(self) -> None:
        if self._asynchrony_panel is not None:
            self._asynchrony_panel.restore_controls()
            self._asynchrony_panel.deleteLater()
            self._asynchrony_panel = None

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
        self.set_top_mode(False)
        self.release_asynchrony()
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
        if self._asynchrony_panel is not None:
            self._asynchrony_panel.refresh_state()

    def set_raw_preview(self, frames) -> None:
        self._raw_preview_timer.stop()
        self._raw_preview_frames = list(frames)
        self._raw_preview_index = 0
        self._raw_preview_direction = 1
        if not self._raw_preview_frames:
            self.stop_raw_preview()
            return
        if self._top_mode:
            self._top_pin_btn.setChecked(True)
            self._expand_top_console()
        self._results_stack.setCurrentWidget(self._raw_preview_label)
        self._scale_raw_preview()
        if self.isVisible() and len(self._raw_preview_frames) > 1:
            self._raw_preview_timer.start()

    def compose_raw_preview_pair(self, left_frame, right_frame, left_label, right_label):
        height = max(left_frame.height(), right_frame.height())
        left_frame = left_frame.scaledToHeight(height, Qt.TransformationMode.SmoothTransformation)
        right_frame = right_frame.scaledToHeight(height, Qt.TransformationMode.SmoothTransformation)
        gap = 4
        title_height = 16
        right_x = left_frame.width() + gap
        canvas = QPixmap(right_x + right_frame.width(), height + title_height)
        canvas.fill(QColor("black"))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        font = painter.font()
        font.setPixelSize(9)
        painter.setFont(font)
        painter.fillRect(0, 0, canvas.width(), title_height, QColor("#07171f"))
        painter.setPen(QColor("#d9b86f"))
        painter.drawText(4, 12, str(left_label))
        painter.drawText(right_x + 4, 12, str(right_label))
        painter.drawPixmap(0, title_height, left_frame)
        painter.drawPixmap(right_x, title_height, right_frame)
        painter.end()
        return canvas

    def _scale_raw_preview(self) -> None:
        if not self._raw_preview_frames:
            return
        size = self._raw_preview_label.contentsRect().size()
        if size.isEmpty():
            return
        self._raw_preview_scaled_frames = [
            frame.scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            for frame in self._raw_preview_frames
        ]
        self._raw_preview_label.setPixmap(self._raw_preview_scaled_frames[self._raw_preview_index])

    def _advance_raw_preview(self) -> None:
        count = len(self._raw_preview_scaled_frames)
        if count < 2:
            return
        next_index = self._raw_preview_index + self._raw_preview_direction
        if next_index >= count - 1:
            next_index = count - 1
            self._raw_preview_direction = -1
        elif next_index <= 0:
            next_index = 0
            self._raw_preview_direction = 1
        self._raw_preview_index = next_index
        self._raw_preview_label.setPixmap(self._raw_preview_scaled_frames[next_index])

    def stop_raw_preview(self) -> None:
        self._raw_preview_timer.stop()
        self._raw_preview_frames = []
        self._raw_preview_scaled_frames = []
        self._raw_preview_label.clear()
        self._results_stack.setCurrentWidget(self._status_label)

    def showEvent(self, event):
        super().showEvent(event)
        if self._top_mode:
            self._top_collapsed = False
            self._top_tab.hide()
            self._position_top_console()
        if len(self._raw_preview_frames) > 1:
            self._scale_raw_preview()
            self._raw_preview_timer.start()

    def hideEvent(self, event):
        self._top_slide.stop()
        if not self._top_collapsed:
            self._top_tab.hide()
        self._top_hide_timer.stop()
        self._raw_preview_timer.stop()
        super().hideEvent(event)

    def bring_to_front(self) -> None:
        if self._top_mode:
            self._expand_top_console()
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
        self.set_top_mode(False)
        self.release_asynchrony()
        self.stop_raw_preview()
        self._closing_for_app = True
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._closing_for_app:
            event.accept()
            return
        event.ignore()
        self._owner.request_application_close()

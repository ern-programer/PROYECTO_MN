"""Panel de asincronia exclusivo de EPar+ Modern, con los controles reales."""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QPoint, Qt
from PyQt6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractButton, QButtonGroup, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSizeGrip, QSizePolicy, QStyle, QToolButton,
    QVBoxLayout, QWidget,
)


class EParModernAsynchrony(QWidget):
    def __init__(self, owner, console):
        super().__init__(console, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self._owner = owner
        self._console = console
        self._borrowed = []
        self._drag_offset = None
        self._stage_filter = "both"
        self._stage_panels = []
        self._source_band = getattr(owner, "_lower_cine_panel", None)
        self._source_band_hidden = self._source_band.isHidden() if self._source_band is not None else True
        self.setObjectName("eparModernAsynchrony")
        self.setWindowTitle("GammaSync - Asincronía EPar+ Modern")
        self.setMinimumSize(920, 540)
        self.resize(1120, 590)
        self.setStyleSheet(self._stylesheet())
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(14)

        self._header = QFrame()
        self._header.installEventFilter(self)
        self._header.setCursor(Qt.CursorShape.OpenHandCursor)
        header = QHBoxLayout(self._header)
        header.setContentsMargins(0, 0, 0, 0)
        cap = QLabel("SINCRO")
        cap.setObjectName("asyncCap")
        cap.setFixedSize(100, 52)
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("ASINCRONÍA")
        title.setObjectName("asyncTitle")
        for label in (cap, title):
            label.installEventFilter(self)
        header.addWidget(cap)
        header.addWidget(title)
        header.addStretch(1)
        self._stage_group = QButtonGroup(self)
        self._stage_buttons = []
        for key, label in (("primary", "1ra. Fase"), ("secondary", "2da. Fase"), ("both", "Ambas")):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setChecked(key == "both")
            button.clicked.connect(lambda checked, stage=key: self._select_stage(stage))
            self._stage_group.addButton(button)
            self._stage_buttons.append(button)
            header.addWidget(button)
        close = QToolButton()
        close.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton))
        close.setToolTip("Cerrar panel de asincronía")
        close.clicked.connect(self.hide)
        header.addWidget(close)
        root.addWidget(self._header)

        body = QHBoxLayout()
        body.setSpacing(16)
        navigation = QVBoxLayout()
        navigation.setSpacing(5)
        primary = owner.cine
        for caption, slider, previous, following, label in (
            ("GATE", primary.gate_slider, primary.gate_prev_btn, primary.gate_next_btn, primary.gate_label),
            ("CORTE", primary.slice_slider, primary.slice_prev_btn, primary.slice_next_btn, primary.slice_label),
        ):
            navigation.addWidget(self._caption(caption))
            self._borrow(label, navigation)
            row = QHBoxLayout()
            for widget in (previous, slider, following):
                self._borrow(widget, row)
            navigation.addLayout(row)
            navigation.addSpacing(6)
        self._borrow(primary.play_button, navigation)
        navigation.addSpacing(6)
        navigation.addWidget(self._caption("VELOCIDAD"))
        self._borrow(primary.speed_label, navigation)
        self._borrow(primary.speed_slider, navigation)
        navigation.addStretch(1)
        nav_host = QWidget()
        nav_host.setLayout(navigation)
        nav_host.setFixedWidth(116)
        body.addWidget(nav_host)
        body.addWidget(self._separator())

        center = QVBoxLayout()
        color_row = QHBoxLayout()
        color_row.addWidget(self._caption("COLORMAP"))
        self._borrow(primary.cmap_combo, color_row)
        primary.cmap_combo.setMinimumWidth(100)
        self._borrow(primary.invert_cmap_check, color_row)
        color_row.addStretch(1)
        self._borrow(primary.interp_combo, color_row)
        center.addLayout(color_row)
        stages = QHBoxLayout()
        stages.setSpacing(12)
        for index, cine in enumerate((owner.cine, owner.cine_compare)):
            stages.addWidget(self._build_stage(cine, index), 1)
        center.addLayout(stages, 1)
        body.addLayout(center, 1)
        body.addWidget(self._separator())

        results = QVBoxLayout()
        results.addWidget(self._caption("AJUSTE FINO"))
        self._borrow(owner.asynchrony_review_btn, results)
        owner.asynchrony_review_btn.setText("VISTA ASINCRONÍA")
        owner.asynchrony_review_btn.setProperty("asyncRole", "cyan")
        results.addSpacing(12)
        results.addWidget(self._caption("RESULTADOS"))
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        host = QWidget()
        contents = QVBoxLayout(host)
        contents.setContentsMargins(0, 0, 4, 0)
        self._borrow(owner.main_metrics_readout, contents)
        owner.main_metrics_readout.setWordWrap(True)
        contents.addStretch(1)
        scroll.setWidget(host)
        results.addWidget(scroll, 1)
        self._stage_summary = QLabel("SIN ESTUDIO")
        self._stage_summary.setWordWrap(True)
        results.addWidget(self._stage_summary)
        close_panel = QPushButton("CERRAR PANEL")
        close_panel.clicked.connect(self.hide)
        results.addWidget(close_panel)
        results_host = QWidget()
        results_host.setLayout(results)
        results_host.setFixedWidth(220)
        body.addWidget(results_host)
        root.addLayout(body, 1)

        footer = QHBoxLayout()
        self._state_label = QLabel("SIN ESTUDIO")
        footer.addWidget(self._state_label)
        footer.addStretch(1)
        footer.addWidget(QSizeGrip(self))
        root.addLayout(footer)
        primary.gate_slider.valueChanged.connect(self._update_primary_position)
        primary.slice_slider.valueChanged.connect(self._update_primary_position)
        self.refresh_state()
        if self._source_band is not None:
            self._source_band.hide()

    @staticmethod
    def _caption(text):
        label = QLabel(text)
        label.setProperty("asyncCaption", True)
        return label

    @staticmethod
    def _separator():
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.VLine)
        frame.setStyleSheet("color:#29414a;")
        return frame

    @classmethod
    def _locate(cls, layout, widget):
        if layout is None:
            return None
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item.widget() is widget:
                return layout, index
            if item.layout() is not None:
                found = cls._locate(item.layout(), widget)
                if found is not None:
                    return found
        return None

    def _borrow(self, widget, target, *position):
        parent = widget.parentWidget()
        found = self._locate(parent.layout(), widget)
        if found is None:
            raise ValueError(f"Control sin layout de origen: {type(widget).__name__}")
        layout, index = found
        item = layout.itemAt(index)
        grid = layout.getItemPosition(index) if isinstance(layout, QGridLayout) else None
        stretch = layout.stretch(index) if isinstance(layout, (QHBoxLayout, QVBoxLayout)) else 0
        self._borrowed.append((
            widget, parent, layout, index, grid, item.alignment(), stretch,
            widget.minimumSize(), widget.maximumSize(), widget.sizePolicy(),
            widget.styleSheet(), widget.isHidden(),
            widget.text() if isinstance(widget, QAbstractButton) and widget is not self._owner.cine.play_button else None,
            widget.property("asyncRole"),
        ))
        layout.removeWidget(widget)
        widget.setMinimumSize(0, 0)
        widget.setMaximumSize(16777215, 16777215)
        widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        widget.setStyleSheet("")
        target.addWidget(widget, *position)
        widget.show()

    def _build_stage(self, cine, index):
        frame = QFrame()
        frame.setObjectName("asyncStage")
        self._stage_panels.append(frame)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 6, 8, 8)
        self._borrow(cine.phase_title_label, layout)
        cine.phase_title_label.setProperty("asyncPhaseTitle", True)
        cine.phase_title_label.setFixedHeight(28)
        image_row = QHBoxLayout()
        self._borrow(cine.preview, image_row)
        cine.preview.setMinimumSize(140, 140)
        cine.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        window = QVBoxLayout()
        window.setContentsMargins(0, 0, 0, 0)
        window.setSpacing(3)
        for widget in (cine.window_high_reset_btn, cine.window_high_label, cine.range_slider,
                       cine.window_low_reset_btn, cine.window_low_label):
            self._borrow(widget, window)
        cine.range_slider.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        cine.range_slider.setFixedWidth(28)
        cine.range_slider.setMinimumHeight(140)
        window_host = QWidget()
        window_host.setLayout(window)
        window_host.setFixedWidth(48)
        image_row.addWidget(window_host)
        layout.addLayout(image_row, 1)
        position_host = QWidget()
        position_host.setFixedHeight(18)
        position = QHBoxLayout(position_host)
        position.setContentsMargins(0, 0, 0, 0)
        if index:
            self._borrow(cine.slice_label, position)
            self._borrow(cine.gate_label, position)
        else:
            self._primary_position = QLabel()
            position.addWidget(self._primary_position)
        layout.addWidget(position_host)
        actions = QHBoxLayout()
        manual = QPushButton("ROI manual")
        manual.setProperty("asyncRole", "amber")
        manual.setToolTip("Abrir el editor de ROI manual para esta etapa")
        manual.clicked.connect(lambda checked, stage=index: self._open_manual_roi(stage))
        actions.addWidget(manual)
        self._manual_buttons = getattr(self, "_manual_buttons", []) + [manual]
        intestinal = next(
            button for button in cine.findChildren(QToolButton)
            if button.text().startswith("ROI intestinal")
            and (cine is self._owner.cine_compare or not self._owner.cine_compare.isAncestorOf(button))
        )
        self._borrow(intestinal, actions)
        intestinal.setText("ROI intestinal")
        layout.addLayout(actions)
        return frame

    def _open_manual_roi(self, index):
        self._owner._on_cine_panel_activated("secondary" if index else "main")
        toolbar, _ = self._owner._toolbar_group_menus["roi_manual_por_slice"]
        toolbar.toggle_near(self._manual_buttons[index])

    def _select_stage(self, stage):
        self._stage_filter = stage
        self.refresh_state()

    def _update_primary_position(self, *args):
        cine = self._owner.cine
        self._primary_position.setText(f"{cine.slice_label.text()}   |   {cine.gate_label.text()}")

    def refresh_state(self):
        self._update_primary_position()
        available = [cine._cube is not None for cine in (self._owner.cine, self._owner.cine_compare)]
        titles = [cine.phase_title_label.text() for cine in (self._owner.cine, self._owner.cine_compare)]
        for index in range(2):
            self._stage_buttons[index].setText(titles[index])
            self._stage_buttons[index].setEnabled(available[index])
            self._manual_buttons[index].setEnabled(available[index])
        if self._stage_filter != "both" and not available[0 if self._stage_filter == "primary" else 1]:
            self._stage_filter = "both"
            self._stage_buttons[2].setChecked(True)
        for index, frame in enumerate(self._stage_panels):
            frame.setVisible(self._stage_filter == "both" or self._stage_filter == ("primary" if index == 0 else "secondary"))
        summary = " + ".join(title for title, loaded in zip(titles, available) if loaded)
        self._stage_summary.setText(summary or "SIN ESTUDIO")
        self._state_label.setText("ESTUDIO CARGADO" if any(available) else "SIN ESTUDIO RECONSTRUIDO")

    def restore_controls(self):
        self.hide()
        for saved in reversed(self._borrowed):
            widget, parent, layout, index, grid, alignment, stretch, minimum, maximum, policy, stylesheet, hidden, text, role = saved
            widget.setParent(parent)
            widget.setMinimumSize(minimum)
            widget.setMaximumSize(maximum)
            widget.setSizePolicy(policy)
            widget.setStyleSheet(stylesheet)
            widget.setProperty("asyncRole", role)
            widget.setProperty("asyncPhaseTitle", None)
            if text is not None:
                widget.setText(text)
            if grid is not None:
                layout.addWidget(widget, *grid, alignment)
            else:
                layout.insertWidget(index, widget, stretch, alignment)
            widget.setVisible(not hidden)
        self._borrowed.clear()
        if self._source_band is not None:
            self._source_band.setVisible(not self._source_band_hidden)

    def open_near_console(self):
        self.refresh_state()
        geometry = self._owner._ui_settings.value("epar_modern/asynchrony_geometry", None)
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.move(self._console.pos() - QPoint(self.width() + 8, 0))
        screen = QGuiApplication.screenAt(self._console.frameGeometry().center()) or QGuiApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            self.resize(min(self.width(), area.width()), min(self.height(), area.height()))
            self.move(max(area.left(), min(self.x(), area.right() - self.width() + 1)),
                      max(area.top(), min(self.y(), area.bottom() - self.height() + 1)))
        self.show()
        self.raise_()
        self.activateWindow()

    def hideEvent(self, event):
        for cine in (self._owner.cine, self._owner.cine_compare):
            cine.stop_playback()
        self._owner._ui_settings.setValue("epar_modern/asynchrony_geometry", self.saveGeometry())
        super().hideEvent(event)

    def closeEvent(self, event):
        event.ignore()
        self.hide()

    def eventFilter(self, watched, event):
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

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#090d12"))
        painter.setPen(QPen(QColor("#b77968"), 2))
        painter.drawRoundedRect(self.rect().adjusted(2, 2, -3, -3), 24, 24)
        for left, right, color in ((160, 390, "#b77968"), (398, 650, "#85a4a8"), (658, self.width() - 32, "#1692aa")):
            painter.fillRect(left, self.height() - 10, max(0, right - left), 3, QColor(color))

    @staticmethod
    def _stylesheet():
        return """
        QWidget#eparModernAsynchrony { background:#090d12; color:#d7e0e3; font-family:'Bahnschrift','Segoe UI'; font-size:12px; }
        #eparModernAsynchrony QLabel { color:#d7e0e3; background:transparent; }
        #eparModernAsynchrony QLabel[asyncCaption='true'] { color:#d8bb78; font-size:11px; }
        #eparModernAsynchrony QLabel#asyncCap { background:#85a4a8; color:#102127; border-top-left-radius:24px; font-size:17px; }
        #eparModernAsynchrony QLabel#asyncTitle { font-size:22px; }
        #eparModernAsynchrony QFrame#asyncStage { background:#0d141b; border:1px solid #29414a; border-radius:6px; }
        #eparModernAsynchrony QLabel[asyncPhaseTitle='true'] { background:#85a4a8; color:#102127; padding-left:6px; font-weight:bold; }
        #eparModernAsynchrony QPushButton, #eparModernAsynchrony QToolButton { background:#85a4a8; color:#102127; border:none; border-radius:5px; padding:6px; min-height:20px; }
        #eparModernAsynchrony QPushButton:checked, #eparModernAsynchrony QPushButton[asyncRole='cyan'] { background:#1692aa; }
        #eparModernAsynchrony QPushButton[asyncRole='amber'] { background:#d8bb78; }
        #eparModernAsynchrony QPushButton:hover, #eparModernAsynchrony QToolButton:hover { background:#aac6c8; }
        #eparModernAsynchrony QPushButton:disabled, #eparModernAsynchrony QToolButton:disabled { background:#29414a; color:#718187; }
        #eparModernAsynchrony QComboBox { background:#17242c; color:#d7e0e3; border:1px solid #29414a; border-radius:4px; padding:5px; }
        #eparModernAsynchrony QComboBox QAbstractItemView { background:#17242c; color:#d7e0e3; selection-background-color:#1692aa; }
        #eparModernAsynchrony QCheckBox { color:#d7e0e3; spacing:4px; }
        #eparModernAsynchrony QSlider::groove:horizontal { height:3px; background:#4c6570; }
        #eparModernAsynchrony QSlider::handle:horizontal { width:8px; margin:-5px 0; background:#b77968; border-radius:3px; }
        #eparModernAsynchrony QScrollArea, #eparModernAsynchrony QScrollArea QWidget { background:#090d12; }
        """
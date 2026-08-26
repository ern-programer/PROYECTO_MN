# -*- coding: utf-8 -*-
"""Widget MIP interactivo con rotación 360° controlada por mouse."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPointF, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QTransform
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QScrollArea, QScrollBar, QCheckBox, QSlider, QFrame


class MipRotatorWidget(QWidget):
    """MIP interactivo con rotación 360° controlada por mouse.
    
    - Botones de posición: AP, PA, LI, LD, OAI
    - Arrastrar horizontalmente: rotar ángulo azimutal
    - Rueda mouse: zoom in/out
    - Cine automático: rotación continua
    - Doble click: reset a vista AP
    """
    
    angle_changed = pyqtSignal(float, float)  # azimuth, elevation
    
    # Posiciones predefinidas (azimut, elevación) — calibradas vs vista real
    POSITIONS = {
        "PA": (0.0, 0.0),         # Posterior-Anterior (vista dorsal) — referencia 0°
        "AP": (180.0, 0.0),       # Anterior-Posterior (vista frontal)
        "LD": (90.0, 0.0),        # Lateral Derecho
        "LI": (270.0, 0.0),       # Lateral Izquierdo
        "OAI": (235.0, 0.0),      # Oblicuo Anterior Izquierdo
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)

        # Cine (inicializar antes de _setup_ui porque _setup_ui actualiza el label de velocidad)
        self._cine_speed = 2.0  # grados por frame
        self._cine_speed_min = 0.2
        self._cine_speed_max = 12.0

        self._setup_ui()
        
        # Estado de rotación
        self._azimuth_deg = 0.0    # Rotación horizontal (0 = PA)
        self._elevation_deg = 0.0  # Rotación vertical
        
        # Zoom
        self._zoom_factor = 1.0
        
        # Datos
        self._volume = None
        # Volumen sin post-filtro gaussiano (para toggle)
        self._volume_unfiltered = None
        self._spacing_mm = (4.0, 4.0, 4.0)
        self._cmap_fn = None
        self._voi_heart = None
        self._voi_mediastinum = None
        
        # === Overlays 3D ===
        # Máscara CT segmentada (para wireframe + transparente)
        self._mask_3d = None  # ndarray bool (z,y,x) alineado al volumen SPECT
        # Cubo automático (bounding box de la segmentación auto, antes de edición)
        self._auto_cube_bbox = None  # (z0,z1,y0,y1,x0,x1) en coords volumen
        
        # Visibilidad de overlays
        self._show_wireframe = True
        self._show_mask_transparent = True
        self._show_vois = True
        self._show_auto_cube = False  # off por defecto (el usuario lo considera ruidoso)
        
        # Opacidad de máscara transparente (0-100)
        self._mask_opacity = 25  # 25% por defecto — suave
        # Opacidad del wireframe (0-100)
        self._wireframe_opacity = 80  # 80% — visible pero no opaco
        
        # Interacción
        self._drag_active = False
        self._last_pos = None
        
        # Cine
        self._cine_active = False
        self._cine_timer = QTimer()
        self._cine_timer.timeout.connect(self._cine_step)
        
        # Timer para renderizado suave
        self._render_timer = QTimer()
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_mip)
        
    def _setup_ui(self):
        """Configurar UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # ScrollArea para el MIP (permite zoom sin desordenar UI)
        self._scroll = QScrollArea()
        self._scroll.setMinimumHeight(100)
        self._scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._scroll.setWidgetResizable(False)  # El label mantiene su tamaño real
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setStyleSheet("""
            QScrollArea {
                background: #0a0f1e;
                border: 1px solid #1e293b;
                border-radius: 4px;
            }
            QScrollBar:vertical {
                background: #1e293b;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #475569;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #64748b;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: #1e293b;
                height: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal {
                background: #475569;
                border-radius: 5px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #64748b;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
        layout.addWidget(self._scroll)

        # Label para el MIP (dentro del ScrollArea)
        self._lbl = QLabel()
        self._lbl.setMinimumSize(50, 50)
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl.setStyleSheet("background:#0a0f1e;")
        self._lbl.setText("Cargar SPECT y calcular HMR para ver MIP")
        self._scroll.setWidget(self._lbl)
        
        # Label de ángulo en vivo
        self._angle_lbl = QLabel("Az: 0.0°  El: 0.0°")
        self._angle_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._angle_lbl.setStyleSheet("color:#94a3b8; font-size:11px; font-family:Consolas,monospace;")
        layout.addWidget(self._angle_lbl)
        
        # Toggle filtro (con/sin post-filtro gaussiano)
        self._filter_toggle = QCheckBox("Filtro ON")
        self._filter_toggle.setChecked(True)
        self._filter_toggle.setToolTip("Alternar entre volumen con/sin post-filtro gaussiano de reconstrucción")
        self._filter_toggle.setStyleSheet(
            "QCheckBox { color:#94a3b8; font-size:10px; spacing:4px; }"
            "QCheckBox::indicator { width:14px; height:14px; border-radius:3px; "
            "border:1px solid #475569; background:#1e293b; }"
            "QCheckBox::indicator:checked { background:#3b82f6; border-color:#3b82f6; }"
            "QCheckBox::indicator:hover { border-color:#64748b; }"
        )
        self._filter_toggle.stateChanged.connect(self._on_filter_toggled)
        layout.addWidget(self._filter_toggle)
        
        # === Fila de controles de overlay ===
        overlay_row = QHBoxLayout()
        overlay_row.setSpacing(4)
        overlay_row.setContentsMargins(0, 0, 0, 0)
        
        chk_style = (
            "QCheckBox { color:#94a3b8; font-size:10px; spacing:3px; }"
            "QCheckBox::indicator { width:13px; height:13px; border-radius:2px; "
            "border:1px solid #475569; background:#1e293b; }"
            "QCheckBox::indicator:checked { background:#3b82f6; border-color:#3b82f6; }"
            "QCheckBox::indicator:hover { border-color:#64748b; }"
        )
        
        self._chk_wireframe = QCheckBox("Wireframe")
        self._chk_wireframe.setChecked(True)
        self._chk_wireframe.setToolTip("Malla 3D fina (1px) de la máscara cardíaca proyectada sobre el MIP")
        self._chk_wireframe.setStyleSheet(chk_style)
        self._chk_wireframe.stateChanged.connect(self._on_overlay_visibility_changed)
        overlay_row.addWidget(self._chk_wireframe)
        
        self._chk_mask_transparent = QCheckBox("Máscara")
        self._chk_mask_transparent.setChecked(True)
        self._chk_mask_transparent.setToolTip("Máscara CT semi-transparente superpuesta al MIP")
        self._chk_mask_transparent.setStyleSheet(chk_style)
        self._chk_mask_transparent.stateChanged.connect(self._on_overlay_visibility_changed)
        overlay_row.addWidget(self._chk_mask_transparent)
        
        self._chk_vois = QCheckBox("VOIs")
        self._chk_vois.setChecked(True)
        self._chk_vois.setToolTip("Círculos de VOIs corazón/mediastino")
        self._chk_vois.setStyleSheet(chk_style)
        self._chk_vois.stateChanged.connect(self._on_overlay_visibility_changed)
        overlay_row.addWidget(self._chk_vois)
        
        self._chk_auto_cube = QCheckBox("Cubo auto")
        self._chk_auto_cube.setChecked(False)
        self._chk_auto_cube.setToolTip("Bounding box del cubo automático (segmentación antes de edición manual)")
        self._chk_auto_cube.setStyleSheet(chk_style)
        self._chk_auto_cube.stateChanged.connect(self._on_overlay_visibility_changed)
        overlay_row.addWidget(self._chk_auto_cube)
        
        overlay_row.addStretch()
        layout.addLayout(overlay_row)
        
        # === Slider de opacidad de máscara ===
        opacity_row = QHBoxLayout()
        opacity_row.setSpacing(4)
        opacity_row.setContentsMargins(0, 0, 0, 0)
        
        lbl_op = QLabel("Opacidad:")
        lbl_op.setStyleSheet("color:#64748b; font-size:9px;")
        lbl_op.setFixedWidth(48)
        opacity_row.addWidget(lbl_op)
        
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 100)
        self._opacity_slider.setValue(25)
        self._opacity_slider.setToolTip("Opacidad de la máscara transparente (0=invisible, 100=opaca)")
        self._opacity_slider.setStyleSheet(
            "QSlider::groove:horizontal { background:#1e293b; height:4px; border-radius:2px; }"
            "QSlider::handle:horizontal { background:#3b82f6; width:12px; height:12px; "
            "margin:-4px 0; border-radius:6px; }"
            "QSlider::handle:horizontal:hover { background:#60a5fa; }"
            "QSlider::sub-page:horizontal { background:#1e3a5f; border-radius:2px; }"
        )
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_row.addWidget(self._opacity_slider)
        
        self._lbl_opacity_val = QLabel("25%")
        self._lbl_opacity_val.setFixedWidth(28)
        self._lbl_opacity_val.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lbl_opacity_val.setStyleSheet("color:#94a3b8; font-size:9px;")
        opacity_row.addWidget(self._lbl_opacity_val)
        
        layout.addLayout(opacity_row)
        
        # Fila de botones de posición
        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)
        
        self._btn_ap = QPushButton("AP")
        self._btn_ap.setToolTip("Vista Anterior-Posterior")
        self._btn_ap.setFixedWidth(40)
        self._btn_ap.clicked.connect(lambda: self._set_position("AP"))
        btn_row.addWidget(self._btn_ap)
        
        self._btn_pa = QPushButton("PA")
        self._btn_pa.setToolTip("Vista Posterior-Anterior")
        self._btn_pa.setFixedWidth(40)
        self._btn_pa.clicked.connect(lambda: self._set_position("PA"))
        btn_row.addWidget(self._btn_pa)
        
        self._btn_li = QPushButton("LI")
        self._btn_li.setToolTip("Lateral Izquierdo")
        self._btn_li.setFixedWidth(40)
        self._btn_li.clicked.connect(lambda: self._set_position("LI"))
        btn_row.addWidget(self._btn_li)
        
        self._btn_ld = QPushButton("LD")
        self._btn_ld.setToolTip("Lateral Derecho")
        self._btn_ld.setFixedWidth(40)
        self._btn_ld.clicked.connect(lambda: self._set_position("LD"))
        btn_row.addWidget(self._btn_ld)
        
        self._btn_oai = QPushButton("OAI")
        self._btn_oai.setToolTip("Oblicuo Anterior Izquierdo")
        self._btn_oai.setFixedWidth(40)
        self._btn_oai.clicked.connect(lambda: self._set_position("OAI"))
        btn_row.addWidget(self._btn_oai)
        
        btn_row.addStretch()
        
        # Botón Cine
        self._btn_cine = QPushButton("▶ Cine")
        self._btn_cine.setToolTip("Iniciar/detener rotación automática")
        self._btn_cine.setFixedWidth(60)
        self._btn_cine.setCheckable(True)
        self._btn_cine.clicked.connect(self._toggle_cine)
        btn_row.addWidget(self._btn_cine)

        # Control de velocidad de cine
        self._btn_cine_slower = QPushButton("−")
        self._btn_cine_slower.setToolTip("Disminuir velocidad de cine")
        self._btn_cine_slower.setFixedWidth(24)
        self._btn_cine_slower.clicked.connect(lambda: self._change_cine_speed(-0.4))
        btn_row.addWidget(self._btn_cine_slower)

        self._lbl_cine_speed = QLabel()
        self._lbl_cine_speed.setFixedWidth(46)
        self._lbl_cine_speed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_cine_speed.setStyleSheet("color:#cbd5e1; font-size:10px;")
        btn_row.addWidget(self._lbl_cine_speed)

        self._btn_cine_faster = QPushButton("+")
        self._btn_cine_faster.setToolTip("Aumentar velocidad de cine")
        self._btn_cine_faster.setFixedWidth(24)
        self._btn_cine_faster.clicked.connect(lambda: self._change_cine_speed(+0.4))
        btn_row.addWidget(self._btn_cine_faster)
        
        layout.addLayout(btn_row)
        
        # Estilo botones
        btn_style = """
            QPushButton {
                background: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #334155;
            }
            QPushButton:checked {
                background: #3b82f6;
                border-color: #60a5fa;
            }
        """
        for btn in [
            self._btn_ap,
            self._btn_pa,
            self._btn_li,
            self._btn_ld,
            self._btn_oai,
            self._btn_cine,
            self._btn_cine_slower,
            self._btn_cine_faster,
        ]:
            btn.setStyleSheet(btn_style)

        self._update_cine_speed_ui()
        
    def _set_position(self, name: str):
        """Establecer posición predefinida."""
        if name in self.POSITIONS:
            az, el = self.POSITIONS[name]
            self._azimuth_deg = az
            self._elevation_deg = el
            self._update_angle_label()
            self._schedule_render()
            self.angle_changed.emit(az, el)
            
    def _toggle_cine(self, checked: bool):
        """Activar/desactivar cine."""
        self._cine_active = checked
        if checked:
            self._cine_timer.start(50)  # 20 fps
            self._btn_cine.setText("⏸ Cine")
        else:
            self._cine_timer.stop()
            self._btn_cine.setText("▶ Cine")

    def _change_cine_speed(self, delta: float):
        """Ajustar velocidad del cine en grados por frame."""
        self._cine_speed = float(np.clip(self._cine_speed + delta, self._cine_speed_min, self._cine_speed_max))
        self._update_cine_speed_ui()

    def _update_cine_speed_ui(self):
        """Actualizar indicador visual de velocidad de cine."""
        self._lbl_cine_speed.setText(f"{self._cine_speed:.1f}°")
        self._btn_cine.setToolTip(
            f"Iniciar/detener rotación automática · velocidad {self._cine_speed:.1f}°/frame"
        )
            
    def _cine_step(self):
        """Avanzar un frame de cine."""
        self._azimuth_deg = (self._azimuth_deg - self._cine_speed) % 360.0
        self._update_angle_label()
        self._render_mip()
        self.angle_changed.emit(self._azimuth_deg, self._elevation_deg)
        
    def set_volume(self, volume: np.ndarray, spacing_mm: tuple = (4.0, 4.0, 4.0)):
        """Establecer volumen SPECT para MIP."""
        self._volume = np.asarray(volume, dtype=np.float64) if volume is not None else None
        self._spacing_mm = spacing_mm
        self._schedule_render()
        
    def set_volume_unfiltered(self, volume: np.ndarray | None):
        """Establecer volumen sin post-filtro gaussiano para toggle."""
        self._volume_unfiltered = np.asarray(volume, dtype=np.float64) if volume is not None else None
        # Si no hay unfiltered, deshabilitar toggle y forzar ON
        if self._volume_unfiltered is None:
            self._filter_toggle.setChecked(True)
            self._filter_toggle.setEnabled(False)
        else:
            self._filter_toggle.setEnabled(True)
            
    def _on_filter_toggled(self, state):
        """Alternar entre volumen filtrado y sin filtro."""
        if state != Qt.CheckState.Checked.value:
            # Filtro OFF: verificar que exista el volumen sin filtro
            if self._volume_unfiltered is None:
                self._filter_toggle.setChecked(True)
                return
        self._schedule_render()
    
    def _on_overlay_visibility_changed(self):
        """Actualizar visibilidad de overlays según checkboxes."""
        self._show_wireframe = self._chk_wireframe.isChecked()
        self._show_mask_transparent = self._chk_mask_transparent.isChecked()
        self._show_vois = self._chk_vois.isChecked()
        self._show_auto_cube = self._chk_auto_cube.isChecked()
        self._schedule_render()
    
    def _on_opacity_changed(self, value: int):
        """Actualizar opacidad de la máscara transparente."""
        self._mask_opacity = value
        self._lbl_opacity_val.setText(f"{value}%")
        self._schedule_render()
        
    def set_colormap(self, cmap_fn):
        """Establecer función de colormap (debe aceptar array 2D normalizado y retornar RGB)."""
        self._cmap_fn = cmap_fn
        self._schedule_render()
        
    def set_vois(self, voi_heart=None, voi_mediastinum=None):
        """Establecer VOIs para dibujar sobre el MIP."""
        self._voi_heart = voi_heart
        self._voi_mediastinum = voi_mediastinum
        self._schedule_render()
    
    def set_mask_3d(self, mask: np.ndarray | None):
        """Establecer máscara 3D CT segmentada (bool array z,y,x alineado al SPECT)."""
        self._mask_3d = np.asarray(mask, dtype=bool) if mask is not None else None
        self._schedule_render()
    
    def set_auto_cube_bbox(self, bbox: tuple | None):
        """Establecer bounding box del cubo automático (z0,z1,y0,y1,x0,x1)."""
        self._auto_cube_bbox = bbox
        self._schedule_render()
    
    def reset_view(self):
        """Reset a vista PA sin zoom."""
        self._azimuth_deg = 0.0     # PA
        self._elevation_deg = 0.0
        self._zoom_factor = 1.0
        self._update_angle_label()
        self._schedule_render()
        
    def _schedule_render(self):
        """Programar renderizado con debounce."""
        self._render_timer.start(50)  # 50ms debounce

    def _update_angle_label(self):
        """Actualizar label de ángulo en vivo."""
        self._angle_lbl.setText(
            f"Az: {self._azimuth_deg:6.1f}°  El: {self._elevation_deg:+5.1f}°"
        )
        
    def _render_mip(self):
        """Renderizar MIP con rotación 3D real.
        
        Algoritmo:
          1. Rotar volumen 3D alrededor del eje vertical (Y) según azimut
          2. Aplicar elevación (rotación alrededor de eje X)
          3. Proyectar siempre sobre eje Z (profundidad) → np.max(axis=0)
          4. Esto simula girar el objeto 3D y mirarlo de frente
        
        Convención SPECT cardíaco:
          - vol.shape = (nz, ny, nx) = (cortes SA, ant-post, izq-der)
        """
        if self._volume is None:
            return
            
        try:
            # Elegir volumen según toggle de filtro
            if self._filter_toggle.isChecked() or self._volume_unfiltered is None:
                vol = self._volume
            else:
                vol = self._volume_unfiltered
            from scipy import ndimage
            
            # Paso 1: Rotación azimutal alrededor de eje Z (superior-inferior)
            # en el plano XY (axes=(2,1)); así AP/PA/LI/LD/OAI cambian
            # como vistas alrededor del paciente, no como rotación 2D del MIP.
            if abs(self._azimuth_deg) > 0.5:
                vol_rot = ndimage.rotate(
                    vol,
                    self._azimuth_deg,
                    axes=(2, 1),   # rotar en plano XY (alrededor de Z)
                    reshape=False,
                    order=1,
                    cval=0.0,
                    mode='constant'
                )
            else:
                vol_rot = vol.copy()
            
            # Paso 2: Elevación (inclinación craneal/caudal)
            # Rotamos en plano ZY (alrededor de X) después del azimut.
            if abs(self._elevation_deg) > 1.0:
                vol_rot = ndimage.rotate(
                    vol_rot,
                    self._elevation_deg,
                    axes=(0, 1),   # rotar en plano ZY (alrededor de X)
                    reshape=False,
                    order=1,
                    cval=0.0,
                    mode='constant'
                )
            
            # Paso 3: MIP proyectando sobre eje Y (AP profundidad)
            # Esto mantiene AP correcta y PA/LI/LD salen de la rotación 3D real.
            mip = np.max(vol_rot, axis=1)  # Proyección sobre Y → plano XZ
            
            # Normalizar
            mmin, mmax = np.percentile(mip, [1, 99.5])
            if mmax <= mmin:
                mmax = mmin + 1
            mip_norm = np.clip((mip - mmin) / (mmax - mmin), 0.0, 1.0)
            
            # Paso 4: Aplicar colormap
            if self._cmap_fn is not None:
                mip_rgb = (self._cmap_fn(mip_norm) * 255.0).astype(np.uint8)
            else:
                # Fallback: escala de grises
                mip_gray = (mip_norm * 255.0).astype(np.uint8)
                mip_rgb = np.stack([mip_gray, mip_gray, mip_gray], axis=-1)
            
            h, w = mip_rgb.shape[:2]
            
            # Convertir a QPixmap
            qimg = QImage(mip_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            
            # Dibujar VOIs proyectados
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self._draw_voi_projection(painter, vol_rot.shape, mip.shape, pix.size())
            painter.end()
            
            # Paso 5: Escalar al tamaño del viewport del ScrollArea aplicando zoom
            vp_w = self._scroll.viewport().width()
            vp_h = self._scroll.viewport().height()
            
            if vp_w > 20 and vp_h > 20:
                # Tamaño base ajustado por zoom (el label crece pero el scroll lo contiene)
                base_w = int((vp_w - 8) * self._zoom_factor)
                base_h = int((vp_h - 8) * self._zoom_factor)
                
                scaled_pix = pix.scaled(
                    base_w, base_h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            else:
                scaled_pix = pix
                
            self._lbl.setPixmap(scaled_pix)
            self._lbl.setText("")
            
            # Ajustar tamaño del label al pixmap para que el scroll funcione correctamente
            self._lbl.setFixedSize(scaled_pix.size())
            
        except Exception as e:
            print(f"Error en MIP: {e}")
            import traceback
            traceback.print_exc()
            self._lbl.setText(f"MIP ERROR:\n{e}")
            
    def _draw_voi_projection(self, painter: QPainter, vol_shape: tuple, mip_shape: tuple, pix_size):
        """Dibujar proyección de VOIs y overlays 3D sobre el MIP."""
        # === 1. Máscara transparente (relleno semi-opaco) ===
        if self._show_mask_transparent and self._mask_3d is not None:
            self._draw_mask_transparent(painter, vol_shape, mip_shape)
        
        # === 2. Cubo automático (bounding box) ===
        if self._show_auto_cube and self._auto_cube_bbox is not None:
            self._draw_auto_cube(painter, vol_shape, mip_shape)
        
        # === 3. Wireframe 3D de la máscara (contornos finos) ===
        if self._show_wireframe and self._mask_3d is not None:
            self._draw_wireframe(painter, vol_shape, mip_shape)
        
        # === 4. VOIs (círculos corazón/mediastino) ===
        if self._show_vois:
            if self._voi_heart is not None:
                self._draw_single_voi(painter, self._voi_heart, QColor(239, 68, 68, 255), "♥", vol_shape, mip_shape, pix_size)
            if self._voi_mediastinum is not None:
                self._draw_single_voi(painter, self._voi_mediastinum, QColor(59, 130, 246, 255), "Med", vol_shape, mip_shape, pix_size)
            
    def _draw_single_voi(self, painter: QPainter, voi, color: QColor, label: str, vol_shape: tuple, mip_shape: tuple, pix_size):
        """Dibujar un VOI individual proyectado."""
        try:
            # Centro del VOI en coordenadas de volumen original
            cz, cy, cx = voi.cz, voi.cy, voi.cx
            # VOIAnatomical no tiene radius_mm — calcular desde máscara o usar default
            r_mm = getattr(voi, 'radius_mm', None)
            if r_mm is None:
                # Para VOIAnatomical: estimar radio equivalente desde el volumen de la máscara
                try:
                    mask_vol = getattr(voi, 'mask_3d_data', None)
                    if mask_vol is not None and hasattr(mask_vol, 'sum'):
                        n_voxels = int(mask_vol.sum())
                        if n_voxels > 0:
                            # Radio de esfera equivalente: V = (4/3)πr³ → r = (3V/4π)^(1/3)
                            # Usar spacing REAL del volumen, no 6.8 fijo
                            avg_spacing = float(np.mean(self._spacing_mm)) if self._spacing_mm else 6.8
                            r_mm = (3.0 * n_voxels / (4.0 * np.pi)) ** (1.0 / 3.0) * avg_spacing
                        else:
                            r_mm = 20.0
                    else:
                        r_mm = 20.0
                except Exception:
                    r_mm = 20.0

            # Volumen: (z, y, x)
            nz, ny, nx = vol_shape

            # Centro geométrico (coincide mejor con ndimage.rotate + reshape=False)
            cz0 = (nz - 1) * 0.5
            cy0 = (ny - 1) * 0.5
            cx0 = (nx - 1) * 0.5

            # Coordenadas relativas al centro
            rz = float(cz) - cz0
            ry = float(cy) - cy0
            rx = float(cx) - cx0

            # 1) Rotación azimutal en plano XY (axes=(2,1), alrededor de Z)
            az = np.radians(self._azimuth_deg)
            ry1 = (ry * np.cos(az)) - (rx * np.sin(az))
            rx1 = (ry * np.sin(az)) + (rx * np.cos(az))
            rz1 = rz

            # 2) Elevación en plano ZY (axes=(0,1), alrededor de X)
            el = np.radians(self._elevation_deg)
            rz2 = (rz1 * np.cos(el)) - (ry1 * np.sin(el))
            ry2 = (rz1 * np.sin(el)) + (ry1 * np.cos(el))
            rx2 = rx1

            # 3) Proyección usada por el MIP actual: axis=1 (Y) => imagen (Z, X)
            mip_h, mip_w = int(mip_shape[0]), int(mip_shape[1])
            row = ((rz2 + cz0) / max(1.0, (nz - 1))) * max(0, mip_h - 1)
            col = ((rx2 + cx0) / max(1.0, (nx - 1))) * max(0, mip_w - 1)

            # Clamp al lienzo del MIP
            cy_px = int(np.clip(row, 0, max(0, mip_h - 1)))
            cx_px = int(np.clip(col, 0, max(0, mip_w - 1)))

            # Radio en píxeles adaptado a matriz real del MIP (plano Z-X)
            spacing_z = float(self._spacing_mm[0]) if len(self._spacing_mm) > 0 else 4.0
            spacing_x = float(self._spacing_mm[2]) if len(self._spacing_mm) > 2 else 4.0
            mm_per_px = max(1e-6, 0.5 * (spacing_z + spacing_x))
            r_px = int(max(2, np.round(float(r_mm) / mm_per_px)))
            
            # Dibujar círculo con etiqueta de identificación
            pen = QPen(color, 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(cx_px, cy_px), r_px, r_px)
            
            # Etiqueta de identificación (♥ para corazón, Med para mediastino)
            if r_px > 8:
                font = QFont("Arial", max(7, min(10, r_px // 3)))
                painter.setFont(font)
                # Sombra para legibilidad
                shadow_color = QColor(0, 0, 0, 180)
                painter.setPen(QPen(shadow_color, 1))
                painter.drawText(int(cx_px) + r_px + 2, int(cy_px) + 2, label)
                painter.setPen(QPen(color, 1))
                painter.drawText(int(cx_px) + r_px + 1, int(cy_px) + 1, label)
            
        except Exception as e:
            print(f"Error dibujando VOI: {e}")
    
    # === Overlays 3D: máscara transparente, wireframe, cubo automático ===
    
    def _rotate_mask_to_view(self, mask: np.ndarray) -> np.ndarray:
        """Aplicar la misma rotación azimutal+elevación que al volumen SPECT.
        
        Usa order=0 (nearest) para preservar la máscara booleana sin difuminar.
        """
        from scipy import ndimage as ndi
        m = mask
        if abs(self._azimuth_deg) > 0.5:
            m = ndi.rotate(m, self._azimuth_deg, axes=(2, 1), reshape=False, order=0, cval=0.0, mode='constant')
        if abs(self._elevation_deg) > 0.5:
            m = ndi.rotate(m, self._elevation_deg, axes=(0, 1), reshape=False, order=0, cval=0.0, mode='constant')
        return m
    
    def _draw_mask_transparent(self, painter: QPainter, vol_shape: tuple, mip_shape: tuple):
        """Dibujar la máscara CT como overlay semi-transparente proyectado sobre el MIP.
        
        Proyecta la máscara rotada sobre el eje Y (igual que el MIP) y la superpone
        con opacidad controlable. Usa color verde-cian para distinguirla del SPECT.
        """
        try:
            if self._mask_3d is None:
                return
            if self._mask_3d.shape != vol_shape:
                return
            
            # Rotar máscara a la vista actual
            mask_rot = self._rotate_mask_to_view(self._mask_3d)
            
            # Proyección sobre eje Y (igual que el MIP): any() para máscara
            mask_proj = np.any(mask_rot, axis=1)  # (z, x) bool
            
            mip_h, mip_w = int(mip_shape[0]), int(mip_shape[1])
            nz, nx = mask_proj.shape
            
            # Crear overlay RGBA vectorizado (sin loops)
            overlay = np.zeros((nz, nx, 4), dtype=np.uint8)
            alpha_val = int(255 * self._mask_opacity / 100)
            overlay[mask_proj] = [0, 200, 160, alpha_val]
            
            # Recortar al tamaño del MIP si es necesario
            h = min(nz, mip_h)
            w = min(nx, mip_w)
            if h == 0 or w == 0:
                return
            overlay_crop = overlay[:h, :w].copy()
            
            qimg = QImage(overlay_crop.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
            pix_overlay = QPixmap.fromImage(qimg)
            painter.drawPixmap(0, 0, pix_overlay)
        except Exception as e:
            print(f"Error dibujando máscara transparente: {e}")
    
    def _draw_wireframe(self, painter: QPainter, vol_shape: tuple, mip_shape: tuple):
        """Dibujar wireframe 3D fino (1px) de los bordes de la máscara cardíaca.
        
        Detecta bordes de la máscara proyectada y los dibuja como píxeles individuales
        de 1px. Vectorizado con NumPy para rendimiento.
        """
        try:
            from scipy import ndimage as ndi
            if self._mask_3d is None:
                return
            if self._mask_3d.shape != vol_shape:
                return
            
            # Rotar máscara a la vista actual
            mask_rot = self._rotate_mask_to_view(self._mask_3d)
            
            # Proyectar sobre eje Y → imagen 2D (z, x)
            mask_proj = np.any(mask_rot, axis=1)  # bool (z, x)
            
            # Detectar bordes con gradiente morfológico (contorno externo de 1 voxel)
            struct_2d = ndi.generate_binary_structure(2, 1)
            eroded = ndi.binary_erosion(mask_proj, structure=struct_2d, iterations=1)
            borders = mask_proj & ~eroded  # borde externo
            
            mip_h, mip_w = int(mip_shape[0]), int(mip_shape[1])
            nz, nx = borders.shape
            h = min(nz, mip_h)
            w = min(nx, mip_w)
            if h == 0 or w == 0:
                return
            
            # Crear imagen RGBA vectorizada para el wireframe
            wire_img = np.zeros((h, w, 4), dtype=np.uint8)
            alpha_val = int(255 * self._wireframe_opacity / 100)
            wire_img[borders[:h, :w]] = [100, 220, 255, alpha_val]
            
            qimg = QImage(wire_img.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
            pix_wire = QPixmap.fromImage(qimg)
            painter.drawPixmap(0, 0, pix_wire)
        except Exception as e:
            print(f"Error dibujando wireframe: {e}")
    
    def _draw_auto_cube(self, painter: QPainter, vol_shape: tuple, mip_shape: tuple):
        """Dibujar el bounding box del cubo automático como rectángulo proyectado.
        
        El cubo se dibuja con baja opacidad (gris) para no ser intrusivo.
        """
        try:
            if self._auto_cube_bbox is None:
                return
            z0, z1, y0, y1, x0, x1 = self._auto_cube_bbox
            
            nz, ny, nx = vol_shape
            cz0 = (nz - 1) * 0.5
            cy0 = (ny - 1) * 0.5
            cx0 = (nx - 1) * 0.5
            
            # 8 esquinas del cubo en coords relativas al centro
            corners_3d = []
            for cz_v in [z0, z1]:
                for cy_v in [y0, y1]:
                    for cx_v in [x0, x1]:
                        corners_3d.append((float(cz_v) - cz0, float(cy_v) - cy0, float(cx_v) - cx0))
            
            # Rotar cada esquina (azimut + elevación) y proyectar sobre Y
            az = np.radians(self._azimuth_deg)
            el = np.radians(self._elevation_deg)
            
            projected = []
            for rz, ry, rx in corners_3d:
                # Azimut (alrededor de Z)
                ry1 = ry * np.cos(az) - rx * np.sin(az)
                rx1 = ry * np.sin(az) + rx * np.cos(az)
                rz1 = rz
                # Elevación (alrededor de X)
                rz2 = rz1 * np.cos(el) - ry1 * np.sin(el)
                ry2 = rz1 * np.sin(el) + ry1 * np.cos(el)
                rx2 = rx1
                # Proyección sobre Y → (z, x)
                projected.append((rz2 + cz0, rx2 + cx0))
            
            mip_h, mip_w = int(mip_shape[0]), int(mip_shape[1])
            pts_px = []
            for rz, rx in projected:
                py = int(np.clip((rz / max(1.0, nz - 1)) * max(0, mip_h - 1), 0, mip_h - 1))
                px = int(np.clip((rx / max(1.0, nx - 1)) * max(0, mip_w - 1), 0, mip_w - 1))
                pts_px.append((px, py))
            
            # Dibujar las 12 aristas del cubo (con baja opacidad)
            # Índices de esquinas: 0-7 (orden z0/z1 × y0/y1 × x0/x1)
            # Aristas: conectar adyacentes
            edges = [
                (0,1),(0,2),(1,3),(2,3),  # cara z0
                (4,5),(4,6),(5,7),(6,7),  # cara z1
                (0,4),(1,5),(2,6),(3,7),  # conexiones z
            ]
            
            cube_color = QColor(120, 130, 150, 100)  # gris azulado, ~40% opacidad
            pen = QPen(cube_color, 1, Qt.PenStyle.DashLine)  # línea punteada
            painter.setPen(pen)
            from PyQt6.QtCore import QLineF
            for i, j in edges:
                if i < len(pts_px) and j < len(pts_px):
                    painter.drawLine(QLineF(pts_px[i][0], pts_px[i][1], pts_px[j][0], pts_px[j][1]))
        except Exception as e:
            print(f"Error dibujando cubo automático: {e}")
    
    # === Eventos de mouse ===
    
    def mousePressEvent(self, event):
        """Iniciar arrastre para rotación."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = True
            self._last_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)
        
    def mouseReleaseEvent(self, event):
        """Finalizar arrastre."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)
        
    def mouseMoveEvent(self, event):
        """Arrastrar para rotar."""
        if self._drag_active and self._last_pos is not None:
            delta = event.pos() - self._last_pos
            
            # Rotación horizontal (azimut)
            self._azimuth_deg -= delta.x() * 0.8
            self._azimuth_deg = self._azimuth_deg % 360.0
            
            # Rotación vertical (elevación)
            self._elevation_deg += delta.y() * 0.4
            self._elevation_deg = np.clip(self._elevation_deg, -45.0, 45.0)
            
            self._last_pos = event.pos()
            self._update_angle_label()
            self._render_mip()
            self.angle_changed.emit(self._azimuth_deg, self._elevation_deg)
            
        super().mouseMoveEvent(event)
        
    def mouseDoubleClickEvent(self, event):
        """Doble click: reset a vista AP."""
        self.reset_view()
        super().mouseDoubleClickEvent(event)
        
    def wheelEvent(self, event):
        """Zoom con rueda del mouse."""
        try:
            delta = event.angleDelta().y()
            factor = 1.12 if delta > 0 else 0.89
            self._zoom_factor *= factor
            self._zoom_factor = np.clip(self._zoom_factor, 0.25, 5.0)
            self._render_mip()
        except Exception as e:
            print(f"Error en wheelEvent: {e}")
        super().wheelEvent(event)
        
    def enterEvent(self, event):
        """Mouse entra al widget."""
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        """Mouse sale del widget."""
        self.unsetCursor()
        super().leaveEvent(event)

# -*- coding: utf-8 -*-
"""Widget MIP interactivo con rotación 360° controlada por mouse."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPointF, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QTransform
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QScrollArea, QScrollBar, QCheckBox


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
        
    def set_colormap(self, cmap_fn):
        """Establecer función de colormap (debe aceptar array 2D normalizado y retornar RGB)."""
        self._cmap_fn = cmap_fn
        self._schedule_render()
        
    def set_vois(self, voi_heart=None, voi_mediastinum=None):
        """Establecer VOIs para dibujar sobre el MIP."""
        self._voi_heart = voi_heart
        self._voi_mediastinum = voi_mediastinum
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
        """Dibujar proyección de VOIs sobre el MIP."""
        # VOI corazón (rojo)
        if self._voi_heart is not None:
            self._draw_single_voi(painter, self._voi_heart, QColor(239, 68, 68, 255), "♥", vol_shape, mip_shape, pix_size)
            
        # VOI mediastino (azul)
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

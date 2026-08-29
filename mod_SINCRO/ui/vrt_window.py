# -*- coding: utf-8 -*-
"""Ventana VRT 3D: render volumétrico esquelético de la CT con fusión SPECT.

Ray-casting CPU vectorizado: rotación del volumen (scipy), transfer function
de opacidad por HU, shading difuso por gradiente y composición front-to-back.
Los focos SPECT se integran como fuente emisiva violeta dentro del mismo rayo,
por lo que quedan parcialmente ocluidos por el hueso delantero (efecto GE-like).
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QSlider, QVBoxLayout,
)
from scipy import ndimage as ndi


class VrtWindow(QDialog):
    """Render volumétrico 3D interactivo de CT (hueso) + focos SPECT."""

    POSITIONS = {
        "AP": (0.0, 0.0),
        "PA": (180.0, 0.0),
        "LI": (90.0, 0.0),
        "LD": (270.0, 0.0),
        "OAI": (55.0, 0.0),
    }

    #: lado máximo del volumen de render (interactividad CPU)
    MAX_SIDE_HQ = 176
    MAX_SIDE_FAST = 112

    def __init__(self, parent=None, *, ct_volume: np.ndarray,
                 spect_volume: np.ndarray | None = None,
                 spacing_zyx: tuple | None = None,
                 vois: list[dict] | None = None,
                 mask_3d: np.ndarray | None = None):
        super().__init__(parent)
        self.setWindowTitle("VRT 3D — Render volumétrico óseo (experimental)")
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )
        self.resize(760, 820)

        ct = np.asarray(ct_volume, dtype=np.float32)
        sp = np.asarray(spect_volume, dtype=np.float32) if spect_volume is not None else None

        # Remoción de camilla (estilo estaciones comerciales): se calcula una vez
        # sobre la CT completa y se cachean ambas variantes.
        self._table_removed_ok = False
        try:
            from core.amyloid_spect import remove_ct_table
            ct_clean, _body, notes = remove_ct_table(ct)
            self._table_removed_ok = any("se conserva la mayor" in n for n in notes)
            ct_clean = ct_clean.astype(np.float32)
        except Exception:
            ct_clean = ct
        self._remove_table = True

        # Volúmenes de render en dos calidades (HQ para vista fija, FAST para drag/cine)
        self._ct_hq_raw, self._sp_hq = self._prepare(ct, sp, self.MAX_SIDE_HQ)
        self._ct_fast_raw, self._sp_fast = self._prepare(ct, sp, self.MAX_SIDE_FAST)
        self._ct_hq_clean, _ = self._prepare(ct_clean, None, self.MAX_SIDE_HQ)
        self._ct_fast_clean, _ = self._prepare(ct_clean, None, self.MAX_SIDE_FAST)

        # Overlays 3D: VOIs (coords y radio en voxels de la grilla ct_volume) y
        # máscara segmentada; se decimen con el mismo factor que los volúmenes.
        self._factor_hq = min(1.0, float(self.MAX_SIDE_HQ) / float(max(ct.shape)))
        self._factor_fast = min(1.0, float(self.MAX_SIDE_FAST) / float(max(ct.shape)))
        self._vois = list(vois) if vois else []
        self._mask_hq = self._mask_fast = None
        if mask_3d is not None and np.any(mask_3d):
            m = np.asarray(mask_3d, dtype=np.float32)
            m_hq = ndi.zoom(m, self._factor_hq, order=0, prefilter=False) if self._factor_hq < 0.999 else m
            m_fast = ndi.zoom(m, self._factor_fast, order=0, prefilter=False) if self._factor_fast < 0.999 else m
            self._mask_hq = np.ascontiguousarray(m_hq > 0.5)
            self._mask_fast = np.ascontiguousarray(m_fast > 0.5)
        self._show_overlays = bool(self._vois or self._mask_hq is not None)

        # Percentil de focos SPECT (sobre el volumen sin rotar, una sola vez)
        self._sp_p100 = float(np.max(self._sp_hq)) if self._sp_hq is not None else 0.0

        self._azimuth = 0.0   # AP
        self._elevation = 0.0
        self._zoom = 1.0
        self._hu_threshold = 150.0
        self._density = 0.55
        self._view_mode = "fusion" if sp is not None else "bone"
        self._cmap_name = "hot"
        self._sp_base = 20.0     # % del máximo SPECT: por debajo no se muestra (quita fondo)
        self._sp_top = 100.0
        self._fusion_mix = 0.65
        self._brightness = 1.35
        self._cine_speed = 3.0   # grados por frame
        self._fast_mode = False
        self._drag_pos = None

        self._cine_timer = QTimer(self)
        self._cine_timer.timeout.connect(self._cine_step)
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render)
        # Al soltar drag/cine: re-render en alta calidad
        self._hq_timer = QTimer(self)
        self._hq_timer.setSingleShot(True)
        self._hq_timer.timeout.connect(self._render_hq)

        self._setup_ui()
        self._schedule(fast=False)

    # ------------------------------------------------------------------
    @staticmethod
    def _prepare(ct: np.ndarray, sp: np.ndarray | None, max_side: int):
        """Decima a lado máximo manejable (misma escala para CT y SPECT)."""
        factor = min(1.0, float(max_side) / float(max(ct.shape)))
        if factor < 0.999:
            ct_o = ndi.zoom(ct, factor, order=1, prefilter=False)
            sp_o = ndi.zoom(sp, factor, order=1, prefilter=False) if sp is not None else None
        else:
            ct_o = ct.copy()
            sp_o = sp.copy() if sp is not None else None
        return np.ascontiguousarray(ct_o), (np.ascontiguousarray(sp_o) if sp_o is not None else None)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._lbl = QLabel("Renderizando...")
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl.setMinimumSize(420, 420)
        self._lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._lbl.setStyleSheet("background:#05080f; border:1px solid #1e293b; border-radius:4px; color:#64748b;")
        self._lbl.setMouseTracking(False)
        self._lbl.mousePressEvent = self._on_press
        self._lbl.mouseMoveEvent = self._on_move
        self._lbl.mouseReleaseEvent = self._on_release
        self._lbl.wheelEvent = self._on_wheel
        layout.addWidget(self._lbl, 1)

        self._angle_lbl = QLabel("Az: 0.0°  El: +0.0°")
        self._angle_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._angle_lbl.setStyleSheet("color:#94a3b8; font-size:11px; font-family:Consolas,monospace;")
        layout.addWidget(self._angle_lbl)

        has_sp = self._sp_hq is not None

        # Fila 1: posiciones + cine + velocidad + guardar
        row1 = QHBoxLayout()
        for name in ("AP", "PA", "LI", "LD", "OAI"):
            btn = QPushButton(name)
            btn.setFixedWidth(42)
            btn.clicked.connect(lambda _=False, n=name: self._goto(n))
            row1.addWidget(btn)
        self._btn_cine = QPushButton("▶ Cine")
        self._btn_cine.setCheckable(True)
        self._btn_cine.toggled.connect(self._on_cine_toggled)
        row1.addWidget(self._btn_cine)
        row1.addWidget(QLabel("Vel:"))
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(5, 100)   # 0.5 .. 10 °/frame
        self._speed_slider.setValue(int(self._cine_speed * 10))
        self._speed_slider.setFixedWidth(90)
        self._speed_slider.setToolTip("Velocidad del cine (grados por cuadro)")
        self._speed_slider.valueChanged.connect(self._on_speed_changed)
        row1.addWidget(self._speed_slider)
        self._speed_lbl = QLabel("3.0°")
        self._speed_lbl.setFixedWidth(34)
        row1.addWidget(self._speed_lbl)
        row1.addStretch()
        btn_png = QPushButton("Guardar PNG")
        btn_png.clicked.connect(self._save_png)
        row1.addWidget(btn_png)
        layout.addLayout(row1)

        # Fila 2: umbral HU + densidad
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Umbral óseo (HU):"))
        self._thr_slider = QSlider(Qt.Orientation.Horizontal)
        self._thr_slider.setRange(80, 600)
        self._thr_slider.setValue(int(self._hu_threshold))
        self._thr_slider.valueChanged.connect(self._on_params_changed)
        row2.addWidget(self._thr_slider, 1)
        self._thr_lbl = QLabel("150")
        self._thr_lbl.setFixedWidth(34)
        row2.addWidget(self._thr_lbl)
        row2.addWidget(QLabel("Densidad:"))
        self._den_slider = QSlider(Qt.Orientation.Horizontal)
        self._den_slider.setRange(10, 100)
        self._den_slider.setValue(int(self._density * 100))
        self._den_slider.valueChanged.connect(self._on_params_changed)
        row2.addWidget(self._den_slider, 1)
        self._den_lbl = QLabel("55%")
        self._den_lbl.setFixedWidth(34)
        row2.addWidget(self._den_lbl)
        row2.addWidget(QLabel("Brillo:"))
        self._bri_slider = QSlider(Qt.Orientation.Horizontal)
        self._bri_slider.setRange(50, 300)
        self._bri_slider.setValue(int(self._brightness * 100))
        self._bri_slider.setToolTip("Brillo global del render (ganancia sobre el color final).")
        self._bri_slider.valueChanged.connect(self._on_params_changed)
        row2.addWidget(self._bri_slider, 1)
        self._bri_lbl = QLabel("135%")
        self._bri_lbl.setFixedWidth(40)
        row2.addWidget(self._bri_lbl)
        layout.addLayout(row2)

        # Fila 3: camilla + modo de vista + colormap SPECT
        row3 = QHBoxLayout()
        self._chk_table = QCheckBox("Quitar camilla")
        self._chk_table.setChecked(True)
        self._chk_table.setToolTip(
            "Elimina la camilla automáticamente (mayor componente conexa de tejido).\n"
            "Desmarcar si el algoritmo recorta parte del paciente."
        )
        self._chk_table.toggled.connect(self._on_params_changed)
        row3.addWidget(self._chk_table)
        self._chk_overlays = QCheckBox("VOIs/Máscara")
        self._chk_overlays.setChecked(self._show_overlays)
        self._chk_overlays.setEnabled(bool(self._vois or self._mask_hq is not None))
        self._chk_overlays.setToolTip("Proyectar VOIs (círculos) y contorno de la máscara CT sobre el 3D.")
        self._chk_overlays.toggled.connect(self._on_params_changed)
        row3.addWidget(self._chk_overlays)
        row3.addSpacing(12)
        row3.addWidget(QLabel("Ver:"))
        self._view_combo = QComboBox()
        self._view_combo.addItem("Fusión", "fusion")
        self._view_combo.addItem("Solo hueso (CT)", "bone")
        self._view_combo.addItem("Solo SPECT", "spect")
        if not has_sp:
            self._view_combo.setCurrentIndex(1)
            self._view_combo.setEnabled(False)
        self._view_combo.currentIndexChanged.connect(self._on_params_changed)
        row3.addWidget(self._view_combo)
        row3.addSpacing(12)
        row3.addWidget(QLabel("Color SPECT:"))
        self._cmap_combo = QComboBox()
        for name in self._available_cmaps():
            self._cmap_combo.addItem(name)
        if self._cmap_combo.findText("hot") >= 0:
            self._cmap_combo.setCurrentText("hot")
        self._cmap_combo.setEnabled(has_sp)
        self._cmap_combo.currentIndexChanged.connect(self._on_params_changed)
        row3.addWidget(self._cmap_combo)
        row3.addStretch()
        layout.addLayout(row3)

        # Fila 4: ventana SPECT (base/top) + % fusión
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("SPECT fondo:"))
        self._sp_base_slider = QSlider(Qt.Orientation.Horizontal)
        self._sp_base_slider.setRange(0, 95)
        self._sp_base_slider.setValue(int(self._sp_base))
        self._sp_base_slider.setToolTip(
            "Base de la ventana SPECT (% del máximo). Subirla elimina fondo y\n"
            "deja solo los focos calientes (I-131, paratiroides, etc.)."
        )
        self._sp_base_slider.setEnabled(has_sp)
        self._sp_base_slider.valueChanged.connect(self._on_params_changed)
        row4.addWidget(self._sp_base_slider, 1)
        self._sp_base_lbl = QLabel("20%")
        self._sp_base_lbl.setFixedWidth(34)
        row4.addWidget(self._sp_base_lbl)
        row4.addWidget(QLabel("Top:"))
        self._sp_top_slider = QSlider(Qt.Orientation.Horizontal)
        self._sp_top_slider.setRange(10, 100)
        self._sp_top_slider.setValue(int(self._sp_top))
        self._sp_top_slider.setToolTip("Tope de la ventana SPECT (% del máximo): saturación del colormap.")
        self._sp_top_slider.setEnabled(has_sp)
        self._sp_top_slider.valueChanged.connect(self._on_params_changed)
        row4.addWidget(self._sp_top_slider, 1)
        self._sp_top_lbl = QLabel("100%")
        self._sp_top_lbl.setFixedWidth(38)
        row4.addWidget(self._sp_top_lbl)
        row4.addWidget(QLabel("Fusión:"))
        self._mix_slider = QSlider(Qt.Orientation.Horizontal)
        self._mix_slider.setRange(0, 100)
        self._mix_slider.setValue(int(self._fusion_mix * 100))
        self._mix_slider.setToolTip("Peso del SPECT sobre el hueso en modo Fusión.")
        self._mix_slider.setEnabled(has_sp)
        self._mix_slider.valueChanged.connect(self._on_params_changed)
        row4.addWidget(self._mix_slider, 1)
        self._mix_lbl = QLabel("65%")
        self._mix_lbl.setFixedWidth(34)
        row4.addWidget(self._mix_lbl)
        layout.addLayout(row4)

        hint = QLabel("Arrastrar: rotar · Rueda: zoom · Doble clic: vista AP")
        hint.setStyleSheet("color:#64748b; font-size:10px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

    # ------------------------------------------------------------------
    # Interacción
    def _on_press(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = ev.position().toPoint()

    def _on_move(self, ev):
        if self._drag_pos is None:
            return
        pos = ev.position().toPoint()
        dx = pos.x() - self._drag_pos.x()
        dy = pos.y() - self._drag_pos.y()
        self._drag_pos = pos
        self._azimuth = (self._azimuth - dx * 0.7) % 360.0
        self._elevation = float(np.clip(self._elevation + dy * 0.4, -45.0, 45.0))
        self._update_angle_lbl()
        self._schedule(fast=True)

    def _on_release(self, ev):
        self._drag_pos = None
        self._hq_timer.start(220)

    def _on_wheel(self, ev):
        step = 1.12 if ev.angleDelta().y() > 0 else 1 / 1.12
        self._zoom = float(np.clip(self._zoom * step, 0.5, 4.0))
        self._schedule(fast=False)

    def mouseDoubleClickEvent(self, ev):
        self._goto("AP")

    def _goto(self, name: str):
        self._azimuth, self._elevation = self.POSITIONS.get(name, (0.0, 0.0))
        self._update_angle_lbl()
        self._schedule(fast=False)

    def _on_cine_toggled(self, active: bool):
        self._btn_cine.setText("⏸ Cine" if active else "▶ Cine")
        if active:
            self._cine_timer.start(90)
        else:
            self._cine_timer.stop()
            self._hq_timer.start(220)

    def _on_speed_changed(self, value: int):
        self._cine_speed = float(value) / 10.0
        self._speed_lbl.setText(f"{self._cine_speed:.1f}°")

    def _cine_step(self):
        self._azimuth = (self._azimuth + self._cine_speed) % 360.0
        self._update_angle_lbl()
        self._schedule(fast=True)

    def _on_params_changed(self):
        self._hu_threshold = float(self._thr_slider.value())
        self._density = float(self._den_slider.value()) / 100.0
        self._brightness = float(self._bri_slider.value()) / 100.0
        self._remove_table = bool(self._chk_table.isChecked())
        self._show_overlays = bool(self._chk_overlays.isChecked())
        self._view_mode = str(self._view_combo.currentData() or "bone")
        self._cmap_name = str(self._cmap_combo.currentText() or "hot")
        self._sp_base = float(self._sp_base_slider.value())
        self._sp_top = max(float(self._sp_top_slider.value()), self._sp_base + 2.0)
        self._fusion_mix = float(self._mix_slider.value()) / 100.0
        self._thr_lbl.setText(f"{int(self._hu_threshold)}")
        self._den_lbl.setText(f"{int(self._density * 100)}%")
        self._bri_lbl.setText(f"{int(self._brightness * 100)}%")
        self._sp_base_lbl.setText(f"{int(self._sp_base)}%")
        self._sp_top_lbl.setText(f"{int(self._sp_top)}%")
        self._mix_lbl.setText(f"{int(self._fusion_mix * 100)}%")
        self._schedule(fast=True)
        self._hq_timer.start(350)

    @staticmethod
    def _available_cmaps() -> list[str]:
        base = ["hot", "gist_heat", "afmhot", "turbo", "jet", "plasma", "viridis", "gray"]
        try:
            from viz.colormaps import available_colormaps, register_all_colormaps
            register_all_colormaps()
            extra = [n for n in available_colormaps() if n not in base]
            return base + extra
        except Exception:
            return base

    def _apply_spect_cmap(self, arr01: np.ndarray) -> np.ndarray:
        try:
            import matplotlib as mpl
            cm = mpl.colormaps[self._cmap_name]
            return np.asarray(cm(np.clip(arr01, 0.0, 1.0))[..., :3], dtype=np.float32)
        except Exception:
            a = np.clip(arr01, 0.0, 1.0).astype(np.float32)
            return np.stack([a, a * 0.5, np.zeros_like(a)], axis=-1)

    def _update_angle_lbl(self):
        self._angle_lbl.setText(f"Az: {self._azimuth:6.1f}°  El: {self._elevation:+5.1f}°")

    def _schedule(self, *, fast: bool):
        self._fast_mode = fast
        self._render_timer.start(15 if fast else 40)

    def _render_hq(self):
        self._schedule(fast=False)

    # ------------------------------------------------------------------
    # Render volumétrico
    @staticmethod
    def _bone_rgb(t: np.ndarray) -> np.ndarray:
        """Colormap óseo: marrón → beige → blanco marfil según densidad."""
        c0 = np.array([0.42, 0.26, 0.16])
        c1 = np.array([0.87, 0.72, 0.55])
        c2 = np.array([1.00, 0.97, 0.90])
        t = np.clip(t, 0.0, 1.0)[..., None]
        lo = c0 + (c1 - c0) * np.clip(t / 0.45, 0.0, 1.0)
        hi = c1 + (c2 - c1) * np.clip((t - 0.45) / 0.55, 0.0, 1.0)
        return np.where(t < 0.45, lo, hi)

    def _render(self):
        try:
            mode = str(self._view_mode or "bone")
            sp = self._sp_fast if self._fast_mode else self._sp_hq
            use_ct = mode in ("fusion", "bone")
            use_sp = mode in ("fusion", "spect") and sp is not None
            if not use_ct and not use_sp:
                use_ct = True

            def _rot3d(v: np.ndarray, cval: float) -> np.ndarray:
                out = v
                if abs(self._azimuth) > 0.25:
                    out = ndi.rotate(out, self._azimuth, axes=(2, 1), reshape=False,
                                     order=1, cval=cval, mode="constant", prefilter=False)
                if abs(self._elevation) > 0.5:
                    out = ndi.rotate(out, self._elevation, axes=(0, 1), reshape=False,
                                     order=1, cval=cval, mode="constant", prefilter=False)
                return out

            alpha_vox = shade = dens = None
            rot = None
            if use_ct:
                if self._remove_table:
                    ct = self._ct_fast_clean if self._fast_mode else self._ct_hq_clean
                else:
                    ct = self._ct_fast_raw if self._fast_mode else self._ct_hq_raw
                rot = _rot3d(ct, -1000.0)
                thr = self._hu_threshold
                # Opacidad por voxel: rampa desde el umbral (300 HU de transición)
                alpha_vox = np.clip((rot - thr) / 300.0, 0.0, 1.0) ** 1.4
                alpha_vox *= self._density * 0.28
                # Shading difuso: normal desde el gradiente (luz frontal desde cámara)
                g = np.gradient(ndi.gaussian_filter(rot, 1.0) if not self._fast_mode else rot)
                gnorm = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2) + 1e-3
                ndotl = np.clip(-g[1] / gnorm, 0.0, 1.0)
                shade = 0.30 + 0.62 * ndotl + 0.22 * ndotl ** 8  # ambiente + difuso + brillo
                dens = np.clip((rot - thr) / 1100.0, 0.0, 1.0)

            sp_w = sp_alpha = None
            if use_sp:
                sp_rot = _rot3d(sp, 0.0)
                ref = self._sp_p100 if self._sp_p100 > 0 else float(np.max(sp))
                base = self._sp_base / 100.0
                top = max(self._sp_top / 100.0, base + 0.02)
                sp_n = sp_rot / max(ref, 1e-6)
                sp_w = np.clip((sp_n - base) / (top - base), 0.0, 1.0)
                strength = self._fusion_mix if mode == "fusion" else self._density
                sp_alpha = (sp_w ** 1.3) * 0.40 * max(strength, 0.02)

            shape = rot.shape if rot is not None else sp_w.shape
            nz, ny, nx = shape
            color_acc = np.zeros((nz, nx, 3), dtype=np.float32)
            trans = np.ones((nz, nx), dtype=np.float32)

            for j in range(ny):
                a_ct = alpha_vox[:, j, :] if alpha_vox is not None else None
                a_sp = sp_alpha[:, j, :] if sp_alpha is not None else None
                has_ct = a_ct is not None and a_ct.any()
                has_sp = a_sp is not None and a_sp.any()
                if not has_ct and not has_sp:
                    continue
                if has_ct:
                    col = self._bone_rgb(dens[:, j, :]).astype(np.float32) * shade[:, j, :, None]
                    a = a_ct
                    if has_sp:
                        col_sp = self._apply_spect_cmap(sp_w[:, j, :])
                        a_tot = np.clip(a_ct + a_sp, 0.0, 1.0)
                        w_sp = np.divide(a_sp, a_tot, out=np.zeros_like(a_sp), where=a_tot > 1e-6)[..., None]
                        col = col * (1.0 - w_sp) + col_sp * w_sp
                        a = a_tot
                else:
                    col = self._apply_spect_cmap(sp_w[:, j, :])
                    a = a_sp
                contrib = (trans * a)[..., None]
                color_acc += contrib * col
                trans *= (1.0 - a)
                if float(trans.max()) < 0.02:
                    break

            img = np.clip(color_acc * self._brightness * 255.0, 0, 255).astype(np.uint8)
            img = np.ascontiguousarray(img)

            # Overlay: contorno de la máscara CT proyectada (rotada igual que el volumen)
            if self._show_overlays:
                mask = self._mask_fast if self._fast_mode else self._mask_hq
                if mask is not None and mask.shape == shape:
                    m = mask.astype(np.float32)
                    if abs(self._azimuth) > 0.25:
                        m = ndi.rotate(m, self._azimuth, axes=(2, 1), reshape=False,
                                       order=0, cval=0.0, mode="constant", prefilter=False)
                    if abs(self._elevation) > 0.5:
                        m = ndi.rotate(m, self._elevation, axes=(0, 1), reshape=False,
                                       order=0, cval=0.0, mode="constant", prefilter=False)
                    m2d = np.max(m, axis=1) > 0.5
                    if m2d.any():
                        edge = m2d ^ ndi.binary_erosion(m2d, iterations=1)
                        img[m2d] = (img[m2d].astype(np.float32) * 0.82
                                    + np.array([0, 46, 46], dtype=np.float32)).clip(0, 255).astype(np.uint8)
                        img[edge] = (64, 224, 208)  # contorno turquesa

            h, w = img.shape[:2]
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)

            # Overlay: círculos VOI proyectados con la misma rotación del volumen
            if self._show_overlays and self._vois:
                factor = self._factor_fast if self._fast_mode else self._factor_hq
                painter = QPainter(pix)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                self._draw_vois(painter, shape, factor)
                painter.end()

            vw = max(64, self._lbl.width() - 8)
            vh = max(64, self._lbl.height() - 8)
            side = int(min(vw, vh) * self._zoom)
            pix = pix.scaled(side, side, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
            self._lbl.setPixmap(pix)
        except Exception as exc:  # nunca dejar que una excepción mate la app (slot Qt)
            self._lbl.setText(f"VRT error:\n{exc}")

    def _draw_vois(self, painter: QPainter, shape: tuple, factor: float):
        """Proyecta cada VOI (esfera) con la rotación actual (misma convención que ndi.rotate)."""
        nz, ny, nx = shape
        cz0, cy0, cx0 = (nz - 1) * 0.5, (ny - 1) * 0.5, (nx - 1) * 0.5
        az = np.radians(self._azimuth)
        el = np.radians(self._elevation)
        for voi in self._vois:
            try:
                rz = float(voi["cz"]) * factor - cz0
                ry = float(voi["cy"]) * factor - cy0
                rx = float(voi["cx"]) * factor - cx0
                ry1 = ry * np.cos(az) - rx * np.sin(az)
                rx1 = ry * np.sin(az) + rx * np.cos(az)
                rz1 = rz
                rz2 = rz1 * np.cos(el) - ry1 * np.sin(el)
                rx2 = rx1
                row = rz2 + cz0
                col = rx2 + cx0
                r_px = max(2.0, float(voi.get("radius_vox", 4.0)) * factor)
                color = QColor(*voi.get("rgb", (239, 68, 68)))
                painter.setPen(QPen(color, 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPointF(col, row), r_px, r_px)
                label = str(voi.get("label", ""))
                if label:
                    painter.setFont(QFont("Arial", 8))
                    painter.setPen(QPen(QColor(0, 0, 0, 180), 1))
                    painter.drawText(QPointF(col + r_px + 3, row + 3), label)
                    painter.setPen(QPen(color, 1))
                    painter.drawText(QPointF(col + r_px + 2, row + 2), label)
            except Exception:
                continue

    def _save_png(self):
        pm = self._lbl.pixmap()
        if pm is None or pm.isNull():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar VRT como PNG", "", "PNG (*.png)")
        if path:
            if not path.lower().endswith(".png"):
                path += ".png"
            pm.save(path, "PNG")

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._render_timer.start(120)

    def closeEvent(self, ev):
        self._cine_timer.stop()
        super().closeEvent(ev)

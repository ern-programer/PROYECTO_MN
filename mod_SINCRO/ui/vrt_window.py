# -*- coding: utf-8 -*-
"""Ventana VRT 3D: render volumétrico esquelético de la CT con fusión SPECT.

Ray-casting CPU vectorizado: rotación del volumen (scipy), transfer function
de opacidad por HU, shading difuso por gradiente y composición front-to-back.
Los focos SPECT se integran como fuente emisiva violeta dentro del mismo rayo,
por lo que quedan parcialmente ocluidos por el hueso delantero (efecto GE-like).
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QSlider, QVBoxLayout,
)
from scipy import ndimage as ndi


class VrtWindow(QDialog):
    """Render volumétrico 3D interactivo de CT (hueso) + focos SPECT."""

    POSITIONS = {
        "PA": (0.0, 0.0),
        "AP": (180.0, 0.0),
        "LD": (90.0, 0.0),
        "LI": (270.0, 0.0),
        "OAI": (235.0, 0.0),
    }

    #: lado máximo del volumen de render (interactividad CPU)
    MAX_SIDE_HQ = 176
    MAX_SIDE_FAST = 112

    def __init__(self, parent=None, *, ct_volume: np.ndarray,
                 spect_volume: np.ndarray | None = None,
                 spacing_zyx: tuple | None = None):
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

        # Percentil de focos SPECT (sobre el volumen sin rotar, una sola vez)
        self._sp_p100 = float(np.max(self._sp_hq)) if self._sp_hq is not None else 0.0

        self._azimuth = 180.0   # AP
        self._elevation = 0.0
        self._zoom = 1.0
        self._hu_threshold = 150.0
        self._density = 0.55
        self._fusion_on = sp is not None
        self._fusion_pct = 96.0
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

        self._angle_lbl = QLabel("Az: 180.0°  El: +0.0°")
        self._angle_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._angle_lbl.setStyleSheet("color:#94a3b8; font-size:11px; font-family:Consolas,monospace;")
        layout.addWidget(self._angle_lbl)

        # Fila 1: posiciones + cine + guardar
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
        layout.addLayout(row2)

        # Fila 3: fusión SPECT + camilla
        row3 = QHBoxLayout()
        self._chk_table = QCheckBox("Quitar camilla")
        self._chk_table.setChecked(True)
        self._chk_table.setToolTip(
            "Elimina la camilla automáticamente (mayor componente conexa de tejido).\n"
            "Desmarcar si el algoritmo recorta parte del paciente."
        )
        self._chk_table.toggled.connect(self._on_params_changed)
        row3.addWidget(self._chk_table)
        self._chk_fusion = QCheckBox("Fusión SPECT (focos)")
        self._chk_fusion.setChecked(self._fusion_on)
        self._chk_fusion.setEnabled(self._sp_hq is not None)
        self._chk_fusion.toggled.connect(self._on_params_changed)
        row3.addWidget(self._chk_fusion)
        row3.addWidget(QLabel("Umbral focos (percentil):"))
        self._pct_slider = QSlider(Qt.Orientation.Horizontal)
        self._pct_slider.setRange(80, 99)
        self._pct_slider.setValue(int(self._fusion_pct))
        self._pct_slider.setEnabled(self._sp_hq is not None)
        self._pct_slider.valueChanged.connect(self._on_params_changed)
        row3.addWidget(self._pct_slider, 1)
        self._pct_lbl = QLabel("96")
        self._pct_lbl.setFixedWidth(28)
        row3.addWidget(self._pct_lbl)
        layout.addLayout(row3)

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
        self._azimuth, self._elevation = self.POSITIONS.get(name, (180.0, 0.0))
        self._update_angle_lbl()
        self._schedule(fast=False)

    def _on_cine_toggled(self, active: bool):
        self._btn_cine.setText("⏸ Cine" if active else "▶ Cine")
        if active:
            self._cine_timer.start(90)
        else:
            self._cine_timer.stop()
            self._hq_timer.start(220)

    def _cine_step(self):
        self._azimuth = (self._azimuth + 3.0) % 360.0
        self._update_angle_lbl()
        self._schedule(fast=True)

    def _on_params_changed(self):
        self._hu_threshold = float(self._thr_slider.value())
        self._density = float(self._den_slider.value()) / 100.0
        self._fusion_on = bool(self._chk_fusion.isChecked())
        self._fusion_pct = float(self._pct_slider.value())
        self._remove_table = bool(self._chk_table.isChecked())
        self._thr_lbl.setText(f"{int(self._hu_threshold)}")
        self._den_lbl.setText(f"{int(self._density * 100)}%")
        self._pct_lbl.setText(f"{int(self._fusion_pct)}")
        self._schedule(fast=True)
        self._hq_timer.start(350)

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
            if self._remove_table:
                ct = self._ct_fast_clean if self._fast_mode else self._ct_hq_clean
            else:
                ct = self._ct_fast_raw if self._fast_mode else self._ct_hq_raw
            sp = self._sp_fast if self._fast_mode else self._sp_hq

            rot = ct
            if abs(self._azimuth) > 0.25:
                rot = ndi.rotate(rot, self._azimuth, axes=(2, 1), reshape=False,
                                 order=1, cval=-1000.0, mode="constant", prefilter=False)
            if abs(self._elevation) > 0.5:
                rot = ndi.rotate(rot, self._elevation, axes=(0, 1), reshape=False,
                                 order=1, cval=-1000.0, mode="constant", prefilter=False)

            sp_rot = None
            if self._fusion_on and sp is not None:
                sp_rot = sp
                if abs(self._azimuth) > 0.25:
                    sp_rot = ndi.rotate(sp_rot, self._azimuth, axes=(2, 1), reshape=False,
                                        order=1, cval=0.0, mode="constant", prefilter=False)
                if abs(self._elevation) > 0.5:
                    sp_rot = ndi.rotate(sp_rot, self._elevation, axes=(0, 1), reshape=False,
                                        order=1, cval=0.0, mode="constant", prefilter=False)

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

            sp_alpha = None
            if sp_rot is not None and self._sp_p100 > 0:
                sp_thr = np.percentile(sp, self._fusion_pct)
                rng = max(self._sp_p100 - sp_thr, 1e-6)
                sp_alpha = np.clip((sp_rot - sp_thr) / rng, 0.0, 1.0) * 0.55

            nz, ny, nx = rot.shape
            color_acc = np.zeros((nz, nx, 3), dtype=np.float32)
            trans = np.ones((nz, nx), dtype=np.float32)
            sp_color = np.array([0.80, 0.42, 0.98], dtype=np.float32)  # violeta GE-like

            for j in range(ny):
                a_ct = alpha_vox[:, j, :]
                if sp_alpha is not None:
                    a_sp = sp_alpha[:, j, :]
                else:
                    a_sp = None
                if not a_ct.any() and (a_sp is None or not a_sp.any()):
                    continue
                col = self._bone_rgb(dens[:, j, :]).astype(np.float32) * shade[:, j, :, None]
                a = a_ct
                if a_sp is not None and a_sp.any():
                    a_tot = np.clip(a_ct + a_sp, 0.0, 1.0)
                    w_sp = np.divide(a_sp, a_tot, out=np.zeros_like(a_sp), where=a_tot > 1e-6)[..., None]
                    col = col * (1.0 - w_sp) + sp_color * w_sp
                    a = a_tot
                contrib = (trans * a)[..., None]
                color_acc += contrib * col
                trans *= (1.0 - a)
                if float(trans.max()) < 0.02:
                    break

            img = np.clip(color_acc * 255.0, 0, 255).astype(np.uint8)
            img = np.ascontiguousarray(img)
            h, w = img.shape[:2]
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)

            vw = max(64, self._lbl.width() - 8)
            vh = max(64, self._lbl.height() - 8)
            side = int(min(vw, vh) * self._zoom)
            pix = pix.scaled(side, side, Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
            self._lbl.setPixmap(pix)
        except Exception as exc:  # nunca dejar que una excepción mate la app (slot Qt)
            self._lbl.setText(f"VRT error:\n{exc}")

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

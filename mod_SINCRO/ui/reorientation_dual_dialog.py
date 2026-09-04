# -*- coding: utf-8 -*-
"""Ventana de reorientación DUAL (Esfuerzo | Reposo) en paralelo.

Aloja dos CardiacReorientationDialog embebidos lado a lado, uno por etapa.
Reglas clínicas:
- La ELIPSE (zoom/VOI) es COMPARTIDA: al aplicar, los semiejes del reposo se
  igualan a los del esfuerzo (misma magnificación, no simular dilatación).
- La ORIENTACIÓN puede sincronizarse (botón Copiar E→R) pero admite ajuste
  fino independiente en cada panel.
- Un solo botón "Aplicar AMBAS" resuelve las dos etapas en un paso.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ui.reorientation_dialog import CardiacReorientationDialog

_EMBED_HIDE_BUTTONS = {"Cancelar", "Aplicar y generar cortes"}


def _embed_panel(dlg: CardiacReorientationDialog, title: str, color: str) -> QWidget:
    """Convierte un CardiacReorientationDialog en panel embebible con título."""
    dlg.setWindowFlags(Qt.WindowType.Widget)
    dlg.setModal(False)
    dlg.setMinimumSize(640, 560)
    # Ocultar los botones propios del diálogo (los reemplaza la botonera dual).
    for btn in dlg.findChildren(QPushButton):
        if btn.text().strip() in _EMBED_HIDE_BUTTONS:
            btn.hide()
    wrap = QWidget()
    lay = QVBoxLayout(wrap)
    lay.setContentsMargins(2, 2, 2, 2)
    lay.setSpacing(2)
    head = QLabel(title)
    head.setAlignment(Qt.AlignmentFlag.AlignCenter)
    head.setStyleSheet(
        f"background:{color}; color:#f8fafc; font-weight:bold; font-size:13px; "
        "padding:4px; border-radius:3px;"
    )
    lay.addWidget(head)
    lay.addWidget(dlg, 1)
    return wrap


class DualCardiacReorientationDialog(QDialog):
    """Reorientación en paralelo de las dos etapas (pantalla partida)."""

    def __init__(self, stress_kwargs: dict, rest_kwargs: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reorientación DUAL · Esfuerzo | Reposo")
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinMaxButtonsHint
        )
        self.setModal(True)
        self.resize(1860, 860)
        self.setMinimumSize(1200, 640)
        self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)

        # Panel de esfuerzo primero: su VOI automática define el zoom compartido
        # inicial del reposo (si no vino ya un lock de la sesión).
        self.panel_stress = CardiacReorientationDialog(parent=self, **stress_kwargs)
        if rest_kwargs.get("locked_voi") is None:
            rest_kwargs = dict(rest_kwargs)
            rest_kwargs["locked_voi"] = {
                "rz": float(self.panel_stress._voi_rz),
                "ry": float(self.panel_stress._voi_ry),
                "rx": float(self.panel_stress._voi_rx),
            }
        self.panel_rest = CardiacReorientationDialog(parent=self, **rest_kwargs)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        hint = QLabel(
            "Reorientación en paralelo. La <b>elipse (zoom)</b> queda "
            "<span style='color:#b91c1c'>IGUALADA</span> entre etapas al aplicar. "
            "La orientación se puede copiar Esfuerzo→Reposo y luego ajustar fino en cada panel."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#475569; font-size:11px;")
        root.addWidget(hint)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(_embed_panel(self.panel_stress, "ESFUERZO", "#7f1d1d"))
        split.addWidget(_embed_panel(self.panel_rest, "REPOSO", "#1e3a8a"))
        split.setSizes([1, 1])
        root.addWidget(split, 1)

        brow = QHBoxLayout()
        btn_copy = QPushButton("Copiar orientación Esfuerzo → Reposo")
        btn_copy.setToolTip(
            "Copia el eje largo y los límites Base/Ápex del panel de esfuerzo al de "
            "reposo (queda editable para ajuste fino)."
        )
        btn_copy.clicked.connect(self._copy_orientation_stress_to_rest)
        brow.addWidget(btn_copy)
        brow.addStretch(1)
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        brow.addWidget(btn_cancel)
        btn_ok = QPushButton("Aplicar AMBAS y generar cortes")
        btn_ok.setDefault(True)
        btn_ok.setStyleSheet("font-weight:bold;")
        btn_ok.clicked.connect(self._accept_both)
        brow.addWidget(btn_ok)
        root.addLayout(brow)

    # ------------------------------------------------------------- helpers
    def _stress_half_length(self) -> float:
        s = self.panel_stress
        try:
            dz = 0.5 * ((s.h_tra2.y - s.h_tra1.y) + (s.h_cor2.y - s.h_cor1.y))
            dx = s._ap_vol_from_disp_x(s.h_tra2.x) - s._ap_vol_from_disp_x(s.h_tra1.x)
            dy = s._ll_vol_from_disp_y(s.h_cor2.x) - s._ll_vol_from_disp_y(s.h_cor1.x)
            return 0.5 * float(np.linalg.norm([dz, dy, dx]))
        except Exception:
            return 0.0

    def _copy_orientation_stress_to_rest(self):
        try:
            u = self.panel_stress._long_axis_vector()
            half = self._stress_half_length()
            if half <= 0.0:
                return
            # CRÍTICO: copiar explícitamente semiejes de VOI además del eje.
            # Sin esto, ambos ejes coinciden pero la elipse conserva el auto-VOI
            # del reposo, simulando erróneamente dilatación/diferente zoom.
            self._sync_rest_voi_to_stress()
            self.panel_rest._set_handles_from_long_axis(
                np.asarray(u, dtype=np.float64), float(half)
            )
            # Copiar también límites Base/Ápex (fracciones equivalentes).
            self.panel_rest._base_k = int(self.panel_stress._base_k)
            self.panel_rest._apex_k = int(self.panel_stress._apex_k)
            try:
                self.panel_rest.spin_thick.setValue(int(self.panel_stress.spin_thick.value()))
            except Exception:
                pass
            self.panel_rest._recompute_and_draw(full=True)
        except Exception as exc:
            QMessageBox.warning(self, "SINCRO", f"No se pudo copiar la orientación:\n{exc}")

    def _sync_rest_voi_to_stress(self):
        """Fuerza los semiejes (zoom) del reposo = esfuerzo (elipses iguales)."""
        r = self.panel_rest
        s = self.panel_stress
        locked = {
            "rz": float(s._voi_rz),
            "ry": float(s._voi_ry),
            "rx": float(s._voi_rx),
        }
        # Actualizar también el lock interno: _recompute_and_draw respeta este
        # valor y no puede restaurar por error el VOI anterior del reposo.
        r._locked_voi = dict(locked)
        r._voi_rz = locked["rz"]
        r._voi_ry = locked["ry"]
        r._voi_rx = locked["rx"]

    def _accept_both(self):
        # Elipses IGUALES entre etapas: igualar zoom del reposo antes de resolver.
        self._sync_rest_voi_to_stress()
        try:
            self.panel_rest._recompute_and_draw(full=True)
        except Exception:
            pass
        self.panel_stress._accept()
        self.panel_rest._accept()
        if self.panel_stress.reoriented_gated is None or self.panel_rest.reoriented_gated is None:
            QMessageBox.warning(
                self, "SINCRO",
                "No se pudo resolver la reorientación de alguna etapa.\n"
                "Revisá eje largo y límites Base/Ápex en ambos paneles.",
            )
            return
        self.accept()

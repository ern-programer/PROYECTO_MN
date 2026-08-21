# -*- coding: utf-8 -*-
"""Panel de visualización 3D anatómica del corazón.

Muestra un modelo 3D anatómico del corazón con superposición opcional de
captación (PYP, perfusión SPECT, etc.). Usado en amiloidosis, perfusión y
como referencia para la representación del ventrículo.

Modos de visualización:
- Anatomía: modelo gris sin captación.
- Captación: colormap sobre el modelo.
- Fusión: anatomía semitransparente + captación.
"""
from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QWidget, QSlider, QFrame,
)

from core.anatomical_heart import (
    load_heart_model, map_uptake_to_model, to_pyvista,
    list_available_models, ANATOMY_DIR,
)


class Anatomical3DPanel(QDialog):
    """Ventana de visualización 3D anatómica del corazón."""

    def __init__(self, parent=None, uptake_values=None, title="Corazón 3D anatómico"):
        super().__init__(parent)
        self.setWindowTitle(f"SINCRO — {title}")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        self.resize(900, 700)
        self._uptake_values = uptake_values
        self._model = None
        self._plotter = None
        self._cmap = "hot"
        self._mode = "captacion"

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Toolbar ────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        toolbar.addWidget(QLabel("Modelo:"))
        self._model_combo = QComboBox()
        self._model_combo.addItem("Procedural (sin archivo)", "")
        for m in list_available_models():
            self._model_combo.addItem(m, m)
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        toolbar.addWidget(self._model_combo)

        toolbar.addWidget(QLabel("Modo:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Captación", "captacion")
        self._mode_combo.addItem("Anatomía", "anatomia")
        self._mode_combo.addItem("Fusión", "fusion")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        toolbar.addWidget(self._mode_combo)

        toolbar.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["hot", "jet", "viridis", "coolwarm", "gray"])
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        toolbar.addWidget(self._cmap_combo)

        toolbar.addStretch(1)

        btn_reset = QPushButton("Reset vista")
        btn_reset.clicked.connect(self._reset_view)
        toolbar.addWidget(btn_reset)

        root.addLayout(toolbar)

        # ── Host del plotter 3D ────────────────────────────────────
        self._3d_host = QWidget()
        host_layout = QVBoxLayout(self._3d_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._3d_host, 1)

        # ── Info de atribución ─────────────────────────────────────
        self._attribution_lbl = QLabel("")
        self._attribution_lbl.setStyleSheet("font-size: 10px; color: #94a3b8; padding: 2px;")
        self._attribution_lbl.setWordWrap(True)
        root.addWidget(self._attribution_lbl)

        # Inicializar plotter y modelo.
        self._init_plotter()
        self._load_and_render("")

    def _init_plotter(self):
        """Inicializa el QtInteractor de PyVista."""
        try:
            from pyvistaqt import QtInteractor
            self._plotter = QtInteractor(self._3d_host)
            self._3d_host.layout().addWidget(self._plotter.interactor)
            self._plotter.set_background("#0b1220")
        except Exception as exc:
            err_lbl = QLabel(f"No se pudo inicializar el visor 3D:\n{exc}")
            err_lbl.setStyleSheet("color: #f87171; padding: 20px;")
            err_lbl.setWordWrap(True)
            self._3d_host.layout().addWidget(err_lbl)
            self._plotter = None

    def _load_and_render(self, model_name: str):
        """Carga el modelo y lo renderiza."""
        import os
        path = os.path.join(ANATOMY_DIR, model_name) if model_name else None
        self._model = load_heart_model(path)
        self._attribution_lbl.setText(self._model.attribution)
        self._render()

    def _render(self):
        """Renderiza el modelo según el modo actual."""
        if self._plotter is None or self._model is None:
            return
        self._plotter.clear()

        scalars = map_uptake_to_model(self._model, self._uptake_values)

        if self._mode == "anatomia":
            poly = to_pyvista(self._model)
            self._plotter.add_mesh(poly, color="#c0392b", smooth_shading=True, opacity=1.0)
        elif self._mode == "fusion":
            # Anatomía semitransparente + captación.
            poly_anat = to_pyvista(self._model)
            self._plotter.add_mesh(poly_anat, color="#7f8c8d", smooth_shading=True, opacity=0.3)
            poly_up = to_pyvista(self._model, scalars)
            self._plotter.add_mesh(poly_up, scalars="uptake", cmap=self._cmap,
                                   smooth_shading=True, opacity=0.85, show_scalar_bar=True)
        else:  # captacion
            poly = to_pyvista(self._model, scalars)
            self._plotter.add_mesh(poly, scalars="uptake", cmap=self._cmap,
                                   smooth_shading=True, show_scalar_bar=True)

        self._plotter.reset_camera()
        self._plotter.render()

    def _on_model_changed(self, idx: int):
        model_name = self._model_combo.currentData()
        self._load_and_render(model_name or "")

    def _on_mode_changed(self, idx: int):
        self._mode = self._mode_combo.currentData()
        self._render()

    def _on_cmap_changed(self, cmap: str):
        self._cmap = cmap
        self._render()

    def _reset_view(self):
        if self._plotter is not None:
            self._plotter.reset_camera()
            self._plotter.render()

    def closeEvent(self, event):
        """Limpia el plotter al cerrar para evitar leaks."""
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass
        super().closeEvent(event)

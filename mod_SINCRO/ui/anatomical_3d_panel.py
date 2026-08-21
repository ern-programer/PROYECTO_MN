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

import os
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QWidget,
)

from core.anatomical_heart import (
    load_heart_model, map_uptake_to_model, map_planar_ap_to_model,
    apply_uptake_preset, to_pyvista,
    list_available_models, ANATOMY_DIR,
)


class Anatomical3DPanel(QDialog):
    """Ventana de visualización 3D anatómica del corazón."""

    def __init__(self, parent=None, uptake_values=None, image_ap=None, title="Corazón 3D anatómico"):
        super().__init__(parent)
        self.setWindowTitle(f"SINCRO — {title}")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        self.resize(900, 700)
        self._uptake_values = uptake_values
        self._image_ap = image_ap
        self._model = None
        self._plotter = None
        self._cmap = "hot"
        self._mode = "captacion"
        self._preset = "amyloid"

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Toolbar ────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        toolbar.addWidget(QLabel("Modelo:"))
        self._model_combo = QComboBox()
        self._model_combo.addItem("Template VI (sin archivo)", "")
        models = list_available_models()
        heart_models = [m for m in models if os.path.basename(m).lower().startswith("heart_")]
        extra_models = [m for m in models if m not in heart_models]

        for m in heart_models:
            self._model_combo.addItem(f"[Cardíaco] {m}", m)
        for m in extra_models:
            self._model_combo.addItem(f"[Complementario] {m}", m)

        # Preferir por defecto un corazón real si existe.
        if heart_models:
            self._model_combo.setCurrentIndex(1)
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

        toolbar.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("Amiloidosis PYP", "amyloid")
        self._preset_combo.addItem("Perfusión SPECT", "perfusion")
        self._preset_combo.addItem("Referencia VI", "vi")
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        toolbar.addWidget(self._preset_combo)

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

        self._model_status_lbl = QLabel("")
        self._model_status_lbl.setStyleSheet("font-size: 10px; color: #fbbf24; padding: 2px;")
        self._model_status_lbl.setWordWrap(True)
        root.addWidget(self._model_status_lbl)

        self._help_lbl = QLabel(
            "Uso rápido: 1) Elegí preset clínico. 2) Modo Captación para ver señal, "
            "Fusión para comparar anatomía+captación. 3) Rotá con botón izquierdo, zoom con rueda.\n"
            "Nota: visualización exploratoria 2D→3D (no cuantificación tomográfica)."
        )
        self._help_lbl.setStyleSheet("font-size: 10px; color: #93c5fd; padding: 2px;")
        self._help_lbl.setWordWrap(True)
        root.addWidget(self._help_lbl)

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
        path = os.path.join(ANATOMY_DIR, model_name) if model_name else None
        self._model = load_heart_model(path)
        self._attribution_lbl.setText(self._model.attribution)
        if self._model.source.startswith("file:"):
            self._model_status_lbl.setText("Modelo anatómico real cargado.")
        elif self._model.source == "template_lv":
            self._model_status_lbl.setText(
                "Template VI activo (sintético avanzado). Para anatomía completa real, cargá una malla OBJ/STL en assets/anatomy/."
            )
        elif self._model.is_procedural:
            self._model_status_lbl.setText(
                "Modelo procedural activo (demo). Para anatomía cardíaca real, cargá una malla OBJ/STL en assets/anatomy/."
            )
        else:
            self._model_status_lbl.setText("Modelo sintético cargado.")
        self._render()
        self._set_default_view()

    def _render(self):
        """Renderiza el modelo según el modo actual."""
        if self._plotter is None or self._model is None:
            return
        self._plotter.clear()

        if self._image_ap is not None:
            scalars = map_planar_ap_to_model(self._model, self._image_ap)
        else:
            scalars = map_uptake_to_model(self._model, self._uptake_values)
        scalars = apply_uptake_preset(self._model, scalars, self._preset)

        if self._mode == "anatomia":
            poly = to_pyvista(self._model)
            self._plotter.add_mesh(poly, color="#c0392b", smooth_shading=True, opacity=1.0, specular=0.25)
        elif self._mode == "fusion":
            # Anatomía semitransparente + captación.
            poly_anat = to_pyvista(self._model)
            self._plotter.add_mesh(poly_anat, color="#7f8c8d", smooth_shading=True, opacity=0.35, specular=0.2)
            poly_up = to_pyvista(self._model, scalars)
            self._plotter.add_mesh(poly_up, scalars="uptake", cmap=self._cmap,
                                   smooth_shading=True, opacity=0.85, show_scalar_bar=True)
        else:  # captacion
            poly = to_pyvista(self._model, scalars)
            self._plotter.add_mesh(poly, scalars="uptake", cmap=self._cmap,
                                   smooth_shading=True, show_scalar_bar=True)

        self._plotter.add_axes(interactive=False)
        self._plotter.render()

    def _set_default_view(self):
        """Vista anatómica inicial estable (anterior oblicua suave)."""
        if self._plotter is None:
            return
        try:
            # Convención interna: x=lateral-septal, y=anterior-inferior, z=ápex-base.
            # Vista inicial desde anterior con leve oblicuidad para dar profundidad.
            self._plotter.view_vector((0.3, -1.0, 0.2), viewup=(0.0, 0.0, 1.0))
            self._plotter.reset_camera()
            self._plotter.render()
        except Exception:
            pass

    def _on_model_changed(self, idx: int):
        model_name = self._model_combo.currentData()
        self._load_and_render(model_name or "")

    def _on_mode_changed(self, idx: int):
        self._mode = self._mode_combo.currentData()
        self._render()

    def _on_cmap_changed(self, cmap: str):
        self._cmap = cmap
        self._render()

    def _on_preset_changed(self, idx: int):
        self._preset = self._preset_combo.currentData() or "amyloid"
        self._render()

    def _reset_view(self):
        self._set_default_view()

    def closeEvent(self, event):
        """Limpia el plotter al cerrar para evitar leaks."""
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass
        super().closeEvent(event)

# -*- coding: utf-8 -*-
"""Layouts de cuadrantes para el visor de amiloidosis.

Define layouts de 4, 8, 9, 12 y 16 cuadrantes para mostrar imágenes
planar/SPECT con diferentes configuraciones.

Cada cuadrante tiene:
- Imagen (np.ndarray)
- Rótulo (str) — del DICOM o editable
- Colormap (str) — por defecto "gray"
- Ventana (low, high) — por defecto (0, 100)%
- Filtros aplicados (list[str]) — por defecto []
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class Quadrant:
    """Un cuadrante individual del visor."""
    image: np.ndarray | None = None
    label: str = ""
    cmap: str = "gray"
    win_low: float = 0.0
    win_high: float = 100.0
    filters: list[str] = field(default_factory=list)
    selected: bool = False
    # Metadatos para el informe.
    hmr: float | None = None
    perugini: int | None = None
    roi_overlay: bool = False  # True = dibujar ROIs sobre la imagen


@dataclass
class Layout:
    """Un layout de cuadrantes."""
    name: str
    description: str
    rows: int
    cols: int
    quadrants: list[Quadrant]

    def total(self) -> int:
        return self.rows * self.cols


# ============================================================
# Plantillas de layouts
# ============================================================

def layout_4q(
    ap_roi: np.ndarray | None = None,
    ap_clean: np.ndarray | None = None,
    oai: np.ndarray | None = None,
    lat: np.ndarray | None = None,
    ap_label: str = "AP",
    oai_label: str = "OAI 45°",
    lat_label: str = "LAT. IZQ.",
) -> Layout:
    """Layout de 4 cuadrantes: AP+ROIs, AP limpia, OAI, LAT. IZQ."""
    return Layout(
        name="4 cuadrantes (planar básico)",
        description="AP con ROIs + HMR · AP limpia · OAI · LAT. IZQ.",
        rows=2,
        cols=2,
        quadrants=[
            Quadrant(image=ap_roi, label=f"{ap_label} + ROIs", roi_overlay=True),
            Quadrant(image=ap_clean, label=f"{ap_label} (limpio)"),
            Quadrant(image=oai, label=oai_label),
            Quadrant(image=lat, label=lat_label),
        ],
    )


def layout_8q(
    images_1h: list[np.ndarray | None] | None = None,
    images_3h: list[np.ndarray | None] | None = None,
    labels: list[str] | None = None,
) -> Layout:
    """Layout de 8 cuadrantes: 4 arriba (1h), 4 abajo (3h)."""
    labels = labels or ["AP + ROIs", "AP (limpio)", "OAI 45°", "LAT. IZQ."]
    imgs_1h = images_1h or [None, None, None, None]
    imgs_3h = images_3h or [None, None, None, None]
    quads = []
    for i, lbl in enumerate(labels):
        q1h = Quadrant(image=imgs_1h[i] if i < len(imgs_1h) else None, label=f"{lbl} (1h)")
        q3h = Quadrant(image=imgs_3h[i] if i < len(imgs_3h) else None, label=f"{lbl} (3h)")
        if i == 0:
            q1h.roi_overlay = True
            q3h.roi_overlay = True
        quads.append(q1h)
    for i, lbl in enumerate(labels):
        q3h = Quadrant(image=imgs_3h[i] if i < len(imgs_3h) else None, label=f"{lbl} (3h)")
        if i == 0:
            q3h.roi_overlay = True
        quads.append(q3h)
    return Layout(
        name="8 cuadrantes (washout 1h vs 3h)",
        description="4 imágenes (1 hora) arriba · 4 imágenes (3 horas) abajo",
        rows=2,
        cols=4,
        quadrants=quads,
    )


def layout_9q(
    images: list[np.ndarray | None] | None = None,
    labels: list[str] | None = None,
) -> Layout:
    """Layout de 9 cuadrantes (3×3): 3 estudios × 3 vistas."""
    labels = labels or ["AP", "OAI 45°", "LAT. IZQ."]
    imgs = images or [None] * 9
    quads = []
    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            lbl = labels[col] if col < len(labels) else f"Img {idx+1}"
            q = Quadrant(
                image=imgs[idx] if idx < len(imgs) else None,
                label=lbl,
            )
            if col == 0:
                q.roi_overlay = True
            quads.append(q)
    return Layout(
        name="9 cuadrantes (3 estudios × 3 vistas)",
        description="3 estudios comparados en 3 vistas: AP, OAI, LAT. IZQ.",
        rows=3,
        cols=3,
        quadrants=quads,
    )


def layout_12q(
    images: list[np.ndarray | None] | None = None,
    labels: list[str] | None = None,
) -> Layout:
    """Layout de 12 cuadrantes (4×3): 4 estudios × 3 vistas."""
    labels = labels or ["AP + ROIs", "OAI 45°", "LAT. IZQ."]
    imgs = images or [None] * 12
    quads = []
    for row in range(4):
        for col in range(3):
            idx = row * 3 + col
            lbl = labels[col] if col < len(labels) else f"Img {idx+1}"
            q = Quadrant(
                image=imgs[idx] if idx < len(imgs) else None,
                label=lbl,
            )
            if col == 0:
                q.roi_overlay = True
            quads.append(q)
    return Layout(
        name="12 cuadrantes (4 estudios × 3 vistas)",
        description="4 estudios comparados en 3 vistas: AP, OAI, LAT. IZQ.",
        rows=4,
        cols=3,
        quadrants=quads,
    )


def layout_16q(
    images: list[np.ndarray | None] | None = None,
    labels: list[str] | None = None,
) -> Layout:
    """Layout de 16 cuadrantes (4×4): 4 tiempos × 4 vistas."""
    labels = labels or ["AP + ROIs", "AP (limpio)", "OAI 45°", "LAT. IZQ."]
    imgs = images or [None] * 16
    quads = []
    for row in range(4):
        for col in range(4):
            idx = row * 4 + col
            lbl = labels[col] if col < len(labels) else f"Img {idx+1}"
            q = Quadrant(
                image=imgs[idx] if idx < len(imgs) else None,
                label=lbl,
            )
            if col == 0:
                q.roi_overlay = True
            quads.append(q)
    return Layout(
        name="16 cuadrantes (4 tiempos × 4 vistas)",
        description="4 tiempos (p.ej. basal, 1h, 3h, follow-up) × 4 vistas",
        rows=4,
        cols=4,
        quadrants=quads,
    )


# Catálogo de layouts disponibles.
LAYOUT_CATALOG: dict[str, callable] = {
    "4q": layout_4q,
    "8q": layout_8q,
    "9q": layout_9q,
    "12q": layout_12q,
    "16q": layout_16q,
}

LAYOUT_NAMES: list[tuple[str, str]] = [
    ("4q", "4 cuadrantes (planar básico)"),
    ("8q", "8 cuadrantes (washout 1h vs 3h)"),
    ("9q", "9 cuadrantes (3 estudios × 3 vistas)"),
    ("12q", "12 cuadrantes (4 estudios × 3 vistas)"),
    ("16q", "16 cuadrantes (4 tiempos × 4 vistas)"),
]

# -*- coding: utf-8 -*-
"""Carga y visualización de modelos anatómicos 3D del corazón.

Provee un modelo de referencia anatómica para tres usos en SINCRO:
1. Amiloidosis: superponer captación PYP sobre anatomía real.
2. Perfusión SPECT: mapear la perfusión sobre el modelo.
3. Ventrículo 3D: referencia anatómica para validar la segmentación del VI.

Fuentes de modelos gratuitos/open source (colocar en assets/anatomy/):
- BodyParts3D (CC BY-SA): https://lifesciencedb.jp/bp3d/
- NIH 3D Print Exchange (dominio público): https://3dprint.nih.gov/
- Open Anatomy Project (MIT): https://www.openanatomy.org/

El módulo NO descarga automáticamente (respeta licencias y evita dependencias
de red en tiempo de ejecución). Si no encuentra un modelo, genera un corazón
procedural simplificado como fallback para no bloquear la visualización.

Formatos soportados: OBJ, STL, PLY, VTP (vía PyVista/meshio).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Directorio estándar donde el usuario coloca modelos descargados.
ANATOMY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "anatomy",
)

SUPPORTED_EXTENSIONS = (".obj", ".stl", ".ply", ".vtp", ".vtk")

# Créditos de atribución (obligatorios para CC BY-SA).
MODEL_ATTRIBUTIONS = {
    "bodyparts3d": "BodyParts3D, © The Database Center for Life Science, CC BY-SA 2.1 JP",
    "nih": "NIH 3D Print Exchange, dominio público",
    "openanatomy": "Open Anatomy Project, MIT License",
}


@dataclass
class HeartModel:
    """Modelo 3D del corazón cargado o generado."""
    vertices: np.ndarray          # (N, 3)
    faces: np.ndarray             # (M, 3) índices de triángulos
    source: str                   # "file:<nombre>" o "procedural"
    attribution: str = ""         # crédito de licencia si aplica

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def is_procedural(self) -> bool:
        return self.source == "procedural"

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (min, max) de las coordenadas."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def center(self) -> np.ndarray:
        lo, hi = self.bounds()
        return (lo + hi) / 2.0


def list_available_models() -> list[str]:
    """Lista los modelos anatómicos disponibles en el directorio de assets."""
    if not os.path.isdir(ANATOMY_DIR):
        return []
    models = []
    for fn in os.listdir(ANATOMY_DIR):
        if fn.lower().endswith(SUPPORTED_EXTENSIONS):
            models.append(fn)
    return sorted(models)


def load_heart_model(path: str | None = None) -> HeartModel:
    """Carga un modelo 3D del corazón desde archivo, o genera uno procedural.

    Args:
        path: ruta al archivo de malla. Si es None o no existe, genera fallback.

    Returns:
        HeartModel con vértices y caras.
    """
    if path and os.path.isfile(path):
        try:
            return _load_mesh_file(path)
        except Exception:
            pass  # Cae al procedural si falla la carga.
    return generate_procedural_heart()


def _load_mesh_file(path: str) -> HeartModel:
    """Carga una malla usando PyVista (soporta OBJ/STL/PLY/VTP)."""
    import pyvista as pv

    mesh = pv.read(path)
    # Triangular si no lo está.
    mesh = mesh.triangulate()
    vertices = np.asarray(mesh.points, dtype=np.float64)
    # Extraer caras (PyVista usa formato [n, i0, i1, ..., in]).
    faces_raw = np.asarray(mesh.faces).reshape(-1, 4)
    faces = faces_raw[:, 1:4].astype(np.int64)

    # Determinar atribución por nombre de archivo.
    fname = os.path.basename(path).lower()
    attribution = ""
    for key, credit in MODEL_ATTRIBUTIONS.items():
        if key in fname:
            attribution = credit
            break

    return HeartModel(
        vertices=vertices,
        faces=faces,
        source=f"file:{os.path.basename(path)}",
        attribution=attribution,
    )


def generate_procedural_heart() -> HeartModel:
    """Genera un corazón procedural simplificado (fallback sin archivo).

    Modela un elipsoide con una depresión apical y una hendidura interventricular,
    suficiente para orientación anatómica básica. No es anatómicamente exacto.
    """
    # Parametrización esférica.
    n_theta, n_phi = 48, 48
    theta = np.linspace(0, np.pi, n_theta)       # polar
    phi = np.linspace(0, 2 * np.pi, n_phi)       # azimutal
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    # Elipsoide base (corazón más largo que ancho).
    a, b, c = 4.0, 4.0, 5.5  # semiejes (x, y, z)
    x = a * np.sin(theta_grid) * np.cos(phi_grid)
    y = b * np.sin(theta_grid) * np.sin(phi_grid)
    z = c * np.cos(theta_grid)

    # Depresión apical: estrechar la punta (z bajo).
    apex_factor = 1.0 - 0.35 * np.clip((-z / c), 0, 1) ** 2
    x *= apex_factor
    y *= apex_factor

    # Hendidura interventricular (surco en un lado).
    groove = 1.0 - 0.15 * np.exp(-((phi_grid - np.pi / 2) ** 2) / 0.3)
    x *= groove

    # Construir vértices.
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Construir caras (triángulos de la grilla).
    faces = []
    for i in range(n_theta - 1):
        for j in range(n_phi - 1):
            v0 = i * n_phi + j
            v1 = i * n_phi + (j + 1)
            v2 = (i + 1) * n_phi + j
            v3 = (i + 1) * n_phi + (j + 1)
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    faces = np.asarray(faces, dtype=np.int64)

    return HeartModel(
        vertices=vertices,
        faces=faces,
        source="procedural",
        attribution="Modelo procedural SINCRO (no anatómicamente exacto)",
    )


def map_uptake_to_model(
    model: HeartModel,
    uptake_values: np.ndarray | None = None,
) -> np.ndarray:
    """Mapea valores de captación a los vértices del modelo.

    Args:
        model: modelo del corazón.
        uptake_values: array de valores por vértice. Si es None o no coincide,
            genera un patrón radial de ejemplo.

    Returns:
        Array de valores escalares por vértice (normalizado 0-1).
    """
    n = model.n_vertices
    if uptake_values is not None and len(uptake_values) == n:
        vals = np.asarray(uptake_values, dtype=np.float64)
    else:
        # Patrón de ejemplo: gradiente base-ápex.
        z = model.vertices[:, 2]
        vals = (z - z.min()) / max(z.max() - z.min(), 1e-8)

    # Normalizar a 0-1.
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax - vmin < 1e-8:
        return np.zeros(n, dtype=np.float64)
    return (vals - vmin) / (vmax - vmin)


def to_pyvista(model: HeartModel, scalars: np.ndarray | None = None):
    """Convierte un HeartModel a un objeto PyVista PolyData."""
    import pyvista as pv

    # PyVista requiere formato de caras [n, i0, i1, i2, ...].
    n_faces = model.faces.shape[0]
    faces_pv = np.column_stack([
        np.full(n_faces, 3, dtype=np.int64),
        model.faces,
    ]).ravel()
    poly = pv.PolyData(model.vertices, faces_pv)
    if scalars is not None and len(scalars) == model.n_vertices:
        poly["uptake"] = scalars
    return poly

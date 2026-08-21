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
    "totalsegmentator": "TotalSegmentator dataset (subject s1397), CC BY 4.0, doi:10.1148/ryai.230024",
    "ccby4": "TotalSegmentator dataset (subject s1397), CC BY 4.0, doi:10.1148/ryai.230024",
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
    try:
        return generate_lv_template_heart()
    except Exception:
        return generate_procedural_heart()


def _polydata_to_heart_model(mesh, source: str, attribution: str = "") -> HeartModel:
    """Convierte un PolyData de PyVista a HeartModel triangulado."""
    mesh = mesh.triangulate()
    vertices = np.asarray(mesh.points, dtype=np.float64)
    faces_raw = np.asarray(mesh.faces).reshape(-1, 4)
    faces = faces_raw[:, 1:4].astype(np.int64)
    return HeartModel(vertices=vertices, faces=faces, source=source, attribution=attribution)


def _load_mesh_file(path: str) -> HeartModel:
    """Carga una malla usando PyVista (soporta OBJ/STL/PLY/VTP)."""
    import pyvista as pv

    mesh = pv.read(path)

    # Determinar atribución por nombre de archivo.
    fname = os.path.basename(path).lower()
    attribution = ""
    for key, credit in MODEL_ATTRIBUTIONS.items():
        if key in fname:
            attribution = credit
            break

    return _polydata_to_heart_model(
        mesh,
        source=f"file:{os.path.basename(path)}",
        attribution=attribution,
    )


def generate_lv_template_heart() -> HeartModel:
    """Genera una malla templada de VI+pared miocárdica (más anatómica que el fallback)."""
    from core.lv_mesh import myocardium_shell_mesh

    n_slices = 22
    n_angles = 96
    z_mm = np.linspace(0.0, 80.0, n_slices)

    # Radio base de VI (ápex pequeño, cuerpo medio, base más ancha).
    z_n = (z_mm - z_mm.min()) / max(float(z_mm.max() - z_mm.min()), 1e-8)
    long_profile = (np.sin(np.pi * np.clip(z_n, 0.0, 1.0)) ** 0.80)
    long_profile *= 0.80 + 0.32 * z_n

    theta = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    # Elipticidad y leve deformación septo-lateral.
    ellipse = 1.0 + 0.14 * np.cos(theta - np.pi / 2.0) - 0.08 * np.cos(2.0 * theta)

    epi_base = 22.0 * long_profile[:, None] * ellipse[None, :]
    wall = 6.5 - 1.7 * z_n[:, None]  # espesor decrece hacia base
    wall = np.clip(wall, 4.5, 7.0)
    endo = np.clip(epi_base - wall, 0.8, None)
    epi = np.clip(epi_base, 1.2, None)

    # Línea central levemente curva para evitar geometría rígida.
    centers = np.zeros((n_slices, 2), dtype=np.float64)
    centers[:, 0] = 1.8 * np.sin(1.15 * np.pi * z_n)   # lateral-septal
    centers[:, 1] = -1.2 * np.cos(0.9 * np.pi * z_n)   # anterior-inferior

    shell = myocardium_shell_mesh(
        endo,
        epi,
        z_mm,
        smooth_iter=8,
        smooth_relax=0.06,
        interp_z=2.2,
        smooth_angular=0.7,
        centers_mm=centers,
        apex_virtual_rings=8,
        apex_taper=0.28,
    )

    return _polydata_to_heart_model(
        shell,
        source="template_lv",
        attribution="Template anatómico VI SINCRO (sintético basado en geometría ventricular)",
    )


def generate_procedural_heart() -> HeartModel:
    """Genera un corazón procedural simplificado (fallback sin archivo).

    Modela un elipsoide con una depresión apical y una hendidura interventricular,
    suficiente para orientación anatómica básica. No es anatómicamente exacto.
    """
    # Parametrización (z axial + phi circunferencial) para una forma
    # ventricular/miocárdica más reconocible que un elipsoide puro.
    n_z, n_phi = 64, 72
    z_n = np.linspace(0.0, 1.0, n_z)  # 0=ápex, 1=base
    phi = np.linspace(0.0, 2 * np.pi, n_phi)
    z_grid, phi_grid = np.meshgrid(z_n, phi, indexing="ij")

    # Perfil longitudinal tipo VI: máximo en tercio medio, base más ancha,
    # ápex angosto.
    radius_main = (np.sin(np.pi * np.clip(z_grid, 0.0, 1.0)) ** 0.82)
    radius_main *= 0.82 + 0.30 * z_grid

    # Asimetrías anatómicas suaves.
    # - Septum levemente más plano (x negativo)
    # - Bulbo de VD anterior-septal en tercio medio-basal
    phi_center_rv = 0.35 * np.pi
    rv_bulge = 0.18 * np.exp(-((phi_grid - phi_center_rv) ** 2) / 0.20) * np.exp(-((z_grid - 0.58) ** 2) / 0.06)
    septal_flat = 1.0 - 0.16 * np.exp(-((phi_grid - np.pi) ** 2) / 0.22)

    radius = radius_main * septal_flat + rv_bulge

    # Ápex más puntiagudo (sin colapsar) y base valvular ligeramente truncada.
    apex_taper = 0.45 + 0.55 * np.clip(z_grid / 0.25, 0.0, 1.0)
    base_taper = 1.0 - 0.08 * np.clip((z_grid - 0.9) / 0.1, 0.0, 1.0)
    radius *= apex_taper * base_taper

    # Escala final (mm relativos) y ejes:
    # x=lateral-septal, y=anterior-inferior, z=ápex→base
    a, b, c = 32.0, 28.0, 42.0
    x = a * radius * np.cos(phi_grid)
    y = b * radius * np.sin(phi_grid)
    z = c * (z_grid - 0.55)

    # Surco interventricular anterior (hendidura suave visual).
    groove = 1.0 - 0.12 * np.exp(-((phi_grid - 0.15 * np.pi) ** 2) / 0.09)
    x *= groove

    # Construir vértices.
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Construir caras (triángulos de la grilla).
    faces = []
    for i in range(n_z - 1):
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


def map_planar_ap_to_model(model: HeartModel, image_ap: np.ndarray | None) -> np.ndarray:
    """Proyecta una planar AP 2D sobre la superficie 3D del corazón.

    Mapeo práctico para visualización clínica rápida (NO cuantitativo 3D):
    - X del modelo ↔ eje horizontal de la AP
    - Z del modelo ↔ eje vertical de la AP
    - Y modula la intensidad para priorizar la cara anterior
    """
    if image_ap is None:
        return map_uptake_to_model(model, None)

    img = np.asarray(image_ap, dtype=np.float64)
    while img.ndim > 2:
        img = img[img.shape[0] // 2]
    if img.ndim != 2 or img.size == 0:
        return map_uptake_to_model(model, None)

    # Robustez frente a fondo/ruido: normalizar por percentiles.
    p1, p99 = np.percentile(img, [1, 99])
    denom = max(float(p99 - p1), 1e-8)
    img_n = np.clip((img - p1) / denom, 0.0, 1.0)

    verts = model.vertices
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]

    # Coordenadas normalizadas del modelo para mapear a píxeles.
    u = (x - x.min()) / max(float(x.max() - x.min()), 1e-8)
    v = (z - z.min()) / max(float(z.max() - z.min()), 1e-8)

    h, w = img_n.shape
    px = np.clip(u * (w - 1), 0, w - 1)
    py = np.clip((1.0 - v) * (h - 1), 0, h - 1)

    # Muestreo bilineal.
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    dx = px - x0
    dy = py - y0

    i00 = img_n[y0, x0]
    i10 = img_n[y0, x1]
    i01 = img_n[y1, x0]
    i11 = img_n[y1, x1]
    sampled = (
        i00 * (1 - dx) * (1 - dy)
        + i10 * dx * (1 - dy)
        + i01 * (1 - dx) * dy
        + i11 * dx * dy
    )

    # Peso anterior fuerte: AP representa principalmente la cara anterior.
    y_n = (y - y.min()) / max(float(y.max() - y.min()), 1e-8)
    anterior_weight = np.clip(1.0 - y_n, 0.0, 1.0) ** 1.8
    vals = sampled * anterior_weight

    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax - vmin < 1e-8:
        return np.zeros(model.n_vertices, dtype=np.float64)
    return (vals - vmin) / (vmax - vmin)


def apply_uptake_preset(model: HeartModel, scalars: np.ndarray, preset: str) -> np.ndarray:
    """Ajusta la visualización según preset clínico (heurístico, no diagnóstico)."""
    s = np.asarray(scalars, dtype=np.float64)
    if s.size != model.n_vertices:
        return map_uptake_to_model(model, s)

    p = (preset or "amyloid").lower()
    if p == "perfusion":
        # Resaltar pared media (anillo) para lectura tipo perfusión.
        r = np.linalg.norm(model.vertices[:, :2], axis=1)
        r_n = (r - r.min()) / max(float(r.max() - r.min()), 1e-8)
        ring = np.exp(-((r_n - 0.55) ** 2) / 0.035)
        out = 0.55 * s + 0.45 * ring
    elif p == "vi":
        # Priorizar anatomía del VI: menor saturación de captación.
        z = model.vertices[:, 2]
        z_n = (z - z.min()) / max(float(z.max() - z.min()), 1e-8)
        out = 0.35 * s + 0.65 * (1.0 - z_n)
    else:
        # amyloid
        out = np.clip(s, 0.0, 1.0) ** 0.85

    # Suavizado muy leve para evitar apariencia "ruido pegado" al mesh.
    # Se aplica por vecindad geométrica simple usando promedio de triángulos.
    try:
        acc = np.zeros_like(out)
        cnt = np.zeros_like(out)
        f = model.faces
        for tri in f:
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            m = (out[i] + out[j] + out[k]) / 3.0
            acc[i] += m
            acc[j] += m
            acc[k] += m
            cnt[i] += 1.0
            cnt[j] += 1.0
            cnt[k] += 1.0
        sm = acc / np.maximum(cnt, 1e-8)
        out = 0.75 * out + 0.25 * sm
    except Exception:
        pass

    vmin, vmax = float(out.min()), float(out.max())
    if vmax - vmin < 1e-8:
        return np.zeros(model.n_vertices, dtype=np.float64)
    return (out - vmin) / (vmax - vmin)


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

"""SINCRO - core.col_registry

Registro centralizado de colormaps: carga los archivos .col de Odyssey/Xeleris
(TABLAS FX y TABLAS LX) y los registra como colormaps de matplotlib junto con
los colormaps estándar ya disponibles.

Uso:
    from core.col_registry import register_all_colormaps, available_colormaps

    register_all_colormaps()  # llamar una vez al inicio
    names = available_colormaps()  # lista ordenada de todos los colormaps
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import matplotlib

# Tamaño fijo de paleta Odyssey: 256 colores, 3 canales, uint32 big-endian por canal.
_PALETTE_SIZE = 256
_ENTRY_SIZE = 4 * 3  # 3 x uint32
_FILE_SIZE = _PALETTE_SIZE * _ENTRY_SIZE

# Colormaps estándar que siempre deben estar presentes.
_BUILTIN_NAMES: list[str] = [
    "gray", "hot", "cool", "prism", "viridis", "plasma", "inferno",
    "magma", "cividis", "turbo", "bone", "cubehelix",
]

# Colormaps cíclicos (para fase).
_CYCLIC_NAMES: list[str] = [
    "hsv", "twilight", "twilight_shifted",
]

# Colormap "french" propio de GammaSync.
_FRENCH_CMAP = LinearSegmentedColormap.from_list(
    "french",
    [
        (0.0, "#0b3fa5"),
        (0.5, "#ffffff"),
        (1.0, "#d62828"),
    ],
)

# Colormap clínico de perfusión (bajo->alto): cian -> magenta -> naranja -> amarillo -> blanco.
_PERF_CLINICAL_CMAP = LinearSegmentedColormap.from_list(
    "perf_clinical",
    [
        (0.00, "#19b8c8"),
        (0.35, "#c653c8"),
        (0.62, "#ff9800"),
        (0.84, "#ffe082"),
        (1.00, "#ffffff"),
    ],
)

# Directorios donde buscar .col (relativos a la raíz del módulo o absolutos conocidos).
_COL_SEARCH_DIRS: list[str] = []

_registered: bool = False
_all_names: list[str] = []


def _find_col_dirs() -> list[Path]:
    """Busca las carpetas TABLAS FX y TABLAS LX de Odyssey."""
    candidates = []

    # Ruta absoluta conocida del usuario.
    base = Path(r"D:\- GAMMASYS\- PROGRAMAS\- COL")
    for subdir in ("TABLAS FX", "TABLAS LX"):
        p = base / subdir
        if p.is_dir():
            candidates.append(p)

    return candidates


def _read_col_binary(path: Path) -> list[tuple[int, int, int]]:
    """Lee un archivo .col binario de Odyssey (256 colores, uint32 BE por canal)."""
    data = path.read_bytes()
    if len(data) != _FILE_SIZE:
        return []
    palette = []
    for i in range(_PALETTE_SIZE):
        chunk = data[i * _ENTRY_SIZE : (i + 1) * _ENTRY_SIZE]
        r, g, b = struct.unpack(">III", chunk)
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            return []
        palette.append((r, g, b))
    return palette


def _palette_to_cmap(name: str, palette: list[tuple[int, int, int]]) -> LinearSegmentedColormap:
    """Convierte una paleta de 256 colores a un LinearSegmentedColormap de matplotlib."""
    colors = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in palette]
    return LinearSegmentedColormap.from_list(name, colors, N=256)


def register_all_colormaps() -> list[str]:
    """Registra todos los colormaps disponibles en matplotlib y devuelve la lista de nombres."""
    global _registered, _all_names
    if _registered:
        return _all_names

    names: list[str] = []

    # 1) Colormaps propios.
    matplotlib.colormaps.register(_FRENCH_CMAP, force=True)
    names.append("french")
    matplotlib.colormaps.register(_PERF_CLINICAL_CMAP, force=True)
    names.append("perf_clinical")

    # 2) Cargar archivos .col de Odyssey.
    col_dirs = _find_col_dirs()
    seen_col_names: set[str] = set()
    for col_dir in col_dirs:
        for col_file in sorted(col_dir.glob("*.col")):
            stem = col_file.stem.strip()
            if not stem:
                continue
            # Prefijo para distinguir FX vs LX si hay nombre duplicado.
            tag = col_dir.name.replace(" ", "_").replace("TABLAS_", "")
            reg_name = f"odyssey_{tag}_{stem}" if stem in seen_col_names else f"odyssey_{stem}"
            seen_col_names.add(stem)
            palette = _read_col_binary(col_file)
            if not palette:
                continue
            cmap = _palette_to_cmap(reg_name, palette)
            try:
                matplotlib.colormaps.register(cmap, force=True)
            except Exception:
                continue
            names.append(reg_name)

    # 3) Agregar colormaps estándar (pueden ya existir, solo listamos).
    for n in _BUILTIN_NAMES:
        if n not in names:
            names.append(n)

    # 4) Agregar colormaps cíclicos.
    for n in _CYCLIC_NAMES:
        if n not in names:
            names.append(n)

    # Ordenar: french, odyssey_*, luego estándar, luego cíclicos.
    def _sort_key(name: str) -> tuple[int, str]:
        if name == "french":
            return (0, name)
        if name == "perf_clinical":
            return (0, name)
        if name.startswith("odyssey_"):
            return (1, name)
        if name in _CYCLIC_NAMES:
            return (3, name)
        return (2, name)

    names = sorted(set(names), key=_sort_key)
    _all_names = names
    _registered = True
    return names


def available_colormaps() -> list[str]:
    """Devuelve la lista de todos los colormaps registrados."""
    if not _registered:
        register_all_colormaps()
    return list(_all_names)


def get_cmap(name: str):
    """Obtiene un colormap por nombre (registrado o estándar de matplotlib)."""
    if not _registered:
        register_all_colormaps()
    return matplotlib.colormaps.get_cmap(name)

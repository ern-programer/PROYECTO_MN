"""SINCRO - core.axis_normals — BD local de normales del eje largo del VI.

Cada vez que el usuario ACEPTA una reorientación (eje manual validado), el
caso puede registrarse acá. Con los casos acumulados se recalibra el prior
anatómico ``LV_AXIS_PRIOR`` usado por la auto-orientación: a más casuística
propia (población + equipo + reconstrucción del sitio), mejor el prior.

Almacén: JSON en mod_SINCRO/research_data/axis_normals.json.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research_data")
_DB_PATH = os.path.join(_DB_DIR, "axis_normals.json")
# Eje de respaldo del diálogo cuando no hay eje válido: nunca es una corrección real.
_DEFAULT_AXIS = np.array([1.0, 0.0, 0.0])
OUTLIER_DEG = 25.0


def db_path() -> str:
    return _DB_PATH


def _load() -> dict:
    try:
        with open(_DB_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data.setdefault("cases", [])
            data.setdefault("prior", None)
            return data
    except Exception:
        pass
    return {"cases": [], "prior": None}


def _save(data: dict) -> None:
    os.makedirs(_DB_DIR, exist_ok=True)
    with open(_DB_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)


def record_case(manual_axis, auto_axis=None, center=None, stage: str = "", study_key: str = "") -> bool:
    """Registra un eje validado; reemplaza el caso previo del mismo estudio y etapa."""
    u = np.asarray(manual_axis, dtype=np.float64)
    n = float(np.linalg.norm(u))
    if n <= 0 or u.shape != (3,):
        return False
    u = u / n
    if abs(float(np.dot(u, _DEFAULT_AXIS))) > 0.9999:
        return False
    case = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "axis": np.round(u, 4).tolist(),
        "stage": str(stage or ""),
        "study_key": str(study_key or ""),
    }
    if auto_axis is not None:
        ua = np.asarray(auto_axis, dtype=np.float64)
        na = float(np.linalg.norm(ua))
        if na > 0:
            ua = ua / na
            case["auto_axis"] = np.round(ua, 4).tolist()
            d = abs(float(np.dot(u, ua)))
            case["auto_error_deg"] = round(float(np.degrees(np.arccos(min(1.0, d)))), 1)
    if center is not None:
        case["center"] = [round(float(v), 1) for v in center]
    data = _load()
    if case["study_key"]:
        data["cases"] = [
            c for c in data["cases"]
            if (c.get("study_key"), c.get("stage")) != (case["study_key"], case["stage"])
        ]
    data["cases"].append(case)
    _save(data)
    return True


def count_cases() -> int:
    return len(_load()["cases"])


def calibrated_prior():
    """Prior recalibrado localmente (lista [z,y,x]) o None si no se calibró."""
    p = _load().get("prior")
    if isinstance(p, (list, tuple)) and len(p) == 3:
        return list(p)
    return None


def get_prior(default) -> np.ndarray:
    """Prior a usar: el calibrado local si existe, si no ``default``."""
    p = calibrated_prior()
    if p is not None:
        u = np.asarray(p, dtype=np.float64)
        n = float(np.linalg.norm(u))
        if n > 0:
            return u / n
    return np.asarray(default, dtype=np.float64)


def robust_mean_axis(axes, outlier_deg: float = OUTLIER_DEG, iterations: int = 5):
    """Media direccional descartando ejes a más de ``outlier_deg`` de la media vigente."""
    axes = np.asarray(axes, dtype=np.float64)
    keep = np.ones(len(axes), dtype=bool)
    mean = axes.mean(axis=0)
    for _ in range(iterations):
        mean = axes[keep].mean(axis=0)
        mean /= float(np.linalg.norm(mean)) or 1.0
        new_keep = np.degrees(np.arccos(np.clip(np.abs(axes @ mean), 0.0, 1.0))) <= outlier_deg
        if not new_keep.any() or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return mean, keep


def recalibrate(min_cases: int = 3):
    """Recalcula el prior como media robusta de los casos acumulados.

    Devuelve (prior_lista, n_casos_usados) o (None, n_casos) si no hay suficientes.
    """
    data = _load()
    axes = []
    for c in data["cases"]:
        a = c.get("axis")
        if isinstance(a, (list, tuple)) and len(a) == 3:
            v = np.asarray(a, dtype=np.float64)
            n = float(np.linalg.norm(v))
            if n > 0 and abs(float(np.dot(v / n, _DEFAULT_AXIS))) <= 0.9999:
                axes.append(v / n)
    if len(axes) < int(min_cases):
        return None, len(axes)
    m, keep = robust_mean_axis(axes)
    if int(keep.sum()) < int(min_cases):
        return None, int(keep.sum())
    prior = np.round(m, 4).tolist()
    data["prior"] = prior
    _save(data)
    return prior, int(keep.sum())


def reset_prior() -> None:
    """Vuelve al prior de fábrica (borra la calibración local, no los casos)."""
    data = _load()
    data["prior"] = None
    _save(data)

# -*- coding: utf-8 -*-
"""Análisis temporal exploratorio de Tc-99m PYP planar.

MÉTODO EXPERIMENTAL / DE INVESTIGACIÓN.
No diagnostica ATTR ni AL, no reemplaza SPECT/CT, laboratorio de cadenas
livianas, inmunofijación, biopsia ni evaluación clínica. No debe guiar por sí
solo tafamidis, quimioterapia u otra decisión terapéutica.

El módulo prioriza medidas observables y reproducibles:
- curvas tiempo-actividad (TAC) tempranas del corazón y mediastino;
- cuentas por segundo (cps), corregidas por duración de frame;
- corrección opcional por decaimiento físico de Tc-99m;
- retención entre 1 h y 3 h;
- localización cardíaca temprana para ayudar a posicionar ROIs tardíos.

Una descomposición lineal en pool / componente creciente lento / componente
retenido está disponible solo como hipótesis condicionada. Sus amplitudes no
se denominan "amiloide puro" ni se interpretan como subtipo ATTR/AL.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TC99M_HALF_LIFE_MIN = 360.6


@dataclass(frozen=True)
class DynamicSeries:
    """Serie dinámica normalizada a tasas de conteo."""

    frames_cps: np.ndarray
    frame_mid_times_min: np.ndarray
    frame_durations_s: np.ndarray
    decay_corrected: bool
    reference_time_min: float


@dataclass(frozen=True)
class TemporalMetrics:
    """Métricas observables, sin inferencia diagnóstica de subtipo."""

    early_heart_cps: np.ndarray
    early_mediastinum_cps: np.ndarray
    early_hmr: np.ndarray
    time_min: np.ndarray
    heart_cps_1h: float
    heart_cps_3h: float
    mediastinum_cps_1h: float
    mediastinum_cps_3h: float
    hmr_1h: float
    hmr_3h: float
    heart_retention_3h_over_1h: float
    heart_change_pct: float
    hmr_change: float


@dataclass(frozen=True)
class ExploratoryDecomposition:
    """Descomposición condicionada a constantes temporales asumidas."""

    pool_amplitude: float
    slow_rising_amplitude: float
    retained_amplitude: float
    condition_number: float
    residual_rms: float
    stable: bool


EXPERIMENTAL_EXPLANATION = {
    "title": "Análisis temporal PYP — método experimental",
    "summary": (
        "Usa un dinámico temprano y las imágenes de 1 h/3 h para describir "
        "curvas tiempo-actividad y retención. Puede ayudar a ubicar el corazón "
        "cuando la captación tardía es baja y a reconocer contaminación por "
        "pool sanguíneo, pero no separa de forma validada ATTR de AL."
    ),
    "physics": (
        "Se comparan tasas de conteo (cps), no cuentas totales, porque cada "
        "adquisición puede durar distinto. Opcionalmente las cps se corrigen "
        "por el decaimiento físico de Tc-99m a un tiempo de referencia común."
    ),
    "differential": (
        "Una caída rápida de la señal temprana es compatible con lavado de pool; "
        "una señal tardía retenida puede corresponder a fijación tisular, hueso "
        "superpuesto o ambos. Estas observaciones pueden motivar SPECT/CT o "
        "laboratorio, pero no descartan otras patologías ni subtipifican amiloidosis."
    ),
    "limitations": (
        "Requiere misma geometría o registro espacial, calibración temporal y "
        "corrección por duración. Tres tiempos no identifican de manera única "
        "todos los compartimentos; cualquier descomposición depende de supuestos."
    ),
    "clinical_warning": (
        "No usar como criterio único para diagnóstico, pronóstico o tratamiento. "
        "ATTR/AL requiere correlación con inmunofijación sérica/urinaria, cadenas "
        "livianas libres, SPECT/CT y/o biopsia según el contexto clínico."
    ),
}


def normalize_dynamic_frames(
    frames: np.ndarray,
    frame_durations_s: np.ndarray,
    frame_start_times_s: np.ndarray | None = None,
    *,
    decay_correct: bool = True,
    reference_time_min: float = 0.0,
    half_life_min: float = TC99M_HALF_LIFE_MIN,
) -> DynamicSeries:
    """Convierte una serie dinámica a cps y opcionalmente corrige decaimiento."""
    arr = np.asarray(frames, dtype=np.float64)
    durations = np.asarray(frame_durations_s, dtype=np.float64).reshape(-1)
    if arr.ndim != 3:
        raise ValueError(f"Se esperaba dinámico [frames, rows, cols], recibido {arr.shape}")
    if durations.size != arr.shape[0] or np.any(durations <= 0):
        raise ValueError("Duraciones de frame inválidas o incompatibles")

    if frame_start_times_s is None:
        starts = np.concatenate(([0.0], np.cumsum(durations[:-1])))
    else:
        starts = np.asarray(frame_start_times_s, dtype=np.float64).reshape(-1)
        if starts.size != arr.shape[0]:
            raise ValueError("Tiempos de inicio incompatibles")
    mid_min = (starts + durations / 2.0) / 60.0
    cps = arr / durations[:, None, None]

    if decay_correct:
        decay_constant = np.log(2.0) / float(half_life_min)
        factors = np.exp(decay_constant * (mid_min - float(reference_time_min)))
        cps = cps * factors[:, None, None]

    return DynamicSeries(
        frames_cps=cps,
        frame_mid_times_min=mid_min,
        frame_durations_s=durations,
        decay_corrected=decay_correct,
        reference_time_min=float(reference_time_min),
    )


def normalize_static_image(
    image: np.ndarray,
    duration_s: float,
    acquisition_time_min: float,
    *,
    decay_correct: bool = True,
    reference_time_min: float = 0.0,
    half_life_min: float = TC99M_HALF_LIFE_MIN,
) -> np.ndarray:
    """Normaliza imagen estática a cps y corrige decaimiento al tiempo común."""
    if duration_s <= 0:
        raise ValueError("La duración de la adquisición estática debe ser > 0")
    cps = np.asarray(image, dtype=np.float64) / float(duration_s)
    if decay_correct:
        lam = np.log(2.0) / float(half_life_min)
        cps = cps * np.exp(lam * (float(acquisition_time_min) - float(reference_time_min)))
    return cps


def roi_tac(series: DynamicSeries, roi) -> np.ndarray:
    """Extrae curva tiempo-actividad promedio (cps/píxel) dentro de un ROI."""
    mask = roi.mask(series.frames_cps.shape[1:])
    if not np.any(mask):
        raise ValueError("ROI vacío")
    return series.frames_cps[:, mask].mean(axis=1)


def temporal_metrics(
    dynamic: DynamicSeries,
    image_1h_cps: np.ndarray,
    image_3h_cps: np.ndarray,
    roi_heart,
    roi_mediastinum,
) -> TemporalMetrics:
    """Calcula TAC temprana, HMR temporal y retención 3h/1h."""
    shape = dynamic.frames_cps.shape[1:]
    if image_1h_cps.shape != shape or image_3h_cps.shape != shape:
        raise ValueError("Las imágenes deben compartir matriz; registrar antes del análisis")
    hm = roi_heart.mask(shape)
    mm = roi_mediastinum.mask(shape)
    heart_tac = roi_tac(dynamic, roi_heart)
    medi_tac = roi_tac(dynamic, roi_mediastinum)
    early_hmr = heart_tac / np.maximum(medi_tac, 1e-12)

    h1 = float(np.asarray(image_1h_cps)[hm].mean())
    h3 = float(np.asarray(image_3h_cps)[hm].mean())
    m1 = float(np.asarray(image_1h_cps)[mm].mean())
    m3 = float(np.asarray(image_3h_cps)[mm].mean())
    hmr1 = h1 / max(m1, 1e-12)
    hmr3 = h3 / max(m3, 1e-12)
    retention = h3 / max(h1, 1e-12)

    return TemporalMetrics(
        early_heart_cps=heart_tac,
        early_mediastinum_cps=medi_tac,
        early_hmr=early_hmr,
        time_min=dynamic.frame_mid_times_min,
        heart_cps_1h=h1,
        heart_cps_3h=h3,
        mediastinum_cps_1h=m1,
        mediastinum_cps_3h=m3,
        hmr_1h=hmr1,
        hmr_3h=hmr3,
        heart_retention_3h_over_1h=retention,
        heart_change_pct=(retention - 1.0) * 100.0,
        hmr_change=hmr3 - hmr1,
    )


def exploratory_three_component_fit(
    values: np.ndarray,
    times_min: np.ndarray,
    *,
    tau_pool_min: float = 3.0,
    tau_slow_min: float = 90.0,
    tau_retained_min: float = 10.0,
    max_condition_number: float = 1e4,
) -> ExploratoryDecomposition:
    """Ajusta 3 componentes asumidos y reporta estabilidad numérica.

    Los nombres describen formas cinéticas, no identidades histológicas.
    """
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    t = np.asarray(times_min, dtype=np.float64).reshape(-1)
    if y.size != t.size or y.size < 3:
        raise ValueError("Se requieren al menos tres observaciones temporales")
    design = np.column_stack((
        np.exp(-t / tau_pool_min),
        1.0 - np.exp(-t / tau_slow_min),
        1.0 - np.exp(-t / tau_retained_min),
    ))
    condition = float(np.linalg.cond(design))
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    coeffs = np.clip(coeffs, 0.0, None)
    residual = float(np.sqrt(np.mean((y - design @ coeffs) ** 2)))
    return ExploratoryDecomposition(
        pool_amplitude=float(coeffs[0]),
        slow_rising_amplitude=float(coeffs[1]),
        retained_amplitude=float(coeffs[2]),
        condition_number=condition,
        residual_rms=residual,
        stable=bool(np.isfinite(condition) and condition <= max_condition_number),
    )

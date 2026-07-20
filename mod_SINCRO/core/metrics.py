"""
SINCRO - core.metrics
======================

Métricas de asincronía a partir de la distribución de fase (grados 0-360°).

Métricas (Emory / literatura):
- Phase SD (desviación estándar CIRCULAR)
- Bandwidth (P95 - P5 sobre distribución centrada)
- Skewness, Kurtosis (forma de la distribución)
- Entropy Shannon (bits) e Entropy normalizada (%)
- Asynchrony Index (% voxels a >2σ de la media)
- Peak phase + peak width (moda del histograma y su ancho)
- Latest activation site (fase máxima → clave para CRT lead positioning)
- Clasificación técnica: NORMAL / MILD / MODERATE / SEVERE (por Phase SD)

NOTA sobre circularidad: la fase es angular (0°=360°). La media y el SD se calculan
de forma CIRCULAR para no romperse en el wrap. Para bandwidth/skew/kurt/entropy se
"descentra" la distribución alrededor de su media circular y se opera en lineal.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# Umbrales técnicos históricos por Phase SD (grados). La interpretación clínica
# final debe compararse contra una DB normal/software específica (core.normal_db).
CLASS_THRESHOLDS = {"NORMAL": 20.0, "MILD": 40.0, "MODERATE": 60.0}  # >60 = SEVERE


def circular_mean_deg(phases_deg: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Media circular en grados (0-360)."""
    if phases_deg.size == 0:
        return 0.0
    rad = np.radians(phases_deg)
    if weights is None:
        c, s = np.mean(np.cos(rad)), np.mean(np.sin(rad))
    else:
        w = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        c, s = np.sum(w * np.cos(rad)), np.sum(w * np.sin(rad))
    return float(np.degrees(np.arctan2(s, c)) % 360.0)


def circular_std_deg(phases_deg: np.ndarray) -> float:
    """
    Desviación estándar circular en grados (Mardia).
    SD = sqrt(-2 ln R) donde R = |mean(e^{iθ})|. Se devuelve en grados.
    """
    if phases_deg.size < 2:
        return 0.0
    rad = np.radians(phases_deg)
    R = np.hypot(np.mean(np.cos(rad)), np.mean(np.sin(rad)))
    R = min(max(R, 1e-12), 1.0)
    return float(abs(np.degrees(np.sqrt(-2.0 * np.log(R)))))


def _center_around_mean(phases_deg: np.ndarray, mean_deg: float) -> np.ndarray:
    """Descentra la distribución angular a (-180, 180] alrededor de su media circular."""
    d = (phases_deg - mean_deg + 180.0) % 360.0 - 180.0
    return d


def calculate_phase_metrics(phases_deg: np.ndarray, hist_bins: int = 360) -> dict:
    """
    Calcula todas las métricas de asincronía sobre las fases (grados 0-360).

    Parameters
    ----------
    phases_deg : ndarray (n_voxels,)
        Fases de los voxels miocárdicos.
    hist_bins : int
        Nº de bins del histograma para Entropy / peak (360 = 1°/bin, estándar Emory).

    Returns
    -------
    dict con las métricas + clasificación.
    """
    phases_deg = np.asarray(phases_deg, dtype=np.float64)
    n = phases_deg.size
    if n == 0:
        raise ValueError("phases_deg está vacío.")

    mean_deg = circular_mean_deg(phases_deg)
    phase_sd = circular_std_deg(phases_deg)

    # Distribución centrada (lineal) para BW/skew/kurt
    centered = _center_around_mean(phases_deg, mean_deg)
    bandwidth = float(np.percentile(centered, 95) - np.percentile(centered, 5))
    skewness = float(stats.skew(centered)) if n > 2 else 0.0
    kurtosis = float(stats.kurtosis(centered)) if n > 3 else 0.0

    # Entropy Shannon sobre histograma 0-360. La literatura de paquetes clínicos
    # suele reportar entropy normalizada en porcentaje, por eso se exponen ambas.
    hist, edges = np.histogram(phases_deg, bins=hist_bins, range=(0.0, 360.0))
    p = hist / hist.sum() if hist.sum() > 0 else hist
    nz = p[p > 0]
    entropy_shannon = float(-np.sum(nz * np.log2(nz))) if nz.size else 0.0
    entropy_percent = float(100.0 * entropy_shannon / np.log2(hist_bins)) if hist_bins > 1 else 0.0

    # Peak phase (moda) + peak width (FWHM aproximado en nº de bins con >50% del pico)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    peak_idx = int(np.argmax(hist))
    peak_phase = float(bin_centers[peak_idx])
    half = hist.max() / 2.0 if hist.max() > 0 else 0.0
    peak_width = float((hist >= half).sum() * (360.0 / hist_bins))

    # Asynchrony index: % de voxels a más de 2 SD de la media (usando distribución centrada)
    ai = float(np.mean(np.abs(centered) > 2.0 * phase_sd) * 100.0) if phase_sd > 0 else 0.0

    # Latest activation site: fase (0-360) del voxel más tardío respecto de la media
    latest_offset = float(centered.max())               # cuánto más tarde que la media
    latest_phase = float((mean_deg + latest_offset) % 360.0)

    metrics = {
        "n_voxels": int(n),
        "mean_phase": round(mean_deg, 2),
        "phase_sd": round(phase_sd, 2),
        "bandwidth": round(bandwidth, 2),
        "skewness": round(skewness, 3),
        "kurtosis": round(kurtosis, 3),
        "entropy": round(entropy_shannon, 3),
        "entropy_shannon_bits": round(entropy_shannon, 3),
        "entropy_normalized_pct": round(entropy_percent, 2),
        "asynchrony_index": round(ai, 2),
        "peak_phase": round(peak_phase, 2),
        "peak_width": round(peak_width, 2),
        "latest_activation_phase": round(latest_phase, 2),
        "classification": classify_dyssynchrony(phase_sd),
        "technical_classification": classify_dyssynchrony(phase_sd),
    }
    return metrics


def classify_dyssynchrony(phase_sd: float) -> str:
    """Clasifica el grado de disincronía por Phase SD (grados)."""
    if phase_sd < CLASS_THRESHOLDS["NORMAL"]:
        return "NORMAL"
    if phase_sd < CLASS_THRESHOLDS["MILD"]:
        return "MILD"
    if phase_sd < CLASS_THRESHOLDS["MODERATE"]:
        return "MODERATE"
    return "SEVERE"

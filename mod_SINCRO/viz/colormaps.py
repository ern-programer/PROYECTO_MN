"""SINCRO - viz.colormaps — colormaps cíclicos para fase (0-360°)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def get_phase_cmap(name: str = "hsv"):
	"""
	Devuelve un colormap matplotlib CÍCLICO para mapear fase 0-360°.
	Opciones válidas: 'hsv', 'twilight', 'twilight_shifted'.

	Nota: en fase cardíaca se suele interpretar inicio (temprano) en rojo y colores
	posteriores como contracción más tardía según la referencia de fase utilizada.
	"""
	valid = {"hsv", "twilight", "twilight_shifted"}
	if name not in valid:
		raise ValueError(f"cmap inválido '{name}'. Válidos: {sorted(valid)}")
	return plt.get_cmap(name)


def phase_to_rgb(phase_deg, cmap_name: str = "hsv", nan_color=(0.1, 0.1, 0.1)):
	"""
	Mapea un array de fase (0-360°, puede tener NaN) a RGB (...,3) float 0-1.
	NaN se reemplaza por nan_color.
	"""
	phase = np.asarray(phase_deg, dtype=np.float64)
	norm = (phase % 360.0) / 360.0

	cmap = get_phase_cmap(cmap_name)
	rgba = cmap(norm)
	rgb = np.asarray(rgba[..., :3], dtype=np.float64)

	nan_mask = np.isnan(phase)
	if np.any(nan_mask):
		rgb = rgb.copy()
		rgb[nan_mask] = np.asarray(nan_color, dtype=np.float64)

	return rgb

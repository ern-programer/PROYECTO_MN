"""Genera logos PNG para GammaSync.

Salida:
- logo_gammasync_512.png
- logo_gammasync_256.png
- logo_gammasync_128.png
- logo_gammasync_64.png
- logo_gammasync_banner.png
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = BASE_DIR


def _ecg_wave(n: int = 1200):
	"""Señal tipo ECG con QRS y ST simplificados."""
	t = np.linspace(0.0, 1.0, n)
	y = np.zeros_like(t)
	# Línea de base
	y += 0.03 * np.sin(2 * np.pi * 1.2 * t)
	# Onda P pequeña
	y += 0.06 * np.exp(-((t - 0.14) / 0.025) ** 2)
	# QRS azul: caída Q, pico R, S
	y += -0.09 * np.exp(-((t - 0.30) / 0.008) ** 2)
	y += 0.62 * np.exp(-((t - 0.325) / 0.006) ** 2)
	y += -0.20 * np.exp(-((t - 0.348) / 0.010) ** 2)
	# Segmento ST ligeramente elevado
	y += 0.08 * np.exp(-((t - 0.48) / 0.07) ** 2)
	# T wave
	y += 0.18 * np.exp(-((t - 0.70) / 0.05) ** 2)
	return t, y


def _draw_logo(path: str, size: int = 256, banner: bool = False):
	fig_w = 8.0 if banner else 4.0
	fig_h = 2.1 if banner else 4.0
	fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=size / (fig_w if banner else 4.0))
	fig.patch.set_facecolor("white")
	ax.set_facecolor("white")
	ax.axis("off")

	t, y = _ecg_wave()
	# Ajuste de encuadre: pequeño logo limpio, línea azul sobre blanco.
	if banner:
		x = 0.10 + 0.40 * t
		y_plot = 0.52 + 0.26 * y
		ax.text(0.58, 0.58, "GammaSync", transform=ax.transAxes, ha="left", va="center",
			fontsize=28, fontweight="bold", color="#123b8b")
		ax.text(0.58, 0.38, "Cardiac synchrony analysis", transform=ax.transAxes, ha="left", va="center",
			fontsize=10, color="#6b7a90")
	else:
		x = 0.12 + 0.76 * t
		y_plot = 0.42 + 0.20 * y

	# Tramo de base y trazado ECG.
	ax.plot(x, y_plot, color="#2f6bff", linewidth=4.2, solid_capstyle="round")
	ax.plot([x[0], x[-1]], [0.42, 0.42], color="#d7e3ff", linewidth=6, solid_capstyle="round", zorder=0)

	# Segmentos QRS/ST resaltados.
	qrs_mask = (t >= 0.285) & (t <= 0.365)
	st_mask = (t >= 0.365) & (t <= 0.55)
	ax.plot(0.12 + (0.76 if not banner else 0.40) * t[qrs_mask], 0.42 + 0.20 * y[qrs_mask], color="#0d47a1", linewidth=5.2)
	ax.plot(0.12 + (0.76 if not banner else 0.40) * t[st_mask], 0.42 + 0.20 * y[st_mask], color="#64b5f6", linewidth=4.2)

	# Punto de latido principal.
	peak_idx = int(np.argmax(y))
	ax.scatter([x[peak_idx]], [y_plot[peak_idx]], s=220 if banner else 180, color="#1e88e5", edgecolor="white", linewidth=2, zorder=5)

	# Efecto visual leve de electrocardiograma.
	ax.text(0.12, 0.15 if not banner else 0.22, "QRS", transform=ax.transAxes, fontsize=10 if not banner else 11,
		fontweight="bold", color="#0d47a1")
	ax.text(0.24, 0.15 if not banner else 0.22, "ST", transform=ax.transAxes, fontsize=10 if not banner else 11,
		fontweight="bold", color="#64b5f6")

	ax.set_xlim(0, 1)
	ax.set_ylim(0.18, 0.92)
	fig.tight_layout(pad=0)
	fig.savefig(path, dpi=size, facecolor="white", bbox_inches="tight", pad_inches=0.02)
	plt.close(fig)


if __name__ == "__main__":
	outputs = [
		("logo_gammasync_512.png", 512, False),
		("logo_gammasync_256.png", 256, False),
		("logo_gammasync_128.png", 128, False),
		("logo_gammasync_64.png", 64, False),
		("logo_gammasync_banner.png", 512, True),
	]
	for filename, size, banner in outputs:
		_draw_logo(os.path.join(OUT_DIR, filename), size=size, banner=banner)
		print(f"[OK] {filename}")

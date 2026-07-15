"""Demo visual: genera todas las salidas de SINCRO con datos reales."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.console_utf8 import enable_utf8
enable_utf8()

import numpy as np
from core import dicom_loader
from core.segmentation import segment_myocardium
from core.phase_analysis import phase_analysis
from core.aha_segments import map_to_17_segments, phase_by_segment
from core.metrics import calculate_phase_metrics
from viz.colormaps import phase_to_rgb, get_phase_cmap
from viz.polar_map import build_polar_map, save_polar_map
from viz.histogram import build_phase_histogram, save_histogram

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Ruta al DICOM SA gated ---
SA_GATED_PATH = (
    r"C:\Users\Ernesto\Desktop\INTERCAMBIO\varios stress cardiacos de xeleris"
    r"\estudio uno\MYOMETRIX\myometrix results\REST_IRNCG_SA001_DS.dcm"
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_demo")
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(SA_GATED_PATH):
    print(f"[ERROR] No existe el DICOM: {SA_GATED_PATH}")
    print("Copiá el archivo ahí o ajustá la ruta.")
    sys.exit(1)

# ============================================================
# 1. Cargar estudio
# ============================================================
print("=" * 60)
print("1. Cargando DICOM SA gated...")
study = dicom_loader.load(SA_GATED_PATH, verbose=True)
cube = study.cube
print(f"   Cube shape: {cube.shape}  (gates, slices, H, W)")
print(f"   Rango valores: [{cube.min():.1f}, {cube.max():.1f}]")

# ============================================================
# 2. Segmentación
# ============================================================
print("\n2. Segmentando miocardio...")
seg = segment_myocardium(cube, method="auto")
mask = seg.mask
n_voxels = int(mask.sum())
mid_sl = cube.shape[1] // 2
cy, cx = seg.center_per_slice[mid_sl]
print(f"   Máscara: {n_voxels} voxels miocárdicos")
print(f"   Centro slice {mid_sl}: (y={cy:.1f}, x={cx:.1f})")
print(f"   Radio inner (slice {mid_sl}): {seg.inner_radius[mid_sl]:.1f}")
print(f"   Radio outer (slice {mid_sl}): {seg.outer_radius[mid_sl]:.1f}")

# ============================================================
# 3. Análisis de fase
# ============================================================
print("\n3. Analizando fase (FFT 1er armónico)...")
res = phase_analysis(cube, mask, harmonics=1, amplitude_threshold_frac=0.10)
phases = res.phases_deg
print(f"   Voxels con fase válida: {phases.size}")
from core.metrics import circular_mean_deg
mean_deg = circular_mean_deg(phases)
print(f"   Fase media: {mean_deg:.1f}°")

metrics = calculate_phase_metrics(phases)
print(f"   Phase SD: {metrics['phase_sd']:.1f}°")
print(f"   Bandwidth: {metrics['bandwidth']:.1f}°")
print(f"   Entropy: {metrics['entropy']:.3f}")
print(f"   Clasificación: {metrics['classification']}")

# ============================================================
# 4. Segmentos AHA
# ============================================================
print("\n4. Mapeando a 17 segmentos AHA...")
aha = map_to_17_segments(seg)
pbs = phase_by_segment(res.phase_map, aha)

# Tabla de segmentos
print(f"   {'Seg':>3}  {'Fase (°)':>10}")
print("   " + "-" * 16)
for sid in sorted(pbs):
    print(f"   {sid:>3}  {pbs[sid]:>10.1f}")

# Territorios
from core.aha_segments import territory_analysis
terr = territory_analysis(pbs)
print(f"\n   Territorios:")
for name, d in terr.items():
    print(f"   {name:>3}: mean={d['mean']:.1f}°  SD={d['std']:.1f}°  n={d['n']}")

# ============================================================
# 5. VISUALIZACIONES
# ============================================================
print("\n5. Generando visualizaciones...")

# --- 5a. Slice medio con máscara overlay ---
mid_slice = cube.shape[1] // 2
mid_gate = cube.shape[0] // 2
frame = cube[mid_gate, mid_slice]
frame_norm = frame / (frame.max() + 1e-8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

# Original
axes[0].imshow(frame_norm, cmap="gray")
axes[0].set_title(f"Slice {mid_slice}, Gate {mid_gate}")

# Máscara overlay
axes[1].imshow(frame_norm, cmap="gray")
mask_slice = mask[mid_slice].astype(float)
mask_overlay = np.zeros((*mask_slice.shape, 4))
mask_overlay[..., 0] = 1.0
mask_overlay[..., 3] = mask_slice * 0.45
axes[1].imshow(mask_overlay)
axes[1].set_title("Máscara miocardio")

# Fase overlay
axes[2].imshow(frame_norm, cmap="gray")
phase_map_slice = res.phase_map[mid_slice].copy()
pm_overlay = np.zeros((*phase_map_slice.shape, 4))
valid = np.isfinite(phase_map_slice)
if valid.any():
    rgb = phase_to_rgb(phase_map_slice[valid])
    pm_overlay[valid, :3] = rgb
    pm_overlay[valid, 3] = 0.7
axes[2].imshow(pm_overlay)
axes[2].set_title("Fase superpuesta (HSV)")

fig.suptitle("SINCRO — Visualización de Fase Cardíaca", fontsize=14, fontweight="bold")
fig.tight_layout()
path_slices = os.path.join(OUT_DIR, "slices_fase.png")
fig.savefig(path_slices, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   [OK] {path_slices}")

# --- 5b. Polar Map (Bullseye AHA 17) ---
pmfig = build_polar_map(pbs, cmap_name="hsv", title="Phase Polar Map — REST_IRNCG")
path_polar = os.path.join(OUT_DIR, "polar_map.png")
save_polar_map(pmfig, path_polar, dpi=150)
plt.close(pmfig.fig)
print(f"   [OK] {path_polar}")

# --- 5c. Histograma de fase ---
hfig = build_phase_histogram(phases, metrics=metrics, bins=72,
                              title="Phase Histogram — REST_IRNCG")
path_hist = os.path.join(OUT_DIR, "histograma.png")
save_histogram(hfig, path_hist, dpi=150)
plt.close(hfig)
print(f"   [OK] {path_hist}")

# --- 5d. Animación de gates (montaje) ---
print("\n6. Generando montaje de gates...")
n_gates = cube.shape[0]
n_slices_show = min(5, cube.shape[1])
gate_indices = list(range(0, n_gates, max(1, n_gates // 4)))[:4]
if gate_indices[-1] != n_gates - 1:
    gate_indices.append(n_gates - 1)

fig, axes = plt.subplots(len(gate_indices), n_slices_show, figsize=(3 * n_slices_show, 3 * len(gate_indices)))
if len(gate_indices) == 1:
    axes = axes[np.newaxis, :]

for row, gi in enumerate(gate_indices):
    for col in range(n_slices_show):
        si = col * (cube.shape[1] - 1) // max(1, n_slices_show - 1)
        frame = cube[gi, si]
        fmax = frame.max()
        if fmax > 0:
            frame = frame / fmax
        axes[row, col].imshow(frame, cmap="hot")
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        if row == 0:
            axes[row, col].set_title(f"Slice {si}", fontsize=9)
    axes[row, 0].set_ylabel(f"Gate {gi}", fontsize=9, fontweight="bold")

fig.suptitle("Gates × Slices — Dinámica cardíaca", fontsize=13, fontweight="bold")
fig.tight_layout()
path_gates = os.path.join(OUT_DIR, "montaje_gates.png")
fig.savefig(path_gates, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   [OK] {path_gates}")

# --- 5e. Curva TAC (Time-Activity Curve) global ---
print("\n7. Generando curva TAC global...")
fig = plt.figure(figsize=(14, 5))
ax1 = fig.add_subplot(121)
gate_times = np.arange(n_gates)

# Curva global miocardio
mean_per_gate = []
for gi in range(n_gates):
    vals = cube[gi][mask]
    mean_per_gate.append(float(vals.mean()))
mean_per_gate = np.array(mean_per_gate)
mean_per_gate_norm = mean_per_gate / (mean_per_gate.max() + 1e-8)

ax1.plot(gate_times, mean_per_gate_norm, "o-", color="#2c7fb8", linewidth=2, markersize=5)
ax1.fill_between(gate_times, mean_per_gate_norm, alpha=0.2, color="#2c7fb8")
ax1.set_xlabel("Gate (frame)")
ax1.set_ylabel("Intensidad normalizada")
ax1.set_title("Curva TAC — Promedio miocardio global")
ax1.set_xticks(gate_times)
ax1.grid(True, alpha=0.3)

# Radar chart: fase por segmento AHA
seg_ids = sorted(pbs.keys())
seg_phases = [pbs[s] for s in seg_ids]
# Cerrar el polígono
angles = np.linspace(0, 2 * np.pi, len(seg_ids), endpoint=False).tolist()
seg_phases_plot = seg_phases + [seg_phases[0]]
angles += [angles[0]]

ax2 = fig.add_subplot(122, polar=True)
ax2.plot(angles, seg_phases_plot, "o-", color="#d7191c", linewidth=2, markersize=4)
ax2.fill(angles, seg_phases_plot, alpha=0.15, color="#d7191c")
ax2.set_thetagrids(np.degrees(angles[:-1]), [str(s) for s in seg_ids], fontsize=8)
ax2.set_title("Fase por segmento AHA (radar)", pad=20, fontsize=10)
ax2.set_ylim(0, 360)
ax2.set_yticks([0, 90, 180, 270, 360])

fig.tight_layout()
path_tac = os.path.join(OUT_DIR, "curva_tac.png")
fig.savefig(path_tac, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   [OK] {path_tac}")

# ============================================================
# Resumen
# ============================================================
print("\n" + "=" * 60)
print("RESUMEN DE MÉTRICAS")
print("=" * 60)
print(f"  Estudio: REST_IRNCG_SA001_DS.dcm")
print(f"  Shape: {cube.shape} = (gates={cube.shape[0]}, slices={cube.shape[1]}, {cube.shape[2]}×{cube.shape[3]})")
print(f"  Voxels miocardio: {n_voxels}")
print(f"  Fase válida: {phases.size} voxels")
print(f"  Phase SD: {metrics['phase_sd']:.1f}°")
print(f"  Bandwidth: {metrics['bandwidth']:.1f}°")
print(f"  Entropy: {metrics['entropy']:.3f}")
print(f"  Peak Phase: {metrics['peak_phase']:.1f}°")
print(f"  Clasificación: {metrics['classification']}")
print(f"\n  Archivos generados en: {OUT_DIR}")
for f in sorted(os.listdir(OUT_DIR)):
    sz = os.path.getsize(os.path.join(OUT_DIR, f))
    print(f"    {f}  ({sz/1024:.0f} KB)")
print("=" * 60)
print("¡Listo! Abrí las imágenes desde la carpeta output_demo.")

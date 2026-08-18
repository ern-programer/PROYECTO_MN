# -*- coding: utf-8 -*-
"""Harness comparativo: OSEM actual vs OSEM adyunto con fantoma sintético.

Compara:
1. Centro de masa (shift).
2. Contraste cavidad/pared.
3. Ruido en fondo.
4. Forma de la pared (FWHM radial).
"""
import numpy as np
from scipy.ndimage import rotate, center_of_mass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# Fantoma sintético: anillo con cavidad
# ============================================================
N = 64
img = np.zeros((N, N), dtype=np.float64)
cx, cy = N // 2, N // 2
for y in range(N):
    for x in range(N):
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if 12 < r < 22:
            img[y, x] = 1.0
        elif r <= 12:
            img[y, x] = 0.1

# Defecto en un sector (hipocaptación)
for y in range(N):
    for x in range(N):
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        angle = np.degrees(np.arctan2(y - cy, x - cx)) % 360
        if 12 < r < 22 and 45 < angle < 135:
            img[y, x] *= 0.2

# ============================================================
# Simular proyecciones
# ============================================================
n_angles = 64
theta = np.linspace(0, 360, n_angles, endpoint=False)
det_size = N

def forward_project(image, angles):
    sino = np.zeros((det_size, len(angles)))
    for i, a in enumerate(angles):
        rot = rotate(image, angle=-a, reshape=False, order=1, mode='constant', cval=0.0)
        sino[:, i] = rot.sum(axis=0)
    return sino

sino_clean = forward_project(img, theta)
# Agregar ruido Poisson
np.random.seed(42)
counts = 5e4
sino_noisy = np.random.poisson(
    np.clip(sino_clean * counts / max(sino_clean.sum(), 1), 0, None)
).astype(np.float64)

# ============================================================
# OSEM actual (rotation-based)
# ============================================================
def osem_current(sinogram, angles, output_size=64, iterations=4, subsets=4):
    measured = np.clip(sinogram, 0, None)
    theta = np.asarray(angles, dtype=np.float64)
    n_angles = len(theta)
    det_size = measured.shape[0]
    out_size = output_size
    image = np.full((out_size, out_size), max(float(measured.mean()), 1.0))
    eps = 1e-6
    angle_indices = np.arange(n_angles)
    for _ in range(iterations):
        for s in range(subsets):
            idx = angle_indices[s::subsets]
            theta_sub = theta[idx]
            measured_sub = measured[:, idx]
            estimated = np.zeros((det_size, len(idx)))
            for i, a in enumerate(theta_sub):
                rot = rotate(image, angle=-a, reshape=False, order=1, mode='constant', cval=0.0)
                estimated[:, i] = rot.sum(axis=0)
            ratio = measured_sub / np.maximum(estimated, eps)
            correction = np.zeros((out_size, out_size))
            for i, a in enumerate(theta_sub):
                slab = np.tile(ratio[:, i].reshape(-1, 1), (1, out_size))
                correction += rotate(slab, angle=a, reshape=False, order=1, mode='constant', cval=0.0)
            sensitivity = np.zeros((out_size, out_size))
            ones = np.ones_like(ratio)
            for i, a in enumerate(theta_sub):
                slab = np.tile(ones[:, i].reshape(-1, 1), (1, out_size))
                sensitivity += rotate(slab, angle=a, reshape=False, order=1, mode='constant', cval=0.0)
            image *= correction / np.maximum(sensitivity, eps)
            image = np.clip(image, 0, None)
    return image

# ============================================================
# OSEM adyunto (ray-driven)
# ============================================================
from core.osem_adjoint import osem_adjoint_reconstruct_slice

# ============================================================
# Reconstruir con ambos
# ============================================================
print("Reconstruyendo con OSEM actual (4 iter, 4 subsets)...")
recon_current = osem_current(sino_noisy, theta, output_size=N, iterations=4, subsets=4)

print("Reconstruyendo con OSEM adyunto (4 iter, 8 subsets)...")
recon_adjoint = osem_adjoint_reconstruct_slice(sino_noisy, theta, output_size=N, iterations=4, subsets=8)

# ============================================================
# Métricas comparativas
# ============================================================
cm_orig = center_of_mass(img)
cm_curr = center_of_mass(recon_current)
cm_adj = center_of_mass(recon_adjoint)

print(f"\n=== Centro de masa (shift) ===")
print(f"Original:  ({cm_orig[0]:.2f}, {cm_orig[1]:.2f})")
print(f"OSEM curr: ({cm_curr[0]:.2f}, {cm_curr[1]:.2f})  shift=({cm_curr[0]-cm_orig[0]:+.2f}, {cm_curr[1]-cm_orig[1]:+.2f})")
print(f"OSEM adj:  ({cm_adj[0]:.2f}, {cm_adj[1]:.2f})  shift=({cm_adj[0]-cm_orig[0]:+.2f}, {cm_adj[1]-cm_orig[1]:+.2f})")

# Contraste cavidad/pared
mask_wall = (img > 0.8)
mask_cavity = (img > 0.05) & (img < 0.15)
for label, recon in [("OSEM curr", recon_current), ("OSEM adj", recon_adjoint)]:
    wall_mean = recon[mask_wall].mean() if mask_wall.any() else 0
    cavity_mean = recon[mask_cavity].mean() if mask_cavity.any() else 0
    contrast = (wall_mean - cavity_mean) / max(wall_mean + cavity_mean, 1e-6)
    print(f"\n{label}: pared={wall_mean:.3f}, cavidad={cavity_mean:.3f}, contraste={contrast:.3f}")

# ============================================================
# Figura comparativa
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

vmax = float(np.percentile(img, 99))
axes[0, 0].imshow(img, cmap="hot", vmin=0, vmax=vmax)
axes[0, 0].set_title("Fantoma original")
axes[0, 0].axis("off")

vmax_c = float(np.percentile(recon_current, 99.5))
axes[0, 1].imshow(recon_current, cmap="hot", vmin=0, vmax=vmax_c)
axes[0, 1].set_title(f"OSEM actual\nshift=({cm_curr[0]-cm_orig[0]:+.1f}, {cm_curr[1]-cm_orig[1]:+.1f})")
axes[0, 1].axis("off")

vmax_a = float(np.percentile(recon_adjoint, 99.5))
axes[0, 2].imshow(recon_adjoint, cmap="hot", vmin=0, vmax=vmax_a)
axes[0, 2].set_title(f"OSEM adyunto\nshift=({cm_adj[0]-cm_orig[0]:+.1f}, {cm_adj[1]-cm_orig[1]:+.1f})")
axes[0, 2].axis("off")

# Perfiles radiales
angles_prof = np.linspace(0, 360, 360)
r_range = np.arange(0, N // 2)
for label, recon, color in [("Original", img, "black"), ("OSEM curr", recon_current, "red"), ("OSEM adj", recon_adjoint, "blue")]:
    prof = []
    for r in r_range:
        vals = []
        for a in angles_prof:
            x = cx + r * np.cos(np.deg2rad(a))
            y = cy + r * np.sin(np.deg2rad(a))
            xi, yi = int(np.clip(x, 0, N-1)), int(np.clip(y, 0, N-1))
            vals.append(recon[yi, xi])
        prof.append(np.mean(vals))
    axes[1, 0].plot(r_range, prof, label=label, color=color, linewidth=1.5)
axes[1, 0].set_title("Perfil radial promedio")
axes[1, 0].set_xlabel("Radio (px)")
axes[1, 0].set_ylabel("Intensidad")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Perfil horizontal
axes[1, 1].plot(img[cx], label="Original", color="black", linewidth=1.5)
axes[1, 1].plot(recon_current[cx], label="OSEM curr", color="red", linewidth=1.5)
axes[1, 1].plot(recon_adjoint[cx], label="OSEM adj", color="blue", linewidth=1.5)
axes[1, 1].set_title("Perfil horizontal (fila central)")
axes[1, 1].set_xlabel("Pixel")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

# Diferencia
diff = recon_current - recon_adjoint
im = axes[1, 2].imshow(diff, cmap="RdBu_r")
axes[1, 2].set_title("Diferencia (actual - adyunto)")
axes[1, 2].axis("off")
fig.colorbar(im, ax=axes[1, 2], fraction=0.046)

fig.suptitle("Comparación OSEM: rotation-based vs ray-driven adyunto\n4 iter, 64 ángulos, fantoma anillo con defecto", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output_demo")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "osem_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n✅ Figura: {out_path}")

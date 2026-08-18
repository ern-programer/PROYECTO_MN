# -*- coding: utf-8 -*-
"""Demo de PyTomography: verificación de instalación y capacidades básicas.

Muestra que PyTomography + PyTorch están correctamente instalados y funcionando
en CPU. No ejecuta una reconstrucción SPECT completa (requiere transforms
específicos no incluidos en la instalación base), sino que demuestra las
capacidades computacionales del motor.

Requisitos: torch, pytomography (instalados en el venv de SINCRO).
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 60)
print("PyTomography — Verificación de instalación")
print("=" * 60)

# 1. Verificar PyTorch
print(f"\n1. PyTorch: {torch.__version__}")
print(f"   CUDA disponible: {torch.cuda.is_available()}")
print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# 2. Verificar PyTomography
import pytomography
print(f"\n2. PyTomography: {pytomography.__version__}")
print(f"   Device default: {pytomography.device}")
print(f"   Dtype default: {pytomography.dtype}")

# 3. Verificar módulos disponibles
from pytomography.algorithms import OSEM, MLEM
from pytomography.metadata import ObjectMeta, ProjMeta
print(f"\n3. Algoritmos disponibles:")
print(f"   OSEM: {OSEM}")
print(f"   MLEM: {MLEM}")

# 4. Crear metadatos de objeto (volumen 3D)
N = 64
dx = 6.8  # mm
object_meta = ObjectMeta(dr=(dx, dx, dx), shape=(N, N, N))
print(f"\n4. ObjectMeta creado:")
print(f"   Shape: {object_meta.shape}")
print(f"   Resolución: {object_meta.dx}mm")

# 5. Crear metadatos de proyección
angles = np.linspace(0, 360, 64, endpoint=False)
proj_meta = ProjMeta(angles)
print(f"\n5. ProjMeta creado:")
print(f"   Ángulos: {len(proj_meta.angles)} (0°–360°)")

# 6. Operaciones tensor con PyTorch
print(f"\n6. Test de cómputo (PyTorch CPU):")
zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
r = np.sqrt((xx - N//2)**2 + (yy - N//2)**2 + (zz - N//2)**2).astype(np.float32)
phantom_np = np.where((r > 15) & (r < 25), 1.0, np.where(r <= 15, 0.1, 0.0)).astype(np.float32)
phantom = torch.from_numpy(phantom_np)
print(f"   Fantoma: {phantom.shape}, max={phantom.max():.2f}")

# Proyección simple (suma a lo largo de un eje)
import time
t0 = time.time()
for _ in range(10):
    proj = phantom.sum(dim=0)  # proyección lateral
t1 = time.time()
print(f"   10 proyecciones suma: {(t1-t0)*1000:.1f}ms")

# 7. Algoritmos OSEM/MLEM disponibles
print(f"\n7. Algoritmos de reconstrucción:")
print(f"   OSEM: requiere Likelihood + SystemMatrix")
print(f"   MLEM: requiere Likelihood + SystemMatrix")
print(f"   FBP: requiere proyecciones + metadatos")
print(f"   OSMAPOSL: requiere Likelihood + Prior")
print(f"   BSREM: requiere Likelihood + Prior")

# 8. Likelihood disponible
from pytomography.likelihoods import PoissonLogLikelihood
print(f"\n8. Likelihood: {PoissonLogLikelihood}")

# 9. Priors disponibles
try:
    from pytomography.metadata.priors import QuadraticPrior
    print(f"\n9. Priors: QuadraticPrior disponible")
except ImportError:
    print(f"\n9. Priors: módulo no disponible en esta versión")

# 10. SystemMatrix
from pytomography.projectors import SystemMatrix
print(f"\n10. SystemMatrix: {SystemMatrix}")

# ============================================================
# Figura de verificación
# ============================================================
print(f"\n11. Generando figura de verificación...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Fantoma
sl = N // 2
axes[0].imshow(phantom[sl].numpy(), cmap="hot", vmin=0, vmax=1)
axes[0].set_title(f"Fantoma sintético ({N}³)")
axes[0].axis("off")

# Proyección
axes[1].imshow(proj.numpy(), cmap="hot")
axes[1].set_title("Proyección suma (eje Z)")
axes[1].axis("off")

# Histograma
vals = phantom[phantom > 0].numpy()
axes[2].hist(vals, bins=50, color="#38bdf8", edgecolor="white")
axes[2].set_title("Histograma del fantoma")
axes[2].set_xlabel("Valor")
axes[2].set_ylabel("Frecuencia")

fig.suptitle(
    f"PyTomography {pytomography.__version__} — Verificación OK\n"
    f"PyTorch {torch.__version__} ({'CUDA' if torch.cuda.is_available() else 'CPU'})",
    fontsize=13, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.92])

import os
out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output_demo", "pytomography_demo.png",
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\n{'=' * 60}")
print(f"✅ PyTomography verificado correctamente.")
print(f"   Figura: {out_path}")
print(f"")
print(f"   Lo que SÍ puede hacer PyTomography:")
print(f"   - Reconstrucción SPECT/PET iterativa (OSEM, MLEM, MAP)")
print(f"   - Modelado del sistema (colimador, atenuación, scatter)")
print(f"   - Reconstrucción GPU (si hay CUDA)")
print(f"   - Integración con 3D Slicer (SlicerSPECTRecon)")
print(f"")
print(f"   Lo que NO hace (SINCRO sí):")
print(f"   - Análisis cardíaco (AHA 17, fase, FEVI, bullseye)")
print(f"   - Segmentación miocárdica automática")
print(f"   - Informe clínico (PDF/HTML)")
print(f"   - Corrección de movimiento")
print(f"   - Filtros NITIDA / Denoise+ / FBP CLEAN")
print(f"{'=' * 60}")

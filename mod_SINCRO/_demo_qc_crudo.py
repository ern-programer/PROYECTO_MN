"""Demo temporal: panel QC de crudo gated."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.raw_projections import load_raw_projections, build_sinograms, center_of_mass_tracking

p = r"D:\- GAMMASYS\varios laburo\DICOMS\DICOM EXP\GATED\1.2.840.114080.1.0.25.210.122.122.1132007120716553910.4.dcm"
raw = load_raw_projections(p)
sh, sv = build_sinograms(raw.projections)
ty = center_of_mass_tracking(raw.projections, axis="y")
tx = center_of_mass_tracking(raw.projections, axis="x")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.patch.set_facecolor("#0b1220")

axes[0, 0].imshow(sh.T, cmap="hot", aspect="auto")
axes[0, 0].set_title("Sinograma H (perfil vertical vs angulo)", color="white")
axes[0, 0].set_xlabel("angulo"); axes[0, 0].set_ylabel("y")
axes[1, 0].imshow(sv.T, cmap="hot", aspect="auto")
axes[1, 0].set_title("Sinograma V (perfil horizontal vs angulo)", color="white")
axes[1, 0].set_xlabel("angulo"); axes[1, 0].set_ylabel("x")

summed = raw.projections.sum(axis=0)
pos = [(0, 1, 0), (0, 2, raw.n_angles // 3), (1, 1, 2 * raw.n_angles // 3)]
for r, c, a in pos:
    axes[r, c].imshow(summed[a], cmap="hot")
    axes[r, c].set_title(f"Proyeccion ang {a} ({raw.angles_deg[a]:.0f} deg)", color="white")
    axes[r, c].axis("off")

axc = axes[1, 2]
ang = np.arange(raw.n_angles)
axc.plot(ang, ty["com_series"], "o-", color="cyan", label="COM Y", ms=4)
axc.plot(ang, tx["com_series"], "s-", color="orange", label="COM X", ms=4)
out_y = np.where(ty["outliers"])[0]
axc.plot(out_y, ty["com_series"][out_y], "r*", ms=14, label=f"outliers Y ({ty['n_outliers']})")
axc.set_title(f"COM tracking: mov Y={ty['motion_suspected']} (max {ty['max_shift_px']}px)", color="white", fontsize=10)
axc.set_xlabel("angulo"); axc.set_ylabel("centro de masa (px)")
axc.legend(fontsize=8); axc.grid(alpha=0.3)

for ax in axes.ravel():
    ax.set_facecolor("#0b1220")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("#334155")

fig.suptitle(
    f"QC Crudo Gated — {raw.patient_name} | {raw.study_description} | "
    f"{raw.n_gates} gates x {raw.n_angles} ang | FC {raw.gating_info.get('heart_rate', 'N/D')} lpm",
    color="white", fontsize=12, fontweight="bold",
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = r"D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\output_demo\qc_crudo_demo.png"
fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
print("guardado:", out)

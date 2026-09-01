"""Punto de entrada NEUTRO de CT + fusión + atenuación para todo SINCRO.

Fase 0 del plan PERFU_FUSION: el código maduro vive en ``core.amyloid_spect``
(validado en el flujo AMYLO). Este módulo lo re-exporta con nombres neutros
para que perfusión (y cualquier flujo futuro) NO importe nada con nombre
"amyloid". Cuando la extracción física se justifique, las implementaciones se
mueven acá sin tocar a los consumidores.

Aporta además la conversión bilineal HU→μ (``mu_map_from_ct_hu``) para generar
el mapa de atenuación desde un CT cuando el equipo no exporta ATTMAP/CTAC.
"""

from __future__ import annotations

import numpy as np

# --- Re-export del núcleo maduro (validado en AMYLO_SPECT) -----------------
from core.amyloid_spect import (  # noqa: F401
    AttenuationMapResult,
    CTVolumeResult,
    align_ct_orientation_to_spect,
    apply_attenuation_correction_chang,
    apply_attenuation_correction_prototype,
    central_slices_preview,
    list_ct_series_in_path,
    load_attenuation_map_from_path,
    load_ct_volume_from_path,
    refine_ct_to_spect_rotation,
    refine_ct_to_spect_translation,
    register_ct_to_spect_rigid,
    remove_ct_table,
    resample_volume_to_spect_grid,
)

# μ del agua a 140 keV (Tc-99m), haz angosto, cm^-1.
MU_WATER_140KEV_CM = 0.154
# μ de hueso cortical (~1000 HU) a 140 keV, cm^-1.
MU_BONE_140KEV_CM = 0.250


def mu_map_from_ct_hu(
    ct_hu: np.ndarray,
    *,
    mu_water_cm: float = MU_WATER_140KEV_CM,
    mu_bone_cm: float = MU_BONE_140KEV_CM,
    bone_breakpoint_hu: float = 0.0,
    bone_reference_hu: float = 1000.0,
) -> tuple[np.ndarray, list[str]]:
    """Convierte un CT en HU a mapa de atenuación μ (cm^-1) por modelo bilineal.

    Modelo CTAC estándar (breakpoint en 0 HU):
    - HU <= 0 (aire..agua):  μ = μ_agua · (HU + 1000) / 1000, clip a 0.
    - HU > 0  (tejido..hueso): μ = μ_agua + HU · (μ_hueso − μ_agua) / HU_ref.

    Defaults calibrados para Tc-99m (140 keV). Para otros isótopos pasar los
    μ correspondientes a la energía del fotón.

    Returns
    -------
    (mu_map, notes) — mu_map en cm^-1, mismo shape que el CT.
    """
    hu = np.asarray(ct_hu, dtype=np.float64)
    mu = np.empty_like(hu)
    soft = hu <= float(bone_breakpoint_hu)
    mu[soft] = float(mu_water_cm) * (hu[soft] + 1000.0) / 1000.0
    bone_slope = (float(mu_bone_cm) - float(mu_water_cm)) / max(float(bone_reference_hu), 1e-6)
    mu[~soft] = float(mu_water_cm) + hu[~soft] * bone_slope
    mu = np.clip(mu, 0.0, None)
    notes = [
        "μ-map generado desde CT (modelo bilineal CTAC): "
        f"μ_agua={float(mu_water_cm):.3f}/cm, μ_hueso={float(mu_bone_cm):.3f}/cm @ {float(bone_reference_hu):.0f} HU, "
        f"breakpoint={float(bone_breakpoint_hu):.0f} HU."
    ]
    return mu, notes


def validate_mu_map(mu_map: np.ndarray) -> tuple[bool, list[str]]:
    """QC mínimo de un μ-map antes de usarlo en AC.

    Detecta los exports rotos vistos en campo (mapa vacío/2D/valores absurdos).
    """
    mu = np.asarray(mu_map, dtype=np.float64)
    notes: list[str] = []
    if mu.ndim != 3:
        return False, [f"μ-map inválido: se esperaba 3D, llegó shape {mu.shape}."]
    if mu.size == 0 or not np.isfinite(mu).any():
        return False, ["μ-map inválido: vacío o sin valores finitos."]
    mx = float(np.nanmax(mu))
    if mx <= 0.0:
        return False, ["μ-map inválido: todos los voxels son 0 (export CTAC roto, caso HWK)."]
    if mx > 2.0:
        notes.append(f"μ-map sospechoso: máximo {mx:.3f}/cm supera lo físico (~0.5/cm en hueso denso). Revisar unidades/escala.")
    body_frac = float(np.mean(mu > 0.01))
    if body_frac < 0.005:
        notes.append(f"μ-map con muy poco cuerpo ({body_frac * 100:.2f}% de voxels > 0.01/cm). Revisar registro/FOV.")
    notes.insert(0, f"μ-map OK: max={mx:.3f}/cm, cuerpo={body_frac * 100:.1f}% de voxels.")
    return True, notes


def lung_mask_from_ct_hu(
    ct_hu: np.ndarray,
    *,
    lung_hu_max: float = -400.0,
    body_hu_min: float = -500.0,
    min_component_frac: float = 0.002,
) -> np.ndarray:
    """Máscara de pulmones desde CT en HU (aire dentro del cuerpo).

    Distingue pulmón del aire exterior exigiendo que el componente de baja
    densidad NO toque el borde lateral del FOV.
    """
    import scipy.ndimage as ndi

    hu = np.asarray(ct_hu, dtype=np.float64)
    low = hu < float(lung_hu_max)
    body = hu > float(body_hu_min)
    # Aire exterior: componente low conectado al borde del volumen.
    lbl, n = ndi.label(low)
    if n == 0:
        return np.zeros_like(low, dtype=bool)
    border_labels = set(np.unique(np.concatenate([
        lbl[0].ravel(), lbl[-1].ravel(),
        lbl[:, 0, :].ravel(), lbl[:, -1, :].ravel(),
        lbl[:, :, 0].ravel(), lbl[:, :, -1].ravel(),
    ])))
    border_labels.discard(0)
    min_vox = max(32, int(hu.size * float(min_component_frac)))
    lungs = np.zeros_like(low, dtype=bool)
    for lab in range(1, n + 1):
        if lab in border_labels:
            continue
        comp = lbl == lab
        if int(comp.sum()) >= min_vox:
            lungs |= comp
    return lungs


def subdiaphragmatic_mask_from_ct(
    ct_hu: np.ndarray,
    *,
    soft_hu_range: tuple[float, float] = (-100.0, 200.0),
) -> tuple[np.ndarray, list[str]]:
    """Máscara de tejido blando SUBDIAFRAGMÁTICO (hígado/intestino) desde CT.

    Idea: el corazón queda ENTRE los pulmones (dentro del rango z pulmonar);
    hígado/intestino quedan más allá de la base pulmonar en dirección caudal.
    Por columna (y,x): el límite es el último voxel de pulmón en dirección
    caudal; el tejido blando más allá de ese límite se marca. La dirección
    caudal del eje z se detecta por dónde hay más tejido blando fuera del
    rango pulmonar.
    """
    hu = np.asarray(ct_hu, dtype=np.float64)
    notes: list[str] = []
    if hu.ndim != 3:
        raise ValueError(f"CT debe ser 3D, llegó {hu.shape}")
    lungs = lung_mask_from_ct_hu(hu)
    soft = (hu >= float(soft_hu_range[0])) & (hu <= float(soft_hu_range[1]))
    if not lungs.any():
        notes.append("Sin pulmones detectables en el CT: no se genera máscara subdiafragmática.")
        return np.zeros_like(soft, dtype=bool), notes

    zs = np.where(lungs.any(axis=(1, 2)))[0]
    z_lo, z_hi = int(zs.min()), int(zs.max())
    n_z = hu.shape[0]
    soft_before = float(soft[:z_lo].sum())
    soft_after = float(soft[z_hi + 1:].sum())
    caudal_up = soft_after >= soft_before  # caudal hacia índices z crecientes
    notes.append(
        f"Pulmones en z=[{z_lo},{z_hi}] de {n_z}; caudal hacia z "
        f"{'crecientes' if caudal_up else 'decrecientes'} "
        f"(soft antes={soft_before:.0f}, después={soft_after:.0f})."
    )

    mask = np.zeros_like(soft, dtype=bool)
    zi = np.arange(n_z)[:, None, None]
    has_lung_col = lungs.any(axis=0)
    if caudal_up:
        # Límite por columna: último z de pulmón; fallback global z_hi.
        zlung = np.where(lungs, zi, -1).max(axis=0)
        boundary = np.where(has_lung_col, zlung, z_hi)
        mask = soft & (zi > boundary[None, :, :])
    else:
        zlung = np.where(lungs, zi, n_z).min(axis=0)
        boundary = np.where(has_lung_col, zlung, z_lo)
        mask = soft & (zi < boundary[None, :, :])
    frac = float(mask.mean() * 100.0)
    notes.append(f"Máscara subdiafragmática: {frac:.1f}% de voxels (tejido blando más allá de la base pulmonar).")
    return mask, notes

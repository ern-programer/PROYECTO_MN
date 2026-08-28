# -*- coding: utf-8 -*-
"""SINCRO - core.amyloid_spect

Pipeline base para AMYLO SPECT 3D (fase 2, experimental).

Objetivo:
- Cargar estudio SPECT desde DICOM.
- Si viene en crudo (proyecciones), reconstruir un volumen ungated básico.
- Si viene reconstruido gated, generar volumen ungated por promedio de gates.
- Calcular métricas volumétricas iniciales para prototipo clínico.
- Ofrecer una sustracción ósea visual inicial (con y sin CT).

NOTA: módulo experimental y de apoyo visual. No diagnóstico automático.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from core.dicom_loader import load as load_dicom_study
from core.raw_reconstruction import RawReconConfig, reconstruct_raw_gated_pipeline
from core.cardiac_reorientation import (
    anatomical_cuts_gated,
    auto_orient_lv,
    reslice_from_vector_gated,
)
from core.dicom_export import save_cardiac_axes_dicoms


@dataclass
class AmyloidSpectResult:
    """Resultado mínimo del análisis SPECT 3D."""

    volume: np.ndarray
    source_path: str
    was_raw: bool
    notes: list[str]
    metrics: dict[str, float]
    n_gates: int = 1
    spacing_zyx: tuple[float, float, float] | None = None
    affine_ijk_to_lps: np.ndarray | None = None


@dataclass
class BoneSuppressionResult:
    """Resultado de la sustracción ósea visual."""

    enhanced_volume: np.ndarray
    bone_mask: np.ndarray
    method: str
    notes: list[str]


@dataclass
class CTVolumeResult:
    """Volumen CT cargado desde un archivo o una serie DICOM."""

    volume: np.ndarray
    source_path: str
    series_uid: str
    series_description: str
    n_slices: int
    notes: list[str]
    spacing_zyx: tuple[float, float, float] | None = None
    affine_ijk_to_lps: np.ndarray | None = None


@dataclass
class AttenuationMapResult:
    """Mapa de atenuación (ATT MAP) cargado desde DICOM."""

    volume: np.ndarray
    source_path: str
    series_uid: str
    series_description: str
    n_slices: int
    notes: list[str]
    spacing_zyx: tuple[float, float, float] | None = None
    affine_ijk_to_lps: np.ndarray | None = None


@dataclass
class AmyloidReconstructionBundle:
    """Bundle de reconstrucción AMYLO reutilizando pipeline de perfusión."""

    study: Any
    source_path: str
    was_raw: bool
    ungated_volume: np.ndarray
    gated_volume: np.ndarray
    tomo_cuts: dict[str, np.ndarray]
    cardiac_axes: dict[str, np.ndarray]
    notes: list[str]
    spacing_zyx: tuple[float, float, float] | None = None
    affine_ijk_to_lps: np.ndarray | None = None
    # Volumen sin post-filtro gaussiano (para toggle con/sin filtro en UI)
    ungated_volume_unfiltered: np.ndarray | None = None


def _safe_norm(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    if arr.size == 0:
        return arr
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _dicom_sort_key(ds) -> tuple[float, int, str]:
    try:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            return (float(ipp[2]), int(getattr(ds, "InstanceNumber", 0) or 0), str(getattr(ds, "SOPInstanceUID", "")))
    except Exception:
        pass
    try:
        return (float(getattr(ds, "SliceLocation", 0.0) or 0.0), int(getattr(ds, "InstanceNumber", 0) or 0), str(getattr(ds, "SOPInstanceUID", "")))
    except Exception:
        return (0.0, int(getattr(ds, "InstanceNumber", 0) or 0), str(getattr(ds, "SOPInstanceUID", "")))


def _ct_pixels_hu(ds) -> np.ndarray:
    arr = np.asarray(ds.pixel_array, dtype=np.float64)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return arr * slope + intercept


def _ensure_hu_calibration(vol: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Corrige CT sin intercept declarado (valores = HU + 1024, aire en 0).

    Sin esta corrección las ventanas fijas por tejido (pulmón/blanda/ósea)
    caen en rangos equivocados y no aíslan nada.
    """
    notes: list[str] = []
    v = np.asarray(vol, dtype=np.float64)
    vmin = float(v.min())
    p999 = float(np.percentile(v, 99.9))
    if vmin >= -10.0 and p999 > 1200.0:
        v = v - 1024.0
        notes.append(
            "CT sin RescaleIntercept útil: valores detectados como HU+1024; "
            "se aplicó offset -1024 (aire ≈ -1000 HU) para calibrar ventanas por tejido."
        )
    return v, notes


def _affine_from_iop_ipp_spacing(iop, ipp, spacing_zyx: tuple[float, float, float] | None) -> np.ndarray | None:
    if iop is None or ipp is None or spacing_zyx is None:
        return None
    try:
        row = np.asarray([float(iop[0]), float(iop[1]), float(iop[2])], dtype=np.float64)
        col = np.asarray([float(iop[3]), float(iop[4]), float(iop[5])], dtype=np.float64)
        normal = np.cross(row, col)
        origin = np.asarray([float(ipp[0]), float(ipp[1]), float(ipp[2])], dtype=np.float64)
        sz, sy, sx = [float(v) for v in spacing_zyx]
        aff = np.eye(4, dtype=np.float64)
        # Array index order = (z, y(row), x(col)). DICOM LPS axes: ipp + row*y*sy + col*x*sx + normal*z*sz.
        aff[:3, 0] = normal * sz
        aff[:3, 1] = row * sy
        aff[:3, 2] = col * sx
        aff[:3, 3] = origin
        return aff
    except Exception:
        return None


def _extract_iop_ipp_from_dataset(ds) -> tuple[list[float] | None, list[float] | None]:
    """Extrae IOP/IPP desde root DICOM o desde DetectorInformationSequence."""
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if iop is not None and ipp is not None:
        return iop, ipp

    seq = getattr(ds, "DetectorInformationSequence", None)
    if seq:
        try:
            first = seq[0]
            iop = iop if iop is not None else getattr(first, "ImageOrientationPatient", None)
            ipp = ipp if ipp is not None else getattr(first, "ImagePositionPatient", None)
            if iop is not None and ipp is not None:
                return iop, ipp
        except Exception:
            pass
    return None, None


def _affine_from_dicom_file(path: str, spacing_zyx: tuple[float, float, float] | None) -> np.ndarray | None:
    try:
        import pydicom
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        iop, ipp = _extract_iop_ipp_from_dataset(ds)
        return _affine_from_iop_ipp_spacing(iop, ipp, spacing_zyx)
    except Exception:
        return None


def _study_spacing_zyx(study: Any) -> tuple[float, float, float] | None:
    px = getattr(study, "pixel_spacing", None)
    if px is None:
        return None
    try:
        sy = abs(float(px[0]))
        sx = abs(float(px[1]))
        sz = (
            getattr(study, "z_spacing_mm", None)
            or getattr(study, "spacing_between_slices_mm", None)
            or getattr(study, "slice_thickness_mm", None)
            or sy
        )
        return (abs(float(sz)), sy, sx)
    except Exception:
        return None


def _center_crop_or_pad_3d(vol: np.ndarray, target_shape: tuple[int, int, int], fill_value: float = -1024.0) -> np.ndarray:
    src = np.asarray(vol, dtype=np.float64)
    out = np.full(tuple(int(v) for v in target_shape), float(fill_value), dtype=np.float64)
    src_slices = []
    dst_slices = []
    for src_len, dst_len in zip(src.shape, out.shape):
        if src_len <= dst_len:
            src0 = 0
            src1 = src_len
            dst0 = (dst_len - src_len) // 2
            dst1 = dst0 + src_len
        else:
            src0 = (src_len - dst_len) // 2
            src1 = src0 + dst_len
            dst0 = 0
            dst1 = dst_len
        src_slices.append(slice(src0, src1))
        dst_slices.append(slice(dst0, dst1))
    out[tuple(dst_slices)] = src[tuple(src_slices)]
    return out


def _resample_ct_to_spect_affine(
    ct: np.ndarray,
    spect_shape: tuple[int, int, int],
    ct_affine: np.ndarray,
    spect_affine: np.ndarray,
    fill_value: float,
    order: int = 0,
) -> np.ndarray:
    """Remuestrea CT en la grilla IJK del SPECT usando affines IJK→LPS."""
    ct_to_spect = np.linalg.inv(ct_affine) @ spect_affine
    matrix = ct_to_spect[:3, :3]
    offset = ct_to_spect[:3, 3]
    return ndi.affine_transform(
        np.asarray(ct, dtype=np.float64),
        matrix=matrix,
        offset=offset,
        output_shape=tuple(int(v) for v in spect_shape),
        order=order,
        mode="constant",
        cval=float(fill_value),
    )


def resample_volume_to_spect_grid(
    source_volume: np.ndarray,
    spect_volume: np.ndarray,
    *,
    source_spacing_zyx: tuple[float, float, float] | None = None,
    spect_spacing_zyx: tuple[float, float, float] | None = None,
    source_affine_ijk_to_lps: np.ndarray | None = None,
    spect_affine_ijk_to_lps: np.ndarray | None = None,
    fill_value: float | None = None,
    order: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Remuestrea un volumen fuente a la grilla del SPECT sin corrimiento por máscara.
    
    Args:
        order: Orden de interpolación. 0=nearest-neighbor (rápido, ideal para CT
            downsampling), 1=bilineal (suave, ideal para SPECT upsampling).
    """
    src = np.asarray(source_volume, dtype=np.float64)
    sp = np.asarray(spect_volume, dtype=np.float64)
    if src.ndim != 3 or sp.ndim != 3:
        raise ValueError(f"source_volume y spect_volume deben ser 3D. source={src.shape}, spect={sp.shape}")
    notes: list[str] = []
    fv = float(np.min(src)) if fill_value is None else float(fill_value)

    if src.shape == sp.shape:
        notes.append("Volumen fuente ya está en misma grilla que SPECT (shape idéntico).")
        return src.copy(), notes

    if source_affine_ijk_to_lps is not None and spect_affine_ijk_to_lps is not None:
        rs = _resample_ct_to_spect_affine(
            src,
            sp.shape,
            np.asarray(source_affine_ijk_to_lps, dtype=np.float64),
            np.asarray(spect_affine_ijk_to_lps, dtype=np.float64),
            fill_value=fv,
            order=order,
        )
        notes.append(
            "Remuestreo a grilla SPECT por geometría DICOM completa "
            f"(IPP/IOP/spacing): {src.shape} -> {rs.shape} (order={order})."
        )
        return rs, notes

    if source_spacing_zyx is not None and spect_spacing_zyx is not None:
        zoom_factors = tuple(
            max(1e-6, float(source_spacing_zyx[i])) / max(1e-6, float(spect_spacing_zyx[i]))
            for i in range(3)
        )
        phys = ndi.zoom(src, zoom_factors, order=order)
        rs = _center_crop_or_pad_3d(phys, sp.shape, fill_value=fv)
        notes.append(
            "Remuestreo a grilla SPECT por espaciado físico "
            f"shape {src.shape} -> físico {phys.shape} -> {rs.shape}; "
            f"spacing src z/y/x={source_spacing_zyx}, SPECT z/y/x={spect_spacing_zyx}, zoom={zoom_factors} (order={order})."
        )
        return rs, notes

    zoom_factors = (
        sp.shape[0] / max(src.shape[0], 1),
        sp.shape[1] / max(src.shape[1], 1),
        sp.shape[2] / max(src.shape[2], 1),
    )
    rs = ndi.zoom(src, zoom_factors, order=order)
    notes.append(
        "Remuestreo a grilla SPECT por shape "
        f"{src.shape} -> {rs.shape} (zoom={zoom_factors}, order={order}); sin spacing físico disponible."
    )
    return rs, notes


def load_ct_volume_from_path(path: str) -> CTVolumeResult:
    """Carga CT desde un DICOM o desde la serie completa del archivo elegido.

    Si ``path`` es archivo, usa su ``SeriesInstanceUID`` y busca los cortes hermanos
    en la misma carpeta. Si ``path`` es carpeta, elige la serie CT con más cortes.
    """
    import pydicom
    import os

    if not path:
        raise ValueError("Ruta CT vacía")
    root = os.path.abspath(path)
    notes: list[str] = []

    selected_uid = ""
    search_dir = root
    if os.path.isfile(root):
        ds0 = pydicom.dcmread(root, stop_before_pixels=True, force=True)
        selected_uid = str(getattr(ds0, "SeriesInstanceUID", "") or "")
        search_dir = os.path.dirname(root)
        notes.append("Archivo CT elegido: se intenta cargar la serie completa de la misma carpeta.")
    elif os.path.isdir(root):
        notes.append("Carpeta CT elegida: se selecciona la serie CT con más cortes.")
    else:
        raise FileNotFoundError(root)

    series: dict[str, list[tuple[str, Any]]] = {}
    if os.path.isdir(search_dir):
        for dirpath, _, filenames in os.walk(search_dir):
            for name in filenames:
                fpath = os.path.join(dirpath, name)
                try:
                    ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
                    if str(getattr(ds, "Modality", "") or "").upper() != "CT":
                        continue
                    uid = str(getattr(ds, "SeriesInstanceUID", "") or fpath)
                    if selected_uid and uid != selected_uid:
                        continue
                    series.setdefault(uid, []).append((fpath, ds))
                except Exception:
                    continue

    if not series and os.path.isfile(root):
        ds = pydicom.dcmread(root, force=True)
        uid = str(getattr(ds, "SeriesInstanceUID", "") or root)
        series[uid] = [(root, ds)]

    if not series:
        raise ValueError("No se encontraron DICOM CT en la ruta seleccionada.")

    uid, items = max(series.items(), key=lambda kv: len(kv[1]))
    items = sorted(items, key=lambda item: _dicom_sort_key(item[1]))

    first_path, first_ds = items[0]
    if len(items) == 1:
        ds = pydicom.dcmread(first_path, force=True)
        vol = _ct_pixels_hu(ds)
        if vol.ndim == 2:
            vol = vol[np.newaxis, :, :]
        elif vol.ndim > 3:
            vol = np.squeeze(vol)
        if vol.ndim != 3:
            raise ValueError(f"CT no convertible a volumen 3D: {vol.shape}")
    else:
        slices = []
        for fpath, _ in items:
            ds = pydicom.dcmread(fpath, force=True)
            img = _ct_pixels_hu(ds)
            while img.ndim > 2:
                img = img[0]
            if img.ndim == 2:
                slices.append(img)
        if not slices:
            raise ValueError("La serie CT no contiene píxeles 2D utilizables.")
        vol = np.stack(slices, axis=0)

    desc = str(getattr(first_ds, "SeriesDescription", "") or "CT")
    px = getattr(first_ds, "PixelSpacing", None)
    spacing_zyx = None
    if px is not None:
        try:
            sy = abs(float(px[0]))
            sx = abs(float(px[1]))
            if len(items) > 1:
                zs = []
                for _, ds in items:
                    ipp = getattr(ds, "ImagePositionPatient", None)
                    if ipp is not None and len(ipp) >= 3:
                        zs.append(float(ipp[2]))
                zsort = sorted(zs)
                diffs = [abs(zsort[i + 1] - zsort[i]) for i in range(len(zsort) - 1) if abs(zsort[i + 1] - zsort[i]) > 1e-6]
                sz = float(np.median(diffs)) if diffs else float(getattr(first_ds, "SliceThickness", sy) or sy)
            else:
                sz = float(getattr(first_ds, "SliceThickness", sy) or sy)
            spacing_zyx = (abs(sz), sy, sx)
        except Exception:
            spacing_zyx = None
    affine_ijk_to_lps = _affine_from_iop_ipp_spacing(
        getattr(first_ds, "ImageOrientationPatient", None),
        getattr(first_ds, "ImagePositionPatient", None),
        spacing_zyx,
    )
    notes.append(f"Serie CT cargada: {desc} · cortes={int(vol.shape[0])} · shape={tuple(vol.shape)}.")
    vol, hu_notes = _ensure_hu_calibration(vol)
    notes.extend(hu_notes)
    notes.append(f"Rango CT: {float(vol.min()):.0f} a {float(vol.max()):.0f} HU.")
    if spacing_zyx is not None:
        notes.append(f"Spacing CT z/y/x={spacing_zyx[0]:.3f}/{spacing_zyx[1]:.3f}/{spacing_zyx[2]:.3f} mm.")
    if len(series) > 1 and not selected_uid:
        notes.append(f"Se eligió automáticamente la serie con más cortes entre {len(series)} series CT.")
    if affine_ijk_to_lps is not None:
        notes.append("Geometría CT DICOM disponible (IPP/IOP/spacing).")

    return CTVolumeResult(
        volume=np.asarray(vol, dtype=np.float64),
        source_path=root,
        series_uid=uid,
        series_description=desc,
        n_slices=int(vol.shape[0]),
        notes=notes,
        spacing_zyx=spacing_zyx,
        affine_ijk_to_lps=affine_ijk_to_lps,
    )


def _looks_like_att_map(description: str, file_name: str) -> bool:
    text = f"{description} {file_name}".upper()
    return ("ATT" in text and "MAP" in text) or ("ATTMAP" in text)


def load_attenuation_map_from_path(path: str) -> AttenuationMapResult:
    """Carga un ATT MAP DICOM (volumen μ aproximado) desde archivo o carpeta."""
    import os
    import pydicom

    if not path:
        raise ValueError("Ruta ATT MAP vacía")

    root = os.path.abspath(path)
    notes: list[str] = []
    selected_uid = ""
    search_dir = root
    selected_modality = "NM"

    if os.path.isfile(root):
        ds0 = pydicom.dcmread(root, stop_before_pixels=True, force=True)
        selected_uid = str(getattr(ds0, "SeriesInstanceUID", "") or "")
        selected_modality = str(getattr(ds0, "Modality", "NM") or "NM").upper()
        search_dir = os.path.dirname(root)
        notes.append("Archivo ATT MAP elegido: se intenta cargar la serie completa de la misma carpeta.")
    elif os.path.isdir(root):
        notes.append("Carpeta elegida: se buscará la mejor serie ATT MAP.")
    else:
        raise FileNotFoundError(root)

    series: dict[str, list[tuple[str, Any]]] = {}
    if os.path.isdir(search_dir):
        for dirpath, _, filenames in os.walk(search_dir):
            for name in filenames:
                fpath = os.path.join(dirpath, name)
                try:
                    ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
                except Exception:
                    continue
                mod = str(getattr(ds, "Modality", "") or "").upper()
                if mod not in {"NM", "PT"}:
                    continue
                uid = str(getattr(ds, "SeriesInstanceUID", "") or fpath)
                if selected_uid and uid != selected_uid:
                    continue
                if selected_modality and mod != selected_modality and selected_uid:
                    continue
                series.setdefault(uid, []).append((fpath, ds))

    if not series and os.path.isfile(root):
        ds = pydicom.dcmread(root, force=True)
        uid = str(getattr(ds, "SeriesInstanceUID", "") or root)
        series[uid] = [(root, ds)]

    if not series:
        raise ValueError("No se encontraron series NM/PT para ATT MAP en la ruta seleccionada.")

    scored: list[tuple[int, int, str]] = []
    for uid, items in series.items():
        first_ds = items[0][1]
        desc = str(getattr(first_ds, "SeriesDescription", "") or "")
        looks = _looks_like_att_map(desc, os.path.basename(items[0][0]))
        n = len(items)
        score = (1000 if looks else 0) + n
        scored.append((score, n, uid))
    _, _, best_uid = max(scored, key=lambda x: (x[0], x[1]))
    items = sorted(series[best_uid], key=lambda item: _dicom_sort_key(item[1]))

    first_path, first_ds = items[0]
    arrs = []
    for fpath, _ in items:
        ds = pydicom.dcmread(fpath, force=True)
        a = np.asarray(ds.pixel_array, dtype=np.float64)
        while a.ndim > 2:
            a = a[0]
        if a.ndim == 2:
            arrs.append(a)

    if not arrs:
        # fallback multiframe
        ds = pydicom.dcmread(first_path, force=True)
        a = np.asarray(ds.pixel_array, dtype=np.float64)
        if a.ndim == 3:
            vol = a
        elif a.ndim == 2:
            vol = a[np.newaxis, :, :]
        else:
            raise ValueError(f"ATT MAP no convertible a volumen 3D: {a.shape}")
    else:
        vol = np.stack(arrs, axis=0)

    desc = str(getattr(first_ds, "SeriesDescription", "") or "ATT MAP")
    px = getattr(first_ds, "PixelSpacing", None)
    spacing_zyx = None
    if px is not None:
        try:
            sy = abs(float(px[0]))
            sx = abs(float(px[1]))
            if len(items) > 1:
                zs = []
                for _, ds in items:
                    ipp = getattr(ds, "ImagePositionPatient", None)
                    if ipp is not None and len(ipp) >= 3:
                        zs.append(float(ipp[2]))
                zsort = sorted(zs)
                diffs = [abs(zsort[i + 1] - zsort[i]) for i in range(len(zsort) - 1) if abs(zsort[i + 1] - zsort[i]) > 1e-6]
                sz = float(np.median(diffs)) if diffs else float(getattr(first_ds, "SliceThickness", sy) or sy)
            else:
                sz = float(getattr(first_ds, "SliceThickness", sy) or sy)
            spacing_zyx = (abs(sz), sy, sx)
        except Exception:
            spacing_zyx = None

    iop, ipp = _extract_iop_ipp_from_dataset(first_ds)
    affine_ijk_to_lps = _affine_from_iop_ipp_spacing(iop, ipp, spacing_zyx)
    notes.append(f"Serie ATT MAP cargada: {desc} · cortes={int(vol.shape[0])} · shape={tuple(vol.shape)}.")
    # Diagnóstico de unidades µ: para Tc-99m (140 keV) el agua/tejido blando
    # debe quedar en ~0.15 cm⁻¹ tras aplicar µ-scale. Se estima la mediana del
    # tejido (voxeles no-cero) y se sugiere la escala que la lleva a 0.154.
    _nz = np.asarray(vol, dtype=np.float64)
    _nz = _nz[_nz > 0]
    if _nz.size > 100:
        _med = float(np.median(_nz))
        _sug = 0.154 / _med if _med > 1e-9 else 1.0
        notes.append(
            f"µ mediana (no-cero): {_med:.4g}. Referencia Tc-99m 140 keV: agua=0.154 cm⁻¹, "
            f"hueso≈0.25 cm⁻¹. µ-scale sugerido ≈ {_sug:.4g} "
            "(si el mapa ya está en cm⁻¹, dejar 1.0)."
        )
    if spacing_zyx is not None:
        notes.append(f"Spacing ATT MAP z/y/x={spacing_zyx[0]:.3f}/{spacing_zyx[1]:.3f}/{spacing_zyx[2]:.3f} mm.")
    if affine_ijk_to_lps is not None:
        notes.append("Geometría ATT MAP DICOM disponible (IPP/IOP/spacing).")

    return AttenuationMapResult(
        volume=np.asarray(vol, dtype=np.float64),
        source_path=root,
        series_uid=best_uid,
        series_description=desc,
        n_slices=int(vol.shape[0]),
        notes=notes,
        spacing_zyx=spacing_zyx,
        affine_ijk_to_lps=affine_ijk_to_lps,
    )


def apply_attenuation_correction_prototype(
    spect_volume: np.ndarray,
    att_map_volume: np.ndarray,
    *,
    mu_scale: float = 0.12,
) -> tuple[np.ndarray, list[str]]:
    """Aplica AC prototipo sobre SPECT usando ATT MAP (heurístico).

    Se normaliza μ-map a [0..1] y se aplica factor multiplicativo
    ``exp(mu_scale * mu_norm)`` sobre SPECT.
    """
    sp = np.asarray(spect_volume, dtype=np.float64)
    mu = np.asarray(att_map_volume, dtype=np.float64)
    if sp.ndim != 3 or mu.ndim != 3:
        raise ValueError(f"SPECT y ATT MAP deben ser 3D. SPECT={sp.shape}, ATT={mu.shape}")
    if sp.shape != mu.shape:
        raise ValueError(f"ATT MAP y SPECT deben estar en la misma grilla. SPECT={sp.shape}, ATT={mu.shape}")

    notes: list[str] = []
    mu_n = _safe_norm(mu)
    gain = np.exp(float(mu_scale) * mu_n)
    corrected = np.clip(sp, 0.0, None) * gain
    notes.append(
        "AC prototipo aplicada con ATT MAP normalizado "
        f"(mu_scale={float(mu_scale):.3f}, gain≈[{float(np.min(gain)):.3f},{float(np.max(gain)):.3f}])."
    )
    return corrected, notes


def apply_attenuation_correction_chang(
    spect_volume: np.ndarray,
    att_map_volume: np.ndarray,
    *,
    spect_spacing_zyx: tuple[float, float, float] | None = None,
    mu_scale: float = 1.0,
    n_angles: int = 36,
) -> tuple[np.ndarray, list[str]]:
    """AC Chang 2D slice-wise (experimental) usando ATT MAP en grilla SPECT."""
    sp = np.asarray(spect_volume, dtype=np.float64)
    mu = np.asarray(att_map_volume, dtype=np.float64)
    if sp.ndim != 3 or mu.ndim != 3:
        raise ValueError(f"SPECT y ATT MAP deben ser 3D. SPECT={sp.shape}, ATT={mu.shape}")
    if sp.shape != mu.shape:
        raise ValueError(f"ATT MAP y SPECT deben estar en la misma grilla. SPECT={sp.shape}, ATT={mu.shape}")

    notes: list[str] = []
    sp_nonneg = np.clip(sp, 0.0, None)
    q99 = float(np.percentile(mu, 99.0)) if mu.size else 0.0
    if q99 > 0.5:
        mu_cm = _safe_norm(mu) * 0.15
        notes.append("ATT MAP escalado a μ 1/cm por normalización robusta (p99>0.5).")
    else:
        mu_cm = np.clip(mu, 0.0, None)
        notes.append("ATT MAP interpretado directamente como μ aproximado (1/cm).")

    px_mm = float(spect_spacing_zyx[2]) if spect_spacing_zyx is not None else 6.8
    px_cm = max(1e-4, px_mm / 10.0)
    angles = np.linspace(0.0, 180.0, max(8, int(n_angles)), endpoint=False)

    corrected = np.zeros_like(sp_nonneg)
    body_mask = mu_cm > max(1e-6, float(np.percentile(mu_cm, 15.0)))
    if not np.any(body_mask):
        body_mask = sp_nonneg > float(np.percentile(sp_nonneg, 25.0))

    for z in range(sp_nonneg.shape[0]):
        mu_sl = np.asarray(mu_cm[z], dtype=np.float64)
        cf_acc = np.zeros_like(mu_sl, dtype=np.float64)
        for ang in angles:
            rot = ndi.rotate(mu_sl, angle=float(ang), reshape=False, order=1, mode="constant", cval=0.0)
            tau = np.cumsum(rot[:, ::-1], axis=1)[:, ::-1] * px_cm
            cf_rot = np.exp(float(mu_scale) * tau)
            cf = ndi.rotate(cf_rot, angle=-float(ang), reshape=False, order=1, mode="constant", cval=1.0)
            cf_acc += cf
        cf_avg = cf_acc / float(len(angles))
        m = np.asarray(body_mask[z], dtype=bool)
        if np.any(m):
            norm = float(np.median(cf_avg[m]))
            if norm > 1e-9:
                cf_avg = cf_avg / norm
        cf_avg = np.clip(cf_avg, 0.25, 4.0)
        corrected[z] = sp_nonneg[z] * cf_avg

    notes.append(
        "AC Chang slice-wise aplicada "
        f"(mu_scale={float(mu_scale):.2f}, ángulos={int(len(angles))}, px={px_mm:.3f} mm)."
    )
    return corrected, notes


def _compute_lv_like_metrics(volume: np.ndarray) -> dict[str, float]:
    """Métricas 3D iniciales (proxy) para fase 2.

    Heurística:
    - Máscara miocárdica aproximada por percentil alto.
    - Fondo mediastinal aproximado por percentil bajo global.
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"Se esperaba volumen 3D, recibido {vol.shape}")

    p80 = float(np.percentile(vol, 80.0))
    p95 = float(np.percentile(vol, 95.0))
    p20 = float(np.percentile(vol, 20.0))

    lv_mask = vol >= p80
    bg_mask = vol <= p20

    lv_mean = float(np.mean(vol[lv_mask])) if np.any(lv_mask) else 0.0
    lv_peak = float(np.mean(vol[vol >= p95])) if np.any(vol >= p95) else lv_mean
    bg_mean = float(np.mean(vol[bg_mask])) if np.any(bg_mask) else 1.0

    ratio_lv_bg = lv_mean / max(bg_mean, 1e-9)
    heterogeneity = float(np.std(vol[lv_mask]) / max(lv_mean, 1e-9)) if np.any(lv_mask) else 0.0

    return {
        "lv_mean": lv_mean,
        "lv_peak": lv_peak,
        "bg_mean": bg_mean,
        "ratio_lv_bg": ratio_lv_bg,
        "heterogeneity_cv": heterogeneity,
        "p80_threshold": p80,
    }


def load_spect_volume_from_dicom(
    dicom_path: str,
    *,
    recon_method: str = "fbp",
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Carga un volumen SPECT desde DICOM y retorna el volumen 3D + spacing.
    
    Helper simplificado para el módulo de washout dual-SPECT.
    
    Args:
        dicom_path: Ruta al archivo DICOM o directorio.
        recon_method: Método de reconstrucción ('fbp' o 'osem').
    
    Returns:
        Tuple (volume, spacing) donde:
        - volume: np.ndarray 3D con el volumen SPECT
        - spacing: Tuple (dz, dy, dx) en mm
    """
    result = run_amyloid_spect_analysis(dicom_path, recon_method=recon_method)
    volume = np.asarray(result.volume, dtype=np.float64)
    if result.spacing_zyx is not None:
        spacing = result.spacing_zyx
    else:
        spacing = (4.0, 4.0, 4.0)  # Default 4mm isotropic
    return volume, spacing


def run_amyloid_spect_analysis(
    dicom_path: str,
    *,
    recon_method: str = "fbp",
    notes_prefix: str = "AMYLO_SPECT",
) -> AmyloidSpectResult:
    """Ejecuta el flujo base AMYLO SPECT sobre un DICOM.

    Si el DICOM es crudo, reconstruye ungated básico con `reconstruct_raw_gated_pipeline`.
    Si el DICOM ya es reconstruido gated, promedia gates para obtener volumen 3D.
    """
    study = load_dicom_study(dicom_path)
    notes: list[str] = [f"{notes_prefix}: inicio análisis 3D."]
    n_gates = int(getattr(study, "n_gates", 1) or 1)

    if not getattr(study, "reconstructed", True):
        # Modo crudo: cube = (gates, angles, H, W)
        cfg = RawReconConfig(reconstruction_method=str(recon_method).lower())
        raw_res = reconstruct_raw_gated_pipeline(
            study.cube,
            angles_deg=study.angles_deg,
            config=cfg,
            scatter_projections=getattr(study, "scatter_projections", None),
        )
        vol = np.asarray(raw_res.ungated_volume, dtype=np.float64)
        notes.append("Fuente cruda detectada: reconstrucción ungated realizada.")
        notes.extend(getattr(raw_res, "notes", []))
        was_raw = True
    else:
        cube = np.asarray(study.cube, dtype=np.float64)
        if cube.ndim != 4:
            raise ValueError(
                "El estudio reconstruido no tiene formato gated 4D esperado "
                f"(gates, slices, H, W): {cube.shape}"
            )
        vol = np.mean(cube, axis=0)
        notes.append("Fuente reconstruida detectada: volumen ungated por promedio de gates.")
        was_raw = False

    if vol.ndim != 3:
        raise ValueError(f"Volumen 3D inválido tras análisis: {vol.shape}")

    metrics = _compute_lv_like_metrics(vol)
    notes.append("Métricas 3D proxy calculadas (experimental).")

    spacing = _study_spacing_zyx(study)
    return AmyloidSpectResult(
        volume=vol,
        source_path=dicom_path,
        was_raw=was_raw,
        notes=notes,
        metrics=metrics,
        n_gates=n_gates,
        spacing_zyx=spacing,
        affine_ijk_to_lps=_affine_from_dicom_file(dicom_path, spacing),
    )


def reconstruct_amyloid_with_perf_pipeline(
    dicom_path: str,
    *,
    recon_config: RawReconConfig | None = None,
    cuts_mode: str = "mixed",
    attenuation_mu_map: np.ndarray | None = None,
    attenuation_pixel_size_cm: float | None = None,
    progress_callback=None,
) -> AmyloidReconstructionBundle:
    """Reconstruye AMYLO reusando el pipeline de perfusión existente.

    cuts_mode:
    - ``tomo``: solo tomográficos (axial/coronal/sagital).
    - ``cardiac``: solo cardíacos SA/HLA/VLA (auto-orientación si posible).
    - ``mixed``: ambos.
    """
    if progress_callback is not None:
        progress_callback(0.0, "Cargando DICOM SPECT...")
    study = load_dicom_study(dicom_path)
    notes: list[str] = ["AMYLO_SPECT: reconstrucción con pipeline de perfusión."]
    cfg = recon_config or RawReconConfig(reconstruction_method="fbp")

    if not getattr(study, "reconstructed", True):
        raw_res = reconstruct_raw_gated_pipeline(
            np.asarray(study.cube, dtype=np.float64),
            angles_deg=getattr(study, "angles_deg", None),
            config=cfg,
            scatter_projections=getattr(study, "scatter_projections", None),
            attenuation_mu_map=attenuation_mu_map,
            attenuation_pixel_size_cm=attenuation_pixel_size_cm,
            progress_callback=progress_callback,
        )
        ungated = np.asarray(raw_res.ungated_volume, dtype=np.float64)
        gated = np.asarray(raw_res.gated_volume, dtype=np.float64)
        notes.append("Fuente cruda: reconstrucción desde proyecciones completada.")
        notes.extend(list(getattr(raw_res, "notes", []) or []))
        was_raw = True
    else:
        cube = np.asarray(study.cube, dtype=np.float64)
        if cube.ndim != 4:
            raise ValueError(f"Cubo reconstruido inválido: {cube.shape}")
        gated = cube
        ungated = np.mean(cube, axis=0)
        notes.append("Fuente reconstruida: ungated por promedio de gates.")
        was_raw = False
        if progress_callback is not None:
            progress_callback(0.75, "Volumen reconstruido cargado; generando cortes...")

    mode = str(cuts_mode or "mixed").strip().lower()
    is_gated = int(gated.shape[0]) >= 2
    if mode == "cardiac" and not is_gated:
        mode = "tomo"
        notes.append("Estudio no gatillado: modo cardíaco degradado a cortes tomográficos.")
    tomo_cuts: dict[str, np.ndarray] = {}
    cardiac_axes: dict[str, np.ndarray] = {}

    if mode in {"tomo", "mixed"}:
        if progress_callback is not None:
            progress_callback(0.92, "Generando cortes tomográficos...")
        zc = ungated.shape[0] // 2
        yc = ungated.shape[1] // 2
        xc = ungated.shape[2] // 2
        tomo_cuts = {
            "axial": np.asarray(ungated[zc], dtype=np.float64),
            "coronal": np.asarray(ungated[:, yc, :], dtype=np.float64),
            "sagittal": np.asarray(ungated[:, :, xc], dtype=np.float64),
        }
        notes.append("Cortes tomográficos centrales generados.")

    if mode in {"cardiac", "mixed"} and is_gated:
        if progress_callback is not None:
            progress_callback(0.96, "Generando cortes cardíacos SA/HLA/VLA...")
        orient = auto_orient_lv(gated, ungated_volume=ungated)
        if orient is not None:
            center = orient["center"]
            long_axis = np.asarray(orient["long_axis"], dtype=np.float64)
            out_size = int(min(gated.shape[1], 128))
            reo_g = reslice_from_vector_gated(gated, center, long_axis, out_size)
            cuts = anatomical_cuts_gated(reo_g)
            cardiac_axes = {
                "SA": np.asarray(cuts["sa"], dtype=np.float64),
                "HLA": np.asarray(cuts["hla"], dtype=np.float64),
                "VLA": np.asarray(cuts["vla"], dtype=np.float64),
            }
            notes.append(
                "Cortes cardíacos auto (SA/HLA/VLA) generados con reorientación automática."
            )
        else:
            notes.append("No se pudo estimar auto-orientación cardíaca; solo cortes tomográficos.")
    elif mode in {"cardiac", "mixed"}:
        notes.append("Estudio no gatillado: se omiten cortes cardíacos SA/HLA/VLA automáticos.")

    if progress_callback is not None:
        progress_callback(1.0, "Reconstrucción y cortes listos")

    spacing = _study_spacing_zyx(study)
    # Volumen sin post-filtro: si viene de raw_recon, usar la copia guardada; sino es igual al filtrado
    _unfiltered = getattr(raw_res, "ungated_volume_unfiltered", None) if was_raw else None
    return AmyloidReconstructionBundle(
        study=study,
        source_path=dicom_path,
        was_raw=was_raw,
        ungated_volume=ungated,
        gated_volume=gated,
        tomo_cuts=tomo_cuts,
        cardiac_axes=cardiac_axes,
        notes=notes,
        spacing_zyx=spacing,
        affine_ijk_to_lps=_affine_from_dicom_file(dicom_path, spacing),
        ungated_volume_unfiltered=_unfiltered,
    )


def export_amyloid_cardiac_axes_dicom(
    bundle: AmyloidReconstructionBundle,
    output_dir: str,
    *,
    base_name: str = "AMYLO_SPECT",
) -> dict[str, str]:
    """Exporta SA/HLA/VLA derivados AMYLO como DICOM multiframe."""
    axes = dict(bundle.cardiac_axes or {})
    if not axes:
        return {}
    st = (
        getattr(bundle.study, "slice_thickness_mm", None)
        or getattr(bundle.study, "z_spacing_mm", None)
        or None
    )
    return save_cardiac_axes_dicoms(
        axes,
        output_dir,
        source_study=bundle.study,
        base_name=base_name,
        slice_thickness_mm=st,
        extra_description="AMYLO SPECT",
    )


def apply_visual_bone_suppression(
    spect_volume: np.ndarray,
    *,
    ct_volume: np.ndarray | None = None,
    ct_hu_threshold: float = 200.0,
    spect_bone_percentile: float = 92.0,
    suppression_factor: float = 0.45,
) -> BoneSuppressionResult:
    """Sustracción ósea visual inicial (experimental).

    - Con CT: máscara ósea por umbral HU.
    - Sin CT: máscara probable ósea por percentil alto de actividad SPECT.
    """
    vol = np.asarray(spect_volume, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"Se esperaba volumen SPECT 3D, recibido {vol.shape}")

    notes: list[str] = []
    method = "spect_heuristic"

    if ct_volume is not None:
        ct = np.asarray(ct_volume, dtype=np.float64)
        if ct.shape == vol.shape:
            bone_mask = ct >= float(ct_hu_threshold)
            method = "ct_threshold"
            notes.append(f"Máscara ósea por CT (HU >= {ct_hu_threshold:.0f}).")
        else:
            bone_mask = vol >= np.percentile(vol, float(spect_bone_percentile))
            notes.append(
                "CT provisto con shape distinto al SPECT: fallback a heurística SPECT por percentil."
            )
    else:
        bone_mask = vol >= np.percentile(vol, float(spect_bone_percentile))
        notes.append(
            f"Sin CT: máscara ósea heurística por percentil SPECT >= {spect_bone_percentile:.1f}."
        )

    enhanced = vol.copy()
    enhanced[bone_mask] = enhanced[bone_mask] * max(0.0, 1.0 - float(suppression_factor))
    notes.append(f"Factor de supresión aplicado: {suppression_factor:.2f}.")

    return BoneSuppressionResult(
        enhanced_volume=enhanced,
        bone_mask=bone_mask.astype(np.uint8),
        method=method,
        notes=notes,
    )


def register_ct_to_spect_rigid(
    ct_volume: np.ndarray,
    spect_volume: np.ndarray,
    *,
    ct_spacing_zyx: tuple[float, float, float] | None = None,
    spect_spacing_zyx: tuple[float, float, float] | None = None,
    ct_affine_ijk_to_lps: np.ndarray | None = None,
    spect_affine_ijk_to_lps: np.ndarray | None = None,
    ct_bone_hu_threshold: float = 200.0,
    spect_focus_percentile: float = 85.0,
    refine_ncc: bool = True,
    ncc_search_radius_zyx: tuple[int, int, int] = (2, 4, 4),
) -> tuple[np.ndarray, tuple[float, float, float], list[str]]:
    """Registro rígido inicial CT→SPECT (experimental).

    Estrategia liviana para prototipo:
    1) Re-muestreo CT al shape del SPECT.
    2) Alineación por centros de masa de máscaras (hueso CT vs foco SPECT).
    3) Corrimiento traslacional 3D con interpolación lineal.
    """
    ct = np.asarray(ct_volume, dtype=np.float64)
    sp = np.asarray(spect_volume, dtype=np.float64)
    if ct.ndim != 3 or sp.ndim != 3:
        raise ValueError(f"CT y SPECT deben ser 3D. CT={ct.shape}, SPECT={sp.shape}")

    notes: list[str] = []
    if ct.shape != sp.shape:
        if ct_affine_ijk_to_lps is not None and spect_affine_ijk_to_lps is not None:
            ct_rs = _resample_ct_to_spect_affine(
                ct,
                sp.shape,
                np.asarray(ct_affine_ijk_to_lps, dtype=np.float64),
                np.asarray(spect_affine_ijk_to_lps, dtype=np.float64),
                fill_value=float(np.min(ct)),
            )
            if float(np.ptp(ct)) > 1e-6 and float(np.ptp(ct_rs)) <= 1e-6 and ct_spacing_zyx is not None and spect_spacing_zyx is not None:
                zoom_factors = tuple(
                    max(1e-6, float(ct_spacing_zyx[i])) / max(1e-6, float(spect_spacing_zyx[i]))
                    for i in range(3)
                )
                ct_phys = ndi.zoom(ct, zoom_factors, order=1)
                ct_rs = _center_crop_or_pad_3d(ct_phys, sp.shape, fill_value=float(np.min(ct)))
                notes.append(
                    "Geometría DICOM incompatible con la grilla reconstruida SPECT: "
                    "el remuestreo affine quedó vacío. Se aplicó fallback por espaciado físico "
                    f"{ct.shape} -> {ct_phys.shape} -> {ct_rs.shape}."
                )
            else:
                notes.append(
                    "CT remuestreado a grilla SPECT usando geometría DICOM completa "
                    f"(IPP/IOP/spacing): {ct.shape} -> {ct_rs.shape}."
                )
        elif ct_spacing_zyx is not None and spect_spacing_zyx is not None:
            zoom_factors = tuple(
                max(1e-6, float(ct_spacing_zyx[i])) / max(1e-6, float(spect_spacing_zyx[i]))
                for i in range(3)
            )
            ct_phys = ndi.zoom(ct, zoom_factors, order=1)
            ct_rs = _center_crop_or_pad_3d(ct_phys, sp.shape, fill_value=float(np.min(ct)))
            notes.append(
                "CT remuestreado por espaciado físico a grilla SPECT "
                f"shape {ct.shape} -> físico {ct_phys.shape} -> {ct_rs.shape}; "
                f"spacing CT z/y/x={ct_spacing_zyx}, SPECT z/y/x={spect_spacing_zyx}, zoom={zoom_factors}."
            )
        else:
            zoom_factors = (
                sp.shape[0] / max(ct.shape[0], 1),
                sp.shape[1] / max(ct.shape[1], 1),
                sp.shape[2] / max(ct.shape[2], 1),
            )
            ct_rs = ndi.zoom(ct, zoom_factors, order=1)
            notes.append(
                "CT remuestreado por shape a grilla SPECT "
                f"{ct.shape} -> {ct_rs.shape} (zoom={zoom_factors}; sin spacing físico disponible."
            )
    else:
        ct_rs = ct

    ct_mask = ct_rs >= float(ct_bone_hu_threshold)
    sp_thr = float(np.percentile(sp, float(spect_focus_percentile)))
    sp_mask = sp >= sp_thr

    if not np.any(ct_mask):
        ct_mask = ct_rs >= float(np.percentile(ct_rs, 85.0))
        notes.append("CT sin voxels sobre HU de hueso; fallback a percentil 85 CT.")
    if not np.any(sp_mask):
        sp_mask = sp >= float(np.percentile(sp, 75.0))
        notes.append("Máscara SPECT vacía; fallback a percentil 75 SPECT.")

    ct_pts = np.argwhere(ct_mask)
    sp_pts = np.argwhere(sp_mask)
    if ct_pts.size == 0 or sp_pts.size == 0:
        notes.append("Registro sin corrimiento: máscaras insuficientes.")
        return ct_rs, (0.0, 0.0, 0.0), notes

    ct_center = np.mean(ct_pts, axis=0)
    sp_center = np.mean(sp_pts, axis=0)
    shift_zyx = tuple((sp_center - ct_center).tolist())

    ct_reg = ndi.shift(ct_rs, shift=shift_zyx, order=1, mode="nearest")

    def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
        m = np.asarray(mask, dtype=bool)
        if np.count_nonzero(m) < 64:
            return -1.0
        av = np.asarray(a[m], dtype=np.float64)
        bv = np.asarray(b[m], dtype=np.float64)
        av = av - float(np.mean(av))
        bv = bv - float(np.mean(bv))
        den = float(np.linalg.norm(av) * np.linalg.norm(bv))
        if den < 1e-12:
            return -1.0
        return float(np.dot(av, bv) / den)

    if refine_ncc:
        rz, ry, rx = [max(0, int(v)) for v in ncc_search_radius_zyx]
        sp_thr2 = float(np.percentile(sp, max(70.0, float(spect_focus_percentile) - 10.0)))
        sp_feat = np.clip(sp - sp_thr2, 0.0, None)
        sp_feat = _safe_norm(sp_feat)
        base_bone = np.clip(ct_rs - float(ct_bone_hu_threshold), 0.0, None)
        base_bone = _safe_norm(base_bone)
        roi_mask = sp_feat > 0.05
        if not np.any(roi_mask):
            roi_mask = sp > np.percentile(sp, 75.0)

        best_score = -2.0
        best_shift = shift_zyx
        for dz in range(-rz, rz + 1):
            for dy in range(-ry, ry + 1):
                for dx in range(-rx, rx + 1):
                    cand_shift = (
                        float(shift_zyx[0] + dz),
                        float(shift_zyx[1] + dy),
                        float(shift_zyx[2] + dx),
                    )
                    ct_cand = ndi.shift(base_bone, shift=cand_shift, order=1, mode="nearest")
                    score = _ncc(ct_cand, sp_feat, roi_mask)
                    if score > best_score:
                        best_score = score
                        best_shift = cand_shift

        shift_zyx = best_shift
        ct_reg = ndi.shift(ct_rs, shift=shift_zyx, order=1, mode="nearest")
        notes.append(
            "Refinamiento NCC local aplicado "
            f"(radio z/y/x={rz}/{ry}/{rx}, score={best_score:.4f})."
        )

    notes.append(
        "Registro rígido traslacional aplicado "
        f"Δ(z,y,x)=({shift_zyx[0]:.2f},{shift_zyx[1]:.2f},{shift_zyx[2]:.2f})."
    )
    return ct_reg, shift_zyx, notes


def align_ct_orientation_to_spect(
    ct_volume: np.ndarray,
    spect_volume: np.ndarray,
    *,
    try_flip_x: bool = True,
    try_flip_y: bool = True,
    try_flip_z: bool = False,
    try_flip_xy: bool = True,
    try_rot90_inplane: bool = True,
    min_score_gain: float = 0.03,
    min_abs_score: float = 0.05,
) -> tuple[np.ndarray, dict[str, bool | int], list[str]]:
    """Ajuste de orientación CT→SPECT por evaluación de flips globales.

    Evalúa correlación normalizada entre máscara ósea CT (gradiente + altas HU)
    y foco SPECT para elegir la orientación que más coincide.
    """
    ct = np.asarray(ct_volume, dtype=np.float64)
    sp = np.asarray(spect_volume, dtype=np.float64)
    if ct.ndim != 3 or sp.ndim != 3:
        raise ValueError(f"CT y SPECT deben ser 3D. CT={ct.shape}, SPECT={sp.shape}")
    if ct.shape != sp.shape:
        raise ValueError(f"CT y SPECT deben tener misma grilla. CT={ct.shape}, SPECT={sp.shape}")

    notes: list[str] = []

    sp_feat = np.clip(sp - float(np.percentile(sp, 75.0)), 0.0, None)
    sp_feat = _safe_norm(sp_feat)
    roi = sp_feat > 0.05
    if np.count_nonzero(roi) < 64:
        roi = sp > float(np.percentile(sp, 70.0))
    sp_mask = sp > float(np.percentile(sp, 85.0))

    def _ct_feat(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        # mezcla simple: estructuras densas + bordes
        hu = np.clip(a - float(np.percentile(a, 70.0)), 0.0, None)
        gz, gy, gx = np.gradient(a)
        grad = np.sqrt(gz * gz + gy * gy + gx * gx)
        f = 0.7 * _safe_norm(hu) + 0.3 * _safe_norm(grad)
        return _safe_norm(f)

    def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
        m = np.asarray(mask, dtype=bool)
        if np.count_nonzero(m) < 64:
            return -1.0
        av = np.asarray(a[m], dtype=np.float64)
        bv = np.asarray(b[m], dtype=np.float64)
        av = av - float(np.mean(av))
        bv = bv - float(np.mean(bv))
        den = float(np.linalg.norm(av) * np.linalg.norm(bv))
        if den < 1e-12:
            return -1.0
        return float(np.dot(av, bv) / den)

    candidates: list[tuple[str, np.ndarray, dict[str, bool | int]]] = []

    rot_ks = [0, 1, 2, 3] if try_rot90_inplane else [0]
    flip_opts_xy: list[tuple[bool, bool]] = [(False, False)]
    if try_flip_x:
        flip_opts_xy.append((False, True))
    if try_flip_y:
        flip_opts_xy.append((True, False))
    if try_flip_xy and try_flip_x and try_flip_y:
        flip_opts_xy.append((True, True))

    for k in rot_ks:
        ct_rot = np.rot90(ct, k=int(k), axes=(1, 2)) if int(k) != 0 else ct
        for fy, fx in flip_opts_xy:
            arr = ct_rot
            if fy:
                arr = np.flip(arr, axis=1)
            if fx:
                arr = np.flip(arr, axis=2)
            name = f"rot{k*90}"
            if fy:
                name += "_flip_y"
            if fx:
                name += "_flip_x"
            candidates.append(
                (
                    name,
                    arr,
                    {
                        "flip_z": False,
                        "flip_y": bool(fy),
                        "flip_x": bool(fx),
                        "rot_k": int(k),
                    },
                )
            )

    if try_flip_z:
        z_augmented: list[tuple[str, np.ndarray, dict[str, bool | int]]] = []
        for name, arr, flags in candidates:
            z_augmented.append((name, arr, flags))
            flags_z = dict(flags)
            flags_z["flip_z"] = True
            z_augmented.append((name + "_flip_z", np.flip(arr, axis=0), flags_z))
        candidates = z_augmented

    best_name = "rot0"
    best_arr = ct
    best_flags: dict[str, bool | int] = {"flip_z": False, "flip_y": False, "flip_x": False, "rot_k": 0}
    best_score = -2.0
    cand_best: dict[str, tuple[float, np.ndarray, dict[str, bool | int]]] = {}
    scores: list[str] = []
    for name, arr, flags in candidates:
        # Reajuste traslacional por candidato para no sesgar la comparación.
        ct_mask = arr > float(np.percentile(arr, 85.0))
        if np.count_nonzero(ct_mask) >= 64 and np.count_nonzero(sp_mask) >= 64:
            ct_center = np.mean(np.argwhere(ct_mask), axis=0)
            sp_center = np.mean(np.argwhere(sp_mask), axis=0)
            shift_zyx = tuple((sp_center - ct_center).tolist())
            arr_aligned = ndi.shift(arr, shift=shift_zyx, order=1, mode="nearest")
        else:
            shift_zyx = (0.0, 0.0, 0.0)
            arr_aligned = arr

        sc = _ncc(_ct_feat(arr_aligned), sp_feat, roi)
        scores.append(
            f"{name}:{sc:.4f} Δ({float(shift_zyx[0]):.1f},{float(shift_zyx[1]):.1f},{float(shift_zyx[2]):.1f})"
        )
        cand_best[name] = (sc, np.asarray(arr_aligned, dtype=np.float64), flags)
        if sc > best_score:
            best_score = sc
            best_name = name
            best_arr = arr_aligned
            best_flags = flags

    notes.append("Auto-orient CT por NCC (candidatos): " + ", ".join(scores) + ".")
    ranked = sorted([(k, v[0]) for k, v in cand_best.items()], key=lambda kv: kv[1], reverse=True)
    second_score = float(ranked[1][1]) if len(ranked) > 1 else -2.0
    gain = float(best_score - second_score)

    if best_name != "none" and (best_score < float(min_abs_score) or gain < float(min_score_gain)):
        none_key = "rot0"
        none_sc, none_arr, none_flags = cand_best.get(none_key, (best_score, best_arr, best_flags))
        notes.append(
            "Auto-orient CT: decisión de flip con baja confianza "
            f"(best={best_score:.4f}, gain={gain:.4f}); se fuerza 'rot0' (score={none_sc:.4f})."
        )
        best_name = "rot0"
        best_score = float(none_sc)
        best_arr = none_arr
        best_flags = none_flags

    if best_name != "rot0":
        notes.append(
            f"Auto-orient CT: aplicado {best_name} "
            f"(score={best_score:.4f}, gain={gain:.4f})."
        )
    else:
        notes.append(
            f"Auto-orient CT: sin flip adicional "
            f"(score={best_score:.4f}, gain={gain:.4f})."
        )

    return np.asarray(best_arr, dtype=np.float64), best_flags, notes


def refine_ct_to_spect_translation(
    ct_volume: np.ndarray,
    spect_volume: np.ndarray,
    *,
    search_radius_zyx: tuple[int, int, int] = (3, 8, 8),
    ct_bone_hu_threshold: float = 200.0,
    spect_focus_percentile: float = 85.0,
) -> tuple[np.ndarray, tuple[float, float, float], list[str]]:
    """Refina traslación CT→SPECT por NCC local en grilla ya orientada.

    Espera CT y SPECT en misma grilla/shape y devuelve un corrimiento incremental
    (delta) respecto de la posición CT de entrada.
    """
    ct = np.asarray(ct_volume, dtype=np.float64)
    sp = np.asarray(spect_volume, dtype=np.float64)
    if ct.ndim != 3 or sp.ndim != 3:
        raise ValueError(f"CT y SPECT deben ser 3D. CT={ct.shape}, SPECT={sp.shape}")
    if ct.shape != sp.shape:
        raise ValueError(f"CT y SPECT deben tener misma grilla. CT={ct.shape}, SPECT={sp.shape}")

    notes: list[str] = []
    rz, ry, rx = [max(0, int(v)) for v in search_radius_zyx]
    if rz == 0 and ry == 0 and rx == 0:
        return ct.copy(), (0.0, 0.0, 0.0), notes

    sp_thr = float(np.percentile(sp, max(70.0, float(spect_focus_percentile) - 10.0)))
    sp_feat = _safe_norm(np.clip(sp - sp_thr, 0.0, None))
    roi_mask = sp_feat > 0.05
    if not np.any(roi_mask):
        roi_mask = sp > np.percentile(sp, 75.0)

    ct_feat_base = _safe_norm(np.clip(ct - float(ct_bone_hu_threshold), 0.0, None))
    if float(np.max(ct_feat_base)) <= 0.0:
        ct_feat_base = _safe_norm(np.clip(ct - float(np.percentile(ct, 80.0)), 0.0, None))

    def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
        m = np.asarray(mask, dtype=bool)
        if np.count_nonzero(m) < 64:
            return -1.0
        av = np.asarray(a[m], dtype=np.float64)
        bv = np.asarray(b[m], dtype=np.float64)
        av = av - float(np.mean(av))
        bv = bv - float(np.mean(bv))
        den = float(np.linalg.norm(av) * np.linalg.norm(bv))
        if den < 1e-12:
            return -1.0
        return float(np.dot(av, bv) / den)

    # Búsqueda en dos pasos: gruesa (step=2) + fina local (step=1).
    best_score = -2.0
    best_shift = (0.0, 0.0, 0.0)

    for dz in range(-rz, rz + 1, 2 if rz >= 2 else 1):
        for dy in range(-ry, ry + 1, 2 if ry >= 2 else 1):
            for dx in range(-rx, rx + 1, 2 if rx >= 2 else 1):
                cand = (float(dz), float(dy), float(dx))
                ct_cand = ndi.shift(ct_feat_base, shift=cand, order=1, mode="nearest")
                sc = _ncc(ct_cand, sp_feat, roi_mask)
                if sc > best_score:
                    best_score = sc
                    best_shift = cand

    bz, by, bx = [int(round(v)) for v in best_shift]
    for dz in range(max(-rz, bz - 1), min(rz, bz + 1) + 1):
        for dy in range(max(-ry, by - 1), min(ry, by + 1) + 1):
            for dx in range(max(-rx, bx - 1), min(rx, bx + 1) + 1):
                cand = (float(dz), float(dy), float(dx))
                ct_cand = ndi.shift(ct_feat_base, shift=cand, order=1, mode="nearest")
                sc = _ncc(ct_cand, sp_feat, roi_mask)
                if sc > best_score:
                    best_score = sc
                    best_shift = cand

    ct_refined = ndi.shift(ct, shift=best_shift, order=1, mode="nearest")
    notes.append(
        "Refinamiento fino CT↔SPECT por NCC aplicado "
        f"(radio z/y/x={rz}/{ry}/{rx}, Δ=({best_shift[0]:.1f},{best_shift[1]:.1f},{best_shift[2]:.1f}), score={best_score:.4f})."
    )
    return ct_refined, best_shift, notes


def central_slices_preview(volume: np.ndarray) -> dict[str, np.ndarray]:
    """Devuelve cortes centrales axial/coronal/sagital normalizados 0..1."""
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"Se esperaba volumen 3D, recibido {vol.shape}")

    zc = vol.shape[0] // 2
    yc = vol.shape[1] // 2
    xc = vol.shape[2] // 2

    axial = _safe_norm(vol[zc])
    coronal = _safe_norm(vol[:, yc, :])
    sagittal = _safe_norm(vol[:, :, xc])

    return {
        "axial": axial,
        "coronal": coronal,
        "sagittal": sagittal,
    }


# =============================================================================
# HMR-SPECT: Heart-to-Mediastinum Ratio en SPECT 3D
# =============================================================================

@dataclass
class VOISphere:
    """VOI (Volume of Interest) esférica 3D en coordenadas ZYX (índices)."""
    
    cz: float      # centro Z (axial)
    cy: float      # centro Y (coronal)
    cx: float      # centro X (sagittal)
    radius_mm: float
    
    def mask_3d(self, shape: tuple[int, int, int], spacing_zyx: tuple[float, float, float]) -> np.ndarray:
        """Genera máscara 3D con radio en mm."""
        nz, ny, nx = shape
        sz, sy, sx = spacing_zyx
        
        zz, yy, xx = np.meshgrid(
            np.arange(nz) * sz,
            np.arange(ny) * sy,
            np.arange(nx) * sx,
            indexing='ij'
        )
        
        dist = np.sqrt(
            (zz - self.cz * sz) ** 2 +
            (yy - self.cy * sy) ** 2 +
            (xx - self.cx * sx) ** 2
        )
        
        return dist <= self.radius_mm
    
    def mask_slice(self, z_idx: int, shape_2d: tuple[int, int], spacing_yx: tuple[float, float]) -> np.ndarray:
        """Genera máscara 2D para un slice axial específico."""
        ny, nx = shape_2d
        sy, sx = spacing_yx
        
        yy, xx = np.meshgrid(np.arange(ny) * sy, np.arange(nx) * sx, indexing='ij')
        dist = np.sqrt((yy - self.cy * sy) ** 2 + (xx - self.cx * sx) ** 2)
        
        return dist <= self.radius_mm
    
    def volume_ml(self) -> float:
        """Calcula volumen de la esfera en mL."""
        vol_mm3 = (4.0 / 3.0) * np.pi * (self.radius_mm ** 3)
        return vol_mm3 / 1000.0
    
    def get_circle_params_axial(self, spacing_yx: tuple[float, float]) -> tuple[float, float, float]:
        """Retorna (cy, cx, radius_px) para dibujar círculo en vista axial."""
        sy, sx = spacing_yx
        radius_px = self.radius_mm / sx  # Asumiendo pixels isotrópicos en YX
        return (self.cy, self.cx, radius_px)
    
    def get_circle_params_coronal(self, spacing_zx: tuple[float, float]) -> tuple[float, float, float]:
        """Retorna (cz, cx, radius_px) para dibujar círculo en vista coronal."""
        sz, sx = spacing_zx
        radius_px = self.radius_mm / max(sz, sx)
        return (self.cz, self.cx, radius_px)
    
    def get_circle_params_sagittal(self, spacing_zy: tuple[float, float]) -> tuple[float, float, float]:
        """Retorna (cz, cy, radius_px) para dibujar círculo en vista sagittal."""
        sz, sy = spacing_zy
        radius_px = self.radius_mm / max(sz, sy)
        return (self.cz, self.cy, radius_px)


# =============================================================================
# VOI ANATÓMICA — Fase 2: ROI conformada a la anatomía del miocardio desde CT
# =============================================================================

@dataclass
class VOIAnatomical:
    """VOI definida por una máscara binaria 3D (ej: miocardio segmentado desde CT).
    
    Reemplaza la esfera genérica por una ROI que sigue exactamente la forma
    anatómica del miocardio, eliminando spill-in de cavidad y tejido adyacente.
    """
    mask_3d_data: np.ndarray          # Máscara binaria bool/uint8 3D (nz, ny, nx)
    centroid_zyx: tuple[float, float, float]  # Centroide para referencia visual
    source: str = "ct_segmentation"   # Origen de la máscara
    volume_mm3: float = 0.0           # Volumen físico en mm³ (si se conoce)
    
    def __post_init__(self):
        arr = np.asarray(self.mask_3d_data)
        # Asegurar que sea escalar 0/1 o bool, no un objeto raro
        if arr.ndim > 0:
            self.mask_3d_data = arr.astype(bool)
        else:
            self.mask_3d_data = np.atleast_1d(arr).astype(bool)
    
    def mask_3d(self, shape: tuple[int, int, int] | None = None,
                spacing_zyx: tuple[float, float, float] | None = None) -> np.ndarray:
        """Retorna la máscara 3D. Compatible interfaz VOISphere."""
        # Forzar que mask_3d_data sea ndarray bool válido
        if not isinstance(self.mask_3d_data, np.ndarray):
            self.mask_3d_data = np.asarray(self.mask_3d_data, dtype=bool)
        if self.mask_3d_data.ndim != 3:
            # Si no es 3D, intentar reshape o retornar vacío con shape solicitado
            if shape is not None:
                return np.zeros(shape, dtype=bool)
            return np.zeros((1,1,1), dtype=bool)
        if shape is not None and tuple(self.mask_3d_data.shape) != tuple(shape):
            # Resize si las dimensiones no coinciden (raro pero posible)
            from scipy.ndimage import zoom
            factors = [s / m for s, m in zip(shape, self.mask_3d_data.shape)]
            return zoom(self.mask_3d_data.astype(np.float32), order=0) > 0.5
        return self.mask_3d_data.copy()
    
    def mask_slice(self, z_idx: int, shape_2d: tuple[int, int] | None = None,
                   spacing_yx: tuple[float, float] | None = None) -> np.ndarray:
        """Extrae máscara 2D para un slice axial. Compatible interfaz VOISphere."""
        z_idx = int(np.clip(z_idx, 0, self.mask_3d_data.shape[0] - 1))
        return self.mask_3d_data[z_idx].copy()
    
    def volume_ml(self) -> float:
        """Volumen en mL. Si se proporcionó volume_mm3, lo usa; sino cuenta voxels."""
        if self.volume_mm3 > 0:
            return self.volume_mm3 / 1000.0
        # Fallback: contar voxels (sin spacing, aproximado)
        return float(self.mask_3d_data.sum())  # voxel count
    
    @property
    def cz(self) -> float:
        return self.centroid_zyx[0]
    
    @property
    def cy(self) -> float:
        return self.centroid_zyx[1]
    
    @property
    def cx(self) -> float:
        return self.centroid_zyx[2]
    
    def _find_contour_2d(self, mask_2d: np.ndarray) -> np.ndarray | None:
        """Extrae contorno mayor de una máscara 2D usando OpenCV.
        
        Args:
            mask_2d: Máscara booleana/uint8 2D (ny, nx) o (nz, nx) o (nz, ny).
            
        Returns:
            Array (N,2) de coordenadas (row, col), o None si está vacío.
        """
        try:
            if not bool(np.any(mask_2d)):
                return None
            import cv2
            m8 = mask_2d.astype(np.uint8)
            contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            biggest = max(contours, key=cv2.contourArea)
            pts = biggest.reshape(-1, 2)  # (N, 2) formato OpenCV (x=col, y=row)
            return pts[:, ::-1].astype(np.float64)  # → (row, col)
        except Exception:
            return None

    def get_contour_for_slice(self, z_idx: int) -> np.ndarray | None:
        """Retorna contorno ORDENADO (N,2) para dibujar overlay en slice axial.
        
        Usa cv2.findContours para obtener los puntos en orden perimetral
        (necesario para drawPolygon; argwhere daría orden de barrido → caos).
        
        Returns:
            Array (N,2) de coordenadas (row, col) del contorno mayor, o None si vacío.
        """
        try:
            slice_mask = self.mask_slice(z_idx)
            return self._find_contour_2d(slice_mask)
        except Exception:
            return None

    def get_contour_for_coronal(self, y_idx: int) -> np.ndarray | None:
        """Retorna contorno (N,2) para un slice coronal (eje Y fijo).
        
        El slice coronal corta el volumen en Y, dando una imagen (Z, X).
        Las coordenadas retornadas son (z, x) para mapeo a pantalla.
        
        Returns:
            Array (N,2) de (z, x), o None si vacío.
        """
        try:
            y_idx = int(np.clip(y_idx, 0, self.mask_3d_data.shape[1] - 1))
            slice_mask = self.mask_3d_data[:, y_idx, :]  # (nz, nx)
            return self._find_contour_2d(slice_mask)
        except Exception:
            return None

    def get_contour_for_sagittal(self, x_idx: int) -> np.ndarray | None:
        """Retorna contorno (N,2) para un slice sagital (eje X fijo).
        
        El slice sagital corta el volumen en X, dando una imagen (Z, Y).
        Las coordenadas retornadas son (z, y) para mapeo a pantalla.
        
        Returns:
            Array (N,2) de (z, y), o None si vacío.
        """
        try:
            x_idx = int(np.clip(x_idx, 0, self.mask_3d_data.shape[2] - 1))
            slice_mask = self.mask_3d_data[:, :, x_idx]  # (nz, ny)
            return self._find_contour_2d(slice_mask)
        except Exception:
            return None


def create_anatomical_heart_voi(
    ct_segmentation: MyocardialSegmentationResult,
    spect_shape: tuple[int, int, int],
    spect_spacing: tuple[float, float, float],
    ct_spacing: tuple[float, float, float],
) -> VOIAnatomical:
    """Crea una VOI anatómica del corazón a partir de la segmentación CT.
    
    La máscara CT (alta resolución) se resamplea al espacio SPECT para que
    cada voxel SPECT tenga su correspondiente etiqueta anatómica.
    
    Args:
        ct_segmenting: Resultado de segment_myocardium_from_ct()
        spect_shape: Shape del volumen SPECT (nz, ny, nx)
        spect_spacing: Spacing del SPECT (sz, sy, sx) en mm
        ct_spacing: Spacing del CT (sz_ct, sy_ct, sx_ct) en mm
        
    Returns:
        VOIAnatomical lista para usar en compute_hmr_spect()
    """
    from scipy.ndimage import zoom
    
    ct_mask = ct_segmentation.mask_3d  # (nz_ct, ny_ct, nx_ct)
    
    if ct_mask.shape == spect_shape:
        # Misma resolución (caso raro)
        resampled = ct_mask.astype(bool)
    else:
        # Resamplear CT → espacio SPECT
        zoom_factors = [
            spect_shape[i] / ct_mask.shape[i] for i in range(3)
        ]
        # Usar orden 0 (nearest neighbor) para mantener binario
        resampled = zoom(ct_mask.astype(np.float32), zoom_factors, order=0) > 0.5
    
    # Asegurar shape correcto (clip/pad si hay diferencias de redondeo)
    if resampled.shape != spect_shape:
        target = np.zeros(spect_shape, dtype=bool)
        slices_t = tuple(slice(0, min(resampled.shape[d], spect_shape[d])) for d in range(3))
        target[slices_t] = resampled[slices_t]
        resampled = target
    
    return VOIAnatomical(
        mask_3d_data=resampled,
        centroid_zyx=ct_segmentation.centroid_zyx,
        source="ct_myocardium",
        volume_mm3=ct_segmentation.volume_mm3,
    )


def create_bone_safe_mediastinum_voi(
    ct_volume: np.ndarray,
    ct_spacing: tuple[float, float, float],
    mediastinum_center_zyx: tuple[float, float, float],
    mediastinum_radius_mm: float,
    spect_shape: tuple[int, int, int],
    spect_spacing: tuple[float, float, float],
    bone_hu_threshold: float = 400.0,
) -> VOIAnatomical:
    """Crea VOI de mediastino que evita hueso (costillas, esternón).
    
    El problema clásico: una VOI esférica de mediastino puede incluir parte
    de una costilla, inflando artificialmente las cuentas y subestimando HMR.
    
    Esta función usa el CT para crear una máscara que excluye regiones óseas.
    
    Args:
        ct_volume: Volumen CT en HU
        ct_spacing: Spacing CT (sz, sy, sx)
        mediastinum_center_zyx: Centro deseado del VOI mediastino
        mediastinum_radius_mm: Radio de la esfera base
        spect_shape: Shape del volumen SPECT destino
        spect_spacing: Spacing del SPECT
        bone_hu_threshold: Umbral HU para detectar hueso
        
    Returns:
        VOIAnatomical con máscara libre de hueso
    """
    from scipy.ndimage import zoom, binary_dilation, binary_erosion
    
    ct_vol = np.asarray(ct_volume, dtype=np.float64)
    nz_ct, ny_ct, nx_ct = ct_vol.shape
    sz_ct, sy_ct, sx_ct = ct_spacing
    
    # Crear esfera base en espacio CT
    zz, yy, xx = np.meshgrid(
        np.arange(nz_ct) * sz_ct,
        np.arange(ny_ct) * sy_ct,
        np.arange(nx_ct) * sx_ct,
        indexing='ij'
    )
    cz, cy, cx = mediastinum_center_zyx
    dist = np.sqrt((zz - cz * sz_ct)**2 + (yy - cy * sy_ct)**2 + (xx - cx * sx_ct)**2)
    sphere_ct = dist <= mediastinum_radius_mm
    
    # Máscara de hueso (HU > threshold)
    bone_mask = ct_vol > bone_hu_threshold
    
    # Dilatar hueso ligeramente para crear margen de seguridad (2-3mm)
    bone_margin_mm = 3.0
    bone_dilated = bone_mask
    for d in range(3):
        iterations = max(1, int(round(bone_margin_mm / ct_spacing[d])))
        if iterations > 0:
            bone_dilated = binary_dilation(bone_dilated, iterations=iterations)
    
    # VOI final = esfera AND NOT hueso
    mediastinum_ct = sphere_ct & ~bone_dilated
    
    # Si quedó muy pequeño o vacío, volver a la esfera sin restricción
    if float(mediastinum_ct.sum()) < float(sphere_ct.sum()) * 0.3:
        mediastinum_ct = sphere_ct.copy()
    
    # Resamplear CT → espacio SPECT
    if mediastinum_ct.shape != spect_shape:
        zoom_factors = [spect_shape[i] / mediastinum_ct.shape[i] for i in range(3)]
        resampled = zoom(mediastinum_ct.astype(np.float32), zoom_factors, order=0) > 0.5
    else:
        resampled = mediastinum_ct.astype(bool)
    
    # Clip/pad si necesario
    if resampled.shape != spect_shape:
        target = np.zeros(spect_shape, dtype=bool)
        slices_t = tuple(slice(0, min(resampled.shape[d], spect_shape[d])) for d in range(3))
        target[slices_t] = resampled[slices_t]
        resampled = target
    
    return VOIAnatomical(
        mask_3d_data=resampled,
        centroid_zyx=mediastinum_center_zyx,
        source="ct_bonesafe_mediastinum",
    )


@dataclass
class HmrSpectResult:
    """Resultado del cálculo HMR en SPECT 3D."""
    
    hmr: float  # HMR sobre volumen filtrado (el que se visualiza)
    hmr_raw: float | None = None  # HMR sobre volumen sin filtrar (el que vale clínicamente)
    heart_counts: float = 0.0
    mediastinum_counts: float = 0.0
    heart_counts_raw: float = 0.0
    mediastinum_counts_raw: float = 0.0
    heart_pixels: int = 0  # Número de píxeles en VOI corazón
    mediastinum_pixels: int = 0  # Número de píxeles en VOI mediastino
    heart_mean: float = 0.0  # Cuentas promedio por píxel
    mediastinum_mean: float = 0.0  # Cuentas promedio por píxel
    heart_volume_ml: float = 0.0
    mediastinum_volume_ml: float = 0.0
    voi_heart: VOISphere | None = None
    voi_mediastinum: VOISphere | None = None
    method: str = ""  # "VOI completa" o "ROI slice central"
    slice_idx: int | None = None
    
    @property
    def classification(self) -> str:
        # Clasificación basada en HMR raw (el clínicamente relevante)
        # HMR ALTO = mucha captación cardíaca = POSITIVO para amiloidosis
        hmr = self.hmr_raw if self.hmr_raw is not None else self.hmr
        if hmr >= 1.6:
            return "POSITIVO"
        if hmr >= 1.5:
            return "EQUIVOCO"
        return "NEGATIVO"
    
    @property
    def hmr_text(self) -> str:
        return f"HMR-SPECT = {self.hmr:.2f} ({self.classification})"


class HmrSpectMethod(enum.Enum):
    """Métodos de cálculo HMR-SPECT."""
    VOI_COMPLETE = "VOI completa"
    SLICE_CENTRAL = "ROI slice central"


def compute_hmr_spect(
    volume: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    voi_heart: VOISphere | VOIAnatomical,
    voi_mediastinum: VOISphere | VOIAnatomical,
    method: HmrSpectMethod = HmrSpectMethod.VOI_COMPLETE,
    slice_idx: int | None = None,
    volume_raw: np.ndarray | None = None,
) -> HmrSpectResult:
    """Calcula HMR-SPECT = media VOI corazón / media VOI mediastino.
    
    Args:
        volume: Volumen SPECT 3D filtrado (nz, ny, nx) - el que se visualiza
        spacing_zyx: Espaciado en mm (sz, sy, sx)
        voi_heart: VOI esférica del corazón
        voi_mediastinum: VOI esférica del mediastino
        method: Método de cálculo
            - VOI_COMPLETE: Integra cuentas de toda la esfera 3D
            - SLICE_CENTRAL: Usa solo el slice axial especificado
        slice_idx: Slice axial para método SLICE_CENTRAL (default: centro del corazón)
        volume_raw: Volumen SPECT sin filtrar (opcional, para HMR raw)
        
    Returns:
        HmrSpectResult con HMR filtrado y HMR raw (si se proporciona volume_raw)
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"Se esperaba volumen 3D, recibido {vol.shape}")
    
    vol_raw = None
    if volume_raw is not None:
        vol_raw = np.asarray(volume_raw, dtype=np.float64)
        if vol_raw.shape != vol.shape:
            vol_raw = None  # Ignorar si no coincide
    
    if method == HmrSpectMethod.VOI_COMPLETE:
        mask_h = voi_heart.mask_3d(vol.shape, spacing_zyx)
        mask_m = voi_mediastinum.mask_3d(vol.shape, spacing_zyx)
        
        heart_counts = float(vol[mask_h].sum())
        mediastinum_counts = float(vol[mask_m].sum())
        
        heart_counts_raw = float(vol_raw[mask_h].sum()) if vol_raw is not None else 0.0
        mediastinum_counts_raw = float(vol_raw[mask_m].sum()) if vol_raw is not None else 0.0
        
        used_slice = None
    else:
        if slice_idx is None:
            slice_idx = int(round(voi_heart.cz))
        slice_idx = int(np.clip(slice_idx, 0, vol.shape[0] - 1))
        
        slice_2d = vol[slice_idx]

        # Cortar las máscaras 3D conserva la distancia al centro en Z. Usar
        # mask_slice() directamente convertía toda esfera en un círculo de radio
        # máximo, incluso cuando su centro estaba muy lejos del corte elegido.
        mask_h = voi_heart.mask_3d(vol.shape, spacing_zyx)[slice_idx]
        mask_m = voi_mediastinum.mask_3d(vol.shape, spacing_zyx)[slice_idx]
        
        heart_counts = float(slice_2d[mask_h].sum())
        mediastinum_counts = float(slice_2d[mask_m].sum())
        
        if vol_raw is not None:
            slice_2d_raw = vol_raw[slice_idx]
            heart_counts_raw = float(slice_2d_raw[mask_h].sum())
            mediastinum_counts_raw = float(slice_2d_raw[mask_m].sum())
        else:
            heart_counts_raw = 0.0
            mediastinum_counts_raw = 0.0
        
        used_slice = slice_idx
    
    # Calcular estadísticas de diagnóstico
    heart_pixels = int(mask_h.sum())
    mediastinum_pixels = int(mask_m.sum())
    if heart_pixels == 0:
        raise ValueError("La VOI del corazón no intersecta el volumen/corte seleccionado.")
    if mediastinum_pixels == 0:
        raise ValueError(
            "La VOI del mediastino no intersecta el volumen/corte seleccionado. "
            "Reubique el punto B al mismo nivel axial o use VOI completa."
        )

    heart_mean = heart_counts / max(heart_pixels, 1) if heart_pixels > 0 else 0.0
    mediastinum_mean = mediastinum_counts / max(mediastinum_pixels, 1) if mediastinum_pixels > 0 else 0.0
    signal_floor = max(float(np.nanmax(np.abs(vol))) * 1e-4, 1e-8)
    if not np.isfinite(mediastinum_mean) or mediastinum_mean <= signal_floor:
        raise ValueError(
            "La VOI del mediastino tiene señal nula o insuficiente. "
            "Revise el punto B y confirme que esté dentro del tórax, no en el fondo."
        )

    hmr = heart_mean / mediastinum_mean
    hmr_raw = None
    if vol_raw is not None:
        heart_mean_raw = heart_counts_raw / heart_pixels
        mediastinum_mean_raw = mediastinum_counts_raw / mediastinum_pixels
        raw_floor = max(float(np.nanmax(np.abs(vol_raw))) * 1e-4, 1e-8)
        if np.isfinite(mediastinum_mean_raw) and mediastinum_mean_raw > raw_floor:
            hmr_raw = heart_mean_raw / mediastinum_mean_raw
    
    return HmrSpectResult(
        hmr=hmr,
        hmr_raw=hmr_raw,
        heart_counts=heart_counts,
        mediastinum_counts=mediastinum_counts,
        heart_counts_raw=heart_counts_raw,
        mediastinum_counts_raw=mediastinum_counts_raw,
        heart_pixels=heart_pixels,
        mediastinum_pixels=mediastinum_pixels,
        heart_mean=heart_mean,
        mediastinum_mean=mediastinum_mean,
        heart_volume_ml=voi_heart.volume_ml(),
        mediastinum_volume_ml=voi_mediastinum.volume_ml(),
        voi_heart=voi_heart,
        voi_mediastinum=voi_mediastinum,
        method=method.value,
        slice_idx=used_slice,
    )


# =============================================================================
# CORRECCIÓN DE EFECTO DE VOLUMEN PARCIAL (PVE) — Fase 1
# =============================================================================
# El PVE subestima la actividad en estructuras pequeñas (pared miocárdica ~10-14 mm)
# cuando la resolución SPECT (FWHM ~10-15 mm) es comparable al tamaño del objeto.
# El CT de alta resolución permite: segmentar el miocardio, medir grosor por segmento,
# y aplicar coeficientes de recuperación (RC) para corregir el HMR.
# =============================================================================

@dataclass
class MyocardialSegmentationResult:
    """Resultado de la segmentación miocárdica desde CT."""
    
    mask_3d: np.ndarray                    # Máscara binaria 3D del miocardio
    mask_axial_slices: list[np.ndarray]   # Lista de máscaras 2D por slice axial
    centroid_zyx: tuple[float, float, float]  # Centroide de la máscara
    volume_mm3: float                      # Volumen segmentado en mm³
    n_slices: int                          # Número de slices axiales con tejido
    wall_thickness_mm: dict[str, float]    # Grosor por segmento AHA simplificado
    mean_wall_thickness_mm: float          # Grosor promedio ponderado
    notes: list[str]                       # Notas del proceso


def segment_myocardium_from_ct(
    ct_volume: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    seed_zyx: tuple[float, float, float] | None = None,
    seed_radius_mm: float = 50.0,
    hu_min: float = -100.0,
    hu_max: float = 300.0,
    min_volume_mm3: float = 5000.0,
    max_volume_mm3: float = 500000.0,
    dilation_mm: float = 3.0,
    erosion_mm: float = 2.0,
) -> MyocardialSegmentationResult:
    """Segmenta el miocardio del ventrículo izquierdo desde un volumen CT.

    Estrategia:
    1. Si hay semilla (seed_zyx), recortar el volumen a una caja 3D alrededor
       de la semilla antes de cualquier procesamiento.  Esto evita que la
       componente conexa mayor sea toda la caja torácica.
    2. Umbralizado HU para incluir miocardio, sangre pool y paredes.
    3. Operaciones morfológicas (dilatación + erosión) para limpiar.
    4. Componente conexa 3D más grande dentro de la caja (asume que el VI es
       la estructura mayor en esa región restringida).
    5. Refinamiento con umbral más ajustado dentro de la máscara.

    Args:
        ct_volume: Volumen CT en HU (nz, ny, nx)
        spacing_zyx: Espaciado físico (sz, sy, sx) en mm
        seed_zyx: Semilla (z, y, x) en coordenadas de voxel del volumen CT.
            Típicamente el ancla A que el usuario coloca sobre el corazón.
            Si es None, segmenta todo el volumen (comportamiento legacy).
        seed_radius_mm: Radio de la caja 3D alrededor de la semilla (mm).
        hu_min/max: Rango HU inicial para tejido blando cardíaco
        min/max_volume_mm3: Filtros de volumen plausible para VI
        dilation_mm: Dilatación morfológica previa (cierra huecos)
        erosion_mm: Erosión morfológica posterior (separa de otras estructuras)

    Returns:
        MyocardialSegmentationResult con máscara y métricas
    """
    ct = np.asarray(ct_volume, dtype=np.float64)
    sz, sy, sx = spacing_zyx
    notes: list[str] = []

    # Paso 0: Restringir a una caja 3D alrededor de la semilla si está dada.
    # Esto evita que la componente conexa mayor sea todo el tórax/hígado.
    bbox = None  # (z0, z1, y0, y1, x0, x1) en coords del volumen completo
    if seed_zyx is not None:
        sz_vox = max(1, int(round(seed_radius_mm / sz)))
        sy_vox = max(1, int(round(seed_radius_mm / sy)))
        sx_vox = max(1, int(round(seed_radius_mm / sx)))
        cz, cy, cx = float(seed_zyx[0]), float(seed_zyx[1]), float(seed_zyx[2])
        z0 = max(0, int(cz) - sz_vox)
        z1 = min(ct.shape[0], int(cz) + sz_vox + 1)
        y0 = max(0, int(cy) - sy_vox)
        y1 = min(ct.shape[1], int(cy) + sy_vox + 1)
        x0 = max(0, int(cx) - sx_vox)
        x1 = min(ct.shape[2], int(cx) + sx_vox + 1)
        bbox = (z0, z1, y0, y1, x0, x1)
        ct_crop = ct[z0:z1, y0:y1, x0:x1]
        notes.append(
            f"Semilla ({cz:.1f},{cy:.1f},{cx:.1f}) → caja "
            f"[{z0}:{z1},{y0}:{y1},{x0}:{x1}] "
            f"({z1-z0}x{y1-y0}x{x1-x0} voxels, r={seed_radius_mm:.0f}mm)"
        )
    else:
        ct_crop = ct
        notes.append("Sin semilla: segmentación global (legacy)")

    # Paso 1: Umbralizado inicial — tejido blando cardíaco (sobre la caja)
    soft_mask = (ct_crop >= hu_min) & (ct_crop <= hu_max)
    notes.append(f"Umbral HU inicial: [{hu_min}, {hu_max}] → {soft_mask.sum()} voxels")
    
    if not np.any(soft_mask):
        notes.append("WARNING: Umbral inicial vacío. Expandiendo rango a [-200, 500].")
        soft_mask = (ct_crop >= -200.0) & (ct_crop <= 500.0)
    
    # Paso 2: Morfología — cerrar pequeños huecos, separar estructuras adyacentes
    dil_iter = max(1, int(round(dilation_mm / min(sy, sx))))
    ero_iter = max(1, int(round(erosion_mm / min(sy, sx))))
    
    from scipy.ndimage import binary_dilation, binary_erosion, binary_closing, label
    
    struct = ndi.generate_binary_structure(3, 1)  # Conectividad 6 (cruz 3D)
    
    closed = binary_closing(soft_mask, structure=struct, iterations=dil_iter)
    dilated = binary_dilation(closed, structure=struct, iterations=dil_iter)
    eroded = binary_erosion(dilated, structure=struct, iterations=ero_iter)
    notes.append(f"Morfología: close({dil_iter})→dilate({dil_iter})→erode({ero_iter})")
    
    # Paso 3: Componente conexa 3D más grande dentro de la caja
    labeled, n_features = label(eroded)
    if n_features == 0:
        notes.append("ERROR: Sin componentes conexos. Retornando máscara vacía.")
        return MyocardialSegmentationResult(
            mask_3d=np.zeros(ct.shape, dtype=bool),
            mask_axial_slices=[],
            centroid_zyx=(0.0, 0.0, 0.0),
            volume_mm3=0.0,
            n_slices=0,
            wall_thickness_mm={},
            mean_wall_thickness_mm=0.0,
            notes=notes,
        )
    
    # Encontrar componente mayor dentro de la caja
    component_sizes = np.bincount(labeled.ravel())[1:]  # Ignorar fondo (0)
    largest_label = int(np.argmax(component_sizes)) + 1
    major_component_crop = labeled == largest_label
    notes.append(f"Componentes 3D: {n_features}. Seleccionado el mayor (label={largest_label}, {component_sizes[largest_label-1]} voxels)")
    
    # Paso 4: Refinar dentro del componente mayor — umbral más estricto para miocardio
    refined_crop = major_component_crop.copy()
    
    # ── MEJORA v2.7: Detección adaptativa de HU del miocardio ─────
    # Analizar el histograma DENTRO del componente mayor para encontrar
    # el pico de tejido miocárdico real.
    ct_inside = ct_crop[major_component_crop]
    if ct_inside.size > 200:
        hist_bins = np.linspace(-200, 400, 121)  # 5 HU por bin
        hist_counts, bin_edges = np.histogram(ct_inside, bins=hist_bins)
        valid_range = (bin_edges[:-1] >= -150) & (bin_edges[:-1] <= 300)
        valid_counts = hist_counts.copy()
        valid_counts[~valid_range] = 0
        if valid_counts.max() > 0:
            peak_bin = np.argmax(valid_counts)
            peak_hu = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2
            adaptive_min = max(-50, peak_hu - 60)
            adaptive_min = min(adaptive_min, 20)
            adaptive_max = min(350, peak_hu + 120)
            adaptive_max = max(adaptive_max, 150)
            myocardium_hu_min, myocardium_hu_max = adaptive_min, adaptive_max
            notes.append(f"HU adaptativo: pico={peak_hu:.0f}HU → rango [{adaptive_min:.0f}, {adaptive_max:.0f}]")
        else:
            myocardium_hu_min, myocardium_hu_max = 0.0, 250.0
            notes.append("HU adaptativo falló, usando default [0, 250]")
    else:
        myocardium_hu_min, myocardium_hu_max = 0.0, 250.0
        notes.append("Pocos voxels para HU adaptativo, usando default [0, 250]")
    
    inner_mask = (ct_crop >= myocardium_hu_min) & (ct_crop <= myocardium_hu_max) & major_component_crop
    
    if inner_mask.sum() > 100:
        refined_crop = inner_mask
        notes.append(f"Refinamiento HU interno: [{myocardium_hu_min}, {myocardium_hu_max}]")
    else:
        notes.append("Refinamiento interno insuficiente (>100 voxels). Usando componente mayor.")
    
    # ── MEJORA v2.7: Shaping anatómico — crecimiento radial desde semilla ─
    # El problema fundamental: la caja de recorte es rectangular y la
    # componente conexa mayor tiende a llenarla → forma cúbica.
    # Solución: en lugar de aceptar toda la componente, hacer crecimiento
    # radial desde la semilla usando distance transform como guía.
    # Esto produce una forma elipsoidal natural centrada en el corazón.
    try:
        from scipy.ndimage import distance_transform_edt, binary_fill_holes
        
        # 4a. Rellenar huecos internos
        filled = binary_fill_holes(refined_crop)
        
        # 4b. Distance transform desde el BORDE de la máscara
        # Valores altos = centro del miocardio, valores bajos = bordes
        dt = distance_transform_edt(filled)
        
        # 4c. Encontrar la semilla dentro del crop (centro de masa o seed)
        if seed_zyx is not None and bbox is not None:
            z0, z1, y0, y1, x0, x1 = bbox
            # Coordenadas de la semilla dentro del crop
            sz_crop = seed_zyx[0] - z0
            sy_crop = seed_zyx[1] - y0
            sx_crop = seed_zyx[2] - x0
            seed_in_crop = (sz_crop, sy_crop, sx_crop)
        else:
            # Usar centroide de la máscara
            coords = np.argwhere(filled)
            if coords.size > 0:
                seed_in_crop = tuple(coords.mean(axis=0).astype(float))
            else:
                seed_in_crop = (ct_crop.shape[0]/2, ct_crop.shape[1]/2, ct_crop.shape[2]/2)
        
        # 4d. Distance transform desde la semilla (distancia euclidiana)
        zz, yy, xx = np.mgrid[0:ct_crop.shape[0], 0:ct_crop.shape[1], 0:ct_crop.shape[2]]
        dist_from_seed = np.sqrt(
            (zz - seed_in_crop[0])**2 +
            (yy - seed_in_crop[1])**2 +
            (xx - seed_in_crop[2])**2
        )
        # Convertir a mm
        dist_from_seed_mm = dist_from_seed * min(sy, sx)  # aproximación isotrópica
        
        # 4e. Crecimiento radial: mantener solo voxels dentro de un radio
        # que contenga el ~80% del volumen del componente mayor.
        # Esto elimina las "esquinas" del cubo que están lejos de la semilla.
        dists_inside = dist_from_seed_mm[filled]
        if dists_inside.size > 0:
            # Radio que contiene el 80% de los voxels (percentil 80)
            radius_80 = float(np.percentile(dists_inside, 80))
            # Usar un radio ligeramente mayor para no cortar demasiado
            effective_radius = radius_80 * 1.15
            
            # Máscara radial: voxels dentro del radio efectivo Y en la máscara
            radial_mask = (dist_from_seed_mm <= effective_radius) & filled
            
            # 4f. Suavizar con morfología esférica
            radius_vox = max(2, int(round(3.0 / min(sy, sx))))  # ~3mm esfera
            sy_, sx_ = np.mgrid[-radius_vox:radius_vox+1, -radius_vox:radius_vox+1]
            sz_ = np.mgrid[-radius_vox:radius_vox+1]
            sphere_struct = (sx_**2 + sy_**2 + sz_**2) <= radius_vox**2
            
            from scipy.ndimage import binary_closing as sph_closing, binary_opening as sph_opening
            shaped = sph_closing(radial_mask, structure=sphere_struct, iterations=2)
            shaped = sph_opening(shaped, structure=sphere_struct, iterations=1)
            shaped = binary_fill_holes(shaped)
            
            # Intersectar con la máscara original (no expandir fuera)
            shaped = shaped & refined_crop
            
            # Solo usar si volumen razonable (>40% del original)
            new_vol = shaped.sum()
            old_vol = refined_crop.sum()
            if new_vol > old_vol * 0.4 and new_vol < old_vol * 2.0:
                refined_crop = shaped
                notes.append(
                    f"Shaping radial: {old_vol}→{new_vol} voxels "
                    f"({100*new_vol/old_vol:.0f}%) — r_eff={effective_radius:.1f}mm"
                )
            else:
                notes.append(
                    f"Shaping radial omitido: vol extremo ({100*new_vol/old_vol:.0f}%). "
                    "Usando forma cruda."
                )
        else:
            notes.append("Shaping radial: sin voxels internos. Usando forma cruda.")
    except Exception as shaping_exc:
        notes.append(f"Shaping anatómico falló: {shaping_exc}. Usando forma cruda.")
    
    # Mapear la máscara recortada de vuelta al volumen completo
    refined = np.zeros(ct.shape, dtype=bool)
    if bbox is not None:
        z0, z1, y0, y1, x0, x1 = bbox
        refined[z0:z1, y0:y1, x0:x1] = refined_crop
    else:
        refined = refined_crop
    
    # Calcular volumen
    voxel_volume_mm3 = sz * sy * sx
    volume_mm3 = float(refined.sum()) * voxel_volume_mm3
    
    if volume_mm3 < min_volume_mm3 or volume_mm3 > max_volume_mm3:
        notes.append(
            f"WARNING: Volumen {volume_mm3:.0f} mm³ fuera de rango esperado "
            f"[{min_volume_mm3:.0f}, {max_volume_mm3:.0f}]. "
            f"La segmentación puede ser incorrecta."
        )
    
    # Centroide
    coords = np.argwhere(refined)
    if coords.size > 0:
        centroid = coords.mean(axis=0).astype(float)
        centroid_zyx = (float(centroid[0]), float(centroid[1]), float(centroid[2]))
    else:
        centroid_zyx = (ct.shape[0] / 2.0, ct.shape[1] / 2.0, ct.shape[2] / 2.0)
    
    # Extraer máscaras 2D por slice axial (para visualización y grosor)
    mask_slices = []
    slices_with_tissue = 0
    for z in range(ct.shape[0]):
        slc = refined[z]
        mask_slices.append(slc)
        if slc.sum() > 0:
            slices_with_tissue += 1
    
    # Medición de grosor parietal (aproximada por distancia al centroide en cada slice)
    wall_thickness = _measure_wall_thickness_per_segment(refined, centroid_zyx, spacing_zyx)
    mean_thickness = float(np.mean(list(wall_thickness.values()))) if wall_thickness else 0.0
    
    notes.append(f"Volumen segmentado: {volume_mm3:.0f} mm³ ({volume_mm3/1000:.1f} mL)")
    notes.append(f"Slices axiales con tejido: {slices_with_tissue}/{ct.shape[0]}")
    notes.append(f"Grosor medio pared: {mean_thickness:.1f} mm")
    
    return MyocardialSegmentationResult(
        mask_3d=refined,
        mask_axial_slices=mask_slices,
        centroid_zyx=centroid_zyx,
        volume_mm3=volume_mm3,
        n_slices=slices_with_tissue,
        wall_thickness_mm=wall_thickness,
        mean_wall_thickness_mm=mean_thickness,
        notes=notes,
    )


def _measure_wall_thickness_per_segment(
    mask_3d: np.ndarray,
    centroid_zyx: tuple[float, float, float],
    spacing_zyx: tuple[float, float, float],
) -> dict[str, float]:
    """Mide el grosor de la pared miocárdica en 6 segmentos (AHA simplificado).
    
    Para cada slice axial con tejido, proyecta rayos desde el centroide en
    6 direcciones (septal, anterior, lateral, inferior, anteroseptal, inferolateral)
    y mide la distancia al borde de la máscara.
    """
    from scipy.ndimage import distance_transform_edt
    
    sz, sy, sx = spacing_zyx
    cz, cy, cx = centroid_zyx
    
    # Direcciones angulares (grados) para 6 segmentos en short-axis
    segments = {
        "anterior":      0,    # arriba (posterior en convención radiológica)
        "anteroseptal":  60,
        "inferior":     120,  # abajo (anterior)
        "inferolateral": 180,
        "lateral":      240,
        "septal":       300,
    }
    
    thicknesses: dict[str, float] = {}
    distances = distance_transform_edt(mask_3d, sampling=spacing_zyx)
    
    cz_idx, cy_idx, cx_idx = int(round(cz)), int(round(cy)), int(round(cx))
    
    # Verificar que el centroide caiga dentro de la máscara (o cerca)
    if not (0 <= cz_idx < mask_3d.shape[0] and 0 <= cy_idx < mask_3d.shape[1] and 0 <= cx_idx < mask_3d.shape[2]):
        # Centroide fuera → usar centro geométrico de la máscara
        coords = np.argwhere(mask_3d)
        if coords.size > 0:
            cz_idx, cy_idx, cx_idx = tuple(int(v) for v in coords.mean(axis=0).round().astype(int))
    
    # Para cada dirección, medir distancia al borde en el plano axial del centroide
    rad = np.pi / 180.0
    for seg_name, angle_deg in segments.items():
        angle_rad = angle_deg * rad
        
        # Muestrear varios puntos a lo largo del radio en esa dirección
        dists = []
        for r_px in range(5, 80):  # hasta ~80 px (~80-160 mm dependiendo de spacing)
            dy = int(round(r_px * np.sin(angle_rad)))
            dx = int(round(r_px * np.cos(angle_rad)))
            yy, xx = cy_idx + dy, cx_idx + dx
            
            if 0 <= yy < mask_3d.shape[1] and 0 <= xx < mask_3d.shape[2]:
                if mask_3d[cz_idx, yy, xx]:
                    # Dentro de la máscara — guardar distancia EDT en mm
                    d_mm = float(distances[cz_idx, yy, xx])
                    dists.append(d_mm)
                else:
                    # Fuera de la máscora — si ya tenemos mediciones, parar
                    if len(dists) > 2:
                        break
        
        if dists:
            # Grosor ≈ 2 × distancia media al borde (desde centro hacia afuera)
            thicknesses[seg_name] = 2.0 * float(np.median(dists))
        else:
            thicknesses[seg_name] = 12.0  # fallback grosor típico
    
    return thicknesses


@dataclass
class PVERecoveryCoefficients:
    """Coeficientes de Recuperación (RC) para corrección PVE."""
    
    rc_heart: float           # RC global para VOI corazón
    rc_per_segment: dict[str, float]  # RC por segmento AHA
    fwhm_mm: float            # FWHM del sistema usado
    mean_wall_thickness_mm: float
    method: str = "analytical_gaussian"
    
    @property
    def pve_correction_factor(self) -> float:
        """Factor por el cual multiplicar cuentas para corregir PVE.
        
        C_corregido = C_medido / RC  →  factor = 1/RC
        Si RC=0.7 (30% de pérdida), factor≈1.43 (+43% de corrección)
        """
        return 1.0 / max(self.rc_heart, 0.1)


def compute_pve_recovery_coefficients(
    wall_thickness_mm: float | dict[str, float],
    fwhm_mm: float = 12.0,
    *,
    method: str = "analytical_gaussian",
) -> PVERecoveryCoefficients:
    """Calcula Coeficientes de Recuperación (RC) para corrección PVE.
    
    Modelo analítico basado en la aproximación gaussiana de la PSF:
    
    RC(d) = 1 - exp(-d² / (4 × σ²))   [modelo simplificado Hoffman 1979]
    
    donde d = grosor del objeto y σ = FWHM / (2√(2ln2))
    
    Para objetos grandes (d >> FWHM), RC → 1.0 (sin pérdida).
    Para objetos pequeños (d << FWHM), RC → 0 (pérdida total).
    
    Args:
        wall_thickness_mm: Grosor parietal (float global o dict por segmento)
        fwhm_mm: Resolución espacial FWHM del sistema SPECT (típico 10-15 mm)
        method: Modelo de RC ('analytical_gaussian' o 'empirical_lookup')
        
    Returns:
        PVERecocoveryCoefficients con RC global y por segmento
    """
    sigma_mm = fwhm_mm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ≈ FWHM / 2.355
    
    def _rc_for_thickness(d_mm: float) -> float:
        """RC para un grosor dado usando modelo gaussiano."""
        if d_mm <= 0:
            return 0.1  # mínimo para evitar división por cero
        if method == "analytical_gaussian":
            # Modelo de Hoffman modificado: RC ≈ erf(d / (2√2 × σ))
            import math
            arg = d_mm / (2.0 * np.sqrt(2.0) * sigma_mm)
            if arg > 5.0:
                return 1.0  # saturación
            return float(np.clip(float(math.erf(arg)), 0.15, 1.0))
        else:
            # Lookup empírico (placeholder)
            return float(np.clip(d_mm / (d_mm + sigma_mm), 0.2, 1.0))
    
    if isinstance(wall_thickness_mm, dict):
        rc_per_seg = {seg: _rc_for_thickness(d) for seg, d in wall_thickness_mm.items()}
        rc_global = float(np.mean(list(rc_per_seg.values())))
        mean_thick = float(np.mean(list(wall_thickness_mm.values())))
    else:
        rc_global = _rc_for_thickness(wall_thickness_mm)
        rc_per_seg = {"global": rc_global}
        mean_thick = wall_thickness_mm
    
    return PVERecoveryCoefficients(
        rc_heart=rc_global,
        rc_per_segment=rc_per_seg,
        fwhm_mm=fwhm_mm,
        mean_wall_thickness_mm=mean_thick,
        method=method,
    )


@dataclass
class HMRSpectPVECorrected:
    """Resultado HMR-SPECT con corrección PVE aplicada."""
    
    hmr_original: float         # HMR sin corrección (el que se venía calculando)
    hmr_pve_corrected: float    # HMR corregido por PVE
    pve_factor: float           # Factor de corrección aplicado (1/RC)
    rc_heart: float             # Coeficiente de recuperación usado
    wall_thickness_mm: float    # Grosor parietal promedio
    fwhm_mm: float              # FWHM asumido
    classification_original: str
    classification_corrected: str
    delta_pct: float            # Cambio porcentual ((HMR_corr/HMR_orig)-1)×100
    notes: list[str]


def apply_pve_correction_to_hmr(
    hmr_result: HmrSpectResult,
    ct_segmentation: MyocardialSegmentationResult | None = None,
    *,
    fwhm_mm: float = 12.0,
    force_wall_thickness_mm: float | None = None,
) -> HMRSpectPVECorrected:
    """Aplica corrección PVE a un resultado HMR-SPECT existente.
    
    La corrección se aplica solo al numerador (cuentas del corazón),
    asumiendo que el ROI contralateral (mediastino/tórax derecho) es
    suficientemente grande como para tener RC ≈ 1.0.
    
    Fórmula:
        HMR_corr = (C_heart / RC_heart) / C_contralateral
                = HMR_original × (1 / RC_heart)
                = HMR_original × PVE_factor
    
    Args:
        hmr_result: Resultado HMR original de compute_hmr_spect()
        ct_segmentation: Segmentación miocárdica desde CT (opcional pero recomendado)
        fwhm_mm: FWHM del sistema si no hay segmentación CT
        force_wall_thickness_mm: Forzar grosor (sin CT)
        
    Returns:
        HMRSpectPVECorrected con HMR original y corregido
    """
    notes: list[str] = []
    
    # Determinar grosor parietal
    if ct_segmentation is not None and ct_segmentation.mean_wall_thickness_mm > 0:
        wall_mm = ct_segmentation.mean_wall_thickness_mm
        wall_source = "CT segmentation"
        seg_notes = ct_segmentation.notes[:3]  # Primeras 3 notas
        notes.extend(seg_notes)
    elif force_wall_thickness_mm is not None:
        wall_mm = force_wall_thickness_mm
        wall_source = "user-specified"
    else:
        wall_mm = 12.0  # grosor típico pared VI normal
        wall_source = "assumed typical"
        notes.append(f"WARNING: Sin segmentación CT ni grosor forzado. Asumiendo grosor típico {wall_mm} mm.")
    
    # Calcular RC
    rc_result = compute_pve_recovery_coefficients(wall_mm, fwhm_mm=fwhm_mm)
    rc = rc_result.rc_heart
    pve_factor = rc_result.pve_correction_factor
    
    # Aplicar corrección
    hmr_orig = hmr_result.hmr_raw if hmr_result.hmr_raw is not None else hmr_result.hmr
    hmr_corr = hmr_orig * pve_factor
    
    # Clasificaciones
    class_orig = hmr_result.classification
    if hmr_corr >= 1.6:
        class_corr = "POSITIVO"
    elif hmr_corr >= 1.5:
        class_corr = "EQUIVOCO"
    else:
        class_corr = "NEGATIVO"
    
    delta_pct = ((hmr_corr / max(hmr_orig, 1e-8)) - 1.0) * 100.0
    
    notes.append(f"PVE: FWHM={fwhm_mm}mm, grosor={wall_mm:.1f}mm ({wall_source}), RC={rc:.3f}")
    notes.append(f"PVE: factor corrección={pve_factor:.3f}, ΔHMR={delta_pct:+.1f}%")
    notes.append(f"HMR original={hmr_orig:.2f} ({class_orig}) → corregido={hmr_corr:.2f} ({class_corr})")
    
    # Alerta si la corrección cambia la clasificación
    if class_orig != class_corr:
        notes.append(
            f"⚠️ LA CORRECCIÓN PVE CAMBIÓ LA CLASIFICACIÓN: {class_orig} → {class_corr}. "
            "Revisar calidad de la segmentación CT."
        )
    
    return HMRSpectPVECorrected(
        hmr_original=hmr_orig,
        hmr_pve_corrected=hmr_corr,
        pve_factor=pve_factor,
        rc_heart=rc,
        wall_thickness_mm=wall_mm,
        fwhm_mm=fwhm_mm,
        classification_original=class_orig,
        classification_corrected=class_corr,
        delta_pct=delta_pct,
        notes=notes,
    )


def create_voi_from_localization(
    anchor_zyx: tuple[float, float, float],
    point_zyx: tuple[float, float, float] | None = None,
    heart_radius_mm: float = 30.0,
    mediastinum_radius_mm: float = 20.0,
) -> tuple[VOISphere, VOISphere]:
    """Crea VOIs corazón y mediastino a partir de puntos de localización.
    
    Args:
        anchor_zyx: Coordenadas del anchor (centro corazón) en índices
        point_zyx: Coordenadas del point B (opcional, para mediastino manual)
        heart_radius_mm: Radio de la VOI corazón en mm
        mediastinum_radius_mm: Radio de la VOI mediastino en mm
        
    Returns:
        (voi_heart, voi_mediastinum)
    """
    voi_heart = VOISphere(
        cz=anchor_zyx[0],
        cy=anchor_zyx[1],
        cx=anchor_zyx[2],
        radius_mm=heart_radius_mm,
    )
    
    if point_zyx is not None:
        voi_mediastinum = VOISphere(
            cz=point_zyx[0],
            cy=point_zyx[1],
            cx=point_zyx[2],
            radius_mm=mediastinum_radius_mm,
        )
    else:
        voi_mediastinum = VOISphere(
            cz=anchor_zyx[0],
            cy=anchor_zyx[1],
            cx=anchor_zyx[2],
            radius_mm=mediastinum_radius_mm,
        )
    
    return voi_heart, voi_mediastinum


# =============================================================================
# RATIO S/VD — Corazón / Vértebra / Aorta descendente (SPECT 3D)
# =============================================================================
# Referencia: Castano et al., J Nucl Cardiol 2024 — ratio S/VD en SPECT/CT
# para amiloidosis cardíaca. Útil en casos equívocos (HMR planar 1.0–1.5).
# S = cuentas ROI corazón, V = cuentas ROI vértebra, D = cuentas ROI aorta.
# Ratio principal: S/VD = S / sqrt(V × D)  (media geométrica de V y D).
# Ratios secundarios: S/V, S/D, V/D.
# =============================================================================

@dataclass
class SvdRatioResult:
    """Resultado del cálculo ratio S/VD en SPECT 3D."""

    # Ratios principales
    s_vd: float          # S / sqrt(V×D) — ratio principal (media geométrica)
    s_v: float           # S / V
    s_d: float           # S / D
    v_d: float           # V / D (ratio de referencia normal)

    # Cuentas por ROI
    s_counts: float = 0.0       # Cuentas totales ROI corazón
    v_counts: float = 0.0       # Cuentas totales ROI vértebra
    d_counts: float = 0.0       # Cuentas totales ROI aorta

    # Cuentas promedio por voxel
    s_mean: float = 0.0
    v_mean: float = 0.0
    d_mean: float = 0.0

    # Número de voxels por ROI
    s_voxels: int = 0
    v_voxels: int = 0
    d_voxels: int = 0

    # Volúmenes en mL
    s_volume_ml: float = 0.0
    v_volume_ml: float = 0.0
    d_volume_ml: float = 0.0

    # VOIs utilizadas
    voi_heart: VOISphere | None = None
    voi_vertebra: VOISphere | None = None
    voi_aorta: VOISphere | None = None

    # Clasificación
    method: str = "VOI completa"

    @property
    def classification(self) -> str:
        """Clasificación basada en S/VD.

        Cutoffs preliminares (literatura en desarrollo):
        - S/VD >= 2.2 → POSITIVO
        - S/VD 1.8–2.2 → EQUIVOCO
        - S/VD < 1.8 → NEGATIVO

        NOTA: Estos cutoffs son orientativos y deben validarse con la
        población local antes de uso clínico definitivo.
        """
        if self.s_vd >= 2.2:
            return "POSITIVO"
        if self.s_vd >= 1.8:
            return "EQUIVOCO"
        return "NEGATIVO"

    @property
    def s_vd_text(self) -> str:
        return f"S/VD = {self.s_vd:.2f} ({self.classification})"


def compute_spect_ratio(
    volume: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    voi_heart: VOISphere | VOIAnatomical,
    voi_vertebra: VOISphere | VOIAnatomical,
    voi_aorta: VOISphere | VOIAnatomical,
    *,
    volume_raw: np.ndarray | None = None,
) -> SvdRatioResult:
    """Calcula ratio S/VD (corazón / vértebra / aorta) en SPECT 3D.

    Integra cuentas totales dentro de cada VOI esférica (o anatómica)
    sobre el volumen SPECT y calcula:

        S/VD = S / sqrt(V × D)
        S/V  = S / V
        S/D  = S / D
        V/D  = V / D

    Args:
        volume: Volumen SPECT 3D filtrado (nz, ny, nx)
        spacing_zyx: Espaciado en mm (sz, sy, sx)
        voi_heart: VOI del corazón (S)
        voi_vertebra: VOI de la vértebra torácica (V)
        voi_aorta: VOI de la aorta descendente (D)
        volume_raw: Volumen SPECT sin filtrar (opcional, para ratios raw)

    Returns:
        SvdRatioResult con todos los ratios y estadísticas.
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"Se esperaba volumen 3D, recibido {vol.shape}")

    # Generar máscaras 3D para cada ROI
    mask_s = voi_heart.mask_3d(vol.shape, spacing_zyx)
    mask_v = voi_vertebra.mask_3d(vol.shape, spacing_zyx)
    mask_d = voi_aorta.mask_3d(vol.shape, spacing_zyx)

    # Cuentas totales (volumen filtrado)
    s_counts = float(vol[mask_s].sum())
    v_counts = float(vol[mask_v].sum())
    d_counts = float(vol[mask_d].sum())

    # Voxels y medias
    s_voxels = int(mask_s.sum())
    v_voxels = int(mask_v.sum())
    d_voxels = int(mask_d.sum())

    s_mean = s_counts / max(s_voxels, 1) if s_voxels > 0 else 0.0
    v_mean = v_counts / max(v_voxels, 1) if v_voxels > 0 else 0.0
    d_mean = d_counts / max(d_voxels, 1) if d_voxels > 0 else 0.0

    if s_voxels == 0 or v_voxels == 0 or d_voxels == 0:
        missing = [
            label
            for label, count in (("S (corazón)", s_voxels), ("V (vértebra)", v_voxels), ("D (aorta)", d_voxels))
            if count == 0
        ]
        raise ValueError(f"Las VOI no intersectan el volumen: {', '.join(missing)}.")

    signal_floor = max(float(np.nanmax(np.abs(vol))) * 1e-4, 1e-8)
    invalid_refs = [
        label
        for label, mean in (("V (vértebra)", v_mean), ("D (aorta)", d_mean))
        if not np.isfinite(mean) or mean <= signal_floor
    ]
    if invalid_refs:
        raise ValueError(
            "Señal nula o insuficiente en " + ", ".join(invalid_refs) +
            ". Revise la colocación de los puntos sobre la fusión SPECT/CT."
        )

    # Los radios pueden ser diferentes; usar medias evita que el cociente dependa
    # del número de voxels incluido por cada VOI.
    vd_geom = float(np.sqrt(max(v_mean, 0.0) * max(d_mean, 0.0)))
    s_vd = s_mean / vd_geom
    s_v = s_mean / v_mean
    s_d = s_mean / d_mean
    v_d = v_mean / d_mean

    # Volúmenes
    s_vol = voi_heart.volume_ml()
    v_vol = voi_vertebra.volume_ml()
    d_vol = voi_aorta.volume_ml()

    return SvdRatioResult(
        s_vd=s_vd,
        s_v=s_v,
        s_d=s_d,
        v_d=v_d,
        s_counts=s_counts,
        v_counts=v_counts,
        d_counts=d_counts,
        s_mean=s_mean,
        v_mean=v_mean,
        d_mean=d_mean,
        s_voxels=s_voxels,
        v_voxels=v_voxels,
        d_voxels=d_voxels,
        s_volume_ml=s_vol,
        v_volume_ml=v_vol,
        d_volume_ml=d_vol,
        voi_heart=voi_heart,
        voi_vertebra=voi_vertebra,
        voi_aorta=voi_aorta,
        method="VOI completa",
    )

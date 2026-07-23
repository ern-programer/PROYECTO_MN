"""Exportación DICOM simple del volumen desgatillado.

Exportador mínimo para compartir/releer el desgatillado (UngRaw).
El módulo de exportación DICOM completo (con todas las series) queda para más adelante.

Estrategia: crear una serie NM nueva con ImageType RECON TOMO (no gated),
preservando geometría (PixelSpacing, orientación si existe) y metadatos del paciente.
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np


def save_ungated_dicom(
    volume: np.ndarray,
    output_path: str,
    source_study=None,
    pixel_spacing: tuple[float, float] | None = None,
    slice_thickness_mm: float | None = None,
) -> str:
    """
    Guarda el volumen desgatillado como DICOM NM no-gated.

    Parameters
    ----------
    volume : ndarray (n_slices, H, W)
        Volumen desgatillado.
    output_path : str
        Ruta del archivo .dcm de salida (una serie, un archivo multiframe).
    source_study : GatedStudy, optional
        Estudio de origen para copiar metadatos del paciente.
    pixel_spacing : tuple, optional
        Espaciado en mm (fallback si no hay source_study).
    slice_thickness_mm : float, optional
        Espesor de corte en mm.

    Returns
    -------
    str
        Ruta del archivo guardado.
    """
    try:
        import pydicom
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import (
            ExplicitVRLittleEndian,
            NuclearMedicineImageStorage,
            generate_uid,
        )
    except ImportError as exc:
        raise ImportError("pydicom requerido para exportar DICOM") from exc

    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"volume debe ser 3D (n_slices,H,W); recibió {vol.shape}")
    n_slices, rows, cols = vol.shape

    # Escalado a uint16
    vmax = float(vol.max()) if vol.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    vol16 = np.clip(vol / vmax * 65535.0, 0, 65535).astype(np.uint16)

    now = datetime.now()
    series_uid = generate_uid()
    study_uid = getattr(source_study, "study_instance_uid", "") or generate_uid()
    sop_uid = generate_uid()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = NuclearMedicineImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = Dataset()
    ds.file_meta = file_meta
    ds.SOPClassUID = NuclearMedicineImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.Modality = "NM"
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesNumber = "900"
    ds.InstanceNumber = "1"

    # Paciente desde el estudio origen
    ds.PatientName = getattr(source_study, "patient_name", "") or ""
    ds.PatientID = getattr(source_study, "patient_id", "") or ""
    ds.PatientSex = getattr(source_study, "patient_sex", "") or ""
    ds.PatientBirthDate = getattr(source_study, "patient_birth_date", "") or ""
    ds.StudyDate = getattr(source_study, "study_date", "") or now.strftime("%Y%m%d")
    ds.StudyTime = getattr(source_study, "study_time", "") or now.strftime("%H%M%S")
    ds.AccessionNumber = str(getattr(source_study, "accession_number", "") or "")[:16]
    ds.StudyDescription = getattr(source_study, "study_description", "") or "GammaSync"
    ds.SeriesDescription = "UNGATED perfusion (GammaSync desgatillado)"

    # Tipo de imagen: reconstruido, no gated
    ds.ImageType = ["DERIVED", "SECONDARY", "RECON TOMO"]

    # Geometría
    ps = pixel_spacing or getattr(source_study, "pixel_spacing", None)
    if ps:
        ds.PixelSpacing = [f"{float(ps[0]):.6f}", f"{float(ps[1]):.6f}"]
    st = slice_thickness_mm or getattr(source_study, "slice_thickness_mm", None) or getattr(source_study, "z_spacing_mm", None)
    if st:
        ds.SliceThickness = f"{float(st):.6f}"
        ds.SpacingBetweenSlices = f"{float(st):.6f}"

    # Imagen multiframe: n_slices frames
    ds.NumberOfFrames = str(n_slices)
    ds.Rows = int(rows)
    ds.Columns = int(cols)
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = vol16.astype("<u2").tobytes()

    # Escalado NM
    ds.RescaleSlope = "1"
    ds.RescaleIntercept = "0"
    ds.RescaleType = "US"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pydicom.dcmwrite(output_path, ds, write_like_original=False)
    return output_path


def save_gated_axis_dicom(
    cube: np.ndarray,
    output_path: str,
    *,
    axis_name: str,
    source_study=None,
    pixel_spacing: tuple[float, float] | None = None,
    slice_thickness_mm: float | None = None,
    series_number: int = 910,
    extra_description: str = "",
) -> str:
    """Guarda un eje cardíaco gated como DICOM NM multiframe.

    Parameters
    ----------
    cube : ndarray (gates,slices,H,W)
        Set de cortes reorientados de un eje (SA, HLA o VLA).
    output_path : str
        Archivo .dcm de salida.
    axis_name : str
        Nombre del eje: SA, HLA o VLA.
    source_study : GatedStudy, optional
        Estudio origen para copiar identidad y geometría.

    Notes
    -----
    Escribe SliceVector y TimeSlotVector para que SINCRO pueda recargarlo como
    cubo 4D. Otros visores DICOM lo verán como serie NM DERIVED/RECON TOMO.
    """
    try:
        import pydicom
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, NuclearMedicineImageStorage, generate_uid
    except ImportError as exc:
        raise ImportError("pydicom requerido para exportar ejes DICOM") from exc

    arr = np.asarray(cube, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"cube debe ser 4D (gates,slices,H,W); recibió {arr.shape}")
    n_gates, n_slices, rows, cols = arr.shape
    if n_gates < 1 or n_slices < 1:
        raise ValueError("cube debe tener al menos 1 gate y 1 slice")

    vmax = float(np.nanmax(arr)) if arr.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    arr16 = np.clip(arr / vmax * 65535.0, 0, 65535).astype(np.uint16)
    flat = arr16.reshape(n_gates * n_slices, rows, cols)
    slice_vector = np.tile(np.arange(1, n_slices + 1, dtype=np.int32), n_gates)
    time_vector = np.repeat(np.arange(1, n_gates + 1, dtype=np.int32), n_slices)

    now = datetime.now()
    series_uid = generate_uid()
    study_uid = getattr(source_study, "study_instance_uid", "") or generate_uid()
    sop_uid = generate_uid()
    axis_key = str(axis_name or "AXIS").strip().upper()
    desc_suffix = f" {extra_description}" if extra_description else ""

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = NuclearMedicineImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = Dataset()
    ds.file_meta = file_meta
    ds.SOPClassUID = NuclearMedicineImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.Modality = "NM"
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesNumber = str(int(series_number))
    ds.InstanceNumber = "1"
    ds.PatientName = getattr(source_study, "patient_name", "") or ""
    ds.PatientID = getattr(source_study, "patient_id", "") or ""
    ds.PatientSex = getattr(source_study, "patient_sex", "") or ""
    ds.PatientBirthDate = getattr(source_study, "patient_birth_date", "") or ""
    ds.StudyDate = getattr(source_study, "study_date", "") or now.strftime("%Y%m%d")
    ds.StudyTime = getattr(source_study, "study_time", "") or now.strftime("%H%M%S")
    ds.AccessionNumber = str(getattr(source_study, "accession_number", "") or "")[:16]
    ds.StudyDescription = getattr(source_study, "study_description", "") or "GammaSync"
    ds.SeriesDescription = f"GammaSync {axis_key} cardiac cuts{desc_suffix}"[:64]
    ds.ImageType = ["DERIVED", "SECONDARY", "RECON TOMO", "GATED", axis_key]

    ps = pixel_spacing or getattr(source_study, "pixel_spacing", None)
    if ps:
        ds.PixelSpacing = [f"{float(ps[0]):.6f}", f"{float(ps[1]):.6f}"]
    st = slice_thickness_mm or getattr(source_study, "slice_thickness_mm", None) or getattr(source_study, "z_spacing_mm", None)
    if st:
        ds.SliceThickness = f"{float(st):.6f}"
        ds.SpacingBetweenSlices = f"{float(st):.6f}"

    ds.NumberOfFrames = str(int(n_gates * n_slices))
    ds.Rows = int(rows)
    ds.Columns = int(cols)
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = flat.astype("<u2").tobytes()
    ds.RescaleSlope = "1"
    ds.RescaleIntercept = "0"
    ds.RescaleType = "US"

    ds.add_new((0x0054, 0x0071), "US", int(n_gates))      # NumberOfTimeSlots
    ds.add_new((0x0054, 0x0101), "US", int(n_gates))      # NumberOfTimeSlices
    ds.add_new((0x0054, 0x0081), "US", int(n_slices))     # NumberOfSlices
    ds.add_new((0x0054, 0x0070), "US", [int(x) for x in time_vector.tolist()])
    ds.add_new((0x0054, 0x0080), "US", [int(x) for x in slice_vector.tolist()])
    ds.add_new((0x0054, 0x1000), "CS", axis_key)          # SeriesType, informativo

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pydicom.dcmwrite(output_path, ds, write_like_original=False)
    return output_path


def save_cardiac_axes_dicoms(
    axes: dict[str, np.ndarray],
    output_dir: str,
    *,
    source_study=None,
    base_name: str = "GammaSync_axes",
    slice_thickness_mm: float | None = None,
    extra_description: str = "",
) -> dict[str, str]:
    """Guarda SA/HLA/VLA como tres DICOM multiframe y devuelve rutas por eje."""
    out: dict[str, str] = {}
    series_numbers = {"SA": 910, "HLA": 911, "VLA": 912}
    for axis_name in ("SA", "HLA", "VLA"):
        if axis_name not in axes:
            continue
        path = os.path.join(output_dir, f"{base_name}_{axis_name}.dcm")
        out[axis_name] = save_gated_axis_dicom(
            axes[axis_name],
            path,
            axis_name=axis_name,
            source_study=source_study,
            slice_thickness_mm=slice_thickness_mm,
            series_number=series_numbers[axis_name],
            extra_description=extra_description,
        )
    return out

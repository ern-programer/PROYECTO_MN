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

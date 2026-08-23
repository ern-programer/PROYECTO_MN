"""Exportación DICOM Structured Report para puntos de localización.

Implementa TID 1411 (Region and Spatial Coordinates) para puntos LOC.
Compatible con visualizadores DICOM estándar (OsiriX, 3D Slicer, etc.).

Referencias:
- DICOM PS3.3: Section A.35.9 (Enhanced SR)
- TID 1411: Region and Spatial Coordinates
- CID 218: Spatial Coordinates
"""

from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DcmSequence
from pydicom.uid import generate_uid

if TYPE_CHECKING:
    import numpy as np


def create_localization_sr(
    localization_points: list[dict],
    spect_ds: Dataset,
    spect_volume: np.ndarray | None = None,
    spacing_zyx: tuple[float, float, float] | None = None,
    output_path: str | Path | None = None,
) -> Dataset:
    """Crea un DICOM-SR (Enhanced SR) con puntos de localización.

    Args:
        localization_points: Lista de dicts con keys:
            - label: str (ej: "A (ancla)", "B (punto)")
            - zyx: list[int] (coordenadas voxel 1-indexed)
            - value_mm: float (opcional, distancia en mm)
        spect_ds: Dataset DICOM del SPECT original (para PatientID, etc.)
        spect_volume: Volumen SPECT (opcional, para validar coordenadas)
        spacing_zyx: Spacing (z, y, x) en mm (opcional)
        output_path: Ruta de salida (opcional, si se provee guarda el archivo)

    Returns:
        Dataset DICOM-SR listo para guardar o transmitir

    Example:
        >>> points = [
        ...     {"label": "A (ancla)", "zyx": [45, 64, 64]},
        ...     {"label": "B (punto)", "zyx": [48, 70, 72]},
        ...     {"label": "Distancia A→B", "value_mm": 12.5},
        ... ]
        >>> sr = create_localization_sr(points, spect_ds, spacing_zyx=(4.8, 4.8, 4.8))
        >>> pydicom.dcmwrite("localizacion_sr.dcm", sr)
    """
    # --- Metadatos básicos ---
    sr = Dataset()
    sr.is_little_endian = True
    sr.is_implicit_VR = False

    # SOP Class: Enhanced SR
    sr.SOPClassUID = "1.2.840.10008.5.1.4.1.1.88.22"
    sr.SOPInstanceUID = generate_uid(prefix="1.2.840.113619.6.325")
    sr.SpecificCharacterSet = "ISO_IR 100"

    # Patient info (copiar del SPECT)
    for tag in ["PatientID", "PatientName", "PatientBirthDate", "PatientSex"]:
        if hasattr(spect_ds, tag):
            setattr(sr, tag, getattr(spect_ds, tag))
        else:
            setattr(sr, tag, "UNKNOWN" if tag != "PatientBirthDate" else "")

    # Study info
    sr.StudyInstanceUID = getattr(spect_ds, "StudyInstanceUID", generate_uid())
    sr.StudyDate = datetime.datetime.now().strftime("%Y%m%d")
    sr.StudyTime = datetime.datetime.now().strftime("%H%M%S")
    sr.AccessionNumber = getattr(spect_ds, "AccessionNumber", "")
    sr.ReferringPhysicianName = getattr(spect_ds, "ReferringPhysicianName", "")
    sr.StudyID = getattr(spect_ds, "StudyID", "1")

    # Series info (nueva serie para el SR)
    sr.SeriesInstanceUID = generate_uid(prefix="1.2.840.113619.6.325")
    sr.SeriesNumber = "999"
    sr.Modality = "SR"

    # Instance info
    sr.InstanceNumber = "1"
    sr.ContentDate = sr.StudyDate
    sr.ContentTime = sr.StudyTime

    # SR-specific
    sr.CompletionFlag = "COMPLETE"
    sr.VerificationFlag = "VERIFIED"
    sr.PreliminaryFlag = "PRELIMINARY"

    # Manufacturer
    sr.Manufacturer = "GAMMASYS SINCRO"
    sr.ManufacturerModelName = "SINCRO AMYLO SPECT"
    sr.SoftwareVersions = "1.14.0"

    # --- Content Sequence (TID 1411) ---
    content_seq = []

    # TID 1411 Row 1: Observation DateTime
    obs_datetime = Dataset()
    obs_datetime.RelationshipType = "HAS OBS CONTEXT"
    obs_datetime.ValueType = "DATETIME"
    obs_datetime.ObservationDateTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    content_seq.append(obs_datetime)

    # TID 1411 Row 2: Person Observer Name
    observer = Dataset()
    observer.RelationshipType = "HAS OBS CONTEXT"
    observer.ValueType = "PNAME"
    observer.PersonName = "SINCRO^Automated"
    content_seq.append(observer)

    # Agregar cada punto de localización
    for pt in localization_points:
        label = pt.get("label", "Punto")
        zyx = pt.get("zyx")
        value_mm = pt.get("value_mm")

        if zyx is not None:
            # Punto espacial (TID 1411 Row 3)
            region = _create_spatial_coordinate(label, zyx, spacing_zyx)
            content_seq.append(region)
        elif value_mm is not None:
            # Medición de distancia
            measurement = _create_distance_measurement(label, value_mm)
            content_seq.append(measurement)

    sr.ContentSequence = DcmSequence(content_seq)

    # --- Referenced SOP Sequence (link al SPECT original) ---
    ref_sop = Dataset()
    ref_sop.ReferencedSOPClassUID = spect_ds.SOPClassUID
    ref_sop.ReferencedSOPInstanceUID = spect_ds.SOPInstanceUID
    sr.ReferencedSeriesSequence = DcmSequence([ref_sop])

    # --- Output ---
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pydicom.dcmwrite(str(output_path), sr)

    return sr


def _create_spatial_coordinate(label: str, zyx: list[int], spacing_zyx: tuple[float, float, float] | None) -> Dataset:
    """Crea un item de coordenada espacial (TID 1411)."""
    region = Dataset()
    region.RelationshipType = "CONTAINS"
    region.ValueType = "SCOORD"

    # Concept Name: CID 218 - Spatial Coordinates
    region.ConceptNameCodeSequence = DcmSequence([
        _create_coded_entry("130456", "DCM", "Spatial Coordinates")
    ])

    # Region name (descripción libre)
    region.ObservationDescription = label

    # Coordenadas en mm si hay spacing
    if spacing_zyx:
        z_mm = (zyx[0] - 1) * spacing_zyx[0]  # Convertir a 0-indexed
        y_mm = (zyx[1] - 1) * spacing_zyx[1]
        x_mm = (zyx[2] - 1) * spacing_zyx[2]
        region.GraphicData = [x_mm, y_mm, z_mm]
    else:
        # Sin spacing, usar coordenadas voxel
        region.GraphicData = [float(zyx[2] - 1), float(zyx[1] - 1), float(zyx[0] - 1)]

    region.GraphicType = "POINT"

    return region


def _create_distance_measurement(label: str, value_mm: float) -> Dataset:
    """Crea un item de medición de distancia."""
    measurement = Dataset()
    measurement.RelationshipType = "CONTAINS"
    measurement.ValueType = "NUM"

    # Concept Name: Distance
    measurement.ConceptNameCodeSequence = DcmSequence([
        _create_coded_entry("130462", "DCM", "Distance")
    ])

    # Measured Value Sequence
    measured_value = Dataset()
    measured_value.NumericValue = value_mm
    measured_value.MeasurementUnitsCodeSequence = DcmSequence([
        _create_coded_entry("mm", "UCUM", "millimeter")
    ])
    measurement.MeasuredValueSequence = DcmSequence([measured_value])

    measurement.ObservationDescription = label

    return measurement


def _create_coded_entry(code_value: str, coding_scheme: str, code_meaning: str) -> Dataset:
    """Crea un Code Sequence item."""
    entry = Dataset()
    entry.CodeValue = code_value
    entry.CodingSchemeDesignator = coding_scheme
    entry.CodeMeaning = code_meaning
    return entry


# --- Función de conveniencia para el panel ---
def export_localization_sr_from_panel(
    panel,
    output_path: str | Path | None = None,
) -> Path | None:
    """Exporta puntos de localización desde AmyloidSpectPanel.

    Args:
        panel: Instancia de AmyloidSpectPanel con get_localization_points()
        output_path: Ruta de salida (opcional, default: misma carpeta que SPECT)

    Returns:
        Path del archivo creado, o None si no hay puntos
    """
    points = panel.get_localization_points()
    if not points:
        return None

    # Obtener dataset SPECT original
    spect_ds = getattr(panel, "_current_spect_ds", None)
    if spect_ds is None:
        # Crear dataset mínimo si no está disponible
        spect_ds = Dataset()
        spect_ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.20"  # Nuclear Medicine Image
        spect_ds.SOPInstanceUID = generate_uid()
        spect_ds.PatientID = getattr(panel, "_patient_id", "UNKNOWN")
        spect_ds.PatientName = getattr(panel, "_patient_name", "UNKNOWN^PATIENT")
        spect_ds.StudyInstanceUID = generate_uid()

    # Obtener spacing
    spacing = getattr(panel, "_spect_spacing_zyx", None)

    # Determinar ruta de salida
    if output_path is None:
        spect_path = getattr(panel, "_current_spect_path", "")
        if spect_path:
            output_path = Path(spect_path).parent / f"LOC_{Path(spect_path).stem}.dcm"
        else:
            output_path = Path.cwd() / f"LOC_{uuid.uuid4().hex[:8]}.dcm"

    sr = create_localization_sr(
        localization_points=points,
        spect_ds=spect_ds,
        spacing_zyx=spacing,
        output_path=output_path,
    )

    return Path(output_path)

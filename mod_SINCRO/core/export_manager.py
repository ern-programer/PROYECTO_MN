"""Exportación de resultados GammaSync a formatos estructurados."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Encoder JSON que maneja tipos numpy."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _safe_get(data: dict, key: str, default: Any = None) -> Any:
    """Obtiene valor de dict de forma segura."""
    value = data.get(key, default)
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def export_json(
    output_path: str,
    study_metadata: dict,
    metrics: dict,
    segmentation_info: dict,
    processing_params: dict,
    robustness: dict | None = None,
    normal_db_eval: dict | None = None,
    qc_info: dict | None = None,
) -> str:
    """
    Exporta resultados completos a JSON estructurado.

    Parameters
    ----------
    output_path : str
        Ruta del archivo JSON de salida.
    study_metadata : dict
        Metadatos del estudio (paciente, fecha, UID, etc.).
    metrics : dict
        Métricas de fase (PSD, BW, entropy, etc.).
    segmentation_info : dict
        Info de segmentación (método, voxels, ROIs).
    processing_params : dict
        Parámetros de procesamiento usados.
    robustness : dict, optional
        Resultados de robustez (segmentario, bootstrap, sensibilidad ROI).
    normal_db_eval : dict, optional
        Evaluación contra base de datos normal.
    qc_info : dict, optional
        Información de control de calidad.

    Returns
    -------
    str
        Ruta del archivo JSON generado.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = {
        "export_version": "1.0",
        "export_timestamp": datetime.now().isoformat(),
        "software": {
            "name": "GammaSync",
            "version": "1.8.0",
            "module": "SINCRO",
        },
        "study": study_metadata,
        "segmentation": segmentation_info,
        "processing": processing_params,
        "metrics": {
            "voxel": {
                "phase_sd_deg": _safe_get(metrics, "phase_sd"),
                "bandwidth_deg": _safe_get(metrics, "bandwidth"),
                "entropy_shannon_bits": _safe_get(metrics, "entropy_shannon_bits"),
                "entropy_normalized_pct": _safe_get(metrics, "entropy_normalized_pct"),
                "asynchrony_index_pct": _safe_get(metrics, "asynchrony_index"),
                "skewness": _safe_get(metrics, "skewness"),
                "kurtosis": _safe_get(metrics, "kurtosis"),
                "peak_phase_deg": _safe_get(metrics, "peak_phase"),
                "peak_width_deg": _safe_get(metrics, "peak_width"),
                "latest_activation_deg": _safe_get(metrics, "latest_activation_phase"),
                "technical_classification": _safe_get(metrics, "technical_classification"),
                "n_voxels_kept": _safe_get(metrics, "n_voxels_kept"),
                "n_voxels_total": _safe_get(metrics, "n_voxels_total"),
                "amp_filter": _safe_get(metrics, "amp_filter"),
            },
        },
    }

    # Agregar robustez si está disponible
    if robustness:
        payload["metrics"]["segmental_aha"] = robustness.get("segmental_aha", {})
        payload["metrics"]["bootstrap"] = robustness.get("bootstrap", {})
        payload["metrics"]["roi_sensitivity"] = robustness.get("roi_sensitivity", {})

    # Agregar evaluación DB normal
    if normal_db_eval:
        payload["normal_db_evaluation"] = normal_db_eval

    # Agregar QC
    if qc_info:
        payload["quality_control"] = qc_info

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

    return output_path


def export_csv(
    output_path: str,
    study_metadata: dict,
    metrics: dict,
    segmentation_info: dict,
    processing_params: dict,
) -> str:
    """
    Exporta métricas principales a CSV tabular.

    Parameters
    ----------
    output_path : str
        Ruta del archivo CSV de salida.
    study_metadata : dict
        Metadatos del estudio.
    metrics : dict
        Métricas de fase.
    segmentation_info : dict
        Info de segmentación.
    processing_params : dict
        Parámetros de procesamiento.

    Returns
    -------
    str
        Ruta del archivo CSV generado.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Header con metadatos
    lines = [
        "# GammaSync Export v1.0",
        f"# Fecha exportación: {datetime.now().isoformat()}",
        f"# Paciente: {study_metadata.get('patient_name', 'N/D')}",
        f"# ID: {study_metadata.get('patient_id', 'N/D')}",
        f"# Estudio: {study_metadata.get('study_description', 'N/D')}",
        f"# Serie: {study_metadata.get('series_description', 'N/D')}",
        "",
        "categoria,metrica,valor,unidad",
    ]

    # Metadatos
    lines.append(f"metadata,patient_name,{study_metadata.get('patient_name', '')},")
    lines.append(f"metadata,patient_id,{study_metadata.get('patient_id', '')},")
    lines.append(f"metadata,patient_sex,{study_metadata.get('patient_sex', '')},")
    lines.append(f"metadata,study_date,{study_metadata.get('study_date', '')},")
    lines.append(f"metadata,accession_number,{study_metadata.get('accession_number', '')},")

    # Segmentación
    lines.append(f"segmentation,method,{segmentation_info.get('method', '')},")
    lines.append(f"segmentation,n_voxels,{segmentation_info.get('n_voxels', '')},voxels")
    lines.append(f"segmentation,n_slices,{segmentation_info.get('n_slices', '')},slices")

    # Procesamiento
    lines.append(f"processing,seg_method,{processing_params.get('seg_method', '')},")
    lines.append(f"processing,threshold,{processing_params.get('threshold', '')},")
    lines.append(f"processing,smooth_sigma,{processing_params.get('smooth_sigma', '')},")
    lines.append(f"processing,harmonics,{processing_params.get('harmonics', '')},")
    lines.append(f"processing,amp_filter,{processing_params.get('amp_filter', '')},")

    # Métricas voxel
    voxel_metrics = [
        ("phase_sd", "°"),
        ("bandwidth", "°"),
        ("entropy_shannon_bits", "bits"),
        ("entropy_normalized_pct", "%"),
        ("asynchrony_index", "%"),
        ("skewness", ""),
        ("kurtosis", ""),
        ("peak_phase", "°"),
        ("peak_width", "°"),
        ("latest_activation_phase", "°"),
        ("technical_classification", ""),
        ("n_voxels_kept", "voxels"),
        ("n_voxels_total", "voxels"),
    ]
    for key, unit in voxel_metrics:
        value = _safe_get(metrics, key, "")
        lines.append(f"metrics_voxel,{key},{value},{unit}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


def export_excel(
    output_path: str,
    study_metadata: dict,
    metrics: dict,
    segmentation_info: dict,
    processing_params: dict,
    robustness: dict | None = None,
) -> str:
    """
    Exporta resultados a Excel con múltiples hojas.

    Parameters
    ----------
    output_path : str
        Ruta del archivo Excel de salida.
    study_metadata : dict
        Metadatos del estudio.
    metrics : dict
        Métricas de fase.
    segmentation_info : dict
        Info de segmentación.
    processing_params : dict
        Parámetros de procesamiento.
    robustness : dict, optional
        Resultados de robustez.

    Returns
    -------
    str
        Ruta del archivo Excel generado.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas es requerido para exportar a Excel. Instalar con: pip install pandas openpyxl")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Hoja 1: Metadatos
        meta_data = {
            "Campo": [
                "Fecha exportación",
                "Paciente",
                "ID",
                "Sexo",
                "Fecha estudio",
                "Accession",
                "Descripción",
                "Serie",
                "Dimensiones",
            ],
            "Valor": [
                datetime.now().isoformat(),
                study_metadata.get("patient_name", ""),
                study_metadata.get("patient_id", ""),
                study_metadata.get("patient_sex", ""),
                study_metadata.get("study_date", ""),
                study_metadata.get("accession_number", ""),
                study_metadata.get("study_description", ""),
                study_metadata.get("series_description", ""),
                study_metadata.get("dimensions", ""),
            ],
        }
        pd.DataFrame(meta_data).to_excel(writer, sheet_name="Metadatos", index=False)

        # Hoja 2: Métricas voxel
        voxel_data = {
            "Métrica": [],
            "Valor": [],
            "Unidad": [],
        }
        for key, unit in [
            ("phase_sd", "°"),
            ("bandwidth", "°"),
            ("entropy_shannon_bits", "bits"),
            ("entropy_normalized_pct", "%"),
            ("asynchrony_index", "%"),
            ("skewness", ""),
            ("kurtosis", ""),
            ("peak_phase", "°"),
            ("peak_width", "°"),
            ("latest_activation_phase", "°"),
            ("technical_classification", ""),
        ]:
            voxel_data["Métrica"].append(key)
            voxel_data["Valor"].append(_safe_get(metrics, key, ""))
            voxel_data["Unidad"].append(unit)
        pd.DataFrame(voxel_data).to_excel(writer, sheet_name="Métricas Voxel", index=False)

        # Hoja 3: Segmentación y procesamiento
        seg_proc_data = {
            "Categoría": [],
            "Parámetro": [],
            "Valor": [],
        }
        for key, value in segmentation_info.items():
            seg_proc_data["Categoría"].append("Segmentación")
            seg_proc_data["Parámetro"].append(key)
            seg_proc_data["Valor"].append(str(value))
        for key, value in processing_params.items():
            seg_proc_data["Categoría"].append("Procesamiento")
            seg_proc_data["Parámetro"].append(key)
            seg_proc_data["Valor"].append(str(value))
        pd.DataFrame(seg_proc_data).to_excel(writer, sheet_name="Segmentación", index=False)

        # Hoja 4: Robustez (si disponible)
        if robustness:
            robust_data = {
                "Tipo": [],
                "Métrica": [],
                "Valor": [],
            }
            # Segmentario AHA
            segm = robustness.get("segmental_aha", {})
            if segm.get("available"):
                for key in ["phase_sd", "bandwidth", "entropy_normalized_pct", "n_segments"]:
                    robust_data["Tipo"].append("Segmentario AHA")
                    robust_data["Métrica"].append(key)
                    robust_data["Valor"].append(segm.get(key, ""))

            # Bootstrap
            boot = robustness.get("bootstrap", {})
            if boot.get("available"):
                for metric_name in ["phase_sd", "bandwidth"]:
                    stats = boot.get(metric_name, {})
                    for stat_name in ["mean", "ci95_low", "ci95_high"]:
                        robust_data["Tipo"].append("Bootstrap")
                        robust_data["Métrica"].append(f"{metric_name}_{stat_name}")
                        robust_data["Valor"].append(stats.get(stat_name, ""))

            # Sensibilidad ROI
            roi_sens = robustness.get("roi_sensitivity", {})
            if roi_sens.get("available"):
                for key in ["phase_sd_min", "phase_sd_max", "max_phase_sd_delta", "warn"]:
                    robust_data["Tipo"].append("Sensibilidad ROI")
                    robust_data["Métrica"].append(key)
                    robust_data["Valor"].append(roi_sens.get(key, ""))

            pd.DataFrame(robust_data).to_excel(writer, sheet_name="Robustez", index=False)

    return output_path


def export_all(
    output_dir: str,
    study_metadata: dict,
    metrics: dict,
    segmentation_info: dict,
    processing_params: dict,
    robustness: dict | None = None,
    normal_db_eval: dict | None = None,
    qc_info: dict | None = None,
    base_name: str | None = None,
) -> dict[str, str]:
    """
    Exporta todos los formatos disponibles.

    Returns
    -------
    dict
        Diccionario con rutas de archivos generados.
    """
    os.makedirs(output_dir, exist_ok=True)

    if base_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        patient_id = study_metadata.get("patient_id", "unknown")
        base_name = f"gammasync_{patient_id}_{timestamp}"

    results = {}

    # JSON
    json_path = os.path.join(output_dir, f"{base_name}.json")
    results["json"] = export_json(
        json_path,
        study_metadata,
        metrics,
        segmentation_info,
        processing_params,
        robustness,
        normal_db_eval,
        qc_info,
    )

    # CSV
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    results["csv"] = export_csv(
        csv_path,
        study_metadata,
        metrics,
        segmentation_info,
        processing_params,
    )

    # Excel (opcional, si pandas está disponible)
    try:
        excel_path = os.path.join(output_dir, f"{base_name}.xlsx")
        results["excel"] = export_excel(
            excel_path,
            study_metadata,
            metrics,
            segmentation_info,
            processing_params,
            robustness,
        )
    except ImportError:
        results["excel"] = None

    return results

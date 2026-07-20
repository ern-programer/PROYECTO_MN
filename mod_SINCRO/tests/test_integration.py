"""Test de integración end-to-end para GammaSync."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import dicom_loader
from core.segmentation import segment_myocardium
from core.phase_analysis import phase_analysis
from core.metrics import calculate_phase_metrics
from core.aha_segments import map_to_17_segments, phase_by_segment
from core.robustness import (
    bootstrap_phase_metrics,
    calculate_segmental_metrics,
    roi_sensitivity_analysis,
)
from core.export_manager import export_json, export_csv
from core.logging_config import get_logger
from core.console_utf8 import enable_utf8

enable_utf8()


def test_full_pipeline_synthetic():
    """Test E2E: pipeline completo con datos sintéticos."""
    # Crear datos sintéticos
    n_gates, n_slices, H, W = 8, 4, 22, 22
    phase_map = np.full((n_slices, H, W), 90.0)
    cube = np.zeros((n_gates, n_slices, H, W), dtype=np.float64)

    t = np.arange(n_gates)
    phi = np.radians(phase_map)
    for k in range(n_gates):
        cube[k] = 200.0 + 100.0 * np.cos(2 * np.pi * k / n_gates - phi)

    # Segmentación
    mask = np.ones((n_slices, H, W), dtype=bool)
    seg = segment_myocardium(cube, method="manual", manual_rois={
        0: (11.0, 11.0, 5.0, 8.0),
        1: (11.0, 11.0, 5.0, 8.0),
        2: (11.0, 11.0, 5.0, 8.0),
        3: (11.0, 11.0, 5.0, 8.0),
    })
    assert seg.n_voxels > 0, "Segmentación debe producir voxels"

    # Análisis de fase
    res = phase_analysis(cube, seg.mask, amplitude_threshold_frac=0.10)
    assert res.n_voxels_kept > 0, "Fase debe conservar voxels"

    # Métricas
    metrics = calculate_phase_metrics(res.phases_deg)
    assert "phase_sd" in metrics
    assert "bandwidth" in metrics
    assert "entropy_normalized_pct" in metrics
    assert abs(metrics["mean_phase"] - 90.0) < 5.0, f"Media debe ser ~90°, got {metrics['mean_phase']}"

    # AHA segments
    aha = map_to_17_segments(seg)
    phase_by_seg = phase_by_segment(res.phase_map, aha)
    assert len(phase_by_seg) > 0, "Debe haber segmentos AHA"

    # Robustez
    segmental = calculate_segmental_metrics(phase_by_seg)
    assert segmental.get("available") is True

    boot = bootstrap_phase_metrics(res.phases_deg, n_iter=10)
    assert boot.get("available") is True

    print("[OK] Pipeline E2E sintético completado")


def test_export_formats():
    """Test: exportación JSON y CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Datos de prueba
        study_meta = {
            "patient_name": "TEST^PATIENT",
            "patient_id": "TEST001",
            "patient_sex": "M",
            "study_date": "20260720",
            "study_description": "Test Study",
            "series_description": "Test Series",
            "dimensions": "8x4x22x22",
        }
        metrics = {
            "phase_sd": 15.5,
            "bandwidth": 45.2,
            "entropy_shannon_bits": 3.2,
            "entropy_normalized_pct": 35.5,
            "technical_classification": "NORMAL",
            "n_voxels_kept": 1500,
            "n_voxels_total": 2000,
        }
        seg_info = {
            "method": "auto",
            "n_voxels": 2000,
            "n_slices": 4,
        }
        proc_params = {
            "seg_method": "auto",
            "threshold": 0.35,
            "smooth_sigma": 1.0,
            "harmonics": 1,
            "amp_filter": 0.40,
        }

        # Test JSON
        json_path = os.path.join(tmpdir, "test_export.json")
        result_json = export_json(
            json_path,
            study_meta,
            metrics,
            seg_info,
            proc_params,
        )
        assert os.path.exists(result_json), "JSON debe existir"

        # Verificar contenido JSON
        import json as json_lib
        with open(result_json, "r", encoding="utf-8") as f:
            data = json_lib.load(f)
        assert data["metrics"]["voxel"]["phase_sd_deg"] == 15.5
        assert data["study"]["patient_id"] == "TEST001"

        # Test CSV
        csv_path = os.path.join(tmpdir, "test_export.csv")
        result_csv = export_csv(
            csv_path,
            study_meta,
            metrics,
            seg_info,
            proc_params,
        )
        assert os.path.exists(result_csv), "CSV debe existir"

        # Verificar contenido CSV
        with open(result_csv, "r", encoding="utf-8") as f:
            content = f.read()
        assert "phase_sd" in content
        assert "TEST001" in content

        print("[OK] Exportación JSON/CSV completada")


def test_logging():
    """Test: logging estructurado."""
    logger = get_logger()

    # Log simple
    logger.info("Test de logging", context={"test": True})

    # Verificar que el archivo de log existe
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs"
    )
    assert os.path.exists(log_dir), "Directorio de logs debe existir"

    log_files = [f for f in os.listdir(log_dir) if f.startswith("gammasync_")]
    assert len(log_files) > 0, "Debe haber al menos un archivo de log"

    print("[OK] Logging estructurado completado")


def test_real_dicom_if_available():
    """Test E2E con DICOM real si está disponible."""
    # Buscar DICOM de prueba
    test_paths = [
        r"C:\Users\Ernesto\Desktop\INTERCAMBIO\varios stress cardiacos de xeleris\uno mas\STRESS_IRNCG_SA001_DS.dcm",
        r"D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\data_test\test.dcm",
    ]

    dicom_path = None
    for path in test_paths:
        if os.path.exists(path):
            dicom_path = path
            break

    if dicom_path is None:
        print("[SKIP] No hay DICOM real disponible para test")
        return

    # Cargar y procesar
    study = dicom_loader.load(dicom_path, verbose=False)
    assert study.cube.ndim == 4, "Cubo debe ser 4D"

    seg = segment_myocardium(study.cube, method="auto", threshold_frac=0.35, smooth_sigma=1.0)
    assert seg.n_voxels > 0, "Segmentación debe producir voxels"

    res = phase_analysis(study.cube, seg.mask, amplitude_threshold_frac=0.40)
    metrics = calculate_phase_metrics(res.phases_deg)

    assert "phase_sd" in metrics
    assert res.n_voxels_kept > 0

    print(f"[OK] DICOM real procesado: {os.path.basename(dicom_path)}")
    print(f"     PSD={metrics['phase_sd']}°, BW={metrics['bandwidth']}°")


def test_ecg_extraction_and_compare():
    """Test: extracción ECG desde texto y comparación manual vs extraído."""
    from core.ecg_extractor import ECGData, compare_ecg_data, extract_from_pdf_text

    data = extract_from_pdf_text("RITMO SINUSAL FC: 72 QRS: 88ms QT: 390ms")
    assert data.ritmo == "Sinusal"
    assert data.fc == 72
    assert data.qrs_ms == 88
    assert data.qt_ms == 390
    assert data.qtc_ms > 0
    assert data.bri is False

    bri = extract_from_pdf_text("BLOQUEO COMPLETO DE RAMA IZQUIERDA QRS: 152 FC: 68")
    assert bri.bri is True
    assert bri.qrs_ms == 152

    neg = extract_from_pdf_text("ECG normal. BRI: NO. QRS 90")
    assert neg.bri is False

    manual = ECGData(ritmo="Sinusal", fc=70, qrs_ms=90)
    extraido = extract_from_pdf_text("RITMO SINUSAL FC: 95 QRS: 88")
    comp = compare_ecg_data(manual, extraido)
    assert comp["has_differences"] is True
    assert any(d["field"] == "fc" for d in comp["differences"])
    print("[OK] Extracción y comparación ECG completada")


def _run_all():
    test_full_pipeline_synthetic()
    test_export_formats()
    test_logging()
    test_ecg_extraction_and_compare()
    test_real_dicom_if_available()
    print("\n[TODOS LOS TESTS DE INTEGRACIÓN PASARON]")


if __name__ == "__main__":
    _run_all()

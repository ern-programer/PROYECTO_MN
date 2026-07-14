"""
Test del dicom_loader con el archivo Short Axis gated REAL de Xeleris (montage).

Referencia externa (NO versionada). Ajustar SA_GATED_PATH si cambia la ubicación.
Estructura esperada (validada en Fase 0): 8 gates × 19 slices, montage 418=19×22,
frame 0 sumado descartado, 1er armónico ~0.73 (late).

Correr:  python -m pytest tests/test_loader.py -v
     o:  python tests/test_loader.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import dicom_loader  # noqa: E402
from core.console_utf8 import enable_utf8  # noqa: E402

enable_utf8()

SA_GATED_PATH = (
    r"C:\Users\Ernesto\Desktop\INTERCAMBIO\varios stress cardiacos de xeleris"
    r"\estudio uno\MYOMETRIX\myometrix results\REST_IRNCG_SA001_DS.dcm"
)


def test_load_montage_gated():
    if not os.path.exists(SA_GATED_PATH):
        print(f"[SKIP] no existe el DICOM de referencia: {SA_GATED_PATH}")
        return
    study = dicom_loader.load(SA_GATED_PATH, verbose=True)
    assert study.was_montage, "Debería detectar montage (cols=19×rows)."
    assert study.had_summed_frame, "Debería detectar y descartar el frame sumado."
    assert study.n_gates == 8, f"Esperaba 8 gates, obtuvo {study.n_gates}."
    assert study.n_slices == 19, f"Esperaba 19 slices, obtuvo {study.n_slices}."
    assert study.qc_passed, f"QC latido debería pasar (frac={study.qc_first_harmonic:.3f})."
    print("\n[OK] Loader validado sobre el SA gated real.")


if __name__ == "__main__":
    test_load_montage_gated()

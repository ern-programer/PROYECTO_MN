"""
SINCRO - entry point.

Por ahora solo expone el loader. La UI (Fase 6) y el pipeline completo se agregan luego.

Uso:
    python main.py <archivo_SA_gated.dcm>     # carga y muestra resumen + auto-QC
"""
import sys

from core import dicom_loader
from core.console_utf8 import enable_utf8


def main(argv: list[str]) -> int:
    enable_utf8()
    if len(argv) < 2:
        print(__doc__)
        return 1
    try:
        study = dicom_loader.load(argv[1], verbose=True)
    except dicom_loader.LoaderError as e:
        print(f"[LoaderError] {e}")
        return 2
    print("\n[Loader OK]" if study.qc_passed else "\n[Loader con advertencias de QC]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

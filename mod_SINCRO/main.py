"""SINCRO - entry point.

Uso:
    python main.py                     # abre la interfaz visual
    python main.py archivo.dcm         # abre la interfaz y carga el estudio
"""
from __future__ import annotations

import sys

from core.console_utf8 import enable_utf8
from version import __version__


def main(argv: list[str]) -> int:
    enable_utf8()

    file_path = argv[1] if len(argv) > 1 and not argv[1].startswith("-") else None

    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print("PyQt6 no está instalado. Instala las dependencias del módulo y vuelve a intentar.")
        return 2

    app = QApplication(argv)
    app.setApplicationName("GammaSync")
    app.setApplicationDisplayName(f"GammaSync v{__version__}")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("Gammasys")

    from ui.main_window import MainWindow

    window = MainWindow(initial_path=file_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

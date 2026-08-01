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

    # Tema visual: se aplica el tema guardado por el usuario (default "classic"
    # = nativo). El QSS moderno queda como opción seleccionable desde el panel de
    # Configuración. Si falta el .qss, el tema moderno cae a nativo sin romper.
    from ui.theme_manager import apply_theme
    apply_theme(app)

    from ui.main_window import MainWindow

    try:
        window = MainWindow(initial_path=file_path)
    except Exception:
        import traceback, os
        log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "startup_error.txt")
        with open(log, "w", encoding="utf-8") as fh:
            traceback.print_exc(file=fh)
        traceback.print_exc()
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

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

    # Capturar excepciones no manejadas en slots Qt (PyQt6 aborta el proceso):
    # deja el traceback en crash_error.txt junto a main.py antes del abort.
    import os
    import traceback
    _crash_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash_error.txt")
    _prev_hook = sys.excepthook

    # Crash duro (access violation / illegal instruction): faulthandler deja
    # el stack nativo-python en crash_native.txt.
    import faulthandler
    try:
        _fh_file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash_native.txt"), "a", encoding="utf-8")
        faulthandler.enable(file=_fh_file)
    except Exception:
        pass

    def _crash_hook(exc_type, exc_value, exc_tb):
        try:
            with open(_crash_path, "a", encoding="utf-8") as fh:
                from datetime import datetime
                fh.write(f"\n=== {datetime.now().isoformat()} ===\n")
                fh.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
        except Exception:
            pass
        _prev_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = _crash_hook

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

    window = MainWindow(initial_path=file_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

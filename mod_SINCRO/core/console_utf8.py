"""
SINCRO - core.console_utf8
==========================

Fuerza la salida de consola a UTF-8. En Windows la consola por defecto usa cp1252
y no puede imprimir caracteres como '→', 'í', etc. → UnicodeEncodeError.

Importar y llamar `enable_utf8()` al inicio de cualquier script con salida por consola.
"""
from __future__ import annotations

import sys


def enable_utf8() -> None:
    """Reconfigura stdout/stderr a UTF-8 si es posible (Python 3.7+)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

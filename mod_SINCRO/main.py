"""SINCRO - entry point.

Uso:
    python main.py                     # abre la interfaz visual
    python main.py archivo.dcm         # abre la interfaz y carga el estudio
"""
from __future__ import annotations

import hashlib
import os
import sys

from core.console_utf8 import enable_utf8
from version import __version__


def _activate_existing_window(window) -> None:
    if window._active_detached_console in ("modern", "plus"):
        window.bring_epar_plus_console_to_front()
        return
    if window.isMinimized():
        window.showNormal()
    else:
        window.show()
    window.raise_()
    window.activateWindow()


def _claim_application_instance(app, activate):
    from PyQt6.QtCore import QLockFile, QStandardPaths
    from PyQt6.QtNetwork import QLocalServer, QLocalSocket

    user_key = hashlib.sha256(os.path.expanduser("~").encode("utf-8")).hexdigest()[:16]
    server_name = f"Gammasys.GammaSync.{user_key}"
    lock = QLockFile(os.path.join(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.TempLocation), f"GammaSync-{user_key}.lock"))
    lock.setStaleLockTime(0)
    server = QLocalServer(app)
    server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)

    def accept_connections():
        while server.hasPendingConnections():
            connection = server.nextPendingConnection()

            def receive_request(connection=connection):
                if connection.canReadLine() and bytes(connection.readLine()).strip() == b"activate":
                    activate()
                    connection.disconnectFromServer()

            connection.readyRead.connect(receive_request)
            connection.disconnected.connect(connection.deleteLater)
            connection.write(f"{os.getpid()}\n".encode("ascii"))
            receive_request()

    server.newConnection.connect(accept_connections)
    if lock.tryLock(0):
        if not server.listen(server_name):
            lock.unlock()
            raise RuntimeError("No se pudo iniciar el canal de activación de GammaSync.")
        server._instance_lock = lock
        app.aboutToQuit.connect(server.close)
        app.aboutToQuit.connect(lock.unlock)
        return server

    socket = QLocalSocket()
    socket.connectToServer(server_name)
    if not socket.waitForConnected(1500):
        raise RuntimeError("No se pudo contactar la instancia existente de GammaSync.")
    if socket.waitForReadyRead(1000) and os.name == "nt":
        try:
            import ctypes
            process_id = int(bytes(socket.readLine()).strip())
            ctypes.windll.user32.AllowSetForegroundWindow(process_id)
        except (ValueError, OSError, AttributeError):
            pass
    if socket.write(b"activate\n") < 0 or (socket.bytesToWrite() and not socket.waitForBytesWritten(1500)):
        raise RuntimeError("No se pudo activar la instancia existente de GammaSync.")
    socket.disconnectFromServer()
    return None


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

    window = None
    activation_pending = False

    def activate_existing():
        nonlocal activation_pending
        if window is None:
            activation_pending = True
        else:
            _activate_existing_window(window)

    try:
        instance_server = _claim_application_instance(app, activate_existing)
    except RuntimeError as exc:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(None, "GammaSync", str(exc))
        return 3
    if instance_server is None:
        return 0

    # Tema visual: se aplica el tema guardado por el usuario (default "classic"
    # = nativo). El QSS moderno queda como opción seleccionable desde el panel de
    # Configuración. Si falta el .qss, el tema moderno cae a nativo sin romper.
    from ui.theme_manager import apply_theme
    apply_theme(app)

    from ui.main_window import MainWindow

    window = MainWindow(initial_path=file_path)
    window.show()
    if activation_pending:
        _activate_existing_window(window)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

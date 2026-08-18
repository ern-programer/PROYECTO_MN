# -*- coding: utf-8 -*-
"""SINCRO — Almacén de hashes SHA-256 para verificación de integridad de informes.

Cada informe HTML generado recibe un hash SHA-256 que se guarda en una carpeta
local de SINCRO (NO junto al HTML). Esto permite verificar que el archivo
entregado no fue modificado después de su generación.

La carpeta de hashes es configurable (default: report_hashes/ junto al módulo).
La retención es configurable: por cantidad máxima de archivos o por días.

Uso:
    from report.hash_store import HashStore
    store = HashStore()
    entry = store.register(html_bytes, html_filename, patient_name, study_uid)
    ok, msg = store.verify(html_path)
    store.cleanup()
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path


# Directorio default: junto al módulo (mod_SINCRO/report_hashes/)
_DEFAULT_STORE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "report_hashes",
)


class HashStore:
    """Almacén de hashes SHA-256 para verificación de integridad."""

    def __init__(self, store_dir: str | None = None):
        self._dir = store_dir or _DEFAULT_STORE_DIR
        os.makedirs(self._dir, exist_ok=True)

    @property
    def store_dir(self) -> str:
        return self._dir

    def register(
        self,
        html_bytes: bytes,
        html_filename: str,
        patient_name: str = "",
        study_uid: str = "",
        study_date: str = "",
    ) -> dict:
        """Calcula SHA-256 del HTML y lo guarda como archivo .json.

        Returns
        -------
        dict con hash, filename, patient, study_uid, study_date, timestamp, path.
        """
        sha = hashlib.sha256(html_bytes).hexdigest()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_patient = "".join(c if c.isalnum() or c in "-_" else "_" for c in (patient_name or "sin_nombre"))[:40]
        safe_uid = (study_uid or "no_uid")[-20:]
        fname = f"{ts}_{safe_patient}_{safe_uid}.json"

        entry = {
            "sha256": sha,
            "html_filename": html_filename,
            "patient_name": patient_name,
            "study_uid": study_uid,
            "study_date": study_date,
            "timestamp": datetime.now().isoformat(),
            "html_size_bytes": len(html_bytes),
        }

        path = os.path.join(self._dir, fname)
        with open(path, "wb") as f:
            f.write(json.dumps(entry, ensure_ascii=False, indent=2).encode("utf-8"))

        entry["path"] = path
        return entry

    def verify(self, html_path: str) -> tuple[bool, str]:
        """Verifica un HTML contra su hash almacenado.

        Busca en todos los archivos .json del almacén el que coincida con el
        nombre del archivo HTML. Si lo encuentra, recalcula el SHA-256 y compara.

        Returns
        -------
        (ok, message) — ok=True si el hash coincide, False si no.
        """
        if not os.path.exists(html_path):
            return False, f"Archivo no encontrado: {html_path}"

        html_bytes = Path(html_path).read_bytes()
        current_hash = hashlib.sha256(html_bytes).hexdigest()
        html_name = os.path.basename(html_path)

        # Buscar la entrada más reciente para este archivo.
        best_entry = None
        best_path = None
        for jf in sorted(Path(self._dir).glob("*.json"), reverse=True):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                if data.get("html_filename") == html_name:
                    best_entry = data
                    best_path = str(jf)
                    break
            except Exception:
                continue

        if best_entry is None:
            return False, f"No se encontró hash registrado para '{html_name}' en el almacén."

        stored_hash = best_entry.get("sha256", "")
        if current_hash == stored_hash:
            ts = best_entry.get("timestamp", "?")
            patient = best_entry.get("patient_name", "?")
            return True, (
                f"INTEGRIDAD VERIFICADA ✓\n"
                f"Hash SHA-256 coincide con el registrado.\n"
                f"Paciente: {patient}\n"
                f"Generado: {ts}\n"
                f"Hash: {stored_hash[:32]}..."
            )
        else:
            return False, (
                f"⚠ ARCHIVO MODIFICADO\n"
                f"El hash actual NO coincide con el registrado.\n"
                f"Registrado: {stored_hash[:32]}...\n"
                f"Actual:    {current_hash[:32]}...\n"
                f"El informe fue alterado después de su generación."
            )

    def cleanup(self, max_files: int = 200, max_days: int = 90) -> int:
        """Limpia hashes antiguos según retención configurada.

        Parameters
        ----------
        max_files : cantidad máxima de archivos a conservar (0 = sin límite).
        max_days : días de retención (0 = sin límite).

        Returns
        -------
        Cantidad de archivos eliminados.
        """
        files = sorted(Path(self._dir).glob("*.json"), key=lambda p: p.stat().st_mtime)
        removed = 0

        # Eliminar por días.
        if max_days > 0:
            cutoff = time.time() - max_days * 86400
            for f in files:
                if f.stat().st_mtime < cutoff:
                    try:
                        f.unlink()
                        removed += 1
                    except OSError:
                        pass
            files = sorted(Path(self._dir).glob("*.json"), key=lambda p: p.stat().st_mtime)

        # Eliminar por cantidad (los más viejos primero).
        if max_files > 0 and len(files) > max_files:
            to_remove = files[:len(files) - max_files]
            for f in to_remove:
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass

        return removed

    def list_entries(self, limit: int = 20) -> list[dict]:
        """Lista las últimas entradas del almacén (más recientes primero)."""
        entries = []
        for jf in sorted(Path(self._dir).glob("*.json"), reverse=True)[:limit]:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                data["_file"] = jf.name
                entries.append(data)
            except Exception:
                continue
        return entries

    def count(self) -> int:
        """Cantidad de hashes almacenados."""
        return len(list(Path(self._dir).glob("*.json")))

"""SINCRO - core.research_store — Almacén local de casos experimentales.

Acumula pares de volumen/masa miocárdica (máscara CT anatómica vs gated SPECT)
para análisis poblacional posterior (validación del método, trabajo de
investigación). Es un dato de investigación, NO clínico: no altera ningún
resultado reportado.

Almacén: SQLite en mod_SINCRO/research_data/experimental_cases.sqlite.
Exportación: CSV con todas las filas para procesar afuera.
"""
from __future__ import annotations

import csv
import os
import sqlite3
from datetime import datetime, timezone

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research_data")
_DB_PATH = os.path.join(_DB_DIR, "experimental_cases.sqlite")

_COLUMNS = (
    "ts", "patient_id", "study_date", "stage",
    "ct_myo_ml", "ct_mass_g",
    "spect_myo_ml", "spect_mass_g",
    "ef_pct", "edv_ml", "esv_ml",
    "camera_manufacturer", "camera_model",
    "diff_mass_g", "diff_pct",
)


def _connect() -> sqlite3.Connection:
    os.makedirs(_DB_DIR, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS myo_volume_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, patient_id TEXT, study_date TEXT, stage TEXT,
            ct_myo_ml REAL, ct_mass_g REAL,
            spect_myo_ml REAL, spect_mass_g REAL,
            ef_pct REAL, edv_ml REAL, esv_ml REAL,
            camera_manufacturer TEXT, camera_model TEXT,
            diff_mass_g REAL, diff_pct REAL
        )
        """
    )
    return conn


def db_path() -> str:
    """Ruta del archivo SQLite del almacén."""
    return _DB_PATH


def count_cases() -> int:
    """Cantidad de casos acumulados (0 si no existe el almacén)."""
    if not os.path.isfile(_DB_PATH):
        return 0
    try:
        conn = _connect()
        try:
            return int(conn.execute("SELECT COUNT(*) FROM myo_volume_cases").fetchone()[0])
        finally:
            conn.close()
    except Exception:
        return 0


def record_case(row: dict) -> bool:
    """Inserta un caso. Devuelve True si se grabó, False si falló.

    ``row`` puede traer cualquier subconjunto de ``_COLUMNS``; los faltantes
    quedan en NULL. ``ts`` se completa con la hora actual si no viene.
    """
    data = dict(row or {})
    data.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    cols = [c for c in _COLUMNS if c in data]
    if not cols:
        return False
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT INTO myo_volume_cases ({', '.join(cols)}) VALUES ({placeholders})"
    try:
        conn = _connect()
        try:
            conn.execute(sql, [data[c] for c in cols])
            conn.commit()
            return True
        finally:
            conn.close()
    except Exception:
        return False


def export_csv(dest_path: str) -> int:
    """Vuelca todas las filas a CSV. Devuelve la cantidad exportada."""
    if not os.path.isfile(_DB_PATH):
        return 0
    conn = _connect()
    try:
        cur = conn.execute(
            f"SELECT id, {', '.join(_COLUMNS)} FROM myo_volume_cases ORDER BY id"
        )
        header = ["id", *_COLUMNS]
        rows = cur.fetchall()
    finally:
        conn.close()
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
    with open(dest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return len(rows)

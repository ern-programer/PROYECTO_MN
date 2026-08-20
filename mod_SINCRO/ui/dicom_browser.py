# -*- coding: utf-8 -*-
"""Navegador de imágenes DICOM con thumbnails.

Escanea una carpeta (recursivamente) buscando archivos DICOM (.dcm),
genera thumbnails y muestra metadatos clave (paciente, fecha, serie,
dimensiones). Permite selección múltiple para cargar en el visor.

Útil cuando los archivos tienen nombres crípticos (ej: 1.2.840.xxx.dcm).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QListWidget, QListWidgetItem, QSplitter,
    QWidget, QSizePolicy, QProgressBar, QTextEdit, QCheckBox,
)


def _is_dicom_file(path: str) -> bool:
    """Detecta si un archivo es DICOM leyendo el magic number (DICM en offset 128)."""
    try:
        with open(path, "rb") as f:
            f.seek(128)
            magic = f.read(4)
            return magic == b"DICM"
    except Exception:
        return False


def _read_dicom_thumb(path: str, thumb_size: int = 96) -> tuple[QPixmap, dict]:
    """Lee un DICOM y devuelve (thumbnail, metadata_dict)."""
    try:
        import pydicom
        ds = pydicom.dcmread(path, stop_before_pixels=False, force=True)
        arr = ds.pixel_array.astype(np.float64)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[-2], arr.shape[-1]) if arr.ndim == 3 else arr[0, 0]
        # Normalizar a 0-255.
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-8:
            norm = np.zeros_like(arr, dtype=np.uint8)
        else:
            norm = np.clip((arr - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)
        h, w = norm.shape
        rgb = np.stack([norm, norm, norm], axis=-1)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())
        pix = pix.scaled(QSize(thumb_size, thumb_size), Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
        meta = {
            "patient": str(getattr(ds, "PatientName", "N/D")),
            "date": str(getattr(ds, "StudyDate", "N/D")),
            "series": str(getattr(ds, "SeriesDescription", "N/D")),
            "modality": str(getattr(ds, "Modality", "N/D")),
            "rows": int(getattr(ds, "Rows", h)),
            "cols": int(getattr(ds, "Columns", w)),
            "path": path,
        }
        return pix, meta
    except Exception:
        # Fallback: pixmap vacío.
        pix = QPixmap(thumb_size, thumb_size)
        pix.fill(QColor("#1e293b"))
        return pix, {"patient": "?", "date": "?", "series": "?", "modality": "?",
                      "rows": 0, "cols": 0, "path": path}


class DicomBrowserDialog(QDialog):
    """Diálogo para explorar y seleccionar archivos DICOM con thumbnails."""

    filesSelected = pyqtSignal(list)  # lista de paths seleccionados

    def __init__(self, parent=None, start_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Navegador DICOM — Seleccionar imágenes")
        self.resize(900, 600)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        self._start_dir = start_dir
        self._items: list[tuple[QPixmap, dict]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Barra superior: carpeta + opciones ─────────────────────
        top = QHBoxLayout()
        self._lbl_folder = QLabel(start_dir or "(ninguna carpeta seleccionada)")
        self._lbl_folder.setStyleSheet("color: #94a3b8; font-size: 11px;")
        top.addWidget(self._lbl_folder, 1)

        self._chk_subdirs = QCheckBox("Subcarpetas")
        self._chk_subdirs.setChecked(True)
        self._chk_subdirs.setStyleSheet("color: #e2e8f0;")
        top.addWidget(self._chk_subdirs)

        btn_folder = QPushButton("Carpeta...")
        btn_folder.clicked.connect(self._select_folder)
        top.addWidget(btn_folder)
        root.addLayout(top)

        # ── Progreso ───────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setMaximumHeight(6)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet("QProgressBar { background: #1e293b; border: none; } QProgressBar::chunk { background: #38bdf8; }")
        root.addWidget(self._progress)

        # ── Contenido: splitter con lista + preview ────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Lista de thumbnails (grid con QListWidget en modo IconMode).
        self._list = QListWidget()
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setIconSize(QSize(96, 96))
        self._list.setGridSize(QSize(120, 130))
        self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._list.setWrapping(True)
        self._list.setSpacing(4)
        self._list.setStyleSheet("QListWidget { background: #0b1220; border: 1px solid #334155; border-radius: 6px; }")
        self._list.currentItemChanged.connect(self._on_selection_changed)
        splitter.addWidget(self._list)

        # Panel de detalle (preview grande + metadata).
        detail = QVBoxLayout()
        self._lbl_preview = QLabel("Seleccioná una imagen")
        self._lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_preview.setMinimumSize(256, 256)
        self._lbl_preview.setStyleSheet("background: #0f172a; border: 1px solid #334155; border-radius: 8px; color: #94a3b8;")
        detail.addWidget(self._lbl_preview, 1)

        self._txt_meta = QTextEdit()
        self._txt_meta.setReadOnly(True)
        self._txt_meta.setMaximumHeight(140)
        self._txt_meta.setStyleSheet("background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 6px; font-size: 11px;")
        detail.addWidget(self._txt_meta)

        detail_widget = QWidget()
        detail_widget.setLayout(detail)
        splitter.addWidget(detail_widget)
        splitter.setSizes([500, 400])
        root.addWidget(splitter, 1)

        # ── Botones inferiores ─────────────────────────────────────
        btns = QHBoxLayout()
        self._lbl_count = QLabel("0 seleccionados")
        self._lbl_count.setStyleSheet("color: #94a3b8; font-size: 11px;")
        btns.addWidget(self._lbl_count)
        btns.addStretch(1)

        btn_all = QPushButton("Seleccionar todo")
        btn_all.clicked.connect(self._list.selectAll)
        btns.addWidget(btn_all)

        btn_ok = QPushButton("Cargar seleccionados")
        btn_ok.setStyleSheet("background: #2563eb; color: white; font-weight: bold; padding: 6px 16px; border-radius: 6px;")
        btn_ok.clicked.connect(self._accept)
        btns.addWidget(btn_ok)

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_cancel)
        root.addLayout(btns)

        # Auto-scanear si hay directorio inicial.
        if start_dir and os.path.isdir(start_dir):
            self._scan_folder(start_dir)

    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta con DICOMs", self._start_dir)
        if folder:
            self._lbl_folder.setText(folder)
            self._scan_folder(folder)

    def _scan_folder(self, folder: str):
        """Escanea la carpeta en un hilo para no bloquear la UI."""
        self._list.clear()
        self._items.clear()
        self._progress.setValue(0)

        # Recolectar archivos DICOM (por extensión o por magic number).
        extensions = (".dcm", ".DCM")
        files = []
        if self._chk_subdirs.isChecked():
            for root_dir, _, fnames in os.walk(folder):
                for fn in fnames:
                    full = os.path.join(root_dir, fn)
                    if fn.endswith(extensions) or _is_dicom_file(full):
                        files.append(full)
        else:
            for fn in os.listdir(folder):
                full = os.path.join(folder, fn)
                if os.path.isfile(full) and (fn.endswith(extensions) or _is_dicom_file(full)):
                    files.append(full)

        if not files:
            self._lbl_preview.setText("No se encontraron archivos .dcm")
            return

        self._progress.setMaximum(len(files))
        self._lbl_count.setText(f"Escaneando {len(files)} archivos...")

        # Escanear en hilo background.
        def _scan():
            for i, fpath in enumerate(files):
                pix, meta = _read_dicom_thumb(fpath)
                self._items.append((pix, meta))
                # Señal para actualizar UI (usando metacall thread-safe).
                self._add_item_safe(pix, meta, i + 1, len(files))

        t = threading.Thread(target=_scan, daemon=True)
        t.start()

    def _add_item_safe(self, pix: QPixmap, meta: dict, current: int, total: int):
        """Agrega un item a la lista (thread-safe via invokeMethod pattern)."""
        from PyQt6.QtCore import QMetaObject, Qt as QtNamespace, Q_ARG
        # En PyQt6 podemos usar QTimer.singleShot(0, ...) para ejecutar en el hilo principal.
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, lambda: self._add_item_ui(pix, meta, current, total))

    def _add_item_ui(self, pix: QPixmap, meta: dict, current: int, total: int):
        label = os.path.basename(meta["path"])
        item = QListWidgetItem(pix, label[:20])
        item.setData(Qt.ItemDataRole.UserRole, meta)
        item.setToolTip(f"{meta['patient']}\n{meta['date']} — {meta['series']}\n{meta['rows']}×{meta['cols']} ({meta['modality']})")
        self._list.addItem(item)
        self._progress.setValue(current)
        if current == total:
            self._lbl_count.setText(f"{total} imágenes encontradas — seleccioná las que necesites")

    def _on_selection_changed(self, current: QListWidgetItem, _previous: QListWidgetItem):
        if current is None:
            return
        meta = current.data(Qt.ItemDataRole.UserRole)
        if not meta:
            return
        # Preview grande.
        path = meta["path"]
        try:
            pix, _ = _read_dicom_thumb(path, thumb_size=400)
            self._lbl_preview.setPixmap(pix)
        except Exception:
            self._lbl_preview.setText("Error cargando preview")
        # Metadata.
        self._txt_meta.setPlainText(
            f"Paciente: {meta['patient']}\n"
            f"Fecha: {meta['date']}\n"
            f"Serie: {meta['series']}\n"
            f"Modalidad: {meta['modality']}\n"
            f"Dimensiones: {meta['rows']}×{meta['cols']}\n"
            f"Archivo: {os.path.basename(path)}"
        )
        # Actualizar contador de selección.
        n = len(self._list.selectedItems())
        self._lbl_count.setText(f"{n} seleccionados")

    def _accept(self):
        selected = []
        for item in self._list.selectedItems():
            meta = item.data(Qt.ItemDataRole.UserRole)
            if meta:
                selected.append(meta["path"])
        if not selected:
            self._lbl_count.setText("⚠ Seleccioná al menos una imagen")
            return
        self.filesSelected.emit(selected)
        self.accept()

    def selected_paths(self) -> list[str]:
        """Devuelve los paths seleccionados (después de accept)."""
        paths = []
        for item in self._list.selectedItems():
            meta = item.data(Qt.ItemDataRole.UserRole)
            if meta:
                paths.append(meta["path"])
        return paths

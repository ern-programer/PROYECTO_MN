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

from PyQt6.QtCore import Qt, QSize, QSettings, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QPixmap, QColor, QPainter
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QListWidget, QListWidgetItem, QSplitter,
    QWidget, QSizePolicy, QProgressBar, QTextEdit, QCheckBox, QComboBox,
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


def _read_dicom_thumb_data(path: str) -> tuple[np.ndarray | None, dict]:
    """Lee un DICOM y devuelve RGB uint8 + metadatos, sin objetos GUI."""
    try:
        import pydicom
        from pydicom.pixels import pixel_array

        # Leer primero solo metadatos: evita cargar PixelData completo.
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        frame_index = n_frames // 2 if n_frames > 1 else None
        # pydicom 3 permite decodificar un único frame directamente desde archivo.
        arr = pixel_array(path, index=frame_index).astype(np.float64)
        while arr.ndim > 2:
            arr = arr[arr.shape[0] // 2]
        if arr.ndim != 2:
            raise ValueError(f"Dimensiones DICOM no soportadas: {arr.shape}")
        # Normalizar a 0-255.
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-8:
            norm = np.zeros_like(arr, dtype=np.uint8)
        else:
            norm = np.clip((arr - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)
        h, w = norm.shape
        rgb = np.stack([norm, norm, norm], axis=-1)
        meta = {
            "patient": str(getattr(ds, "PatientName", "N/D")),
            "date": str(getattr(ds, "StudyDate", "N/D")),
            "series": str(getattr(ds, "SeriesDescription", "N/D")),
            "modality": str(getattr(ds, "Modality", "N/D")),
            "rows": int(getattr(ds, "Rows", h)),
            "cols": int(getattr(ds, "Columns", w)),
            "path": path,
        }
        return rgb, meta
    except Exception as exc:
        return None, {"patient": "?", "date": "?", "series": "?", "modality": "?",
                      "rows": 0, "cols": 0, "path": path, "error": str(exc)}


def _rgb_to_pixmap(rgb: np.ndarray | None, thumb_size: int) -> QPixmap:
    """Crea el QPixmap en el hilo gráfico a partir de RGB uint8."""
    if rgb is None:
        pix = QPixmap(thumb_size, thumb_size)
        pix.fill(QColor("#1e293b"))
        return pix
    h, w = rgb.shape[:2]
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy()).scaled(
        QSize(thumb_size, thumb_size),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def _read_dicom_thumb(path: str, thumb_size: int = 96) -> tuple[QPixmap, dict]:
    """Lectura síncrona para el preview grande, ejecutada en el hilo GUI."""
    rgb, meta = _read_dicom_thumb_data(path)
    return _rgb_to_pixmap(rgb, thumb_size), meta


class DicomBrowserDialog(QDialog):
    """Diálogo para explorar y seleccionar archivos DICOM con thumbnails."""

    filesSelected = pyqtSignal(list)  # lista de paths seleccionados
    thumbReady = pyqtSignal(object, object, int, int)
    MAX_RECENT_FOLDERS = 4

    def __init__(self, parent=None, start_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Navegador DICOM — Seleccionar imágenes")
        self.resize(900, 600)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        self._start_dir = start_dir
        self._items: list[tuple[QPixmap, dict]] = []
        self._scan_generation = 0
        self._settings = QSettings("PROYECTO_MN", "SINCRO")
        self.thumbReady.connect(self._add_item_ui)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Barra superior: carpeta + opciones ─────────────────────
        top = QHBoxLayout()
        self._lbl_folder = QLabel(start_dir or "(ninguna carpeta seleccionada)")
        self._lbl_folder.setStyleSheet("color: #94a3b8; font-size: 11px;")
        top.addWidget(self._lbl_folder, 1)

        self._recent_combo = QComboBox()
        self._recent_combo.setMinimumWidth(220)
        self._recent_combo.setToolTip("Últimas carpetas abiertas")
        self._recent_combo.activated.connect(self._on_recent_activated)
        top.addWidget(self._recent_combo)

        self._chk_subdirs = QCheckBox("Subcarpetas")
        # Desactivado por defecto: una carpeta de pacientes puede contener miles
        # de DICOM y no debe escanearse recursivamente por sorpresa.
        self._chk_subdirs.setChecked(False)
        self._chk_subdirs.setStyleSheet("color: #e2e8f0;")
        top.addWidget(self._chk_subdirs)

        btn_folder = QPushButton("Carpeta...")
        btn_folder.clicked.connect(self._select_folder)
        top.addWidget(btn_folder)
        root.addLayout(top)

        self._refresh_recent_folders()

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

    def _recent_folders(self) -> list[str]:
        """Devuelve las últimas carpetas abiertas, más reciente primero."""
        value = self._settings.value("dicom_browser/recent_folders", [], type=list)
        if isinstance(value, str):
            value = [value]
        folders = []
        for folder in value or []:
            folder = str(folder)
            if folder and os.path.isdir(folder) and folder not in folders:
                folders.append(folder)
        return folders[:self.MAX_RECENT_FOLDERS]

    def _remember_folder(self, folder: str):
        """Guarda la carpeta como la más reciente, con máximo 4 entradas."""
        folder = os.path.normpath(folder)
        folders = [folder]
        folders.extend(f for f in self._recent_folders() if os.path.normpath(f) != folder)
        self._settings.setValue("dicom_browser/recent_folders", folders[:self.MAX_RECENT_FOLDERS])
        self._settings.sync()
        self._refresh_recent_folders()

    def _refresh_recent_folders(self):
        """Actualiza el combo de carpetas recientes."""
        self._recent_combo.blockSignals(True)
        self._recent_combo.clear()
        self._recent_combo.addItem("Recientes…", "")
        for folder in self._recent_folders():
            self._recent_combo.addItem(folder, folder)
        self._recent_combo.setCurrentIndex(0)
        self._recent_combo.blockSignals(False)

    def _on_recent_activated(self, index: int):
        folder = self._recent_combo.itemData(index)
        if folder:
            self._lbl_folder.setText(folder)
            self._scan_folder(folder)

    def _scan_folder(self, folder: str):
        """Escanea la carpeta en un hilo para no bloquear la UI."""
        folder = os.path.normpath(folder)
        self._remember_folder(folder)
        self._list.clear()
        self._items.clear()
        self._progress.setValue(0)
        self._scan_generation += 1
        generation = self._scan_generation

        # Recolectar archivos DICOM: primero por extensión .dcm (rápido),
        # luego por magic number solo si no hay .dcm (fallback lento).
        extensions = (".dcm", ".DCM")
        files = []
        if self._chk_subdirs.isChecked():
            for root_dir, _, fnames in os.walk(folder):
                for fn in fnames:
                    if fn.endswith(extensions):
                        files.append(os.path.join(root_dir, fn))
        else:
            for fn in os.listdir(folder):
                full = os.path.join(folder, fn)
                if os.path.isfile(full) and fn.endswith(extensions):
                    files.append(full)

        # Si no encontró .dcm, buscar por magic number (más lento).
        if not files:
            self._lbl_count.setText("Buscando DICOM por contenido (sin extensión)...")
            if self._chk_subdirs.isChecked():
                for root_dir, _, fnames in os.walk(folder):
                    for fn in fnames:
                        full = os.path.join(root_dir, fn)
                        if _is_dicom_file(full):
                            files.append(full)
            else:
                for fn in os.listdir(folder):
                    full = os.path.join(folder, fn)
                    if os.path.isfile(full) and _is_dicom_file(full):
                        files.append(full)

        # Limitar a 200 archivos para no colgar la UI.
        if len(files) > 200:
            files = files[:200]
            self._lbl_count.setText(f"Mostrando primeros 200 de {len(files)} archivos")

        if not files:
            self._lbl_preview.setText("No se encontraron archivos .dcm")
            return

        self._progress.setMaximum(len(files))
        self._lbl_count.setText(f"Escaneando {len(files)} archivos...")

        # Escanear en hilo background.
        def _scan():
            for i, fpath in enumerate(files):
                if generation != self._scan_generation:
                    return
                rgb, meta = _read_dicom_thumb_data(fpath)
                # La señal cruza al hilo GUI; allí se crean QImage/QPixmap/items.
                self.thumbReady.emit(rgb, meta, i + 1, len(files))

        t = threading.Thread(target=_scan, daemon=True)
        t.start()

    def _add_item_ui(self, rgb: np.ndarray | None, meta: dict, current: int, total: int):
        pix = _rgb_to_pixmap(rgb, 96)
        self._items.append((pix, meta))
        label = os.path.basename(meta["path"])
        item = QListWidgetItem(QIcon(pix), label[:20])
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

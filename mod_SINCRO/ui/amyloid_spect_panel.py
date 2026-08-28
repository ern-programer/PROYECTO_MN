# -*- coding: utf-8 -*-
"""Panel AMYLO SPECT 3D (fase 2, experimental)."""
from __future__ import annotations

import os
import base64
import json
import numpy as np

from PyQt6.QtCore import QObject, Qt, QSettings, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPdfWriter, QPageSize, QPen, QColor, QFont, QTransform, QPolygonF
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QTextEdit,
    QTextBrowser,
    QCheckBox,
    QGridLayout,
    QSlider,
    QProgressBar,
    QGroupBox,
    QDoubleSpinBox,
    QSpinBox,
    QScrollArea,
    QMessageBox,
    QInputDialog,
    QSizePolicy,
)

from pydicom.dataset import Dataset

from scipy import ndimage as ndi

from core.col_registry import available_colormaps, register_all_colormaps
from ui.cine_widget import RangeSlider
from ui.mip_rotator_widget import MipRotatorWidget

from core.amyloid_spect import (
    run_amyloid_spect_analysis,
    reconstruct_amyloid_with_perf_pipeline,
    export_amyloid_cardiac_axes_dicom,
    apply_visual_bone_suppression,
    central_slices_preview,
    load_ct_volume_from_path,
    load_attenuation_map_from_path,
    apply_attenuation_correction_prototype,
    apply_attenuation_correction_chang,
    resample_volume_to_spect_grid,
    register_ct_to_spect_rigid,
    refine_ct_to_spect_translation,
    # HMR-SPECT
    VOISphere,
    VOIAnatomical,
    HmrSpectResult,
    HmrSpectMethod,
    compute_hmr_spect,
    create_voi_from_localization,
    create_anatomical_heart_voi,
    create_bone_safe_mediastinum_voi,
    # S/VD ratio
    SvdRatioResult,
    compute_spect_ratio,
)


class _AmyloidReconWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        dicom_path: str,
        recon_config,
        cuts_mode: str,
        attenuation_mu_map: np.ndarray | None = None,
        attenuation_pixel_size_cm: float | None = None,
    ):
        super().__init__()
        self._dicom_path = dicom_path
        self._recon_config = recon_config
        self._cuts_mode = cuts_mode
        self._attenuation_mu_map = None if attenuation_mu_map is None else np.asarray(attenuation_mu_map, dtype=np.float64)
        self._attenuation_pixel_size_cm = attenuation_pixel_size_cm
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            def _progress(frac: float, message: str):
                if self._cancel_requested:
                    raise RuntimeError("Reconstrucción cancelada por el usuario.")
                pct = int(round(100.0 * max(0.0, min(1.0, float(frac)))))
                self.progress.emit(pct, str(message or "Procesando..."))

            bundle = reconstruct_amyloid_with_perf_pipeline(
                self._dicom_path,
                recon_config=self._recon_config,
                cuts_mode=self._cuts_mode,
                attenuation_mu_map=self._attenuation_mu_map,
                attenuation_pixel_size_cm=self._attenuation_pixel_size_cm,
                progress_callback=_progress,
            )
            self.finished.emit(bundle)
        except Exception as exc:
            self.failed.emit(str(exc))


class FusionReportLayoutDialog(QDialog):
    """Vista de presentación para informe: tiras SPECT + panel fusión 3x3 con referencias."""

    _PRESET_LAYOUTS = {
        "Clásico vertical": {"strip_w": 980, "strip_h": 130, "grid_w": 280, "grid_h": 220, "line_px": 1},
        "Compacto": {"strip_w": 860, "strip_h": 110, "grid_w": 250, "grid_h": 200, "line_px": 1},
        "Presentación amplia": {"strip_w": 1080, "strip_h": 160, "grid_w": 300, "grid_h": 235, "line_px": 1},
        "Informe clínico A4": {"strip_w": 920, "strip_h": 124, "grid_w": 245, "grid_h": 188, "line_px": 1},
    }

    def __init__(
        self,
        parent,
        spect_vol: np.ndarray,
        ct_vol: np.ndarray | None,
        fusion_pct: int,
        spect_window_fn,
        ct_window_fn,
        cmap_fn,
        slice_idx: dict,
        localization_points: list[dict] | None = None,
        display_spacing_zyx: tuple[float, float, float] | None = None,
        hmr_result: HmrSpectResult | None = None,
        svd_result: SvdRatioResult | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("SINCRO — Vista informe fusión")
        self.resize(1240, 860)
        self.setMinimumSize(980, 680)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMinMaxButtonsHint
        )

        self._spect_vol = np.asarray(spect_vol, dtype=np.float64)
        if self._spect_vol.ndim != 3:
            raise ValueError(f"SPECT inválido para vista informe (se esperaba 3D, recibido {self._spect_vol.shape}).")
        self._ct_vol = None if ct_vol is None else np.asarray(ct_vol, dtype=np.float64)
        if self._ct_vol is not None and self._ct_vol.ndim != 3:
            self._ct_vol = None
        self._display_spacing_zyx = display_spacing_zyx
        self._fusion_pct = int(np.clip(fusion_pct, 0, 100))
        self._spect_window_fn = spect_window_fn
        self._ct_window_fn = ct_window_fn
        self._cmap_fn = cmap_fn
        self._slice_idx = {
            "axial": int(slice_idx.get("axial", self._spect_vol.shape[0] // 2)),
            "coronal": int(slice_idx.get("coronal", self._spect_vol.shape[1] // 2)),
            "sagittal": int(slice_idx.get("sagittal", self._spect_vol.shape[2] // 2)),
        }
        self._localization_points = list(localization_points or [])
        self._hmr_result: HmrSpectResult | None = hmr_result
        self._svd_result: SvdRatioResult | None = svd_result
        self._settings = QSettings("GAMMASYS", "SINCRO_AMYLO_SPECT")
        self._custom_layouts = self._load_custom_layouts()
        self._line_px = 1

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        hdr = QLabel(
            "Presentación de fusión · Filas: SPECT / CT / Fusión · "
            "Columnas: Axial / Coronal / Sagital · Referencias de corte activas"
        )
        hdr.setStyleSheet("color:#cbd5e1; font-weight:600;")
        root.addWidget(hdr)

        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Layout:"))
        self._layout_combo = QComboBox()
        self._layout_combo.addItems(list(self._PRESET_LAYOUTS.keys()) + list(self._custom_layouts.keys()))
        self._layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        ctl.addWidget(self._layout_combo)

        self._btn_save_layout = QPushButton("Guardar layout")
        self._btn_save_layout.clicked.connect(self._save_custom_layout)
        ctl.addWidget(self._btn_save_layout)

        self._btn_rename_layout = QPushButton("Renombrar layout")
        self._btn_rename_layout.clicked.connect(self._rename_custom_layout)
        ctl.addWidget(self._btn_rename_layout)

        self._btn_delete_layout = QPushButton("Eliminar layout")
        self._btn_delete_layout.clicked.connect(self._delete_custom_layout)
        ctl.addWidget(self._btn_delete_layout)

        ctl.addWidget(QLabel("Tiras W/H:"))
        self._strip_w_spin = QSpinBox()
        self._strip_w_spin.setRange(500, 1800)
        self._strip_w_spin.setValue(980)
        self._strip_w_spin.valueChanged.connect(self._render)
        ctl.addWidget(self._strip_w_spin)
        self._strip_h_spin = QSpinBox()
        self._strip_h_spin.setRange(70, 260)
        self._strip_h_spin.setValue(130)
        self._strip_h_spin.valueChanged.connect(self._render)
        ctl.addWidget(self._strip_h_spin)

        ctl.addWidget(QLabel("Grilla W/H:"))
        self._grid_w_spin = QSpinBox()
        self._grid_w_spin.setRange(180, 420)
        self._grid_w_spin.setValue(280)
        self._grid_w_spin.valueChanged.connect(self._render)
        ctl.addWidget(self._grid_w_spin)
        self._grid_h_spin = QSpinBox()
        self._grid_h_spin.setRange(150, 360)
        self._grid_h_spin.setValue(220)
        self._grid_h_spin.valueChanged.connect(self._render)
        ctl.addWidget(self._grid_h_spin)

        ctl.addWidget(QLabel("Línea px:"))
        self._line_px_spin = QSpinBox()
        self._line_px_spin.setRange(1, 3)
        self._line_px_spin.setValue(1)
        self._line_px_spin.valueChanged.connect(self._on_line_width_changed)
        ctl.addWidget(self._line_px_spin)

        self._pdf_a4_mode_check = QCheckBox("PDF A4 clínico")
        self._pdf_a4_mode_check.setToolTip("Si está activo, al exportar PDF se usa temporalmente el preset 'Informe clínico A4'.")
        self._pdf_a4_mode_check.setChecked(
            bool(int(self._settings.value("fusion_report/pdf_use_a4_clinical", 1) or 1))
        )
        self._pdf_a4_mode_check.toggled.connect(self._on_pdf_a4_mode_toggled)
        ctl.addWidget(self._pdf_a4_mode_check)
        ctl.addStretch(1)
        root.addLayout(ctl)

        self._body = QWidget()
        body_lay = QVBoxLayout(self._body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(8)

        strip_title = QLabel("Tiras SPECT (axial/coronal/sagital)")
        strip_title.setStyleSheet("color:#94a3b8; font-weight:600;")
        body_lay.addWidget(strip_title)

        self._strip_col = QVBoxLayout()
        self._strip_col.setSpacing(6)
        body_lay.addLayout(self._strip_col)

        grid_title = QLabel("Muestra típica fusión 3x3")
        grid_title.setStyleSheet("color:#94a3b8; font-weight:600;")
        body_lay.addWidget(grid_title)

        self._grid = QGridLayout()
        self._grid.setHorizontalSpacing(8)
        self._grid.setVerticalSpacing(8)
        body_lay.addLayout(self._grid)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._body)
        root.addWidget(scroll, 1)

        btns = QHBoxLayout()
        btn_export_png = QPushButton("Exportar PNG")
        btn_export_png.clicked.connect(self._export_png)
        btns.addWidget(btn_export_png)
        btn_export_pdf = QPushButton("Exportar PDF")
        btn_export_pdf.clicked.connect(self._export_pdf)
        btns.addWidget(btn_export_pdf)
        btn_export_sr = QPushButton("Exportar DICOM-SR")
        btn_export_sr.clicked.connect(self._export_dicom_sr)
        btn_export_sr.setToolTip("Exporta puntos de localización como Structured Report (TID 1411)")
        btns.addWidget(btn_export_sr)
        btns.addStretch(1)
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.accept)
        btns.addWidget(btn_close)
        root.addLayout(btns)

        selected = str(self._settings.value("fusion_report/layout_selected", "Clásico vertical") or "Clásico vertical")
        idx = self._layout_combo.findText(selected)
        if idx >= 0:
            self._layout_combo.setCurrentIndex(idx)
        else:
            self._apply_layout_values(self._PRESET_LAYOUTS["Clásico vertical"])

        try:
            self._render()
        except Exception as exc:
            QMessageBox.critical(self, "SINCRO", f"Error armando vista de informe:\n{exc}")

    def _load_custom_layouts(self) -> dict[str, dict]:
        raw = str(self._settings.value("fusion_report/layouts_json", "") or "")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            out = {}
            for k, v in dict(data).items():
                if not isinstance(v, dict):
                    continue
                out[str(k)] = {
                    "strip_w": int(v.get("strip_w", 980)),
                    "strip_h": int(v.get("strip_h", 130)),
                    "grid_w": int(v.get("grid_w", 280)),
                    "grid_h": int(v.get("grid_h", 220)),
                    "line_px": int(v.get("line_px", 1)),
                }
            return out
        except Exception:
            return {}

    def _save_custom_layouts(self) -> None:
        try:
            self._settings.setValue("fusion_report/layouts_json", json.dumps(self._custom_layouts, ensure_ascii=False))
            self._settings.sync()
        except Exception:
            pass

    def _current_layout_values(self) -> dict:
        return {
            "strip_w": int(self._strip_w_spin.value()),
            "strip_h": int(self._strip_h_spin.value()),
            "grid_w": int(self._grid_w_spin.value()),
            "grid_h": int(self._grid_h_spin.value()),
            "line_px": int(self._line_px_spin.value()),
        }

    def _apply_layout_values(self, values: dict) -> None:
        self._strip_w_spin.blockSignals(True)
        self._strip_h_spin.blockSignals(True)
        self._grid_w_spin.blockSignals(True)
        self._grid_h_spin.blockSignals(True)
        self._line_px_spin.blockSignals(True)
        self._strip_w_spin.setValue(int(values.get("strip_w", 980)))
        self._strip_h_spin.setValue(int(values.get("strip_h", 130)))
        self._grid_w_spin.setValue(int(values.get("grid_w", 280)))
        self._grid_h_spin.setValue(int(values.get("grid_h", 220)))
        self._line_px_spin.setValue(int(values.get("line_px", 1)))
        self._line_px_spin.blockSignals(False)
        self._grid_h_spin.blockSignals(False)
        self._grid_w_spin.blockSignals(False)
        self._strip_h_spin.blockSignals(False)
        self._strip_w_spin.blockSignals(False)
        self._line_px = int(self._line_px_spin.value())
        self._render()

    def _on_layout_changed(self):
        name = str(self._layout_combo.currentText() or "")
        values = self._PRESET_LAYOUTS.get(name) or self._custom_layouts.get(name)
        if values:
            self._apply_layout_values(values)
        self._settings.setValue("fusion_report/layout_selected", name)

    def _on_line_width_changed(self, value: int):
        self._line_px = max(1, int(value))
        self._render()

    def _on_pdf_a4_mode_toggled(self, checked: bool):
        self._settings.setValue("fusion_report/pdf_use_a4_clinical", 1 if checked else 0)
        self._settings.sync()

    def _save_custom_layout(self):
        name, ok = QInputDialog.getText(self, "Guardar layout", "Nombre del layout:")
        if not ok or not str(name).strip():
            return
        layout_name = str(name).strip()
        self._custom_layouts[layout_name] = self._current_layout_values()
        self._save_custom_layouts()
        self._refresh_layout_combo(layout_name)

    def _refresh_layout_combo(self, selected_name: str = "") -> None:
        current = selected_name or str(self._layout_combo.currentText() or "")
        self._layout_combo.blockSignals(True)
        self._layout_combo.clear()
        self._layout_combo.addItems(list(self._PRESET_LAYOUTS.keys()) + list(self._custom_layouts.keys()))
        idx = self._layout_combo.findText(current)
        if idx >= 0:
            self._layout_combo.setCurrentIndex(idx)
        elif self._layout_combo.count() > 0:
            self._layout_combo.setCurrentIndex(0)
        self._layout_combo.blockSignals(False)
        self._on_layout_changed()

    def _rename_custom_layout(self):
        current = str(self._layout_combo.currentText() or "").strip()
        if not current:
            return
        if current in self._PRESET_LAYOUTS:
            QMessageBox.information(self, "SINCRO", "Solo se pueden renombrar layouts personalizados.")
            return
        if current not in self._custom_layouts:
            return
        new_name, ok = QInputDialog.getText(self, "Renombrar layout", f"Nuevo nombre para '{current}':")
        if not ok:
            return
        new_name = str(new_name).strip()
        if not new_name:
            return
        if new_name == current:
            return
        if new_name in self._PRESET_LAYOUTS or new_name in self._custom_layouts:
            QMessageBox.warning(self, "SINCRO", "Ya existe un layout con ese nombre.")
            return
        self._custom_layouts[new_name] = dict(self._custom_layouts[current])
        self._custom_layouts.pop(current, None)
        self._save_custom_layouts()
        self._refresh_layout_combo(new_name)

    def _delete_custom_layout(self):
        current = str(self._layout_combo.currentText() or "").strip()
        if not current:
            return
        if current in self._PRESET_LAYOUTS:
            QMessageBox.information(self, "SINCRO", "Solo se pueden eliminar layouts personalizados.")
            return
        if current not in self._custom_layouts:
            return
        answer = QMessageBox.question(
            self,
            "SINCRO",
            f"¿Eliminar layout personalizado '{current}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._custom_layouts.pop(current, None)
        self._save_custom_layouts()
        fallback = "Clásico vertical" if "Clásico vertical" in self._PRESET_LAYOUTS else next(iter(self._PRESET_LAYOUTS.keys()), "")
        self._refresh_layout_combo(fallback)

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            child_w = item.widget()
            child_l = item.layout()
            if child_l is not None:
                FusionReportLayoutDialog._clear_layout(child_l)
            if child_w is not None:
                child_w.deleteLater()

    @staticmethod
    def _norm(img: np.ndarray) -> np.ndarray:
        a = np.asarray(img, dtype=np.float64)
        mn, mx = float(np.min(a)), float(np.max(a))
        if mx - mn < 1e-9:
            return np.zeros_like(a, dtype=np.float64)
        return np.clip((a - mn) / (mx - mn), 0.0, 1.0)

    def _mk_strip(self, arr3d: np.ndarray, n_tiles: int = 10) -> np.ndarray:
        img = np.asarray(arr3d, dtype=np.float64)
        n = max(1, int(img.shape[0]))
        idxs = np.linspace(0, n - 1, min(n_tiles, n)).round().astype(int)
        tiles = [self._norm(img[i]) for i in idxs]
        h = max(t.shape[0] for t in tiles)
        w = max(t.shape[1] for t in tiles)
        canvas = np.zeros((h, w * len(tiles)), dtype=np.float64)
        for k, t in enumerate(tiles):
            y0 = (h - t.shape[0]) // 2
            x0 = k * w + (w - t.shape[1]) // 2
            canvas[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
        return canvas

    @staticmethod
    def _to_pix(arr: np.ndarray) -> QPixmap:
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        if a.max() <= 1.0:
            a = np.clip(a * 255.0, 0, 255)
        rgb = np.ascontiguousarray(a.astype(np.uint8))
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def _make_fusion_rgb(self, sp2d: np.ndarray, ct2d: np.ndarray) -> np.ndarray:
        sp = np.clip(np.asarray(sp2d, dtype=np.float64), 0.0, 1.0)
        ct = np.clip(np.asarray(ct2d, dtype=np.float64), 0.0, 1.0)
        rgb_sp = self._cmap_fn(sp)
        rgb_ct = np.stack([ct, ct, ct], axis=-1)
        mix = float(self._fusion_pct) / 100.0
        alpha = np.clip(sp * 1.35, 0.0, 1.0)[..., None] * mix
        out = (1.0 - alpha) * rgb_ct + alpha * rgb_sp
        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def _fit_2d_to_shape(img: np.ndarray, target_shape: tuple[int, int], order: int = 1) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float64)
        if arr.ndim != 2:
            return arr
        th, tw = int(target_shape[0]), int(target_shape[1])
        if th <= 0 or tw <= 0:
            return arr
        if arr.shape == (th, tw):
            return arr
        z = (th / max(1, arr.shape[0]), tw / max(1, arr.shape[1]))
        scaled = ndi.zoom(arr, z, order=order)
        out = np.zeros((th, tw), dtype=np.float64)
        sh, sw = int(scaled.shape[0]), int(scaled.shape[1])

        if sh <= th:
            src_y0, src_y1 = 0, sh
            dst_y0 = (th - sh) // 2
            dst_y1 = dst_y0 + sh
        else:
            src_y0 = (sh - th) // 2
            src_y1 = src_y0 + th
            dst_y0, dst_y1 = 0, th

        if sw <= tw:
            src_x0, src_x1 = 0, sw
            dst_x0 = (tw - sw) // 2
            dst_x1 = dst_x0 + sw
        else:
            src_x0 = (sw - tw) // 2
            src_x1 = src_x0 + tw
            dst_x0, dst_x1 = 0, tw

        out[dst_y0:dst_y1, dst_x0:dst_x1] = scaled[src_y0:src_y1, src_x0:src_x1]
        return out

    def _apply_aspect_2d_for_axis(self, img: np.ndarray, axis: str) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float64)
        sp = self._display_spacing_zyx
        if arr.ndim != 2 or sp is None or len(sp) != 3:
            return arr
        z_mm = max(1e-6, float(sp[0]))
        y_mm = max(1e-6, float(sp[1]))
        x_mm = max(1e-6, float(sp[2]))
        if axis == "coronal":
            ratio = z_mm / x_mm
            if abs(ratio - 1.0) > 1e-3:
                return ndi.zoom(arr, (ratio, 1.0), order=1)
            return arr
        if axis == "sagittal":
            ratio = z_mm / y_mm
            if abs(ratio - 1.0) > 1e-3:
                return ndi.zoom(arr, (ratio, 1.0), order=1)
            return arr
        return arr

    @staticmethod
    def _draw_refs(
        rgb: np.ndarray,
        axis: str,
        idx: dict,
        vol_shape: tuple[int, int, int],
        line_px: int = 1,
    ) -> np.ndarray:
        arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8).copy())
        h, w = arr.shape[:2]
        z = int(np.clip(idx.get("axial", 0), 0, max(0, vol_shape[0] - 1)))
        y = int(np.clip(idx.get("coronal", 0), 0, max(0, vol_shape[1] - 1)))
        x = int(np.clip(idx.get("sagittal", 0), 0, max(0, vol_shape[2] - 1)))

        width = max(1, int(line_px))

        def hline(py: int):
            py = int(np.clip(py, 0, h - 1))
            a = width // 2
            b = width - a
            arr[max(0, py - a):min(h, py + b), :, 0] = 255
            arr[max(0, py - a):min(h, py + b), :, 1] = 208
            arr[max(0, py - a):min(h, py + b), :, 2] = 64

        def vline(px: int):
            px = int(np.clip(px, 0, w - 1))
            a = width // 2
            b = width - a
            arr[:, max(0, px - a):min(w, px + b), 0] = 255
            arr[:, max(0, px - a):min(w, px + b), 1] = 208
            arr[:, max(0, px - a):min(w, px + b), 2] = 64

        if axis == "axial":
            py = int(round((y / max(1, vol_shape[1] - 1)) * max(0, h - 1)))
            px = int(round((x / max(1, vol_shape[2] - 1)) * max(0, w - 1)))
            hline(py)
            vline(px)
        elif axis == "coronal":
            py = int(round((z / max(1, vol_shape[0] - 1)) * max(0, h - 1)))
            px = int(round((x / max(1, vol_shape[2] - 1)) * max(0, w - 1)))
            hline(py)
            vline(px)
        else:
            py = int(round((z / max(1, vol_shape[0] - 1)) * max(0, h - 1)))
            px = int(round((y / max(1, vol_shape[1] - 1)) * max(0, w - 1)))
            hline(py)
            vline(px)
        return arr

    def _view2d(self, vol: np.ndarray, axis: str) -> np.ndarray:
        z = int(np.clip(self._slice_idx["axial"], 0, vol.shape[0] - 1))
        y = int(np.clip(self._slice_idx["coronal"], 0, vol.shape[1] - 1))
        x = int(np.clip(self._slice_idx["sagittal"], 0, vol.shape[2] - 1))
        if axis == "axial":
            return vol[z]
        if axis == "coronal":
            return vol[:, y, :]
        return vol[:, :, x]

    def _render(self):
        self._clear_layout(self._strip_col)
        self._clear_layout(self._grid)

        strip_w = int(self._strip_w_spin.value())
        strip_h = int(self._strip_h_spin.value())
        grid_w = int(self._grid_w_spin.value())
        grid_h = int(self._grid_h_spin.value())

        z_strip = self._mk_strip(self._spect_vol)
        y_strip = self._mk_strip(np.transpose(self._spect_vol, (1, 0, 2)))
        x_strip = self._mk_strip(np.transpose(self._spect_vol, (2, 0, 1)))
        for title, arr in (("Axial", z_strip), ("Coronal", y_strip), ("Sagital", x_strip)):
            box = QHBoxLayout()
            t = QLabel(title)
            t.setStyleSheet("color:#cbd5e1; font-weight:600;")
            t.setMinimumWidth(70)
            img = QLabel()
            img.setStyleSheet("background:#0b1220; border:1px solid #334155;")
            img.setPixmap(self._to_pix(arr).scaled(strip_w, strip_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            box.addWidget(t)
            box.addWidget(img, 1)
            w = QWidget()
            w.setLayout(box)
            self._strip_col.addWidget(w)

        axes = ("axial", "coronal", "sagittal")
        col_names = ("Axial", "Coronal", "Sagital")
        row_names = ("SPECT", "CT", "Fusión")
        for c, name in enumerate(col_names):
            h = QLabel(name)
            h.setStyleSheet("color:#cbd5e1; font-weight:600;")
            self._grid.addWidget(h, 0, c + 1)
        for r, name in enumerate(row_names):
            h = QLabel(name)
            h.setStyleSheet("color:#cbd5e1; font-weight:600;")
            self._grid.addWidget(h, r + 1, 0)

        for c, axis in enumerate(axes):
            sp2d = self._spect_window_fn(self._view2d(self._spect_vol, axis))
            sp2d = self._apply_aspect_2d_for_axis(sp2d, axis)
            # Respetar colormap de la ventana principal (inyectado como callback).
            sp_rgb = self._draw_refs(
                (self._cmap_fn(sp2d) * 255.0).astype(np.uint8),
                axis,
                self._slice_idx,
                tuple(int(v) for v in self._spect_vol.shape[:3]),
                self._line_px,
            )
            sp_lbl = QLabel()
            sp_lbl.setStyleSheet("background:#0b1220; border:1px solid #334155;")
            sp_lbl.setPixmap(self._to_pix(sp_rgb).scaled(grid_w, grid_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self._grid.addWidget(sp_lbl, 1, c + 1)

            if self._ct_vol is not None:
                ct2d = self._ct_window_fn(self._view2d(self._ct_vol, axis))
                ct2d = self._apply_aspect_2d_for_axis(ct2d, axis)
                if sp2d.shape != ct2d.shape:
                    sp2d = self._fit_2d_to_shape(sp2d, ct2d.shape, order=1)
                ct_rgb = self._draw_refs(
                    (np.stack([ct2d, ct2d, ct2d], axis=-1) * 255.0).astype(np.uint8),
                    axis,
                    self._slice_idx,
                    tuple(int(v) for v in self._ct_vol.shape[:3]),
                    self._line_px,
                )
                fx_rgb = self._draw_refs(
                    (self._make_fusion_rgb(sp2d, ct2d) * 255.0).astype(np.uint8),
                    axis,
                    self._slice_idx,
                    tuple(int(v) for v in self._ct_vol.shape[:3]),
                    self._line_px,
                )
            else:
                ct_rgb = self._draw_refs(
                    (np.stack([sp2d, sp2d, sp2d], axis=-1) * 255.0).astype(np.uint8),
                    axis,
                    self._slice_idx,
                    tuple(int(v) for v in self._spect_vol.shape[:3]),
                    self._line_px,
                )
                fx_rgb = sp_rgb

            ct_lbl = QLabel()
            ct_lbl.setStyleSheet("background:#0b1220; border:1px solid #334155;")
            ct_lbl.setPixmap(self._to_pix(ct_rgb).scaled(grid_w, grid_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
            self._grid.addWidget(ct_lbl, 2, c + 1)

            fx_lbl = QLabel()
            fx_lbl.setStyleSheet("background:#0b1220; border:1px solid #334155;")
            fx_lbl.setPixmap(self._to_pix(fx_rgb).scaled(grid_w, grid_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self._grid.addWidget(fx_lbl, 3, c + 1)

    def _render_body_to_pixmap(self) -> QPixmap:
        self._body.adjustSize()
        size = self._body.sizeHint()
        w = max(1, int(size.width()))
        h = max(1, int(size.height()))
        pix = QPixmap(w, h)
        pix.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pix)
        self._body.render(painter)
        painter.end()
        return pix

    @staticmethod
    def _last_dir(default_path: str = "") -> str:
        s = QSettings("GAMMASYS", "SINCRO_AMYLO_SPECT")
        remembered = str(s.value("global/last_dir", "") or "")
        if remembered and os.path.isdir(remembered):
            return remembered
        if default_path and os.path.isdir(default_path):
            return default_path
        return os.path.expanduser("~")

    @staticmethod
    def _remember_path(path: str) -> None:
        if not path:
            return
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        if folder and os.path.isdir(folder):
            s = QSettings("GAMMASYS", "SINCRO_AMYLO_SPECT")
            s.setValue("global/last_dir", folder)
            s.sync()

    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar vista de fusión a PNG",
            self._last_dir(),
            "PNG (*.png)",
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            pix = self._render_body_to_pixmap()
            # Crear imagen anotada con métricas si existen
            img = pix.toImage()
            painter2 = QPainter(img)
            painter2.setRenderHint(QPainter.RenderHint.Antialiasing)
            font = QFont("Consolas", 9)
            painter2.setFont(font)
            pen = QPen(QColor(0, 0, 0, 200))
            pen.setWidthF(1.5)
            painter2.setPen(pen)
            # Fondo semi-transparente para legibilidad
            margin_x = 8
            margin_y = img.height() - 12
            line_h = 16
            has_metrics = False
            # Dibujar HMR-SPECT si existe
            if getattr(self, "_hmr_result", None):
                has_metrics = True
                hmr = self._hmr_result
                color = QColor(0, 120, 0) if hmr.classification == "POSITIVO" else (
                    QColor(180, 120, 0) if hmr.classification == "EQUIVOCO" else QColor(180, 0, 0))
                painter2.setPen(QPen(color))
                txt = f"HMR={hmr.hmr:.2f} ({hmr.classification})"
                painter2.drawText(margin_x, margin_y, txt)
                margin_y -= line_h
            # Dibujar S/VD si existe
            if getattr(self, "_svd_result", None):
                has_metrics = True
                svd = self._svd_result
                color = QColor(180, 0, 0) if svd.classification == "POSITIVO" else (
                    QColor(180, 120, 0) if svd.classification == "EQUIVOCO" else QColor(0, 120, 0))
                painter2.setPen(QPen(color))
                txt = f"S/VD={svd.s_vd:.2f} ({svd.classification})"
                painter2.drawText(margin_x, margin_y, txt)
                margin_y -= line_h
                painter2.setPen(pen)
                sub_txt = f"S/V={svd.s_v:.2f} S/D={svd.s_d:.2f} V/D={svd.v_d:.2f}"
                painter2.drawText(margin_x, margin_y, sub_txt)
                margin_y -= line_h
            painter2.end()
            if has_metrics:
                pix_anot = QPixmap.fromImage(img)
            else:
                pix_anot = pix
            ok = pix_anot.save(path, "PNG")
            if not ok:
                raise RuntimeError("No se pudo escribir PNG")
            self._remember_path(path)
            QMessageBox.information(self, "SINCRO", f"PNG exportado:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "SINCRO", f"Error exportando PNG:\n{exc}")

    def _export_pdf(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar vista de fusión a PDF",
            self._last_dir(),
            "PDF (*.pdf)",
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            use_clinical_a4 = bool(self._pdf_a4_mode_check.isChecked())
            current_layout_name = str(self._layout_combo.currentText() or "")
            current_values = self._current_layout_values()
            a4_values = self._PRESET_LAYOUTS.get("Informe clínico A4") if use_clinical_a4 else None
            try:
                if a4_values:
                    self._apply_layout_values(a4_values)
                pix = self._render_body_to_pixmap()
            finally:
                if a4_values:
                    self._apply_layout_values(current_values)
                    if current_layout_name:
                        self._settings.setValue("fusion_report/layout_selected", current_layout_name)
            writer = QPdfWriter(path)
            writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
            writer.setResolution(300)
            painter = QPainter(writer)
            page_w = writer.width()
            page_h = writer.height()
            sw = max(1, pix.width())
            sh = max(1, pix.height())
            scale = min(page_w / float(sw), page_h / float(sh))
            dw = int(sw * scale)
            dh = int(sh * scale)
            dx = int((page_w - dw) / 2)
            dy = int((page_h - dh) / 2)
            painter.drawPixmap(dx, dy, dw, dh, pix)
            # Agregar puntos de localización si existen
            if self._localization_points:
                painter.save()
                font = QFont("Arial", 10)
                painter.setFont(font)
                painter.setPen(QColor(30, 30, 30))
                text_y = dy + dh + 40
                line_h = 18
                painter.drawText(dx, text_y, "Puntos de localización:")
                text_y += line_h + 4
                for pt in self._localization_points:
                    label = pt.get("label", "?")
                    if "zyx" in pt:
                        zyx = pt["zyx"]
                        line = f"  {label}: Z={zyx[0]}, Y={zyx[1]}, X={zyx[2]}"
                    elif "value_mm" in pt:
                        line = f"  {label}: {pt['value_mm']} mm"
                    else:
                        line = f"  {label}"
                    painter.drawText(dx, text_y, line)
                    text_y += line_h
                painter.restore()
            # Agregar HMR-SPECT si fue calculado
            if self._hmr_result:
                painter.save()
                font = QFont("Arial", 10)
                painter.setFont(font)
                painter.setPen(QColor(30, 30, 30))
                text_y = dy + dh + 40
                if self._localization_points:
                    # Si ya hay puntos, el text_y ya avanzó
                    pass
                else:
                    # Si no hay puntos, empezar desde abajo
                    text_y = dy + dh + 40
                line_h = 18
                painter.drawText(dx, text_y, "HMR-SPECT:")
                text_y += line_h + 4
                hmr = self._hmr_result
                painter.drawText(dx, text_y, f"  HMR = {hmr.hmr:.2f} ({hmr.classification})")
                text_y += line_h
                painter.drawText(dx, text_y, f"  Método: {hmr.method}")
                text_y += line_h
                painter.drawText(dx, text_y, f"  Cuentas corazón: {hmr.heart_counts:.0f}")
                text_y += line_h
                painter.drawText(dx, text_y, f"  Cuentas mediastino: {hmr.mediastinum_counts:.0f}")
                text_y += line_h
                painter.drawText(dx, text_y, f"  Volumen corazón: {hmr.heart_volume_ml:.1f} mL")
                text_y += line_h
                painter.drawText(dx, text_y, f"  Volumen mediastino: {hmr.mediastinum_volume_ml:.1f} mL")
                if hmr.slice_idx is not None:
                    text_y += line_h
                    painter.drawText(dx, text_y, f"  Slice axial: {hmr.slice_idx}")
                painter.restore()
            # Agregar S/VD si fue calculado (optativo)
            if self._svd_result:
                painter.save()
                font = QFont("Arial", 10)
                painter.setFont(font)
                painter.setPen(QColor(30, 30, 30))
                text_y = dy + dh + 40
                if self._localization_points:
                    pass  # text_y ya avanzó por puntos
                elif self._hmr_result:
                    # Recalcular text_y después del bloque HMR
                    text_y = dy + dh + 40 + 18 * 8  # aprox
                line_h = 18
                painter.drawText(dx, text_y, "Ratio S/VD (experimental):")
                text_y += line_h + 4
                svd = self._svd_result
                painter.drawText(dx, text_y, f"  S/VD = {svd.s_vd:.2f} ({svd.classification})")
                text_y += line_h
                painter.drawText(dx, text_y, f"  S/V = {svd.s_v:.2f}  ·  S/D = {svd.s_d:.2f}  ·  V/D = {svd.v_d:.2f}")
                text_y += line_h
                painter.drawText(dx, text_y, f"  Cuentas: S={svd.s_counts:.0f}  V={svd.v_counts:.0f}  D={svd.d_counts:.0f}")
                text_y += line_h
                painter.drawText(dx, text_y, f"  Voxels: S={svd.s_voxels}  V={svd.v_voxels}  D={svd.d_voxels}")
                text_y += line_h
                painter.drawText(dx, text_y, "  ⚠ Cutoffs orientativos (validar con población local)")
                painter.restore()
            painter.end()
            self._remember_path(path)
            QMessageBox.information(self, "SINCRO", f"PDF exportado:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "SINCRO", f"Error exportando PDF:\n{exc}")

    def _export_dicom_sr(self):
        """Exporta puntos de localización como DICOM Structured Report."""
        if not self._localization_points:
            QMessageBox.warning(self, "SINCRO", "No hay puntos de localización para exportar.")
            return
        try:
            from report.dicom_sr import create_localization_sr

            path, _ = QFileDialog.getSaveFileName(
                self,
                "Exportar DICOM-SR",
                self._last_dir(),
                "DICOM (*.dcm)",
            )
            if not path:
                return
            if not path.lower().endswith(".dcm"):
                path += ".dcm"

            # Crear dataset mínimo del SPECT para referencia
            spect_ds = Dataset()
            spect_ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.20"
            spect_ds.SOPInstanceUID = "1.2.3.4.5.6.7.8.9"
            spect_ds.PatientID = "SR_EXPORT"
            spect_ds.PatientName = "Exported^From^SINCRO"
            spect_ds.StudyInstanceUID = "1.2.3.4.5.6.7.8.9.10"

            sr = create_localization_sr(
                localization_points=self._localization_points,
                spect_ds=spect_ds,
                output_path=path,
                hmr_result=getattr(self, "_hmr_result", None),
                svd_result=getattr(self, "_svd_result", None),
            )
            self._remember_path(path)
            QMessageBox.information(self, "SINCRO", f"DICOM-SR exportado:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "SINCRO", f"Error exportando DICOM-SR:\n{exc}")



class AmyloidSpectPanel(QDialog):
    """UI mínima para flujo AMYLO SPECT 3D."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SINCRO — AMYLO SPECT 3D (experimental)")
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMinMaxButtonsHint
        )
        self.resize(1060, 700)
        self.setMinimumSize(920, 580)
        self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)

        self._analysis = None
        self._recon_bundle = None
        self._current_volume = None
        self._base_spect_volume = None
        # Volumen sin post-filtro gaussiano (para toggle en MIP)
        self._unfiltered_volume = None
        self._ct_volume = None
        self._ct_registered = None
        self._ct_auto_registered = None
        self._ct_registration_shift_zyx = (0.0, 0.0, 0.0)  # Shift de registro en píxeles SPECT
        self._ct_total_shift_zyx = (0.0, 0.0, 0.0)  # Shift total (registro + nudge) en píxeles SPECT
        self._att_map_volume = None
        self._att_map_registered = None
        self._att_spacing_zyx = None
        self._att_affine_ijk_to_lps = None
        self._att_path = ""
        self._spect_spacing_zyx = None
        self._ct_spacing_zyx = None
        self._spect_affine_ijk_to_lps = None
        self._ct_affine_ijk_to_lps = None
        self._current_spect_path = ""
        self._ct_path = ""
        self._bone_mask = None
        self._settings = QSettings("GAMMASYS", "SINCRO_AMYLO_SPECT")
        self._recon_thread = None
        self._recon_worker = None
        self._pending_recon_cfg = None
        self._pending_recon_preset_label = ""
        self._pending_recon_cuts_mode = ""
        self._pending_recon_ac_iter = False
        self._study_is_gated = False
        self._slice_idx = {"axial": 0, "coronal": 0, "sagittal": 0}
        self._spect_view_offset = {"axial": 0, "coronal": 0, "sagittal": 0}
        self._ct_view_offset = {"axial": 0, "coronal": 0, "sagittal": 0}
        self._spect_zoom_pct = 100
        self._ct_zoom_pct = 100
        self._spect_pan_px = {"axial": [0, 0], "coronal": [0, 0], "sagittal": [0, 0]}
        self._ct_pan_px = {"axial": [0, 0], "coronal": [0, 0], "sagittal": [0, 0]}
        self._drag_state = None
        # Sensibilidad del arrastre triangulado CT (Ctrl+drag).
        # <1.0 = más suave/fino.
        self._ct_drag_sensitivity = 0.35
        self._spect_win_low = 0
        self._spect_win_high = 100
        self._spect_display_sigma = 0.8
        self._fusion_pct = 55
        self._spect_flip_x_test = False
        self._spect_flip_y_test = False
        self._spect_flip_z_test = False
        self._ct_flip_x_test = False
        self._ct_flip_y_test = False
        self._ct_flip_z_test = False
        self._ct_window = "bone"
        self._ct_visual_trial_mode = True
        self._ct_grid_trial_mode = False
        self._trial_cache_signature = None
        self._trial_spect_on_ct = None
        self._trial_ct_native = None
        self._trial_ct_native_spacing = None
        self._trial_ref_shape = None
        self._mip_vol_cache_sig = None  # Caché para evitar recalcular transform en cada scroll
        self._workflow_tag = "perf_spect_ct"
        self._dicom_profile_info = {}
        self._pending_camera_profile_adjust = None
        self._pre_bone_volume = None
        self._triangulation_cross_enabled = False
        self._localization_cross_enabled = False
        self._localization_point_zyx = None
        self._localization_anchor_zyx = None
        self._hmr_result = None
        # VOIs temporales para visualización en vivo
        self._temp_voi_heart = None
        self._temp_voi_mediastinum = None

        # === Sistema S/VD (ratio corazón/vértebra/aorta) ===
        self._svd_points: dict[str, tuple[int, int, int] | None] = {
            "S": None,  # Corazón
            "V": None,  # Vértebra
            "D": None,  # Aorta descendente
        }
        self._svd_active_roi = "S"  # ROI activa para depositar con clic
        self._svd_result = None   # SvdRatioResult
        self._svd_vertebra_radius = 15.0  # mm
        self._svd_aorta_radius = 12.0     # mm
        
        # === F2.4: Estado edición manual de máscara CT ===
        self._mask_edit_active = False          # Si el modo edición está activo
        self._mask_edit_paint_mode = True       # True = pintar, False = borrar
        self._mask_edit_undo_stack = []         # Stack de máscaras previas para undo
        self._mask_edit_original = None         # Máscara original antes de cualquier edición
        self._mask_edit_has_changes = False     # Si hay cambios pendientes de aplicar
        self._reuse_edited_segmentation = False # Reusar máscara editada en vez de re-segmentar

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        flow_box = QGroupBox("Flujo clínico AMYLO SPECT / SPECT-CT")
        flow = QGridLayout(flow_box)
        flow.setContentsMargins(8, 6, 8, 6)
        flow.setHorizontalSpacing(8)
        flow.setVerticalSpacing(4)

        self._btn_load = QPushButton("1. Cargar SPECT")
        self._btn_load.clicked.connect(self._load_spect)
        self._btn_load.setToolTip("Carga el DICOM SPECT AMYLO. Puede ser crudo o reconstruido; no requiere gating.")
        flow.addWidget(self._btn_load, 0, 0)

        flow.addWidget(QLabel("Preset:"), 0, 1)
        self._preset_combo = QComboBox()
        self._preset_combo.addItem("Manual", "manual")
        self._preset_combo.addItem("AMYLO 360 estándar 128", "amylo360_std128")
        self._preset_combo.addItem("AMYLO 360 alta definición", "amylo360_hd")
        self._preset_combo.addItem("Clínico OSEM suave (2×10)", "clinical_osem_soft")
        self._preset_combo.addItem("Clínico OSEM 8×4 + Butter", "clinical_osem")
        self._preset_combo.setToolTip(
            "Presets PYP/SPECT 360. 128x128 es recomendación de adquisición; "
            "si el DICOM trae otra matriz, se reconstruye en la matriz adquirida."
        )
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        flow.addWidget(self._preset_combo, 0, 2)

        flow.addWidget(QLabel("Recon:"), 0, 3)
        self._recon_combo = QComboBox()
        self._recon_combo.addItem("FBP", "fbp")
        self._recon_combo.addItem("OSEM", "osem")
        self._recon_combo.addItem("MLEM", "mlem")
        self._recon_combo.currentIndexChanged.connect(self._on_top_recon_changed)
        flow.addWidget(self._recon_combo, 0, 4)

        flow.addWidget(QLabel("Cortes:"), 0, 5)
        self._cuts_mode_combo = QComboBox()
        self._cuts_mode_combo.addItem("Mixto", "mixed")
        self._cuts_mode_combo.addItem("Tomográficos", "tomo")
        self._cuts_mode_combo.addItem("Cardíacos", "cardiac")
        self._cuts_mode_combo.setToolTip("Selecciona qué familia de cortes generar en recon AMYLO.")
        self._cuts_mode_combo.currentIndexChanged.connect(self._persist_ui_state)
        flow.addWidget(self._cuts_mode_combo, 0, 6)

        self._btn_recon_pipeline = QPushButton("2. Reconstruir + cortes")
        self._btn_recon_pipeline.clicked.connect(self._reconstruct_with_perf_pipeline)
        self._btn_recon_pipeline.setEnabled(False)
        self._btn_recon_pipeline.setToolTip("Reconstruye el SPECT y genera cortes tomográficos; si hay gating, también SA/HLA/VLA.")
        self._btn_recon_pipeline.setStyleSheet(
            "background-color: #16a34a; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;"
        )
        flow.addWidget(self._btn_recon_pipeline, 0, 7)

        self._btn_cancel_recon = QPushButton("Cancelar")
        self._btn_cancel_recon.clicked.connect(self._cancel_reconstruction)
        self._btn_cancel_recon.setEnabled(False)
        self._btn_cancel_recon.setStyleSheet(
            "background-color: #dc2626; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;"
        )
        flow.addWidget(self._btn_cancel_recon, 0, 8)

        self._ct_check = QCheckBox("Usar CT para sustracción ósea (si hay)")
        self._ct_check.setToolTip("Si hay CT cargado/registrado, usa el CT para definir máscara ósea visual.")
        flow.addWidget(self._ct_check, 1, 0, 1, 2)

        self._btn_load_ct = QPushButton("3a. Cargar CT")
        self._btn_load_ct.clicked.connect(self._load_ct)
        self._btn_load_ct.setEnabled(False)
        flow.addWidget(self._btn_load_ct, 1, 2)

        self._btn_load_ct_dir = QPushButton("3b. CT carpeta")
        self._btn_load_ct_dir.clicked.connect(self._load_ct_dir)
        self._btn_load_ct_dir.setEnabled(False)
        flow.addWidget(self._btn_load_ct_dir, 1, 3)

        self._btn_register = QPushButton("4. Registrar CT↔SPECT")
        self._btn_register.clicked.connect(self._register_ct_to_spect)
        self._btn_register.setEnabled(False)
        flow.addWidget(self._btn_register, 1, 4, 1, 2)

        self._btn_save_cam_preset = QPushButton("4a. Guardar preset cámara")
        self._btn_save_cam_preset.clicked.connect(self._save_camera_profile_preset)
        self._btn_save_cam_preset.setEnabled(False)
        self._btn_save_cam_preset.setToolTip("Guarda flips/rotaciones/offsets para esta cámara+protocolo en flujo perfusión SPECT/CT.")
        flow.addWidget(self._btn_save_cam_preset, 2, 4)

        self._btn_apply_cam_preset = QPushButton("4a'. Aplicar preset cámara")
        self._btn_apply_cam_preset.clicked.connect(lambda: self._apply_camera_profile_preset(auto=False))
        self._btn_apply_cam_preset.setEnabled(False)
        self._btn_apply_cam_preset.setToolTip("Aplica preset guardado para esta cámara+protocolo (si existe).")
        flow.addWidget(self._btn_apply_cam_preset, 2, 5)

        self._btn_load_att = QPushButton("4b. Cargar ATT MAP")
        self._btn_load_att.clicked.connect(self._load_att_map)
        self._btn_load_att.setEnabled(False)
        flow.addWidget(self._btn_load_att, 2, 0, 1, 2)

        self._btn_apply_ac = QPushButton("4c. Aplicar AC (Chang)")
        self._btn_apply_ac.clicked.connect(self._apply_ac_prototype)
        self._btn_apply_ac.setEnabled(False)
        self._btn_apply_ac.setToolTip("Aplica corrección por atenuación tipo Chang (slice-wise) usando ATT MAP remuestreado a grilla SPECT.")
        flow.addWidget(self._btn_apply_ac, 2, 2, 1, 2)

        self._btn_bone = QPushButton("5. Sustracción ósea visual")
        self._btn_bone.clicked.connect(self._apply_bone_suppression)
        self._btn_bone.setEnabled(False)
        flow.addWidget(self._btn_bone, 1, 6, 1, 2)

        self._btn_export_axes_dcm = QPushButton("6. Exportar ejes DICOM")
        self._btn_export_axes_dcm.clicked.connect(self._export_axes_dicom)
        self._btn_export_axes_dcm.setEnabled(False)
        flow.addWidget(self._btn_export_axes_dcm, 1, 8)

        self._btn_fusion_layout = QPushButton("6b. Vista informe fusión")
        self._btn_fusion_layout.clicked.connect(self._show_fusion_report_layout)
        self._btn_fusion_layout.setEnabled(False)
        self._btn_fusion_layout.setToolTip("Muestra tiras SPECT y panel 3x3 (SPECT/CT/Fusión) con referencias de corte.")
        flow.addWidget(self._btn_fusion_layout, 2, 8)
        flow.setColumnStretch(9, 1)

        # ═══════════════════════════════════════════════════════════════════
        # FILA HORIZONTAL: Flujo clínico + Reconstrucción (lado a lado)
        # ═══════════════════════════════════════════════════════════════════
        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        
        # Flujo clínico: ancho compacto (no se expande)
        flow_box.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        top_row.addWidget(flow_box)

        recon_box = QGroupBox("Reconstrucción y filtros")
        recon_grid = QGridLayout(recon_box)
        recon_grid.setContentsMargins(8, 6, 8, 6)
        recon_grid.setHorizontalSpacing(8)
        recon_grid.setVerticalSpacing(4)

        self._ung_filter_combo = self._mk_filter_combo()
        self._ung_cutoff_spin = self._mk_cutoff_spin(0.45)
        self._ung_order_spin = self._mk_order_spin(8)
        self._gated_filter_combo = self._mk_filter_combo()
        self._gated_cutoff_spin = self._mk_cutoff_spin(0.40)
        self._gated_order_spin = self._mk_order_spin(10)
        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(1, 30)
        self._iter_spin.setValue(3)
        self._iter_spin.setToolTip("Iteraciones para OSEM/MLEM en modo Manual.")
        self._subsets_spin = QSpinBox()
        self._subsets_spin.setRange(1, 32)
        self._subsets_spin.setValue(6)
        self._subsets_spin.setToolTip("Subsets para OSEM. En MLEM se ignora y se usa 1 subset.")

        for widget in (
            self._ung_filter_combo,
            self._ung_cutoff_spin,
            self._ung_order_spin,
            self._gated_filter_combo,
            self._gated_cutoff_spin,
            self._gated_order_spin,
            self._iter_spin,
            self._subsets_spin,
        ):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._persist_ui_state)
            elif hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._persist_ui_state)

        recon_grid.addWidget(QLabel("Ungated"), 0, 0)
        self._ung_method_combo = QComboBox()
        self._ung_method_combo.addItem("FBP", "fbp")
        self._ung_method_combo.addItem("MLEM", "mlem")
        self._ung_method_combo.addItem("OSEM", "osem")
        self._ung_method_combo.setMaximumWidth(90)
        self._ung_method_combo.setToolTip("Método de reconstrucción para el volumen estático AMYLO.")
        self._ung_method_combo.currentIndexChanged.connect(self._on_manual_method_changed)
        recon_grid.addWidget(self._ung_method_combo, 0, 1)
        recon_grid.addWidget(self._ung_filter_combo, 0, 2)
        recon_grid.addWidget(QLabel("cutoff"), 0, 3)
        recon_grid.addWidget(self._ung_cutoff_spin, 0, 4)
        recon_grid.addWidget(QLabel("orden"), 0, 5)
        recon_grid.addWidget(self._ung_order_spin, 0, 6)
        recon_grid.addWidget(QLabel("Iter"), 0, 7)
        recon_grid.addWidget(self._iter_spin, 0, 8)
        recon_grid.addWidget(QLabel("Subsets"), 0, 9)
        recon_grid.addWidget(self._subsets_spin, 0, 10)

        self._gated_title_lbl = QLabel("Gated")
        recon_grid.addWidget(self._gated_title_lbl, 1, 0)
        self._gated_method_combo = QComboBox()
        self._gated_method_combo.addItem("FBP", "fbp")
        self._gated_method_combo.addItem("MLEM", "mlem")
        self._gated_method_combo.addItem("OSEM", "osem")
        self._gated_method_combo.setMaximumWidth(90)
        self._gated_method_combo.setToolTip("Método de reconstrucción gate-by-gate. Se desactiva en AMYLO no gatillado.")
        recon_grid.addWidget(self._gated_method_combo, 1, 1)
        recon_grid.addWidget(self._gated_filter_combo, 1, 2)
        recon_grid.addWidget(QLabel("cutoff"), 1, 3)
        recon_grid.addWidget(self._gated_cutoff_spin, 1, 4)
        recon_grid.addWidget(QLabel("orden"), 1, 5)
        recon_grid.addWidget(self._gated_order_spin, 1, 6)

        self._view_combo = QComboBox()
        self._view_combo.addItem("MPR volumen actual", "mpr")
        self._view_combo.addItem("Cortes tomográficos", "tomo")
        self._view_combo.addItem("Cortes cardíacos SA/HLA/VLA", "cardiac")
        self._view_combo.setEnabled(False)
        self._view_combo.currentIndexChanged.connect(self._render_selected_view)
        recon_grid.addWidget(QLabel("Vista"), 1, 7)
        recon_grid.addWidget(self._view_combo, 1, 8, 1, 3)

        self._bg_check = QCheckBox("Fondo")
        self._bg_check.setToolTip("Descuento de fondo automático en el sinograma antes de reconstruir.")
        self._scatter_check = QCheckBox("Desc. SC")
        self._scatter_check.setToolTip("Resta scatter si el loader encontró proyecciones SC asociadas.")
        self._scatter_k_spin = QDoubleSpinBox()
        self._scatter_k_spin.setRange(0.0, 2.0)
        self._scatter_k_spin.setSingleStep(0.05)
        self._scatter_k_spin.setDecimals(2)
        self._scatter_k_spin.setValue(1.0)
        self._post_check = QCheckBox("Suavizar")
        self._post_check.setToolTip("Post-filtro gaussiano 3D del volumen AMYLO.")
        self._post_sigma_spin = QDoubleSpinBox()
        self._post_sigma_spin.setRange(0.0, 6.0)
        self._post_sigma_spin.setSingleStep(0.1)
        self._post_sigma_spin.setDecimals(1)
        self._post_sigma_spin.setValue(0.5)
        self._denoise_plus_check = QCheckBox("Denoise+")
        self._denoise_plus_check.setToolTip("Denoise de sinograma + realce por resta en la rama ungated.")
        self._denoise_plus_k_spin = QDoubleSpinBox()
        self._denoise_plus_k_spin.setRange(0.0, 0.70)
        self._denoise_plus_k_spin.setSingleStep(0.05)
        self._denoise_plus_k_spin.setDecimals(2)
        self._denoise_plus_k_spin.setValue(0.20)
        self._ac_iter_check = QCheckBox("AC iterativa")
        self._ac_iter_check.setToolTip("Activa corrección por atenuación dentro del update OSEM/MLEM (requiere ATT MAP cargado).")
        self._ac_mu_scale_spin = QDoubleSpinBox()
        self._ac_mu_scale_spin.setRange(0.0001, 1000.0)
        self._ac_mu_scale_spin.setSingleStep(0.05)
        self._ac_mu_scale_spin.setDecimals(4)
        self._ac_mu_scale_spin.setValue(1.00)
        self._ac_mu_scale_spin.setToolTip(
            "Factor que lleva el ATT MAP a µ en cm⁻¹.\n"
            "Referencia Tc-99m (140 keV): agua/tejido blando = 0.154 cm⁻¹, "
            "pulmón ≈ 0.04, hueso ≈ 0.25.\n"
            "Si el mapa ya está en cm⁻¹ → 1.0. Si está ×1000 (µ ≈ 154) → 0.001.\n"
            "Al cargar el ATT MAP la consola muestra la mediana medida y la escala sugerida."
        )
        for widget in (self._bg_check, self._scatter_check, self._scatter_k_spin, self._post_check, self._post_sigma_spin, self._denoise_plus_check, self._denoise_plus_k_spin, self._ac_iter_check, self._ac_mu_scale_spin, self._gated_method_combo):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._persist_ui_state)
            elif hasattr(widget, "toggled"):
                widget.toggled.connect(self._persist_ui_state)
            elif hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self._persist_ui_state)
        recon_grid.addWidget(self._bg_check, 2, 0)
        recon_grid.addWidget(self._scatter_check, 2, 1)
        recon_grid.addWidget(QLabel("k SC"), 2, 2)
        recon_grid.addWidget(self._scatter_k_spin, 2, 3)
        recon_grid.addWidget(self._post_check, 2, 4)
        recon_grid.addWidget(self._post_sigma_spin, 2, 5)
        recon_grid.addWidget(self._denoise_plus_check, 2, 6)
        recon_grid.addWidget(QLabel("k"), 2, 7)
        recon_grid.addWidget(self._denoise_plus_k_spin, 2, 8)
        recon_grid.addWidget(self._ac_iter_check, 2, 9)
        recon_grid.addWidget(QLabel("μ-scale"), 2, 10)
        recon_grid.addWidget(self._ac_mu_scale_spin, 2, 11)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("Listo")
        recon_grid.addWidget(self._progress, 3, 0, 1, 11)
        
        # Reconstrucción: ancho compacto (no se expande)
        recon_box.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        top_row.addWidget(recon_box)
        
        # Espacio flexible a la derecha para que no se estiren los boxes
        top_row.addStretch(1)
        
        root.addLayout(top_row)

        self._status = QLabel("Cargar un DICOM SPECT para iniciar.")
        self._status.setStyleSheet("color:#93c5fd; font-size:11px;")
        root.addWidget(self._status)

        # ═══════════════════════════════════════════════════════════════════
        # LAYOUT PRINCIPAL: Imágenes (izq) + Controles laterales (der)
        # ═══════════════════════════════════════════════════════════════════
        main_splitter = QHBoxLayout()
        
        # --- COLUMNA IZQUIERDA: Imágenes y controles principales ---
        left_col = QVBoxLayout()
        
        # ═══════════════════════════════════════════════════════════════════
        # GRUPO: Vistas 3D (solo imágenes + botones de localización)
        # ═══════════════════════════════════════════════════════════════════
        vistas_group = QGroupBox("Vistas 3D")
        vistas_layout = QVBoxLayout(vistas_group)
        
        # Grid de 3 imágenes (Axial, Coronal, Sagittal)
        grid = QGridLayout()
        self._axial_lbl = self._mk_image_label("Axial")
        self._cor_lbl = self._mk_image_label("Coronal")
        self._sag_lbl = self._mk_image_label("Sagital")

        grid.addWidget(self._axial_lbl, 0, 0)
        grid.addWidget(self._cor_lbl, 0, 1)
        grid.addWidget(self._sag_lbl, 0, 2)
        vistas_layout.addLayout(grid)

        # Botones de localización (pie derecho dentro del box de imágenes)
        loc_btns_row = QHBoxLayout()
        loc_btns_row.addStretch(1)
        self._btn_triangulation_cross = QPushButton("Cruz triangulación")
        self._btn_triangulation_cross.setCheckable(True)
        self._btn_triangulation_cross.setChecked(False)
        self._btn_triangulation_cross.setToolTip("Activa/desactiva líneas de referencia entre cortes axial, coronal y sagital.")
        self._btn_triangulation_cross.toggled.connect(self._on_triangulation_cross_toggled)
        loc_btns_row.addWidget(self._btn_triangulation_cross)
        self._btn_localization_cross = QPushButton("Localización")
        self._btn_localization_cross.setCheckable(True)
        self._btn_localization_cross.setChecked(False)
        self._btn_localization_cross.setToolTip(
            "MODO LOCALIZACIÓN:\n"
            "• Ctrl+clic en CT → posiciona cruz\n"
            "• Shift+clic en SPECT → posiciona cruz\n\n"
            "Para HMR-SPECT:\n"
            "1. Posicione cruz en centro del corazón\n"
            "2. Click 'Fijar ancla A'\n"
            "3. Posicione cruz en mediastino\n"
            "4. Click 'Calcular HMR-SPECT'"
        )
        self._btn_localization_cross.toggled.connect(self._on_localization_cross_toggled)
        loc_btns_row.addWidget(self._btn_localization_cross)
        self._btn_loc_anchor = QPushButton("Fijar ancla A")
        self._btn_loc_anchor.setToolTip("Guarda la posición actual como punto A (corazón) para HMR-SPECT.")
        self._btn_loc_anchor.clicked.connect(self._on_set_localization_anchor)
        loc_btns_row.addWidget(self._btn_loc_anchor)
        self._btn_loc_clear = QPushButton("Limpiar ancla")
        self._btn_loc_clear.setToolTip("Borra el punto A.")
        self._btn_loc_clear.clicked.connect(self._on_clear_localization_anchor)
        loc_btns_row.addWidget(self._btn_loc_clear)
        vistas_layout.addLayout(loc_btns_row)

        left_col.addWidget(vistas_group)

        # ═══════════════════════════════════════════════════════════════════
        # PANEL MIP (rotación 360° con mouse)
        # ═══════════════════════════════════════════════════════════════════
        mip_group = QGroupBox("MIP 360° (arrastrar para rotar)")
        mip_layout = QHBoxLayout(mip_group)
        mip_layout.setContentsMargins(4, 2, 4, 2)
        mip_layout.setSpacing(4)

        # Widget MIP interactivo con rotación por mouse
        self._mip_widget = MipRotatorWidget()
        mip_layout.addWidget(self._mip_widget)

        # NOTA: mip_group se agrega en right_col (debajo de Ventana/Color), no aquí
        # left_col.addWidget(mip_group)  ← MOVIDO A COLUMNA DERECHA

        # ═══════════════════════════════════════════════════════════════════
        # GRUPO: Overlay / QC / Cortes (aparte, entre imágenes y controles)
        # ═══════════════════════════════════════════════════════════════════
        overlay_qc_group = QGroupBox("Overlay / QC / Cortes")
        overlay_qc_layout = QVBoxLayout(overlay_qc_group)
        overlay_qc_layout.setSpacing(4)
        
        # Overlay óseo
        blend_row = QHBoxLayout()
        blend_row.addWidget(QLabel("Overlay óseo:"))
        self._blend_slider = QSlider(Qt.Orientation.Horizontal)
        self._blend_slider.setRange(0, 100)
        self._blend_slider.setValue(35)
        self._blend_slider.setToolTip("0% sin overlay · 100% overlay máximo")
        self._blend_slider.valueChanged.connect(self._on_visual_controls_changed)
        self._blend_slider.setEnabled(False)
        blend_row.addWidget(self._blend_slider, 1)
        self._blend_lbl = QLabel("35%")
        self._blend_lbl.setStyleSheet("color:#94a3b8;")
        blend_row.addWidget(self._blend_lbl)
        overlay_qc_layout.addLayout(blend_row)

        # QC registro / Split / Fusión
        qc_row = QHBoxLayout()
        qc_row.addWidget(QLabel("QC registro:"))
        self._qc_mode = QComboBox()
        self._qc_mode.addItem("Off", "off")
        self._qc_mode.addItem("Fusión", "fusion")
        self._qc_mode.addItem("Bordes SPECT/CT", "edges")
        self._qc_mode.addItem("Split", "split")
        self._qc_mode.addItem("Checkerboard", "checker")
        self._qc_mode.addItem("Contornos CT", "contours")
        self._qc_mode.currentIndexChanged.connect(self._on_visual_controls_changed)
        self._qc_mode.setEnabled(False)
        qc_row.addWidget(self._qc_mode)
        # Split % (vertical para más recorrido)
        split_col = QVBoxLayout()
        split_col.addWidget(QLabel("Split %:"))
        self._qc_split_slider = QSlider(Qt.Orientation.Vertical)
        self._qc_split_slider.setRange(10, 90)
        self._qc_split_slider.setValue(50)
        self._qc_split_slider.valueChanged.connect(self._on_visual_controls_changed)
        self._qc_split_slider.setEnabled(False)
        self._qc_split_slider.setMinimumHeight(120)
        split_col.addWidget(self._qc_split_slider, 1)
        self._qc_split_lbl = QLabel("50%")
        self._qc_split_lbl.setStyleSheet("color:#94a3b8; font-size:10px;")
        self._qc_split_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        split_col.addWidget(self._qc_split_lbl)
        qc_row.addLayout(split_col, 1)

        # Fusión % (vertical para más recorrido)
        fusion_col = QVBoxLayout()
        fusion_col.addWidget(QLabel("Fusión %:"))
        self._fusion_slider = QSlider(Qt.Orientation.Vertical)
        self._fusion_slider.setRange(0, 100)
        self._fusion_slider.setValue(55)
        self._fusion_slider.setToolTip("0% CT solamente · 100% SPECT coloreado. Ajusta la mezcla de la fusión.")
        self._fusion_slider.valueChanged.connect(self._on_fusion_slider_changed)
        self._fusion_slider.setEnabled(False)
        self._fusion_slider.setMinimumHeight(120)
        fusion_col.addWidget(self._fusion_slider, 1)
        self._fusion_lbl = QLabel("55%")
        self._fusion_lbl.setStyleSheet("color:#94a3b8; font-size:10px;")
        self._fusion_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fusion_col.addWidget(self._fusion_lbl)
        qc_row.addLayout(fusion_col, 1)
        overlay_qc_layout.addLayout(qc_row)

        # Sliders de corte z/y/x
        slice_row = QHBoxLayout()
        slice_row.addWidget(QLabel("Cortes z/y/x:"))
        self._slice_z = QSlider(Qt.Orientation.Horizontal)
        self._slice_y = QSlider(Qt.Orientation.Horizontal)
        self._slice_x = QSlider(Qt.Orientation.Horizontal)
        self._slice_z_lbl = QLabel("z -")
        self._slice_y_lbl = QLabel("y -")
        self._slice_x_lbl = QLabel("x -")
        for slider in (self._slice_z, self._slice_y, self._slice_x):
            slider.setRange(0, 0)
            slider.setEnabled(False)
            slider.valueChanged.connect(self._on_slice_slider_changed)
        slice_row.addWidget(self._slice_z_lbl)
        slice_row.addWidget(self._slice_z, 1)
        slice_row.addWidget(self._slice_y_lbl)
        slice_row.addWidget(self._slice_y, 1)
        slice_row.addWidget(self._slice_x_lbl)
        slice_row.addWidget(self._slice_x, 1)
        overlay_qc_layout.addLayout(slice_row)
        
        # === Toggles de visibilidad de overlays en vistas MPR ===
        vis_row = QHBoxLayout()
        vis_row.setSpacing(4)
        vis_row.setContentsMargins(0, 2, 0, 0)
        
        _vis_style = (
            "QCheckBox { color:#94a3b8; font-size:10px; spacing:3px; }"
            "QCheckBox::indicator { width:13px; height:13px; border-radius:2px; "
            "border:1px solid #475569; background:#1e293b; }"
            "QCheckBox::indicator:checked { background:#3b82f6; border-color:#3b82f6; }"
            "QCheckBox::indicator:hover { border-color:#64748b; }"
        )
        
        self._chk_show_mask_mpr = QCheckBox("Máscara CT")
        self._chk_show_mask_mpr.setChecked(True)
        self._chk_show_mask_mpr.setToolTip("Mostrar/ocultar máscara CT segmentada en vistas MPR")
        self._chk_show_mask_mpr.setStyleSheet(_vis_style)
        self._chk_show_mask_mpr.stateChanged.connect(self._on_mpr_visibility_changed)
        vis_row.addWidget(self._chk_show_mask_mpr)
        
        self._chk_show_vois_mpr = QCheckBox("VOIs")
        self._chk_show_vois_mpr.setChecked(True)
        self._chk_show_vois_mpr.setToolTip("Mostrar/ocultar VOIs corazón/mediastino en vistas MPR")
        self._chk_show_vois_mpr.setStyleSheet(_vis_style)
        self._chk_show_vois_mpr.stateChanged.connect(self._on_mpr_visibility_changed)
        vis_row.addWidget(self._chk_show_vois_mpr)
        
        self._chk_show_cross_mpr = QCheckBox("Cruces")
        self._chk_show_cross_mpr.setChecked(True)
        self._chk_show_cross_mpr.setToolTip("Mostrar/ocultar líneas de corte cruzadas en vistas MPR")
        self._chk_show_cross_mpr.setStyleSheet(_vis_style)
        self._chk_show_cross_mpr.stateChanged.connect(self._on_mpr_visibility_changed)
        vis_row.addWidget(self._chk_show_cross_mpr)
        
        self._chk_show_loc_mpr = QCheckBox("Localización")
        self._chk_show_loc_mpr.setChecked(True)
        self._chk_show_loc_mpr.setToolTip("Mostrar/ocultar puntos de localización A/B en vistas MPR")
        self._chk_show_loc_mpr.setStyleSheet(_vis_style)
        self._chk_show_loc_mpr.stateChanged.connect(self._on_mpr_visibility_changed)
        vis_row.addWidget(self._chk_show_loc_mpr)
        
        vis_row.addStretch()
        overlay_qc_layout.addLayout(vis_row)

        # NOTA: overlay_qc_group se movió a columna derecha (arriba del MIP)
        # left_col.addWidget(overlay_qc_group)  ← MOVIDO A LA FILA DE BOXES
        # ────────────────────────────────────────────────────────────────────────────

        # ═══════════════════════════════════════════════════════════════════
        # CONTENEDOR DE CONTROLES: 2 COLUMNAS
        #   Izquierda:  HMR-SPEC (~50% ancho)
        #   Derecha:    Contenedor vertical:
        #               Arriba: Edición Manual Máscara CT
        #               Abajo:  [Orientación] [Zoom] [Ajuste] juntos
        # ═══════════════════════════════════════════════════════════════════
        controls_container = QWidget()
        controls_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        main_controls_layout = QHBoxLayout(controls_container)
        main_controls_layout.setContentsMargins(0, 0, 0, 0)
        main_controls_layout.setSpacing(6)
        
        # --- HMR-SPECT (angosto, ~60%) ---
        hmr_group = QGroupBox("HMR-SPECT")
        hmr_layout = QVBoxLayout(hmr_group)
        
        # Fila 1: Método
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Método:"))
        self._hmr_method_combo = QComboBox()
        self._hmr_method_combo.addItem(
            "VOI esférica completa (integra cuentas 3D)",
            HmrSpectMethod.VOI_COMPLETE.value
        )
        self._hmr_method_combo.addItem(
            "ROI slice central (slice único comparable a planar)",
            HmrSpectMethod.SLICE_CENTRAL.value
        )
        self._hmr_method_combo.setToolTip(
            "VOI completa: Mayor precisión, integra todo el volumen\n"
            "Slice central: Más simple, comparable con HMR planar"
        )
        method_row.addWidget(self._hmr_method_combo)
        method_row.addStretch(1)
        hmr_layout.addLayout(method_row)
        
        # Fila 2: Radios
        radius_row = QHBoxLayout()
        radius_row.addWidget(QLabel("Radio corazón (mm):"))
        self._heart_radius_spin = QDoubleSpinBox()
        self._heart_radius_spin.setRange(10.0, 60.0)
        self._heart_radius_spin.setValue(30.0)
        self._heart_radius_spin.setSingleStep(5.0)
        radius_row.addWidget(self._heart_radius_spin)
        
        radius_row.addWidget(QLabel("Radio mediastino (mm):"))
        self._mediastinum_radius_spin = QDoubleSpinBox()
        self._mediastinum_radius_spin.setRange(5.0, 40.0)
        self._mediastinum_radius_spin.setValue(20.0)
        self._mediastinum_radius_spin.setSingleStep(5.0)
        radius_row.addWidget(self._mediastinum_radius_spin)
        hmr_layout.addLayout(radius_row)
        
        # Fila 3: Botón calcular, preservar máscara, volumen y resultado
        calc_row = QHBoxLayout()
        self._btn_calc_hmr = QPushButton("Calcular HMR-SPECT")
        self._btn_calc_hmr.clicked.connect(self._calculate_hmr_spect)
        self._btn_calc_hmr.setToolTip(
            "Usa los puntos de localización:\n"
            "- Ancla A = centro VOI corazón\n"
            "- Cruz actual = centro VOI mediastino"
        )
        calc_row.addWidget(self._btn_calc_hmr)

        # Checkbox para preservar máscara manual al recalcular
        self._preserve_mask_check = QCheckBox("🔒 Preservar máscara")
        self._preserve_mask_check.setChecked(False)
        self._preserve_mask_check.setToolTip(
            "Cuando está activado:\n"
            "- Si editaste la máscara manualmente, se REUTILIZA tal cual\n"
            "- Solo se recalculan las VOIs con la nueva posición del punto B\n"
            "- Útil cuando mueves el mediastino pero quieres mantener tu máscara cardíaca"
        )
        calc_row.addWidget(self._preserve_mask_check)

        # Label de volumen visible SIEMPRE (no solo en consola oculta)
        self._lbl_volume_display = QLabel("❤️ -- mL")
        self._lbl_volume_display.setStyleSheet(
            "font-size:12px; font-weight:600; color:#60a5fa; background:#111827; padding:4px 8px;"
        )
        self._lbl_volume_display.setMinimumWidth(120)
        calc_row.addWidget(self._lbl_volume_display)

        self._lbl_hmr_result = QLabel("HMR-SPECT = N/D")
        self._lbl_hmr_result.setStyleSheet(
            "font-size:16px; font-weight:700; color:#ffffff; background:#000000; padding:6px 12px;"
        )
        calc_row.addWidget(self._lbl_hmr_result)
        calc_row.addStretch(1)
        hmr_layout.addLayout(calc_row)
        
        # Fila 4: Referencia de escala clínica
        scale_row = QHBoxLayout()
        scale_lbl = QLabel(
            "<span style='color:#ef4444; font-weight:600;'>≥1.6 POSITIVO</span> · "
            "<span style='color:#f59e0b; font-weight:600;'>1.5-1.6 EQUIVOCO</span> · "
            "<span style='color:#22c55e; font-weight:600;'>&lt;1.5 NEGATIVO</span>"
        )
        scale_lbl.setStyleSheet("font-size:11px; background:transparent;")
        scale_row.addWidget(scale_lbl)
        scale_row.addStretch(1)
        hmr_layout.addLayout(scale_row)

        # La vía estable mantiene las VOIs esféricas exactamente en los
        # puntos A/B. La segmentación CT queda aislada como opción beta.
        self._ct_anatomical_check = QCheckBox("CT anatómica / PVE (experimental)")
        self._ct_anatomical_check.setChecked(False)
        self._ct_anatomical_check.setToolTip(
            "Desactivado (recomendado): usa VOIs manuales esféricas ancladas en A/B.\n"
            "Activado: prueba segmentación CT, edición de máscara y corrección PVE."
        )
        self._ct_anatomical_check.toggled.connect(self._on_ct_anatomical_mode_toggled)
        hmr_layout.addWidget(self._ct_anatomical_check)

        # === Botones de persistencia CT (guardar/cargar/reiniciar) ===
        ct_persist_row = QHBoxLayout()
        
        self._btn_save_ct_state = QPushButton("💾 Guardar CT")
        self._btn_save_ct_state.setToolTip(
            "Guarda el estado actual de la segmentación CT (máscara, HMR, PVE)\n"
            "en un archivo .json para retomar en otra sesión."
        )
        self._btn_save_ct_state.setEnabled(False)
        self._btn_save_ct_state.clicked.connect(self._save_ct_state)
        ct_persist_row.addWidget(self._btn_save_ct_state)
        
        self._btn_load_ct_state = QPushButton("📂 Cargar CT")
        self._btn_load_ct_state.setToolTip(
            "Carga un estado de segmentación CT guardado previamente.\n"
            "Restaura máscara editada, volumen y resultados HMR."
        )
        self._btn_load_ct_state.clicked.connect(self._load_ct_state)
        ct_persist_row.addWidget(self._btn_load_ct_state)
        
        self._btn_restart_ct = QPushButton("🗑️ Reiniciar CT")
        self._btn_restart_ct.setStyleSheet("color:#ef4444; font-weight:bold;")
        self._btn_restart_ct.setToolTip(
            "BORRA toda la segmentación CT: máscara, HMR, PVE, edición manual.\n"
            "Vuelve al estado inicial (VOIs esféricas manuales)."
        )
        self._btn_restart_ct.setEnabled(False)
        self._btn_restart_ct.clicked.connect(self._restart_ct_state)
        ct_persist_row.addWidget(self._btn_restart_ct)
        
        hmr_layout.addLayout(ct_persist_row)

        # ═══════════════════════════════════════════════════════════════════
        # GRUPO: Ratio S/VD (Corazón / Vértebra / Aorta) — EXPERIMENTAL
        # ═══════════════════════════════════════════════════════════════════
        svd_group = QGroupBox("Ratio S/VD (Corazón / Vértebra / Aorta) — Experimental")
        svd_layout = QVBoxLayout(svd_group)
        svd_layout.setSpacing(4)

        # Fila 1: Selector de ROI activa + radios
        svd_roi_row = QHBoxLayout()
        svd_roi_row.addWidget(QLabel("ROI activa:"))
        self._svd_roi_combo = QComboBox()
        self._svd_roi_combo.addItem("S — Corazón", "S")
        self._svd_roi_combo.addItem("V — Vértebra torácica", "V")
        self._svd_roi_combo.addItem("D — Aorta descendente", "D")
        self._svd_roi_combo.setCurrentIndex(0)
        self._svd_roi_combo.setToolTip(
            "Selecciona qué ROI vas a depositar con Ctrl/Shift+clic.\n"
            "S = Corazón (usa radio corazón del HMR)\n"
            "V = Vértebra torácica (radio propio)\n"
            "D = Aorta descendente (radio propio)"
        )
        self._svd_roi_combo.currentIndexChanged.connect(self._on_svd_roi_changed)
        svd_roi_row.addWidget(self._svd_roi_combo)

        svd_roi_row.addWidget(QLabel("rV (mm):"))
        self._svd_vertebra_spin = QDoubleSpinBox()
        self._svd_vertebra_spin.setRange(5.0, 40.0)
        self._svd_vertebra_spin.setValue(15.0)
        self._svd_vertebra_spin.setSingleStep(1.0)
        self._svd_vertebra_spin.setToolTip("Radio de la ROI vértebra (mm)")
        svd_roi_row.addWidget(self._svd_vertebra_spin)

        svd_roi_row.addWidget(QLabel("rD (mm):"))
        self._svd_aorta_spin = QDoubleSpinBox()
        self._svd_aorta_spin.setRange(5.0, 40.0)
        self._svd_aorta_spin.setValue(12.0)
        self._svd_aorta_spin.setSingleStep(1.0)
        self._svd_aorta_spin.setToolTip("Radio de la ROI aorta (mm)")
        svd_roi_row.addWidget(self._svd_aorta_spin)
        svd_layout.addLayout(svd_roi_row)

        # Fila 2: Botones depositar / limpiar / calcular
        svd_btn_row = QHBoxLayout()
        self._btn_svd_deposit = QPushButton("📍 Depositar punto")
        self._btn_svd_deposit.setToolTip(
            "Deposita la cruz actual en la ROI activa seleccionada.\n"
            "Flujo: 1) Localización → clic en estructura → Depositar punto.\n"
            "Repetir para S, V y D."
        )
        self._btn_svd_deposit.clicked.connect(self._on_svd_deposit)
        svd_btn_row.addWidget(self._btn_svd_deposit)

        self._btn_svd_clear = QPushButton("Limpiar")
        self._btn_svd_clear.setToolTip("Borra los 3 puntos S/V/D")
        self._btn_svd_clear.clicked.connect(self._on_svd_clear)
        svd_btn_row.addWidget(self._btn_svd_clear)

        self._btn_calc_svd = QPushButton("Calcular S/VD")
        self._btn_calc_svd.clicked.connect(self._calculate_svd_ratio)
        self._btn_calc_svd.setToolTip(
            "Calcula S/VD = S / sqrt(V×D)\n"
            "Requiere los 3 puntos S, V y D depositados."
        )
        svd_btn_row.addWidget(self._btn_calc_svd)
        svd_layout.addLayout(svd_btn_row)

        # Fila 3: Resultado + botón info
        svd_result_row = QHBoxLayout()
        self._lbl_svd_result = QLabel("S/VD = N/D")
        self._lbl_svd_result.setStyleSheet(
            "font-size:14px; font-weight:700; color:#ffffff; background:#1e1b4b; padding:6px 12px;"
        )
        svd_result_row.addWidget(self._lbl_svd_result)
        svd_result_row.addStretch(1)
        # Botón ℹ de información S/VD
        self._btn_svd_info = QPushButton("ℹ Interpretación S/VD")
        self._btn_svd_info.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_svd_info.setStyleSheet(
            "QPushButton { font-size:11px; font-weight:600; color:#6366f1; background:transparent; "
            "border:1px dashed #6366f1; border-radius:4px; padding:4px 10px; }"
            "QPushButton:hover { background:#eef2ff; }"
        )
        self._btn_svd_info.clicked.connect(self._show_svd_info_dialog)
        svd_result_row.addWidget(self._btn_svd_info)
        svd_layout.addLayout(svd_result_row)

        # Fila 4: Escala de referencia
        svd_scale_row = QHBoxLayout()
        svd_scale_lbl = QLabel(
            "<span style='color:#ef4444; font-weight:600;'>≥2.2 POSITIVO</span> · "
            "<span style='color:#f59e0b; font-weight:600;'>1.8-2.2 EQUIVOCO</span> · "
            "<span style='color:#22c55e; font-weight:600;'>&lt;1.8 NEGATIVO</span>"
        )
        svd_scale_lbl.setStyleSheet("font-size:10px; background:transparent;")
        svd_scale_row.addWidget(svd_scale_lbl)
        svd_scale_row.addStretch(1)
        svd_layout.addLayout(svd_scale_row)

        hmr_layout.addWidget(svd_group)

        # === F2.4: Edición manual de máscara CT (brush/erase) ===
        edit_group = QGroupBox("F2.4 Edición Máscara CT")
        edit_group.setStyleSheet(
            "QGroupBox { font-weight:600; border:1px solid #4b5563; border-radius:6px; margin-top:18px; padding-top:14px; }"
        )
        edit_layout = QVBoxLayout(edit_group)
        edit_layout.setSpacing(2)
        edit_layout.setContentsMargins(4, 10, 4, 4)
        
        # Fila de controles principales
        edit_ctrl_row = QHBoxLayout()
        
        # Toggle modo edición
        self._btn_toggle_mask_edit = QPushButton("✏️ Editar Máscara")
        self._btn_toggle_mask_edit.setCheckable(True)
        self._btn_toggle_mask_edit.setEnabled(False)
        self._btn_toggle_mask_edit.setToolTip(
            "Activa/desactiva el modo de edición manual de la máscara CT segmentada.\n"
            "Una vez activado, pinta sobre la vista AXIAL para agregar o quitar tejido.\n"
            "Requiere que exista una segmentación CT previa (Fase 2)."
        )
        self._btn_toggle_mask_edit.toggled.connect(self._on_mask_edit_toggled)
        edit_ctrl_row.addWidget(self._btn_toggle_mask_edit)
        
        # Toggle pintar / borrar
        self._btn_paint_erase = QPushButton("🖌️ Pintar")
        self._btn_paint_erase.setCheckable(True)
        self._btn_paint_erase.setChecked(True)  # Default: pintar
        self._btn_paint_erase.setEnabled(False)
        self._btn_paint_erase.setToolTip(
            "🖌️ Pintar = Agregar tejido a la máscara (blanco)\n"
            "🧹 Borrar = Quitar tejido de la máscara (negro)"
        )
        self._btn_paint_erase.toggled.connect(self._on_paint_erase_toggled)
        edit_ctrl_row.addWidget(self._btn_paint_erase)
        
        # Radio del brush
        edit_ctrl_row.addWidget(QLabel("Radio:"))
        self._brush_radius_spin = QSpinBox()
        self._brush_radius_spin.setRange(3, 50)
        self._brush_radius_spin.setValue(10)
        self._brush_radius_spin.setSuffix(" px")
        self._brush_radius_spin.setEnabled(False)
        self._brush_radius_spin.setToolTip("Radio del pincel en píxeles de pantalla")
        edit_ctrl_row.addWidget(self._brush_radius_spin)
        
        edit_layout.addLayout(edit_ctrl_row)
        
        # Fila de acciones
        edit_action_row = QHBoxLayout()
        
        # Deshacer última acción
        self._btn_undo_mask = QPushButton("↩️ Deshacer")
        self._btn_undo_mask.setEnabled(False)
        self._btn_undo_mask.setToolTip("Deshacer la última acción de pincel")
        self._btn_undo_mask.clicked.connect(self._undo_mask_edit)
        edit_action_row.addWidget(self._btn_undo_mask)
        
        # Resetear a máscara original
        self._btn_reset_mask = QPushButton("🔄 Reset Original")
        self._btn_reset_mask.setEnabled(False)
        self._btn_reset_mask.setToolTip(
            "Restaurar la máscara CT a su estado original (antes de edición manual)"
        )
        self._btn_reset_mask.clicked.connect(self._reset_mask_to_original)
        edit_action_row.addWidget(self._btn_reset_mask)
        
        # Aplicar y recalcular
        self._btn_apply_mask_edit = QPushButton("✅ Aplicar → Recalcular")
        self._btn_apply_mask_edit.setEnabled(False)
        self._btn_apply_mask_edit.setStyleSheet(
            "background-color:#059669; color:white; font-weight:bold; padding:6px 12px;"
        )
        self._btn_apply_mask_edit.setToolTip(
            "Aplica los cambios de edición manual y recalcula HMR con la nueva máscara"
        )
        self._btn_apply_mask_edit.clicked.connect(self._apply_mask_edit_and_recalc)
        edit_action_row.addWidget(self._btn_apply_mask_edit)
        
        # Exportar máscara como NIfTI para inspección 3D externa
        self._btn_export_mask_nifti = QPushButton("💾 Exportar NIfTI")
        self._btn_export_mask_nifti.setEnabled(False)
        self._btn_export_mask_nifti.setToolTip(
            "Guarda la máscara CT como archivo .nii.gz para abrir en 3D Slicer / ITK-SNAP"
        )
        self._btn_export_mask_nifti.clicked.connect(self._export_mask_nifti)
        edit_action_row.addWidget(self._btn_export_mask_nifti)
        
        edit_layout.addLayout(edit_action_row)
        
        # Label de estado de edición
        self._mask_edit_status = QLabel("Modo edición: INACTIVO")
        self._mask_edit_status.setStyleSheet(
            "color:#9ca3af; font-style:italic; font-size:11px;"
        )
        edit_layout.addWidget(self._mask_edit_status)
        
        # NOTA: edit_group se agrega en top_row (fila superior, al lado de HMR-SPEC)

        # --- Orientación (ULTRA COMPACTO - una fila) ---
        flip_group = QGroupBox("Orientación")
        flip_grid = QGridLayout(flip_group)
        flip_grid.setSpacing(1)
        flip_grid.setContentsMargins(3, 2, 3, 2)
        
        self._spect_flipx_check = QCheckBox("SPECT flip X")
        self._spect_flipx_check.setToolTip("Espeja SPECT en eje X")
        self._spect_flipx_check.toggled.connect(self._on_spect_orientation_test_toggled)
        self._spect_flipx_check.setStyleSheet("font-size:9px;")
        flip_grid.addWidget(self._spect_flipx_check, 0, 0)
        
        self._spect_flipy_check = QCheckBox("SPECT flip Y")
        self._spect_flipy_check.setToolTip("Espeja SPECT en eje Y")
        self._spect_flipy_check.toggled.connect(self._on_spect_orientation_test_toggled)
        self._spect_flipy_check.setStyleSheet("font-size:9px;")
        flip_grid.addWidget(self._spect_flipy_check, 0, 1)
        
        self._spect_flipz_check = QCheckBox("SPECT flip Z")
        self._spect_flipz_check.setToolTip("Espeja SPECT en eje Z")
        self._spect_flipz_check.toggled.connect(self._on_spect_orientation_test_toggled)
        self._spect_flipz_check.setStyleSheet("font-size:9px;")
        flip_grid.addWidget(self._spect_flipz_check, 1, 0)
        
        self._ct_flipx_check = QCheckBox("CT flip X")
        self._ct_flipx_check.setToolTip("Espeja TC en eje X")
        self._ct_flipx_check.toggled.connect(self._on_ct_orientation_test_toggled)
        self._ct_flipx_check.setStyleSheet("font-size:9px;")
        flip_grid.addWidget(self._ct_flipx_check, 1, 1)
        
        self._ct_flipy_check = QCheckBox("CT flip Y")
        self._ct_flipy_check.setToolTip("Espeja TC en eje Y")
        self._ct_flipy_check.toggled.connect(self._on_ct_orientation_test_toggled)
        self._ct_flipy_check.setStyleSheet("font-size:9px;")
        flip_grid.addWidget(self._ct_flipy_check, 2, 0)
        
        self._ct_flipz_check = QCheckBox("CT flip Z")
        self._ct_flipz_check.setToolTip("Espeja TC en eje Z")
        self._ct_flipz_check.toggled.connect(self._on_ct_orientation_test_toggled)
        self._ct_flipz_check.setStyleSheet("font-size:9px;")
        flip_grid.addWidget(self._ct_flipz_check, 2, 1)
        
        main_controls_layout.addWidget(hmr_group, 5)   # HMR-SPEC: 5/10 (izquierda ~50%)

        # --- Overlay / QC / Cortes MOVIDO a columna derecha (arriba del MIP) ---

        # --- Zoom visual ---
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QVBoxLayout(zoom_group)
        zoom_layout.setSpacing(1)
        zoom_layout.setContentsMargins(3, 2, 3, 2)

        spect_zoom_row = QHBoxLayout()
        spect_zoom_row.setSpacing(4)
        self._spect_zoom_lbl = QLabel("SPECT:")
        self._spect_zoom_lbl.setStyleSheet("font-size:9px;")
        spect_zoom_row.addWidget(self._spect_zoom_lbl)
        self._spect_zoom_spin = QSpinBox()
        self._spect_zoom_spin.setRange(50, 200)
        self._spect_zoom_spin.setValue(100)
        self._spect_zoom_spin.setSuffix("%")
        self._spect_zoom_spin.valueChanged.connect(self._on_zoom_changed)
        self._spect_zoom_spin.setStyleSheet("font-size:9px; padding:0px 2px;")
        spect_zoom_row.addWidget(self._spect_zoom_spin, 1)
        zoom_layout.addLayout(spect_zoom_row)

        ct_zoom_row = QHBoxLayout()
        ct_zoom_row.setSpacing(4)
        self._ct_zoom_lbl = QLabel("CT:")
        self._ct_zoom_lbl.setStyleSheet("font-size:9px;")
        ct_zoom_row.addWidget(self._ct_zoom_lbl)
        self._ct_zoom_spin = QSpinBox()
        self._ct_zoom_spin.setRange(50, 200)
        self._ct_zoom_spin.setValue(100)
        self._ct_zoom_spin.setSuffix("%")
        self._ct_zoom_spin.valueChanged.connect(self._on_zoom_changed)
        self._ct_zoom_spin.setStyleSheet("font-size:9px; padding:0px 2px;")
        ct_zoom_row.addWidget(self._ct_zoom_spin, 1)
        zoom_layout.addLayout(ct_zoom_row)

        # Checkbox para anclar/desanclar zooms SPECT y CT
        link_zoom_row = QHBoxLayout()
        link_zoom_row.setSpacing(4)
        self._link_zoom_check = QCheckBox("🔗 Anclar zooms")
        self._link_zoom_check.setChecked(True)  # ANCLADOS por defecto
        self._link_zoom_check.setToolTip(
            "Cuando está activado, cambiar el zoom de SPECT también cambia el CT "
            "y viceversa. Desactívalos para ajustarlos independientemente."
        )
        self._link_zoom_check.setStyleSheet("font-size:9px; color:#94a3b8;")
        self._link_zoom_check.stateChanged.connect(self._on_link_zoom_changed)
        link_zoom_row.addWidget(self._link_zoom_check, 1)
        zoom_layout.addLayout(link_zoom_row)

        # --- COLUMNA DERECHA: Edición Máscara (arriba) + Orient/Zoom/Ajuste (abajo) ---
        right_side_widget = QWidget()
        right_side_layout = QVBoxLayout(right_side_widget)
        right_side_layout.setContentsMargins(0, 0, 0, 0)
        right_side_layout.setSpacing(4)

        # Arriba: Edición Máscara CT (ocupa todo el ancho de la columna derecha)
        right_side_layout.addWidget(edit_group)

        # Abajo: Orientación + Zoom + Ajuste en fila horizontal juntos
        bottom_groups_row = QHBoxLayout()
        bottom_groups_row.setSpacing(4)
        bottom_groups_row.addWidget(flip_group, 3)      # Orientación
        bottom_groups_row.addWidget(zoom_group, 3)      # Zoom

        # --- Ajuste CT (nudge/rot/resets) ---
        ajuste_group = QGroupBox("Ajuste")
        ajuste_layout = QVBoxLayout(ajuste_group)
        ajuste_layout.setSpacing(1)
        ajuste_layout.setContentsMargins(3, 2, 3, 2)

        # Nudge Δ z/y/x
        nudge_row = QHBoxLayout()
        nudge_row.setSpacing(2)
        nudge_row.addWidget(QLabel("Δ CT:"))
        self._nudge_z = self._mk_nudge_spin()
        self._nudge_y = self._mk_nudge_spin()
        self._nudge_x = self._mk_nudge_spin()
        for spin in (self._nudge_z, self._nudge_y, self._nudge_x):
            spin.valueChanged.connect(self._apply_ct_nudge)
            spin.setEnabled(False)
            spin.setStyleSheet("font-size:9px;")
        nudge_row.addWidget(self._nudge_z)
        nudge_row.addWidget(self._nudge_y)
        nudge_row.addWidget(self._nudge_x)
        ajuste_layout.addLayout(nudge_row)

        # Rotación z/y/x
        rot_row = QHBoxLayout()
        rot_row.setSpacing(2)
        rot_row.addWidget(QLabel("Rot:"))
        self._rot_z = self._mk_rotate_spin()
        self._rot_y = self._mk_rotate_spin()
        self._rot_x = self._mk_rotate_spin()
        for spin in (self._rot_z, self._rot_y, self._rot_x):
            spin.valueChanged.connect(self._apply_ct_nudge)
            spin.setEnabled(False)
            spin.setStyleSheet("font-size:9px;")
        rot_row.addWidget(self._rot_z)
        rot_row.addWidget(self._rot_y)
        rot_row.addWidget(self._rot_x)
        ajuste_layout.addLayout(rot_row)

        # Botones de reset
        reset_row = QHBoxLayout()
        reset_row.setSpacing(2)
        self._btn_reset_nudge = QPushButton("Reset Δ")
        self._btn_reset_nudge.clicked.connect(self._reset_ct_nudge)
        self._btn_reset_nudge.setEnabled(False)
        self._btn_reset_nudge.setStyleSheet("font-size:9px; padding:1px 4px;")
        reset_row.addWidget(self._btn_reset_nudge)
        self._btn_reset_rot = QPushButton("Reset Rot")
        self._btn_reset_rot.clicked.connect(self._reset_ct_rotation)
        self._btn_reset_rot.setEnabled(False)
        self._btn_reset_rot.setStyleSheet("font-size:9px; padding:1px 4px;")
        reset_row.addWidget(self._btn_reset_rot)
        self._btn_reset_offsets = QPushButton("Reset vista")
        self._btn_reset_offsets.clicked.connect(self._reset_view_offsets)
        self._btn_reset_offsets.setToolTip("Resetea offsets relativos y zoom visual SPECT/CT.")
        self._btn_reset_offsets.setStyleSheet("font-size:9px; padding:1px 4px;")
        reset_row.addWidget(self._btn_reset_offsets)
        ajuste_layout.addLayout(reset_row)

        bottom_groups_row.addWidget(ajuste_group, 4)   # Ajuste

        right_side_layout.addLayout(bottom_groups_row)

        main_controls_layout.addWidget(right_side_widget, 5)  # Columna derecha: 5/10

        left_col.addWidget(controls_container)
        # ────────────────────────────────────────────────────────────────────────────

        main_splitter.addLayout(left_col, 6)  # 6 partes para imágenes (más espacio)

        # --- COLUMNA DERECHA: Controles + MIP ---
        right_col = QVBoxLayout()
        right_col.setSpacing(6)

        # ═══ FILA SUPERIOR: Ventana/Color (izq) + Overlay/QC/Cortes (der) ═══
        top_right_row = QHBoxLayout()
        top_right_row.setSpacing(6)

        # --- Grupo: Rango y Color SPECT ---
        window_group = QGroupBox("Ventana / Color")
        window_group.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        window_group.setStyleSheet("QGroupBox { font-weight:600; }")
        window_vbox = QVBoxLayout(window_group)
        window_vbox.setAlignment(Qt.AlignmentFlag.AlignRight)
        window_vbox.setSpacing(4)
        
        # Botón Top (arriba) - angosto
        top_btn_row = QHBoxLayout()
        self._btn_range_top = QPushButton("▲ T100")
        self._btn_range_top.clicked.connect(lambda: self._set_spect_range(self._spect_win_low, 100))
        self._btn_range_top.setToolTip("Fija el límite superior al 100%")
        self._btn_range_top.setFixedSize(70, 28)
        self._btn_range_top.setStyleSheet("font-size:10px; padding:2px 6px;")
        top_btn_row.addWidget(self._btn_range_top)
        window_vbox.addLayout(top_btn_row)

        # RangeSlider (centro, vertical) - centrado
        slider_row = QHBoxLayout()
        self._spect_range_slider = RangeSlider()
        self._spect_range_slider.setMinimumHeight(280)
        self._spect_range_slider.setFixedWidth(50)
        self._spect_range_slider.valuesChanged.connect(self._on_spect_range_changed)
        slider_row.addWidget(self._spect_range_slider)
        window_vbox.addLayout(slider_row)

        self._spect_range_lbl = QLabel("Base 0% · Top 100%")
        self._spect_range_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._spect_range_lbl.setStyleSheet("color:#94a3b8; font-size:11px;")
        window_vbox.addWidget(self._spect_range_lbl)

        # Botón Base (abajo) - angosto
        base_btn_row = QHBoxLayout()
        self._btn_range_base = QPushButton("▼ B0")
        self._btn_range_base.clicked.connect(lambda: self._set_spect_range(0, self._spect_win_high))
        self._btn_range_base.setToolTip("Fija el límite inferior al 0%")
        self._btn_range_base.setFixedSize(70, 28)
        self._btn_range_base.setStyleSheet("font-size:10px; padding:2px 6px;")
        base_btn_row.addWidget(self._btn_range_base)
        window_vbox.addLayout(base_btn_row)

        # Selector de color (compacto)
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Color:"))
        register_all_colormaps()
        self._spect_cmap_combo = QComboBox()
        self._spect_cmap_combo.addItems(available_colormaps())
        if self._spect_cmap_combo.findText("hot") >= 0:
            self._spect_cmap_combo.setCurrentText("hot")
        self._spect_cmap_combo.currentIndexChanged.connect(self._render_current_with_overlay)
        self._spect_cmap_combo.setFixedWidth(90)
        self._spect_cmap_combo.setStyleSheet("font-size:10px; padding:1px 4px;")
        color_row.addWidget(self._spect_cmap_combo)
        window_vbox.addLayout(color_row)

        # Suavizado de vista SPECT (solo display, tipo filtro de visor Xeleris)
        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Suavizar vista:"))
        self._spect_smooth_spin = QDoubleSpinBox()
        self._spect_smooth_spin.setRange(0.0, 3.0)
        self._spect_smooth_spin.setSingleStep(0.1)
        self._spect_smooth_spin.setDecimals(1)
        self._spect_smooth_spin.setValue(0.8)
        self._spect_smooth_spin.setToolTip(
            "Suavizado gaussiano SOLO de la vista SPECT (σ en píxeles).\n"
            "No modifica el volumen ni la cuantificación (HMR/S·VD).\n"
            "0 = sin suavizado."
        )
        self._spect_smooth_spin.setFixedWidth(60)
        self._spect_smooth_spin.setStyleSheet("font-size:10px; padding:1px 4px;")
        self._spect_smooth_spin.valueChanged.connect(self._on_spect_display_smooth_changed)
        smooth_row.addWidget(self._spect_smooth_spin)
        window_vbox.addLayout(smooth_row)

        # Ventana CT (compacto)
        ct_win_row = QHBoxLayout()
        ct_win_row.addWidget(QLabel("Ventana CT:"))
        self._ct_window_combo = QComboBox()
        self._ct_window_combo.addItem("Ósea", "bone")
        self._ct_window_combo.addItem("Partes blandas", "soft")
        self._ct_window_combo.addItem("Pulmón", "lung")
        self._ct_window_combo.addItem("Completa", "full")
        self._ct_window_combo.currentIndexChanged.connect(self._on_ct_window_changed)
        self._ct_window_combo.setFixedWidth(90)
        self._ct_window_combo.setStyleSheet("font-size:10px; padding:1px 4px;")
        ct_win_row.addWidget(self._ct_window_combo)
        window_vbox.addLayout(ct_win_row)

        trial_row = QHBoxLayout()
        self._ct_trial_check = QCheckBox("CT nítida")
        self._ct_trial_check.setChecked(True)
        self._ct_trial_check.setToolTip("Realce visual del MPR de CT (activado por defecto). Desmarcar para ver la CT sin realce.")
        self._ct_trial_check.setStyleSheet("font-size:10px; color:#f59e0b; font-weight:600;")
        self._ct_trial_check.toggled.connect(self._on_ct_trial_toggled)
        trial_row.addWidget(self._ct_trial_check)
        window_vbox.addLayout(trial_row)

        trial_grid_row = QHBoxLayout()
        self._ct_grid_trial_check = QCheckBox("PRUEBA CT nativa + SPECT escalado (BETA)")
        self._ct_grid_trial_check.setChecked(False)
        self._ct_grid_trial_check.setToolTip("Modo de prueba reversible: mantiene resolución CT nativa y remuestrea SPECT a grilla CT.")
        self._ct_grid_trial_check.setStyleSheet("font-size:10px; color:#f59e0b; font-weight:600;")
        self._ct_grid_trial_check.toggled.connect(self._on_ct_grid_trial_toggled)
        trial_grid_row.addWidget(self._ct_grid_trial_check)
        trial_grid_row.addStretch()
        # Botón info CT (i en círculo)
        self._ct_info_btn = QPushButton("\u2139\ufe0f")
        self._ct_info_btn.setFixedSize(22, 22)
        self._ct_info_btn.setToolTip("Rol del CT en diagnóstico de amiloidosis")
        self._ct_info_btn.setStyleSheet(
            "font-size:13px; border-radius:11px; background:#3b82f6; color:white;"
            "border:none; font-weight:bold; font-style:italic;"
        )
        self._ct_info_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ct_info_btn.clicked.connect(self._show_ct_role_info)
        trial_grid_row.addWidget(self._ct_info_btn)
        window_vbox.addLayout(trial_grid_row)

        top_right_row.addWidget(window_group, 1)   # Ventana/Color: 1 parte
        
        # Overlay / QC / Cortes (al lado de Ventana/Color)
        top_right_row.addWidget(overlay_qc_group, 1)  # Overlay/QC: 1 parte

        right_col.addLayout(top_right_row)  # Fila con ambos grupos lado a lado
        
        # ═══ MIP 360° (debajo, ocupa todo el ancho) ═══
        right_col.addWidget(mip_group)
        
        # Sin addStretch: el MIP se expande para ocupar todo el espacio disponible

        main_splitter.addLayout(right_col, 1)  # 1 parte para controles
        root.addLayout(main_splitter)

        self._metrics = QTextEdit()
        self._metrics.setReadOnly(True)
        self._metrics.setStyleSheet("background:#0f172a; color:#e2e8f0; border:1px solid #334155;")
        self._metrics.setMaximumHeight(0)  # OCULTA por defecto
        self._metrics.setVisible(False)  # Oculta al inicio

        # Botón toggle para ocultar/mostrar consola
        self._btn_toggle_console = QPushButton("▶ Consola")
        self._btn_toggle_console.setCheckable(True)
        self._btn_toggle_console.setChecked(False)  # OCULTA por defecto
        self._btn_toggle_console.setToolTip("Ocultar / Mostrar consola de métricas")
        self._btn_toggle_console.setStyleSheet(
            "font-size:10px; padding:2px 8px; background:#1e293b; color:#94a3b8; "
            "border:1px solid #334155; border-radius:3px;"
        )
        self._btn_toggle_console.toggled.connect(self._toggle_console)

        root.addWidget(self._btn_toggle_console)
        root.addWidget(self._metrics, 1)

        footer = QLabel(
            "Módulo experimental fase 2: métricas 3D proxy y sustracción ósea visual de apoyo. "
            "No constituye interpretación diagnóstica automática."
        )
        footer.setWordWrap(True)
        footer.setStyleSheet("color:#94a3b8; font-size:10px;")
        root.addWidget(footer)
        self._restore_global_ui_state()

    @staticmethod
    def _mk_filter_combo() -> QComboBox:
        combo = QComboBox()
        combo.addItem("Sin filtro", "none")
        combo.addItem("Butterworth", "butterworth")
        combo.addItem("Low-pass", "lowpass")
        combo.addItem("Wiener", "wiener")
        return combo

    @staticmethod
    def _mk_cutoff_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.01, 1.00)
        spin.setSingleStep(0.01)
        spin.setDecimals(2)
        spin.setValue(float(value))
        spin.setToolTip("Frecuencia normalizada respecto de Nyquist.")
        return spin

    @staticmethod
    def _mk_order_spin(value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(1, 20)
        spin.setValue(int(value))
        return spin

    @staticmethod
    def _mk_nudge_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-64.0, 64.0)
        spin.setSingleStep(0.5)
        spin.setDecimals(1)
        spin.setSuffix(" px")
        spin.setToolTip("Ajuste manual post-registro CT→SPECT en píxeles de la grilla SPECT.")
        return spin

    @staticmethod
    def _mk_rotate_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-45.0, 45.0)
        spin.setSingleStep(0.5)
        spin.setDecimals(1)
        spin.setSuffix("°")
        spin.setToolTip("Rotación manual post-registro CT→SPECT. z=axial, y=cabeceo coronal, x=cabeceo sagital.")
        return spin

    @staticmethod
    def _settings_id(path: str) -> str:
        raw = os.path.abspath(str(path or "")).encode("utf-8", errors="ignore")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=") or "sin_estudio"

    def _last_dir(self, fallback: str = "") -> str:
        remembered = str(self._settings.value("global/last_dir", "") or "")
        if remembered and os.path.isdir(remembered):
            return remembered
        if fallback and os.path.isdir(fallback):
            return fallback
        return os.path.expanduser("~")

    def _remember_path(self, path: str) -> None:
        if not path:
            return
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        if folder and os.path.isdir(folder):
            self._settings.setValue("global/last_dir", folder)

    def _study_settings_prefix(self) -> str:
        return f"studies/{self._settings_id(self._current_spect_path)}"

    @staticmethod
    def _norm_token(value: str) -> str:
        text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value or ""))
        while "__" in text:
            text = text.replace("__", "_")
        return text.strip("_") or "na"

    def _profile_compact_text(self) -> str:
        if not self._dicom_profile_info:
            return "perfil N/D"
        mf = str(self._dicom_profile_info.get("manufacturer") or "N/D")
        model = str(self._dicom_profile_info.get("model") or "N/D")
        proto = str(self._dicom_profile_info.get("protocol") or "N/D")
        return f"{mf} · {model} · {proto}"

    def _camera_profile_key(self) -> str:
        info = dict(self._dicom_profile_info or {})
        parts = [
            self._norm_token(self._workflow_tag),
            self._norm_token(info.get("manufacturer", "")),
            self._norm_token(info.get("model", "")),
            self._norm_token(info.get("station", "")),
            self._norm_token(info.get("protocol", "") or info.get("series_description", "") or info.get("study_description", "")),
        ]
        return "|".join(parts)

    def _read_dicom_profile_info(self, path: str) -> dict:
        info = {
            "manufacturer": "",
            "model": "",
            "station": "",
            "protocol": "",
            "study_description": "",
            "series_description": "",
            "modality": "",
            "patient_name": "",
            "patient_id": "",
            "study_date": "",
            "path": str(path or ""),
        }
        try:
            import pydicom

            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            info["manufacturer"] = str(getattr(ds, "Manufacturer", "") or "")
            info["model"] = str(getattr(ds, "ManufacturerModelName", "") or "")
            info["station"] = str(getattr(ds, "StationName", "") or "")
            info["protocol"] = str(getattr(ds, "ProtocolName", "") or "")
            info["study_description"] = str(getattr(ds, "StudyDescription", "") or "")
            info["series_description"] = str(getattr(ds, "SeriesDescription", "") or "")
            info["modality"] = str(getattr(ds, "Modality", "") or "")
            info["patient_name"] = str(getattr(ds, "PatientName", "") or "")
            info["patient_id"] = str(getattr(ds, "PatientID", "") or "")
            info["study_date"] = str(getattr(ds, "StudyDate", "") or "")
        except Exception:
            pass
        return info

    def _persist_report_bridge_state(self):
        try:
            bridge = QSettings("GAMMASYS", "SINCRO_AMYLO_BRIDGE")
            payload = {
                "spect_path": str(self._current_spect_path or ""),
                "ct_path": str(self._ct_path or ""),
                "att_path": str(self._att_path or ""),
                "workflow_tag": str(self._workflow_tag or ""),
                "profile": dict(self._dicom_profile_info or {}),
                "camera_profile_key": str(self._camera_profile_key() if self._current_spect_path else ""),
                "preset": str(self._preset_combo.currentData() or "manual") if hasattr(self, "_preset_combo") else "manual",
                "ac_iter": bool(self._ac_iter_check.isChecked()) if hasattr(self, "_ac_iter_check") else False,
                "ac_mu_scale": float(self._ac_mu_scale_spin.value()) if hasattr(self, "_ac_mu_scale_spin") else 1.0,
                "qc_mode": str(self._qc_mode.currentData() or "off") if hasattr(self, "_qc_mode") else "off",
                "fusion_pct": int(self._fusion_slider.value()) if hasattr(self, "_fusion_slider") else int(getattr(self, "_fusion_pct", 55)),
                "spect_flip_x": bool(self._spect_flip_x_test),
                "spect_flip_y": bool(self._spect_flip_y_test),
                "spect_flip_z": bool(self._spect_flip_z_test),
                "ct_flip_x": bool(self._ct_flip_x_test),
                "ct_flip_y": bool(self._ct_flip_y_test),
                "ct_flip_z": bool(self._ct_flip_z_test),
                "ct_nudge_zyx": [float(self._nudge_z.value()), float(self._nudge_y.value()), float(self._nudge_x.value())] if hasattr(self, "_nudge_z") else [0.0, 0.0, 0.0],
                "ct_rot_zyx": [float(self._rot_z.value()), float(self._rot_y.value()), float(self._rot_x.value())] if hasattr(self, "_rot_z") else [0.0, 0.0, 0.0],
            }
            bridge.setValue("last_spect_ct_session_json", json.dumps(payload, ensure_ascii=False))
            bridge.sync()
        except Exception:
            pass

    @staticmethod
    def _infer_workflow_tag(info: dict) -> str:
        txt = " ".join(
            [
                str(info.get("protocol", "") or ""),
                str(info.get("study_description", "") or ""),
                str(info.get("series_description", "") or ""),
            ]
        ).lower()
        if any(k in txt for k in ("amylo", "amilo", "pyp", "pyrophosphate")):
            return "amylo"
        if any(k in txt for k in ("perfusion", "perfusion", "mibi", "sestamibi", "cardio", "spect/ct")):
            return "perf_spect_ct"
        return "perf_spect_ct"

    def _camera_preset_settings_key(self) -> str:
        return f"camera_presets/{self._camera_profile_key()}"

    def _collect_current_adjustments(self) -> dict:
        return {
            "workflow_tag": self._workflow_tag,
            "profile_info": dict(self._dicom_profile_info or {}),
            "spect_flip_x": bool(self._spect_flipx_check.isChecked()),
            "spect_flip_y": bool(self._spect_flipy_check.isChecked()),
            "spect_flip_z": bool(self._spect_flipz_check.isChecked()),
            "ct_flip_x": bool(self._ct_flipx_check.isChecked()),
            "ct_flip_y": bool(self._ct_flipy_check.isChecked()),
            "ct_flip_z": bool(self._ct_flipz_check.isChecked()),
            "nudge_z": float(self._nudge_z.value()),
            "nudge_y": float(self._nudge_y.value()),
            "nudge_x": float(self._nudge_x.value()),
            "rot_z": float(self._rot_z.value()),
            "rot_y": float(self._rot_y.value()),
            "rot_x": float(self._rot_x.value()),
        }

    def _apply_adjustments_to_ui(self, payload: dict, *, auto: bool = False) -> None:
        if not payload:
            return
        for check, key in (
            (self._spect_flipx_check, "spect_flip_x"),
            (self._spect_flipy_check, "spect_flip_y"),
            (self._spect_flipz_check, "spect_flip_z"),
            (self._ct_flipx_check, "ct_flip_x"),
            (self._ct_flipy_check, "ct_flip_y"),
            (self._ct_flipz_check, "ct_flip_z"),
        ):
            if key in payload:
                check.blockSignals(True)
                check.setChecked(bool(payload.get(key, False)))
                check.blockSignals(False)
        self._on_spect_orientation_test_toggled(True)
        self._on_ct_orientation_test_toggled(True)

        if self._ct_auto_registered is not None:
            for spin, key in (
                (self._nudge_z, "nudge_z"),
                (self._nudge_y, "nudge_y"),
                (self._nudge_x, "nudge_x"),
                (self._rot_z, "rot_z"),
                (self._rot_y, "rot_y"),
                (self._rot_x, "rot_x"),
            ):
                if key in payload:
                    spin.blockSignals(True)
                    spin.setValue(float(payload.get(key, 0.0)))
                    spin.blockSignals(False)
            self._apply_ct_nudge()
            self._pending_camera_profile_adjust = None
        else:
            self._pending_camera_profile_adjust = dict(payload)

        mode = "auto" if auto else "manual"
        self._status.setText(f"Preset cámara aplicado ({mode}) · {self._profile_compact_text()}")
        self._metrics.append(
            "\n--- Preset cámara aplicado ---\n"
            f"- modo: {mode}\n"
            f"- workflow: {self._workflow_tag}\n"
            f"- perfil: {self._profile_compact_text()}"
        )
        self._persist_ui_state()

    def _save_camera_profile_preset(self):
        if not self._current_spect_path:
            self._status.setText("Cargar primero un SPECT para guardar preset de cámara.")
            return
        payload = self._collect_current_adjustments()
        try:
            self._settings.setValue(self._camera_preset_settings_key(), json.dumps(payload, ensure_ascii=False))
            self._status.setText(f"Preset cámara guardado · {self._profile_compact_text()}")
            self._metrics.append(
                "\n--- Preset cámara guardado ---\n"
                f"- key: {self._camera_profile_key()}\n"
                f"- workflow: {self._workflow_tag}"
            )
        except Exception as exc:
            self._status.setText(f"No se pudo guardar preset cámara: {exc}")

    def _apply_camera_profile_preset(self, *, auto: bool):
        if not self._current_spect_path:
            return
        raw = self._settings.value(self._camera_preset_settings_key(), "")
        payload = None
        if raw:
            try:
                payload = json.loads(str(raw))
            except Exception:
                payload = None
        if not payload:
            if not auto:
                self._status.setText("No hay preset guardado para esta cámara/protocolo.")
            return
        if auto and str(payload.get("workflow_tag", "")) != "perf_spect_ct":
            return
        self._apply_adjustments_to_ui(payload, auto=auto)

    @staticmethod
    def _set_combo_by_data(combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _persist_ui_state(self):
        if not hasattr(self, "_settings"):
            return
        cuts_mode = str(self._cuts_mode_combo.currentData() or "mixed")
        qc_mode = str(self._qc_mode.currentData() or "off")
        preset = str(self._preset_combo.currentData() or "manual")
        split = int(self._qc_split_slider.value())
        overlay = int(self._blend_slider.value())
        fusion = int(self._fusion_slider.value()) if hasattr(self, "_fusion_slider") else int(getattr(self, "_fusion_pct", 55))
        self._settings.setValue("global/preset", preset)
        self._settings.setValue("global/ung_method", str(self._ung_method_combo.currentData() or "fbp"))
        self._settings.setValue("global/gated_method", str(self._gated_method_combo.currentData() or "fbp"))
        self._settings.setValue("global/cuts_mode", cuts_mode)
        self._settings.setValue("global/qc_mode", qc_mode)
        self._settings.setValue("global/split_pct", split)
        self._settings.setValue("global/overlay_pct", overlay)
        self._settings.setValue("global/fusion_pct", fusion)
        self._settings.setValue("global/triangulation_cross", bool(getattr(self, "_triangulation_cross_enabled", False)))
        self._settings.setValue("global/localization_cross", bool(getattr(self, "_localization_cross_enabled", False)))
        self._settings.setValue("global/ung_filter", str(self._ung_filter_combo.currentData() or "butterworth"))
        self._settings.setValue("global/ung_cutoff", float(self._ung_cutoff_spin.value()))
        self._settings.setValue("global/ung_order", int(self._ung_order_spin.value()))
        self._settings.setValue("global/gated_filter", str(self._gated_filter_combo.currentData() or "butterworth"))
        self._settings.setValue("global/gated_cutoff", float(self._gated_cutoff_spin.value()))
        self._settings.setValue("global/gated_order", int(self._gated_order_spin.value()))
        self._settings.setValue("global/iter", int(self._iter_spin.value()))
        self._settings.setValue("global/subsets", int(self._subsets_spin.value()))
        self._settings.setValue("global/background_subtract", bool(self._bg_check.isChecked()))
        self._settings.setValue("global/scatter_subtract", bool(self._scatter_check.isChecked()))
        self._settings.setValue("global/scatter_k", float(self._scatter_k_spin.value()))
        self._settings.setValue("global/post_filter", bool(self._post_check.isChecked()))
        self._settings.setValue("global/post_sigma", float(self._post_sigma_spin.value()))
        self._settings.setValue("global/denoise_plus", bool(self._denoise_plus_check.isChecked()))
        self._settings.setValue("global/denoise_plus_k", float(self._denoise_plus_k_spin.value()))
        self._settings.setValue("global/ac_iter", bool(self._ac_iter_check.isChecked()))
        self._settings.setValue("global/ac_mu_scale", float(self._ac_mu_scale_spin.value()))
        if hasattr(self, "_ct_trial_check"):
            self._settings.setValue("global/ct_sharp", bool(self._ct_trial_check.isChecked()))
        if hasattr(self, "_spect_smooth_spin"):
            self._settings.setValue("global/spect_display_sigma", float(self._spect_smooth_spin.value()))
        if self._current_spect_path:
            prefix = self._study_settings_prefix()
            self._settings.setValue(f"{prefix}/preset", preset)
            self._settings.setValue(f"{prefix}/cuts_mode", cuts_mode)
            self._settings.setValue(f"{prefix}/qc_mode", qc_mode)
            self._settings.setValue(f"{prefix}/split_pct", split)
            self._settings.setValue(f"{prefix}/overlay_pct", overlay)
            self._settings.setValue(f"{prefix}/fusion_pct", fusion)
            if self._ct_path:
                self._settings.setValue(f"{prefix}/ct_path", self._ct_path)
        self._persist_report_bridge_state()

    def _toggle_console(self, visible: bool):
        """Oculta o muestra la consola de métricas para dar más espacio a las imágenes."""
        self._metrics.setVisible(visible)
        if visible:
            self._btn_toggle_console.setText("▼ Consola")
            self._metrics.setMaximumHeight(120)
        else:
            self._btn_toggle_console.setText("▶ Consola")
            self._metrics.setMaximumHeight(0)

    def _restore_global_ui_state(self):
        self._set_combo_by_data(self._preset_combo, str(self._settings.value("global/preset", "amylo360_std128") or "amylo360_std128"))
        self._set_combo_by_data(self._ung_method_combo, str(self._settings.value("global/ung_method", "fbp") or "fbp"))
        self._set_combo_by_data(self._gated_method_combo, str(self._settings.value("global/gated_method", "fbp") or "fbp"))
        self._set_combo_by_data(self._cuts_mode_combo, str(self._settings.value("global/cuts_mode", "mixed") or "mixed"))
        self._set_combo_by_data(self._qc_mode, str(self._settings.value("global/qc_mode", "off") or "off"))
        self._set_combo_by_data(self._ung_filter_combo, str(self._settings.value("global/ung_filter", "butterworth") or "butterworth"))
        self._ung_cutoff_spin.setValue(float(self._settings.value("global/ung_cutoff", 0.45) or 0.45))
        self._ung_order_spin.setValue(int(self._settings.value("global/ung_order", 8) or 8))
        self._set_combo_by_data(self._gated_filter_combo, str(self._settings.value("global/gated_filter", "butterworth") or "butterworth"))
        self._gated_cutoff_spin.setValue(float(self._settings.value("global/gated_cutoff", 0.40) or 0.40))
        self._gated_order_spin.setValue(int(self._settings.value("global/gated_order", 10) or 10))
        self._iter_spin.setValue(int(self._settings.value("global/iter", 3) or 3))
        self._subsets_spin.setValue(int(self._settings.value("global/subsets", 6) or 6))
        self._bg_check.setChecked(str(self._settings.value("global/background_subtract", "false")).lower() == "true")
        self._scatter_check.setChecked(str(self._settings.value("global/scatter_subtract", "false")).lower() == "true")
        self._scatter_k_spin.setValue(float(self._settings.value("global/scatter_k", 1.0) or 1.0))
        self._post_check.setChecked(str(self._settings.value("global/post_filter", "false")).lower() == "true")
        self._post_sigma_spin.setValue(float(self._settings.value("global/post_sigma", 0.5) or 0.5))
        self._denoise_plus_check.setChecked(str(self._settings.value("global/denoise_plus", "false")).lower() == "true")
        self._denoise_plus_k_spin.setValue(float(self._settings.value("global/denoise_plus_k", 0.20) or 0.20))
        self._ac_iter_check.setChecked(str(self._settings.value("global/ac_iter", "false")).lower() == "true")
        self._ac_mu_scale_spin.setValue(float(self._settings.value("global/ac_mu_scale", 1.0) or 1.0))
        if hasattr(self, "_ct_trial_check"):
            ct_sharp = str(self._settings.value("global/ct_sharp", "true")).lower() == "true"
            self._ct_trial_check.setChecked(ct_sharp)
            self._ct_visual_trial_mode = ct_sharp
        if hasattr(self, "_spect_smooth_spin"):
            sigma = float(self._settings.value("global/spect_display_sigma", 0.8) or 0.0)
            self._spect_smooth_spin.blockSignals(True)
            self._spect_smooth_spin.setValue(sigma)
            self._spect_smooth_spin.blockSignals(False)
            self._spect_display_sigma = sigma
        self._qc_split_slider.setValue(int(self._settings.value("global/split_pct", 50) or 50))
        self._blend_slider.setValue(int(self._settings.value("global/overlay_pct", 35) or 35))
        fusion = int(self._settings.value("global/fusion_pct", 55) or 55)
        self._fusion_slider.setValue(fusion)
        self._fusion_pct = fusion
        triang = str(self._settings.value("global/triangulation_cross", "false")).lower() == "true"
        self._triangulation_cross_enabled = bool(triang)
        if hasattr(self, "_btn_triangulation_cross"):
            self._btn_triangulation_cross.blockSignals(True)
            self._btn_triangulation_cross.setChecked(bool(triang))
            self._btn_triangulation_cross.blockSignals(False)
        localize = str(self._settings.value("global/localization_cross", "false")).lower() == "true"
        self._localization_cross_enabled = bool(localize)
        if hasattr(self, "_btn_localization_cross"):
            self._btn_localization_cross.blockSignals(True)
            self._btn_localization_cross.setChecked(bool(localize))
            self._btn_localization_cross.blockSignals(False)
        self._on_preset_changed()

    def _restore_study_ui_state(self):
        if not self._current_spect_path:
            return
        prefix = self._study_settings_prefix()
        self._set_combo_by_data(
            self._preset_combo,
            str(self._settings.value(f"{prefix}/preset", self._preset_combo.currentData()) or "amylo360_std128"),
        )
        self._set_combo_by_data(
            self._cuts_mode_combo,
            str(self._settings.value(f"{prefix}/cuts_mode", self._cuts_mode_combo.currentData()) or "mixed"),
        )
        self._set_combo_by_data(
            self._qc_mode,
            str(self._settings.value(f"{prefix}/qc_mode", self._qc_mode.currentData()) or "off"),
        )
        self._qc_split_slider.setValue(int(self._settings.value(f"{prefix}/split_pct", self._qc_split_slider.value()) or 50))
        self._blend_slider.setValue(int(self._settings.value(f"{prefix}/overlay_pct", self._blend_slider.value()) or 35))
        fusion = int(self._settings.value(f"{prefix}/fusion_pct", self._fusion_slider.value()) or 55)
        self._fusion_slider.setValue(fusion)
        self._fusion_pct = fusion
        self._ct_path = str(self._settings.value(f"{prefix}/ct_path", "") or "")
        self._on_preset_changed()

    def _on_preset_changed(self):
        preset = str(self._preset_combo.currentData() or "manual")
        manual = preset == "manual"
        self._recon_combo.setEnabled(manual)
        if preset == "amylo360_std128":
            self._set_combo_by_data(self._recon_combo, "fbp")
            self._set_combo_by_data(self._ung_method_combo, "fbp")
            self._set_combo_by_data(self._gated_method_combo, "fbp")
            self._set_combo_by_data(self._ung_filter_combo, "butterworth")
            self._ung_cutoff_spin.setValue(0.45)
            self._ung_order_spin.setValue(8)
            self._set_combo_by_data(self._gated_filter_combo, "butterworth")
            self._gated_cutoff_spin.setValue(0.40)
            self._gated_order_spin.setValue(10)
            self._iter_spin.setValue(3)
            self._subsets_spin.setValue(6)
            self._bg_check.setChecked(False)
            self._scatter_check.setChecked(False)
            self._post_check.setChecked(True)
            self._post_sigma_spin.setValue(0.45)
            self._denoise_plus_check.setChecked(False)
            self._ac_iter_check.setChecked(False)
            self._ac_mu_scale_spin.setValue(1.00)
        elif preset == "amylo360_hd":
            self._set_combo_by_data(self._recon_combo, "osem")
            self._set_combo_by_data(self._ung_method_combo, "osem")
            self._set_combo_by_data(self._gated_method_combo, "osem")
            self._set_combo_by_data(self._ung_filter_combo, "none")
            self._ung_cutoff_spin.setValue(0.50)
            self._ung_order_spin.setValue(1)
            self._set_combo_by_data(self._gated_filter_combo, "none")
            self._gated_cutoff_spin.setValue(0.50)
            self._gated_order_spin.setValue(1)
            self._iter_spin.setValue(3)
            self._subsets_spin.setValue(6)
            self._bg_check.setChecked(False)
            self._scatter_check.setChecked(False)
            self._post_check.setChecked(True)
            self._post_sigma_spin.setValue(0.35)
            self._denoise_plus_check.setChecked(True)
            self._denoise_plus_k_spin.setValue(0.20)
            self._ac_iter_check.setChecked(True)
            self._ac_mu_scale_spin.setValue(1.00)
        elif preset == "clinical_osem_soft":
            self._set_combo_by_data(self._recon_combo, "osem")
            self._set_combo_by_data(self._ung_method_combo, "osem")
            self._set_combo_by_data(self._gated_method_combo, "osem")
            self._set_combo_by_data(self._ung_filter_combo, "none")
            self._ung_cutoff_spin.setValue(0.50)
            self._ung_order_spin.setValue(1)
            self._set_combo_by_data(self._gated_filter_combo, "none")
            self._gated_cutoff_spin.setValue(0.50)
            self._gated_order_spin.setValue(1)
            self._iter_spin.setValue(2)
            self._subsets_spin.setValue(10)
            self._bg_check.setChecked(False)
            self._scatter_check.setChecked(False)
            self._post_check.setChecked(True)
            self._post_sigma_spin.setValue(1.0)
            self._denoise_plus_check.setChecked(False)
            # AC iterativa: se respeta el estado actual.
        elif preset == "clinical_osem":
            self._set_combo_by_data(self._recon_combo, "osem")
            self._set_combo_by_data(self._ung_method_combo, "osem")
            self._set_combo_by_data(self._gated_method_combo, "osem")
            self._set_combo_by_data(self._ung_filter_combo, "none")
            self._ung_cutoff_spin.setValue(0.50)
            self._ung_order_spin.setValue(1)
            self._set_combo_by_data(self._gated_filter_combo, "none")
            self._gated_cutoff_spin.setValue(0.50)
            self._gated_order_spin.setValue(1)
            self._iter_spin.setValue(8)
            self._subsets_spin.setValue(4)
            self._bg_check.setChecked(True)
            self._scatter_check.setChecked(False)
            self._post_check.setChecked(False)
            self._denoise_plus_check.setChecked(False)
            # AC iterativa: se respeta el estado actual del checkbox.
        for widget in (
            self._ung_method_combo,
            self._ung_filter_combo,
            self._ung_cutoff_spin,
            self._ung_order_spin,
            self._gated_method_combo,
            self._gated_filter_combo,
            self._gated_cutoff_spin,
            self._gated_order_spin,
            self._iter_spin,
            self._subsets_spin,
            self._bg_check,
            self._scatter_check,
            self._scatter_k_spin,
            self._post_check,
            self._post_sigma_spin,
            self._denoise_plus_check,
            self._denoise_plus_k_spin,
        ):
            widget.setEnabled(manual)
        self._refresh_gated_controls()
        self._persist_ui_state()

    def _on_manual_method_changed(self, *_):
        method = str(self._ung_method_combo.currentData() or "fbp")
        self._set_combo_by_data(self._recon_combo, method)
        self._persist_ui_state()

    def _on_top_recon_changed(self, *_):
        if not hasattr(self, "_ung_method_combo"):
            return
        method = str(self._recon_combo.currentData() or "fbp")
        self._set_combo_by_data(self._ung_method_combo, method)
        self._persist_ui_state()

    def _refresh_gated_controls(self):
        gated_enabled = bool(self._study_is_gated and str(self._preset_combo.currentData() or "manual") == "manual")
        for widget in (self._gated_method_combo, self._gated_filter_combo, self._gated_cutoff_spin, self._gated_order_spin):
            widget.setEnabled(gated_enabled)
        self._gated_title_lbl.setText("Gated" if self._study_is_gated else "Gated (no disponible)")
        cardiac_idx = self._cuts_mode_combo.findData("cardiac")
        if cardiac_idx >= 0:
            model_item = self._cuts_mode_combo.model().item(cardiac_idx)
            if model_item is not None:
                model_item.setEnabled(bool(self._study_is_gated))
        if not self._study_is_gated and str(self._cuts_mode_combo.currentData() or "mixed") == "cardiac":
            self._set_combo_by_data(self._cuts_mode_combo, "tomo")

    def _build_recon_config(self):
        from core.raw_reconstruction import ProjectionFilterConfig, RawReconConfig

        preset = str(self._preset_combo.currentData() or "manual")
        method = str(self._ung_method_combo.currentData() or self._recon_combo.currentData() or "fbp")
        gated_method = str(self._gated_method_combo.currentData() or method) if self._study_is_gated else method
        if preset == "amylo360_std128":
            return RawReconConfig(
                reconstruction_method="fbp",
                gated_method="fbp",
                ungated_filter=ProjectionFilterConfig("butterworth", 0.45, 8),
                gated_filter=ProjectionFilterConfig("butterworth", 0.40, 10),
                post_filter_sigma_ungated_px=0.45,
                post_filter_sigma_gated_px=0.60,
                display_slice_step_px=2,
            )
        if preset == "amylo360_hd":
            return RawReconConfig(
                reconstruction_method="osem",
                gated_method="osem",
                ungated_filter=ProjectionFilterConfig("none", 0.50, 1),
                gated_filter=ProjectionFilterConfig("none", 0.50, 1),
                iterative_iterations=3,
                osem_subsets=6,
                post_filter_sigma_ungated_px=0.35,
                post_filter_sigma_gated_px=0.50,
                ungated_denoise_plus=True,
                ungated_denoise_plus_k=0.20,
                gated_denoise_plus=True,
                gated_denoise_plus_k=0.45,
                display_slice_step_px=1,
            )
        if preset == "clinical_osem_soft":
            # Perfil suave: pocas actualizaciones (2×10) + gaussiano 3D.
            return RawReconConfig(
                reconstruction_method="osem",
                gated_method="osem",
                ungated_filter=ProjectionFilterConfig("none", 0.50, 1),
                gated_filter=ProjectionFilterConfig("none", 0.50, 1),
                iterative_iterations=2,
                osem_subsets=10,
                post_filter_sigma_ungated_px=1.0,
                post_filter_sigma_gated_px=1.1,
                display_slice_step_px=1,
                attenuation_correction=bool(self._ac_iter_check.isChecked()),
                attenuation_mu_scale=float(self._ac_mu_scale_spin.value()),
            )
        if preset == "clinical_osem":
            # Protocolo óseo clásico: OSEM 8it×4sub + Butterworth 3D 0.35/5
            # post-recon + descuento de fondo. Las 32 actualizaciones limpian
            # fondo y definen hueso; el Butterworth corta el ruido fino sin
            # difuminar bordes (a diferencia del gaussiano).
            return RawReconConfig(
                reconstruction_method="osem",
                gated_method="osem",
                ungated_filter=ProjectionFilterConfig("none", 0.50, 1),
                gated_filter=ProjectionFilterConfig("none", 0.50, 1),
                iterative_iterations=8,
                osem_subsets=4,
                background_subtract=True,
                post_filter_kind="butterworth",
                post_filter_cutoff=0.35,
                post_filter_order=5,
                display_slice_step_px=1,
                attenuation_correction=bool(self._ac_iter_check.isChecked()),
                attenuation_mu_scale=float(self._ac_mu_scale_spin.value()),
            )
        return RawReconConfig(
            reconstruction_method=method,
            gated_method=gated_method,
            ungated_filter=ProjectionFilterConfig(
                str(self._ung_filter_combo.currentData() or "butterworth"),
                float(self._ung_cutoff_spin.value()),
                int(self._ung_order_spin.value()),
            ),
            gated_filter=ProjectionFilterConfig(
                str(self._gated_filter_combo.currentData() or "butterworth"),
                float(self._gated_cutoff_spin.value()),
                int(self._gated_order_spin.value()),
            ),
            iterative_iterations=int(self._iter_spin.value()),
            osem_subsets=int(self._subsets_spin.value()),
            background_subtract=bool(self._bg_check.isChecked()),
            scatter_subtract=bool(self._scatter_check.isChecked()),
            scatter_k=float(self._scatter_k_spin.value()),
            post_filter_sigma_ungated_px=float(self._post_sigma_spin.value()) if self._post_check.isChecked() else 0.0,
            post_filter_sigma_gated_px=float(self._post_sigma_spin.value()) if self._study_is_gated and self._post_check.isChecked() else 0.0,
            ungated_denoise_plus=bool(self._denoise_plus_check.isChecked()),
            ungated_denoise_plus_k=float(self._denoise_plus_k_spin.value()),
            gated_denoise_plus=False,
            attenuation_correction=bool(self._ac_iter_check.isChecked()),
            attenuation_mu_scale=float(self._ac_mu_scale_spin.value()),
        )

    def _current_preset_label(self) -> str:
        return str(self._preset_combo.currentText() or "Manual")

    @staticmethod
    def _matrix_recommendation(shape: tuple[int, ...]) -> str:
        if len(shape) < 2:
            return "Matriz no determinada."
        rows, cols = int(shape[-2]), int(shape[-1])
        if rows == 128 and cols == 128:
            return "Matriz 128x128: objetivo recomendado para AMYLO/CT."
        if rows < 128 or cols < 128:
            return f"Matriz {rows}x{cols}: funciona, pero 128x128 facilita correlación con CT."
        return f"Matriz {rows}x{cols}: alta densidad de muestreo; vigilar ruido/tiempo de reconstrucción."

    @staticmethod
    def _fov_text(shape, spacing) -> str:
        if shape is None:
            return "shape=N/D spacing=N/D FOV=N/D"
        if spacing is None:
            return f"shape={tuple(shape)} spacing=N/D FOV=N/D"
        fov = tuple(float(shape[i]) * float(spacing[i]) for i in range(3))
        return (
            f"shape={tuple(shape)} spacing z/y/x="
            f"{spacing[0]:.3f}/{spacing[1]:.3f}/{spacing[2]:.3f} mm "
            f"FOV={fov[0]:.1f}/{fov[1]:.1f}/{fov[2]:.1f} mm"
        )

    def _append_grid_report(self):
        sp_shape = tuple(np.asarray(self._base_spect_volume).shape) if self._base_spect_volume is not None else None
        ct_shape = tuple(np.asarray(self._ct_volume).shape) if self._ct_volume is not None else None
        reg_shape = tuple(np.asarray(self._ct_registered).shape) if self._ct_registered is not None else None
        self._metrics.append("\n--- Grilla / FOV ---")
        self._metrics.append(f"SPECT: {self._fov_text(sp_shape, self._spect_spacing_zyx)}")
        self._metrics.append(f"CT original: {self._fov_text(ct_shape, self._ct_spacing_zyx)}")
        self._metrics.append(
            "Geometría DICOM: "
            f"SPECT={'OK' if self._spect_affine_ijk_to_lps is not None else 'N/D'} · "
            f"CT={'OK' if self._ct_affine_ijk_to_lps is not None else 'N/D'}"
        )
        if reg_shape is not None:
            self._metrics.append(f"CT registrado: shape={reg_shape} (misma grilla de display que SPECT)")

    def _on_visual_controls_changed(self):
        self._persist_ui_state()
        self._render_current_with_overlay()
    
    def _on_mpr_visibility_changed(self):
        """Toggle de visibilidad de overlays en vistas MPR."""
        self._render_current_with_overlay()

    def _on_fusion_slider_changed(self, value: int):
        self._fusion_pct = int(value)
        self._fusion_lbl.setText(f"{int(value)}%")
        self._persist_ui_state()
        self._render_current_with_overlay()

    def _on_spect_orientation_test_toggled(self, checked: bool):
        self._spect_flip_x_test = bool(self._spect_flipx_check.isChecked())
        self._spect_flip_y_test = bool(self._spect_flipy_check.isChecked())
        self._spect_flip_z_test = bool(self._spect_flipz_check.isChecked())
        self._status.setText(
            "Prueba orientación SPECT: "
            f"flip X {'ON' if self._spect_flip_x_test else 'OFF'} · "
            f"flip Y {'ON' if self._spect_flip_y_test else 'OFF'} · "
            f"flip Z {'ON' if self._spect_flip_z_test else 'OFF'}"
        )
        self._render_current_with_overlay()

    def _on_ct_orientation_test_toggled(self, checked: bool):
        self._ct_flip_x_test = bool(self._ct_flipx_check.isChecked())
        self._ct_flip_y_test = bool(self._ct_flipy_check.isChecked())
        self._ct_flip_z_test = bool(self._ct_flipz_check.isChecked())
        self._status.setText(
            "Prueba orientación CT: "
            f"flip X {'ON' if self._ct_flip_x_test else 'OFF'} · "
            f"flip Y {'ON' if self._ct_flip_y_test else 'OFF'} · "
            f"flip Z {'ON' if self._ct_flip_z_test else 'OFF'}"
        )
        self._render_current_with_overlay()

    def _spect_transform_3d(self, volume: np.ndarray) -> np.ndarray:
        vol = np.asarray(volume, dtype=np.float64)
        if bool(getattr(self, "_spect_flip_x_test", False)):
            vol = np.ascontiguousarray(np.flip(vol, axis=2))
        if bool(getattr(self, "_spect_flip_y_test", False)):
            vol = np.ascontiguousarray(np.flip(vol, axis=1))
        if bool(getattr(self, "_spect_flip_z_test", False)):
            vol = np.ascontiguousarray(np.flip(vol, axis=0))
        return vol

    def _spect_display_volume(self, volume: np.ndarray) -> np.ndarray:
        """Devuelve el SPECT en la misma grilla donde se localizan las VOI."""
        vol = self._spect_transform_3d(volume)
        off_z = float(self._spect_view_offset.get("axial", 0.0))
        off_y = float(self._spect_view_offset.get("coronal", 0.0))
        off_x = float(self._spect_view_offset.get("sagittal", 0.0))
        if abs(off_z) > 1e-6 or abs(off_y) > 1e-6 or abs(off_x) > 1e-6:
            vol = ndi.shift(vol, shift=(off_z, off_y, off_x), order=1, mode="nearest")
        return vol

    def _ct_transform_3d(self, volume: np.ndarray) -> np.ndarray:
        vol = np.asarray(volume, dtype=np.float64)
        if bool(getattr(self, "_ct_flip_x_test", False)):
            vol = np.ascontiguousarray(np.flip(vol, axis=2))
        if bool(getattr(self, "_ct_flip_y_test", False)):
            vol = np.ascontiguousarray(np.flip(vol, axis=1))
        if bool(getattr(self, "_ct_flip_z_test", False)):
            vol = np.ascontiguousarray(np.flip(vol, axis=0))
        return vol

    def _ct_registered_visual_transform(self, volume: np.ndarray) -> np.ndarray:
        vol = np.asarray(volume, dtype=np.float64)
        registered = getattr(self, "_ct_registered_flip_signature", (False, False, False))
        current = (
            bool(getattr(self, "_ct_flip_x_test", False)),
            bool(getattr(self, "_ct_flip_y_test", False)),
            bool(getattr(self, "_ct_flip_z_test", False)),
        )
        for axis, changed in ((2, current[0] != registered[0]), (1, current[1] != registered[1]), (0, current[2] != registered[2])):
            if changed:
                vol = np.ascontiguousarray(np.flip(vol, axis=axis))
        return vol

    def _task_progress_start(self, message: str):
        self._progress.setValue(3)
        self._progress.setFormat(f"3% · {message}")
        QApplication.processEvents()

    def _task_progress_step(self, value: int, message: str):
        self._progress.setValue(int(np.clip(value, 0, 100)))
        self._progress.setFormat(f"{int(np.clip(value, 0, 100))}% · {message}")
        QApplication.processEvents()

    def _task_progress_done(self, message: str):
        self._progress.setValue(100)
        self._progress.setFormat(f"100% · {message}")
        QApplication.processEvents()

    def _mk_image_label(self, title: str) -> QLabel:
        lbl = QLabel(title)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setMinimumSize(300, 250)
        lbl.setStyleSheet("background:#0b1220; color:#94a3b8; border:1px solid #334155;")
        lbl.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        if title.lower().startswith("ax"):
            lbl.wheelEvent = lambda ev, axis="axial": self._on_image_wheel(ev, axis)
            lbl.mousePressEvent = lambda ev, axis="axial": self._on_image_mouse_press(ev, axis)
            lbl.mouseMoveEvent = lambda ev, axis="axial": self._on_image_mouse_move(ev, axis)
            lbl.mouseReleaseEvent = lambda ev, axis="axial": self._on_image_mouse_release(ev, axis)
        elif title.lower().startswith("cor"):
            lbl.wheelEvent = lambda ev, axis="coronal": self._on_image_wheel(ev, axis)
            lbl.mousePressEvent = lambda ev, axis="coronal": self._on_image_mouse_press(ev, axis)
            lbl.mouseMoveEvent = lambda ev, axis="coronal": self._on_image_mouse_move(ev, axis)
            lbl.mouseReleaseEvent = lambda ev, axis="coronal": self._on_image_mouse_release(ev, axis)
        else:
            lbl.wheelEvent = lambda ev, axis="sagittal": self._on_image_wheel(ev, axis)
            lbl.mousePressEvent = lambda ev, axis="sagittal": self._on_image_mouse_press(ev, axis)
            lbl.mouseMoveEvent = lambda ev, axis="sagittal": self._on_image_mouse_move(ev, axis)
            lbl.mouseReleaseEvent = lambda ev, axis="sagittal": self._on_image_mouse_release(ev, axis)
        return lbl


    @staticmethod
    def _arr_to_pixmap(arr: np.ndarray) -> QPixmap:
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError("Se esperaba imagen 2D")
        mn, mx = float(np.min(a)), float(np.max(a))
        if mx - mn < 1e-9:
            norm = np.zeros_like(a, dtype=np.uint8)
        else:
            norm = np.clip((a - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)
        rgb = np.stack([norm, norm, norm], axis=-1)
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        # Percentiles en vez de min/max: un voxel caliente aislado no debe
        # oscurecer todo el corte (MPR "negros" de la comparativa vs Xeleris).
        a = np.asarray(arr, dtype=np.float64)
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return np.zeros_like(a, dtype=np.float64)
        mn = float(np.percentile(finite, 0.5))
        mx = float(np.percentile(finite, 99.5))
        if mx - mn < 1e-9:
            mn, mx = float(finite.min()), float(finite.max())
            if mx - mn < 1e-9:
                return np.zeros_like(a, dtype=np.float64)
        return np.clip((a - mn) / (mx - mn), 0.0, 1.0)

    def _window_spect(self, arr: np.ndarray) -> np.ndarray:
        a = self._normalize(arr)
        lo = float(self._spect_win_low) / 100.0
        hi = float(self._spect_win_high) / 100.0
        if hi <= lo:
            hi = lo + 0.01
        return np.clip((a - lo) / (hi - lo), 0.0, 1.0)

    def _window_ct(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        mode = str(self._ct_window or "bone")
        if mode == "soft":
            lo, hi = -160.0, 240.0
        elif mode == "lung":
            lo, hi = -1000.0, 200.0
        elif mode == "full":
            lo, hi = float(np.percentile(a, 1.0)), float(np.percentile(a, 99.0))
        else:
            lo, hi = -200.0, 1000.0
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((a - lo) / (hi - lo), 0.0, 1.0)

    @staticmethod
    def _enhance_ct_trial(img01: np.ndarray) -> np.ndarray:
        """Realce visual SOLO para prueba (no modifica datos clínicos)."""
        a = np.clip(np.asarray(img01, dtype=np.float64), 0.0, 1.0)
        low = ndi.gaussian_filter(a, sigma=0.8)
        amount = 0.9
        sharp = a + amount * (a - low)
        return np.clip(sharp, 0.0, 1.0)

    def _apply_cmap(self, img: np.ndarray) -> np.ndarray:
        a = np.asarray(img, dtype=np.float64)
        cmap_name = str(self._spect_cmap_combo.currentText() or "hot") if hasattr(self, "_spect_cmap_combo") else "hot"
        try:
            import matplotlib as mpl
            cmap = mpl.colormaps[cmap_name]
            return np.asarray(cmap(np.clip(a, 0.0, 1.0))[..., :3], dtype=np.float64)
        except Exception:
            return np.stack([a, np.clip(a * 0.75, 0, 1), np.zeros_like(a)], axis=-1)

    @staticmethod
    @staticmethod
    def _pan_2d_center(img: np.ndarray, pan_yx: list[int] | tuple[int, int], order: int = 1) -> np.ndarray:
        """Pan centrado con control de orden de interpolación.
        
        Args:
            img: Imagen 2D o 3D.
            pan_yx: Desplazamiento en píxeles (dy, dx).
            order: Orden de interpolación scipy.ndi.shift (0=NN, 1=bilineal).
        """
        arr = np.asarray(img, dtype=np.float64)
        if arr.ndim not in (2, 3):
            return arr
        dy, dx = int(pan_yx[0]), int(pan_yx[1])
        if dy == 0 and dx == 0:
            return arr
        return ndi.shift(arr, shift=(dy, dx, 0) if arr.ndim == 3 else (dy, dx), order=order, mode="constant", cval=0.0)

    @staticmethod
    def _zoom_2d_center(img: np.ndarray, zoom_pct: int, order: int = 1) -> np.ndarray:
        """Zoom centrado con control de orden de interpolación.
        
        Args:
            img: Imagen 2D.
            zoom_pct: Porcentaje de zoom (100 = sin cambio).
            order: Orden de interpolación scipy.ndi.zoom:
                0 = nearest-neighbor (nítido, ideal para CT)
                1 = bilineal (suave, ideal para SPECT)
        """
        arr = np.asarray(img, dtype=np.float64)
        if arr.ndim != 2:
            return arr
        z = max(0.05, float(zoom_pct) / 100.0)
        if abs(z - 1.0) < 1e-6:
            return arr
        out_shape = arr.shape
        scaled = ndi.zoom(arr, z, order=order)
        result = np.zeros(out_shape, dtype=np.float64)
        src_slices = []
        dst_slices = []
        for src_len, dst_len in zip(scaled.shape, out_shape):
            if src_len <= dst_len:
                src0 = 0
                src1 = src_len
                dst0 = (dst_len - src_len) // 2
                dst1 = dst0 + src_len
            else:
                src0 = (src_len - dst_len) // 2
                src1 = src0 + dst_len
                dst0 = 0
                dst1 = dst_len
            src_slices.append(slice(src0, src1))
            dst_slices.append(slice(dst0, dst1))
        result[tuple(dst_slices)] = scaled[tuple(src_slices)]
        return result

    @classmethod
    def _make_overlay_rgb(cls, base2d: np.ndarray, mask2d: np.ndarray | None, alpha: float) -> np.ndarray:
        g = cls._normalize(base2d)
        rgb = np.stack([g, g, g], axis=-1)
        if mask2d is None:
            return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        m = np.asarray(mask2d, dtype=bool)
        a = float(np.clip(alpha, 0.0, 1.0))
        if np.any(m) and a > 0.0:
            rgb[m, 0] = (1.0 - a) * rgb[m, 0] + a * 1.0
            rgb[m, 1] = (1.0 - a) * rgb[m, 1] + a * 0.25
            rgb[m, 2] = (1.0 - a) * rgb[m, 2] + a * 0.25
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    def _make_qc_rgb(self, spect2d: np.ndarray, ct2d: np.ndarray, mode: str, split_pct: int) -> np.ndarray:
        sp = np.clip(np.asarray(spect2d, dtype=np.float64), 0.0, 1.0)
        ct = np.clip(np.asarray(ct2d, dtype=np.float64), 0.0, 1.0)
        rgb_sp = self._apply_cmap(sp)
        rgb_ct = np.stack([ct, ct, ct], axis=-1)
        mode = str(mode or "off").lower()

        if mode == "fusion":
            mix = float(np.clip(getattr(self, "_fusion_pct", 55), 0, 100)) / 100.0
            # Alpha espacial: el porcentaje de fusión define cuánto pesa el SPECT,
            # pero la señal baja queda translúcida para no tapar CT de fondo.
            alpha = np.clip(sp * 1.35, 0.0, 1.0)[..., None] * mix
            out = (1.0 - alpha) * rgb_ct + alpha * rgb_sp
            return np.clip(out * 255.0, 0, 255).astype(np.uint8)

        if mode == "edges":
            out = rgb_ct.copy()
            sp_edge = self._edge_mask(sp, 88.0)
            ct_edge = self._edge_mask(ct, 88.0)
            out[ct_edge, 0] = 0.1
            out[ct_edge, 1] = 1.0
            out[ct_edge, 2] = 0.25
            out[sp_edge, 0] = 1.0
            out[sp_edge, 1] = 0.15
            out[sp_edge, 2] = 0.1
            both = sp_edge & ct_edge
            out[both, 0] = 1.0
            out[both, 1] = 1.0
            out[both, 2] = 0.1
            return np.clip(out * 255.0, 0, 255).astype(np.uint8)

        if mode == "split":
            x = int(np.clip(round(sp.shape[1] * (float(split_pct) / 100.0)), 1, sp.shape[1] - 1))
            out = rgb_sp.copy()
            out[:, x:, :] = rgb_ct[:, x:, :]
            out[:, max(0, x - 1):min(sp.shape[1], x + 1), 0] = 1.0
            out[:, max(0, x - 1):min(sp.shape[1], x + 1), 1] = 0.8
            out[:, max(0, x - 1):min(sp.shape[1], x + 1), 2] = 0.2
            return np.clip(out * 255.0, 0, 255).astype(np.uint8)

        if mode == "checker":
            out = rgb_sp.copy()
            tile = 14
            yy, xx = np.indices(sp.shape)
            checker = ((yy // tile) + (xx // tile)) % 2 == 0
            out[checker] = rgb_ct[checker]
            return np.clip(out * 255.0, 0, 255).astype(np.uint8)

        if mode == "contours":
            out = rgb_sp.copy()
            gy, gx = np.gradient(ct)
            edge = np.hypot(gx, gy)
            thr = float(np.percentile(edge, 92.0))
            em = edge >= thr
            out[em, 0] = 0.2
            out[em, 1] = 1.0
            out[em, 2] = 0.3
            return np.clip(out * 255.0, 0, 255).astype(np.uint8)

        return np.clip(rgb_sp * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _edge_mask(img: np.ndarray, percentile: float = 90.0) -> np.ndarray:
        a = np.asarray(img, dtype=np.float64)
        if a.ndim != 2 or a.size == 0:
            return np.zeros_like(a, dtype=bool)
        gy, gx = np.gradient(a)
        edge = np.hypot(gx, gy)
        if not np.any(edge > 0):
            return np.zeros_like(a, dtype=bool)
        return edge >= float(np.percentile(edge, percentile))

    @staticmethod
    def _rgb_to_pixmap(rgb: np.ndarray) -> QPixmap:
        arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        h, w = arr.shape[:2]
        qimg = QImage(arr.data, w, h, arr.strides[0], QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def _draw_triangulation_cross(self, pix: QPixmap, axis: str) -> QPixmap:
        """Dibuja referencias de corte cruzadas y VOIs sobre una vista MPR ya escalada."""
        # === Respetar toggles de visibilidad del usuario ===
        _show_cross = getattr(self, "_chk_show_cross_mpr", None) is None or self._chk_show_cross_mpr.isChecked()
        _show_loc = getattr(self, "_chk_show_loc_mpr", None) is None or self._chk_show_loc_mpr.isChecked()
        _show_vois = getattr(self, "_chk_show_vois_mpr", None) is None or self._chk_show_vois_mpr.isChecked()
        _show_mask = getattr(self, "_chk_show_mask_mpr", None) is None or self._chk_show_mask_mpr.isChecked()
        
        show_triang = _show_cross and bool(getattr(self, "_triangulation_cross_enabled", False))
        show_loc = _show_loc and bool(getattr(self, "_localization_cross_enabled", False)) and getattr(self, "_localization_point_zyx", None) is not None
        show_vois = _show_vois and bool(getattr(self, "_hmr_result", None)) is not None
        show_temp_vois = _show_vois and (bool(getattr(self, "_temp_voi_heart", None)) or bool(getattr(self, "_temp_voi_mediastinum", None)))
        if not (show_triang or show_loc or show_vois or show_temp_vois or _show_mask) or pix.isNull():
            return pix
        vol = self._base_spect_volume if self._base_spect_volume is not None else self._current_volume
        if vol is None:
            return pix
        shape = tuple(int(v) for v in np.asarray(vol).shape[:3])
        if len(shape) != 3 or min(shape) <= 1:
            return pix

        cur_z = int(np.clip(self._slice_idx.get("axial", shape[0] // 2), 0, shape[0] - 1))
        cur_y = int(np.clip(self._slice_idx.get("coronal", shape[1] // 2), 0, shape[1] - 1))
        cur_x = int(np.clip(self._slice_idx.get("sagittal", shape[2] // 2), 0, shape[2] - 1))

        def _coords_from_zyx(zyx: tuple[int, int, int]) -> tuple[int, int, str, str]:
            zz = int(np.clip(zyx[0], 0, shape[0] - 1))
            yy = int(np.clip(zyx[1], 0, shape[1] - 1))
            xx = int(np.clip(zyx[2], 0, shape[2] - 1))
            if axis == "axial":
                return (
                    int(round(xx / max(1, shape[2] - 1) * (w - 1))),
                    int(round(yy / max(1, shape[1] - 1) * (h - 1))),
                    f"X {xx + 1}",
                    f"Y {yy + 1}",
                )
            if axis == "coronal":
                return (
                    int(round(xx / max(1, shape[2] - 1) * (w - 1))),
                    int(round(zz / max(1, shape[0] - 1) * (h - 1))),
                    f"X {xx + 1}",
                    f"Z {zz + 1}",
                )
            return (
                int(round(yy / max(1, shape[1] - 1) * (w - 1))),
                int(round(zz / max(1, shape[0] - 1) * (h - 1))),
                f"Y {yy + 1}",
                f"Z {zz + 1}",
            )
        w = max(1, int(pix.width()))
        h = max(1, int(pix.height()))

        out = QPixmap(pix)
        painter = QPainter(out)
        try:
            # F2.4: overlay de máscara CT RELLENA en los TRES planos MPR.
            # Se dibuja debajo de contornos/cruces y lee directamente de
            # _ct_segmentation (misma fuente que pinta el pincel), así que
            # refleja los cambios en vivo.  Visible siempre que exista
            # segmentación CT (no requiere modo edición activo) para que el
            # usuario pueda ver el resultado de la segmentación automática
            # antes de decidir si necesita editar.
            _ct_seg_edit = getattr(self, "_ct_segmentation", None)
            _show_mask_overlay = _show_mask and _ct_seg_edit is not None and (
                bool(getattr(self, "_mask_edit_active", False))
                or bool(getattr(self, "_ct_anatomical_check", False) and self._ct_anatomical_check.isChecked())
            )
            if _show_mask_overlay:
                try:
                    _m = np.asarray(_ct_seg_edit.mask_3d, dtype=bool)
                    if _m.ndim == 3:
                        _fill = QColor(168, 85, 247, 70)  # violeta semi-transparente
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.setBrush(_fill)

                        if axis == "axial":
                            _mz = int(np.clip(cur_z, 0, _m.shape[0] - 1))
                            _slc = _m[_mz]  # (ny, nx)
                            _s0, _s1 = int(_slc.shape[0]), int(_slc.shape[1])
                            if _s0 > 1 and _s1 > 1 and np.any(_slc):
                                _cw = (w - 1) / max(1, (_s1 - 1))
                                _ch = (h - 1) / max(1, (_s0 - 1))
                                _ys, _xs = np.nonzero(_slc)
                                for _yy, _xx in zip(_ys.tolist(), _xs.tolist()):
                                    _px0 = int(round(_xx * _cw))
                                    _py0 = int(round(_yy * _ch))
                                    painter.drawRect(_px0, _py0, max(1, int(_cw) + 1), max(1, int(_ch) + 1))

                        elif axis == "coronal":
                            _my = int(np.clip(cur_y, 0, _m.shape[1] - 1))
                            _slc = _m[:, _my, :]  # (nz, nx)
                            _s0, _s1 = int(_slc.shape[0]), int(_slc.shape[1])
                            if _s0 > 1 and _s1 > 1 and np.any(_slc):
                                # En coronal: eje horizontal=X, eje vertical=Z
                                _cw = (w - 1) / max(1, (_s1 - 1))   # X → ancho
                                _ch = (h - 1) / max(1, (_s0 - 1))   # Z → alto
                                _zs, _xs = np.nonzero(_slc)
                                for _zz, _xx in zip(_zs.tolist(), _xs.tolist()):
                                    _px0 = int(round(_xx * _cw))
                                    _py0 = int(round(_zz * _ch))
                                    painter.drawRect(_px0, _py0, max(1, int(_cw) + 1), max(1, int(_ch) + 1))

                        else:  # sagittal
                            _mx = int(np.clip(cur_x, 0, _m.shape[2] - 1))
                            _slc = _m[:, :, _mx]  # (nz, ny)
                            _s0, _s1 = int(_slc.shape[0]), int(_slc.shape[1])
                            if _s0 > 1 and _s1 > 1 and np.any(_slc):
                                # En sagital: eje horizontal=Y, eje vertical=Z
                                _cw = (w - 1) / max(1, (_s1 - 1))   # Y → ancho
                                _ch = (h - 1) / max(1, (_s0 - 1))   # Z → alto
                                _zs, _ys = np.nonzero(_slc)
                                for _zz, _yy in zip(_zs.tolist(), _ys.tolist()):
                                    _px0 = int(round(_yy * _cw))
                                    _py0 = int(round(_zz * _ch))
                                    painter.drawRect(_px0, _py0, max(1, int(_cw) + 1), max(1, int(_ch) + 1))

                        painter.setPen(Qt.PenStyle.SolidLine)
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                except Exception:
                    pass
            
            # Dibujar VOIs si existen
            if show_vois and hasattr(self, "_hmr_result") and self._hmr_result is not None:
                hmr_res = self._hmr_result
                spacing = self._spect_spacing_or_default()
                
                # VOI corazón (rojo punteado)
                if hmr_res.voi_heart is not None:
                    voi_h = hmr_res.voi_heart
                    
                    # Manejar tanto V y están visiblesOISphere como VOIAnatomical
                    if isinstance(voi_h, VOIAnatomical):
                        # Dibujar contorno anatómico del miocardio en TODOS los planos
                        pen_h = QPen(QColor(239, 68, 68, 220), 2, Qt.PenStyle.DashLine)
                        painter.setPen(pen_h)

                        if axis == "axial":
                            cur_z_idx = int(round(cur_z))
                            contour = voi_h.get_contour_for_slice(cur_z_idx)
                            if contour is not None and len(contour) > 2:
                                points = [QPointF(
                                    int(round(c[1] / max(1, shape[2] - 1) * (w - 1))),
                                    int(round(c[0] / max(1, shape[1] - 1) * (h - 1)))
                                ) for c in contour]
                                polygon = QPolygonF(points)
                                painter.drawPolygon(polygon)
                                cx_px = int(round(voi_h.cx / max(1, shape[2] - 1) * (w - 1)))
                                cy_px = int(round(voi_h.cy / max(1, shape[1] - 1) * (h - 1)))
                                painter.drawText(cx_px + 10, cy_px - 4, "♥ Anatómico-CT")
                            else:
                                cx_px = int(round(voi_h.cx / max(1, shape[2] - 1) * (w - 1)))
                                cy_px = int(round(voi_h.cy / max(1, shape[1] - 1) * (h - 1)))
                                pen_dot = QPen(QColor(239, 68, 68, 150), 1, Qt.PenStyle.DotLine)
                                painter.setPen(pen_dot)
                                painter.drawEllipse(QPointF(cx_px, cy_px), 4, 4)
                                painter.drawText(cx_px + 6, cy_px - 4, "♥ CT")
                                painter.setPen(pen_h)  # restaurar

                        elif axis == "coronal":
                            cur_y_idx = int(round(cur_y))
                            contour = voi_h.get_contour_for_coronal(cur_y_idx)
                            if contour is not None and len(contour) > 2:
                                # Contorno retorna (z, x); en coronal: X=horizontal, Z=vertical
                                points = [QPointF(
                                    int(round(c[1] / max(1, shape[2] - 1) * (w - 1))),   # x → X screen
                                    int(round(c[0] / max(1, shape[0] - 1) * (h - 1)))    # z → Y screen
                                ) for c in contour]
                                polygon = QPolygonF(points)
                                painter.drawPolygon(polygon)
                                cx_px = int(round(voi_h.cx / max(1, shape[2] - 1) * (w - 1)))
                                cy_px = int(round(voi_h.cz / max(1, shape[0] - 1) * (h - 1)))
                                painter.drawText(cx_px + 10, cy_px - 4, "♥ Anatómico-CT")
                            else:
                                cx_px = int(round(voi_h.cx / max(1, shape[2] - 1) * (w - 1)))
                                cy_px = int(round(voi_h.cz / max(1, shape[0] - 1) * (h - 1)))
                                pen_dot = QPen(QColor(239, 68, 68, 150), 1, Qt.PenStyle.DotLine)
                                painter.setPen(pen_dot)
                                painter.drawEllipse(QPointF(cx_px, cy_px), 4, 4)
                                painter.drawText(cx_px + 6, cy_px - 4, "♥ CT")
                                painter.setPen(pen_h)

                        else:  # sagittal
                            cur_x_idx = int(round(cur_x))
                            contour = voi_h.get_contour_for_sagittal(cur_x_idx)
                            if contour is not None and len(contour) > 2:
                                # Contorno retorna (z, y); en sagital: Y=horizontal, Z=vertical
                                points = [QPointF(
                                    int(round(c[1] / max(1, shape[1] - 1) * (w - 1))),   # y → X screen
                                    int(round(c[0] / max(1, shape[0] - 1) * (h - 1)))    # z → Y screen
                                ) for c in contour]
                                polygon = QPolygonF(points)
                                painter.drawPolygon(polygon)
                                cx_px = int(round(voi_h.cy / max(1, shape[1] - 1) * (w - 1)))
                                cy_px = int(round(voi_h.cz / max(1, shape[0] - 1) * (h - 1)))
                                painter.drawText(cx_px + 10, cy_px - 4, "♥ Anatómico-CT")
                            else:
                                cx_px = int(round(voi_h.cy / max(1, shape[1] - 1) * (w - 1)))
                                cy_px = int(round(voi_h.cz / max(1, shape[0] - 1) * (h - 1)))
                                pen_dot = QPen(QColor(239, 68, 68, 150), 1, Qt.PenStyle.DotLine)
                                painter.setPen(pen_dot)
                                painter.drawEllipse(QPointF(cx_px, cy_px), 4, 4)
                                painter.drawText(cx_px + 6, cy_px - 4, "♥ CT")
                                painter.setPen(pen_h)
                    else:
                        # VOI esférica tradicional (o anatómica en vistas no-axiales)
                        cz_h, cy_h, cx_h = voi_h.cz, voi_h.cy, voi_h.cx
                        r_vol = float(getattr(voi_h, "radius_mm", 30.0)) / max(spacing)
                        
                        if axis == "axial":
                            # Vista axial: X horizontal, Y vertical
                            cx_px = int(round(cx_h / max(1, shape[2] - 1) * (w - 1)))
                            cy_px = int(round(cy_h / max(1, shape[1] - 1) * (h - 1)))
                            r_px = int(round(r_vol / max(1, shape[2] - 1) * w))
                        elif axis == "coronal":
                            # Vista coronal: X horizontal, Z vertical
                            cx_px = int(round(cx_h / max(1, shape[2] - 1) * (w - 1)))
                            cy_px = int(round(cz_h / max(1, shape[0] - 1) * (h - 1)))
                            r_px = int(round(r_vol / max(1, shape[2] - 1) * w))
                        else:  # sagittal
                            # Vista sagittal: Y horizontal, Z vertical
                            cx_px = int(round(cy_h / max(1, shape[1] - 1) * (w - 1)))
                            cy_px = int(round(cz_h / max(1, shape[0] - 1) * (h - 1)))
                            r_px = int(round(r_vol / max(1, shape[1] - 1) * w))
                        
                        pen_h = QPen(QColor(239, 68, 68, 220), 2, Qt.PenStyle.DashLine)
                        painter.setPen(pen_h)
                        painter.drawEllipse(QPointF(cx_px, cy_px), r_px, r_px)
                        _r_label = float(getattr(voi_h, "radius_mm", 0.0))
                        painter.drawText(cx_px + r_px + 4, cy_px - 4, f"Corazón {_r_label:.0f}mm")
                
                # VOI mediastino (azul punteado)
                if hmr_res.voi_mediastinum is not None:
                    voi_m = hmr_res.voi_mediastinum
                    
                    if isinstance(voi_m, VOIAnatomical):
                        # Contorno anatómico del mediastino en TODOS los planos
                        pen_m = QPen(QColor(59, 130, 246, 220), 2, Qt.PenStyle.DashLine)
                        painter.setPen(pen_m)

                        if axis == "axial":
                            cur_z_idx = int(round(cur_z))
                            contour_m = voi_m.get_contour_for_slice(cur_z_idx)
                            if contour_m is not None and len(contour_m) > 2:
                                points_m = [QPointF(
                                    int(round(c[1] / max(1, shape[2] - 1) * (w - 1))),
                                    int(round(c[0] / max(1, shape[1] - 1) * (h - 1)))
                                ) for c in contour_m]
                                painter.drawPolygon(QPolygonF(points_m))
                                cx_px = int(round(voi_m.cx / max(1, shape[2] - 1) * (w - 1)))
                                cy_px = int(round(voi_m.cy / max(1, shape[1] - 1) * (h - 1)))
                                painter.drawText(cx_px + 10, cy_px - 4, "Mediastino-CT")
                            else:
                                cx_px = int(round(voi_m.cx / max(1, shape[2] - 1) * (w - 1)))
                                cy_px = int(round(voi_m.cy / max(1, shape[1] - 1) * (h - 1)))
                                pen_dot = QPen(QColor(59, 130, 246, 150), 1, Qt.PenStyle.DotLine)
                                painter.setPen(pen_dot)
                                painter.drawEllipse(QPointF(cx_px, cy_px), 4, 4)
                                painter.drawText(cx_px + 6, cy_px - 4, "Medi CT")
                                painter.setPen(pen_m)

                        elif axis == "coronal":
                            cur_y_idx = int(round(cur_y))
                            contour_m = voi_m.get_contour_for_coronal(cur_y_idx)
                            if contour_m is not None and len(contour_m) > 2:
                                points_m = [QPointF(
                                    int(round(c[1] / max(1, shape[2] - 1) * (w - 1))),
                                    int(round(c[0] / max(1, shape[0] - 1) * (h - 1)))
                                ) for c in contour_m]
                                painter.drawPolygon(QPolygonF(points_m))
                                cx_px = int(round(voi_m.cx / max(1, shape[2] - 1) * (w - 1)))
                                cy_px = int(round(voi_m.cz / max(1, shape[0] - 1) * (h - 1)))
                                painter.drawText(cx_px + 10, cy_px - 4, "Mediastino-CT")
                            else:
                                cx_px = int(round(voi_m.cx / max(1, shape[2] - 1) * (w - 1)))
                                cy_px = int(round(voi_m.cz / max(1, shape[0] - 1) * (h - 1)))
                                pen_dot = QPen(QColor(59, 130, 246, 150), 1, Qt.PenStyle.DotLine)
                                painter.setPen(pen_dot)
                                painter.drawEllipse(QPointF(cx_px, cy_px), 4, 4)
                                painter.drawText(cx_px + 6, cy_px - 4, "Medi CT")
                                painter.setPen(pen_m)

                        else:  # sagittal
                            cur_x_idx = int(round(cur_x))
                            contour_m = voi_m.get_contour_for_sagittal(cur_x_idx)
                            if contour_m is not None and len(contour_m) > 2:
                                points_m = [QPointF(
                                    int(round(c[1] / max(1, shape[1] - 1) * (w - 1))),
                                    int(round(c[0] / max(1, shape[0] - 1) * (h - 1)))
                                ) for c in contour_m]
                                painter.drawPolygon(QPolygonF(points_m))
                                cx_px = int(round(voi_m.cy / max(1, shape[1] - 1) * (w - 1)))
                                cy_px = int(round(voi_m.cz / max(1, shape[0] - 1) * (h - 1)))
                                painter.drawText(cx_px + 10, cy_px - 4, "Mediastino-CT")
                            else:
                                cx_px = int(round(voi_m.cy / max(1, shape[1] - 1) * (w - 1)))
                                cy_px = int(round(voi_m.cz / max(1, shape[0] - 1) * (h - 1)))
                                pen_dot = QPen(QColor(59, 130, 246, 150), 1, Qt.PenStyle.DotLine)
                                painter.setPen(pen_dot)
                                painter.drawEllipse(QPointF(cx_px, cy_px), 4, 4)
                                painter.drawText(cx_px + 6, cy_px - 4, "Medi CT")
                                painter.setPen(pen_m)
                    else:
                        cz_m, cy_m, cx_m = voi_m.cz, voi_m.cy, voi_m.cx
                        r_vol = float(getattr(voi_m, "radius_mm", 30.0)) / max(spacing)
                        
                        if axis == "axial":
                            cx_px = int(round(cx_m / max(1, shape[2] - 1) * (w - 1)))
                            cy_px = int(round(cy_m / max(1, shape[1] - 1) * (h - 1)))
                            r_px = int(round(r_vol / max(1, shape[2] - 1) * w))
                        elif axis == "coronal":
                            cx_px = int(round(cx_m / max(1, shape[2] - 1) * (w - 1)))
                            cy_px = int(round(cz_m / max(1, shape[0] - 1) * (h - 1)))
                            r_px = int(round(r_vol / max(1, shape[2] - 1) * w))
                        else:  # sagittal
                            cx_px = int(round(cy_m / max(1, shape[1] - 1) * (w - 1)))
                            cy_px = int(round(cz_m / max(1, shape[0] - 1) * (h - 1)))
                            r_px = int(round(r_vol / max(1, shape[1] - 1) * w))
                        
                        pen_m = QPen(QColor(59, 130, 246, 220), 2, Qt.PenStyle.DashLine)
                        painter.setPen(pen_m)
                        painter.drawEllipse(QPointF(cx_px, cy_px), r_px, r_px)
                        _rm_label = float(getattr(voi_m, "radius_mm", 0.0))
                        painter.drawText(cx_px + r_px + 4, cy_px - 4, f"Mediastino {_rm_label:.0f}mm")
            
            # Dibujar VOIs temporales (en vivo durante posicionamiento)
            if show_temp_vois and not show_vois:  # Solo si no hay resultado final
                spacing = self._spect_spacing_or_default()
                
                # VOI corazón temporal (rojo punteado)
                temp_heart = getattr(self, "_temp_voi_heart", None)
                if temp_heart is not None:
                    cz_h, cy_h, cx_h = temp_heart.cz, temp_heart.cy, temp_heart.cx
                    r_vol = temp_heart.radius_mm / max(spacing)
                    
                    if axis == "axial":
                        cx_px = int(round(cx_h / max(1, shape[2] - 1) * (w - 1)))
                        cy_px = int(round(cy_h / max(1, shape[1] - 1) * (h - 1)))
                        r_px = int(round(r_vol / max(1, shape[2] - 1) * w))
                    elif axis == "coronal":
                        cx_px = int(round(cx_h / max(1, shape[2] - 1) * (w - 1)))
                        cy_px = int(round(cz_h / max(1, shape[0] - 1) * (h - 1)))
                        r_px = int(round(r_vol / max(1, shape[2] - 1) * w))
                    else:  # sagittal
                        cx_px = int(round(cy_h / max(1, shape[1] - 1) * (w - 1)))
                        cy_px = int(round(cz_h / max(1, shape[0] - 1) * (h - 1)))
                        r_px = int(round(r_vol / max(1, shape[1] - 1) * w))
                    
                    pen_h = QPen(QColor(239, 68, 68, 180), 2, Qt.PenStyle.DotLine)
                    painter.setPen(pen_h)
                    painter.drawEllipse(QPointF(cx_px, cy_px), r_px, r_px)
                    painter.drawText(cx_px + r_px + 4, cy_px - 4, f"♥ {temp_heart.radius_mm:.0f}mm")
                
                # VOI mediastino temporal (azul punteado)
                temp_med = getattr(self, "_temp_voi_mediastinum", None)
                if temp_med is not None:
                    cz_m, cy_m, cx_m = temp_med.cz, temp_med.cy, temp_med.cx
                    r_vol = temp_med.radius_mm / max(spacing)
                    
                    if axis == "axial":
                        cx_px = int(round(cx_m / max(1, shape[2] - 1) * (w - 1)))
                        cy_px = int(round(cy_m / max(1, shape[1] - 1) * (h - 1)))
                        r_px = int(round(r_vol / max(1, shape[2] - 1) * w))
                    elif axis == "coronal":
                        cx_px = int(round(cx_m / max(1, shape[2] - 1) * (w - 1)))
                        cy_px = int(round(cz_m / max(1, shape[0] - 1) * (h - 1)))
                        r_px = int(round(r_vol / max(1, shape[2] - 1) * w))
                    else:  # sagittal
                        cx_px = int(round(cy_m / max(1, shape[1] - 1) * (w - 1)))
                        cy_px = int(round(cz_m / max(1, shape[0] - 1) * (h - 1)))
                        r_px = int(round(r_vol / max(1, shape[1] - 1) * w))
                    
                    pen_m = QPen(QColor(59, 130, 246, 180), 2, Qt.PenStyle.DotLine)
                    painter.setPen(pen_m)
                    painter.drawEllipse(QPointF(cx_px, cy_px), r_px, r_px)
                    painter.drawText(cx_px + r_px + 4, cy_px - 4, f"M {temp_med.radius_mm:.0f}mm")

                # === Esferas S/VD (corazón/vértebra/aorta) ===
                svd_colors = {
                    "S": QColor(239, 68, 68, 200),    # rojo
                    "V": QColor(34, 197, 94, 200),    # verde
                    "D": QColor(168, 85, 247, 200),   # violeta
                }
                svd_labels = {"S": "S", "V": "V", "D": "D"}
                svd_radii = {
                    "S": float(self._heart_radius_spin.value()) if hasattr(self, "_heart_radius_spin") else 30.0,
                    "V": float(self._svd_vertebra_spin.value()) if hasattr(self, "_svd_vertebra_spin") else 15.0,
                    "D": float(self._svd_aorta_spin.value()) if hasattr(self, "_svd_aorta_spin") else 12.0,
                }
                for roi_key in ("S", "V", "D"):
                    pt = self._svd_points.get(roi_key)
                    if pt is None:
                        continue
                    cz_p, cy_p, cx_p = float(pt[0]), float(pt[1]), float(pt[2])
                    r_vol = svd_radii[roi_key] / max(spacing)
                    if axis == "axial":
                        cx_px = int(round(cx_p / max(1, shape[2] - 1) * (w - 1)))
                        cy_px = int(round(cy_p / max(1, shape[1] - 1) * (h - 1)))
                        r_px = int(round(r_vol / max(1, shape[2] - 1) * w))
                    elif axis == "coronal":
                        cx_px = int(round(cx_p / max(1, shape[2] - 1) * (w - 1)))
                        cy_px = int(round(cz_p / max(1, shape[0] - 1) * (h - 1)))
                        r_px = int(round(r_vol / max(1, shape[2] - 1) * w))
                    else:
                        cx_px = int(round(cy_p / max(1, shape[1] - 1) * (w - 1)))
                        cy_px = int(round(cz_p / max(1, shape[0] - 1) * (h - 1)))
                        r_px = int(round(r_vol / max(1, shape[1] - 1) * w))
                    pen_svd = QPen(svd_colors[roi_key], 2, Qt.PenStyle.SolidLine)
                    painter.setPen(pen_svd)
                    painter.drawEllipse(QPointF(cx_px, cy_px), r_px, r_px)
                    painter.drawText(cx_px + r_px + 4, cy_px - 4, svd_labels[roi_key])
            
            # Dibujar cruz de triangulación
            if show_triang:
                vx, hy, v_label, h_label = _coords_from_zyx((cur_z, cur_y, cur_x))
                shadow = QPen(QColor(0, 0, 0, 220), 3)
                pen_v = QPen(QColor(56, 189, 248, 235), 1)
                pen_h = QPen(QColor(251, 191, 36, 235), 1)
                for off in (-1, 1):
                    painter.setPen(shadow)
                    painter.drawLine(max(0, min(w - 1, vx + off)), 0, max(0, min(w - 1, vx + off)), h - 1)
                    painter.drawLine(0, max(0, min(h - 1, hy + off)), w - 1, max(0, min(h - 1, hy + off)))
                painter.setPen(pen_v)
                painter.drawLine(vx, 0, vx, h - 1)
                painter.setPen(pen_h)
                painter.drawLine(0, hy, w - 1, hy)
                painter.setPen(QColor(255, 255, 255, 230))
                painter.drawText(6, 16, h_label)
                painter.drawText(max(6, min(w - 48, vx + 5)), max(30, min(h - 8, hy - 6)), v_label)
            if show_loc:
                loc_z, loc_y, loc_x = (int(v) for v in getattr(self, "_localization_point_zyx"))
                vx, hy, _v_label, _h_label = _coords_from_zyx((loc_z, loc_y, loc_x))
                mark_pen = QPen(QColor(255, 255, 255, 245), 2)
                painter.setPen(mark_pen)
                painter.drawEllipse(max(0, vx - 5), max(0, hy - 5), 10, 10)
                painter.drawLine(max(0, vx - 10), hy, max(0, vx - 3), hy)
                painter.drawLine(min(w - 1, vx + 3), hy, min(w - 1, vx + 10), hy)
                painter.drawLine(vx, max(0, hy - 10), vx, max(0, hy - 3))
                painter.drawLine(vx, min(h - 1, hy + 3), vx, min(h - 1, hy + 10))
                painter.drawText(6, max(32, h - 10), f"LOC Z/Y/X {loc_z + 1}/{loc_y + 1}/{loc_x + 1}")
            anchor = getattr(self, "_localization_anchor_zyx", None)
            if anchor is not None:
                a_z, a_y, a_x = (int(v) for v in anchor)
                avx, ahy, _av, _ah = _coords_from_zyx((a_z, a_y, a_x))
                anchor_pen = QPen(QColor(74, 222, 128, 245), 2)
                painter.setPen(anchor_pen)
                painter.drawEllipse(max(0, avx - 6), max(0, ahy - 6), 12, 12)
                painter.drawLine(max(0, avx - 12), ahy, max(0, avx - 4), ahy)
                painter.drawLine(min(w - 1, avx + 4), ahy, min(w - 1, avx + 12), ahy)
                painter.drawLine(avx, max(0, ahy - 12), avx, max(0, ahy - 4))
                painter.drawLine(avx, min(h - 1, ahy + 4), avx, min(h - 1, ahy + 12))
                painter.drawText(6, max(48, h - 26), f"ANCLA A Z/Y/X {a_z + 1}/{a_y + 1}/{a_x + 1}")
                if show_loc:
                    dist = self._localization_distance_mm()
                    if dist is not None:
                        painter.setPen(QPen(QColor(74, 222, 128, 220), 1))
                        painter.drawLine(avx, ahy, vx, hy)
                        mid_x = (avx + vx) // 2
                        mid_y = (ahy + hy) // 2
                        painter.setPen(QColor(255, 255, 255, 240))
                        painter.drawText(max(6, min(w - 70, mid_x + 4)), max(14, min(h - 8, mid_y - 4)), f"{dist:.1f} mm")
        finally:
            painter.end()
        return out

    def _set_axis_pixmap_with_cross(self, lbl: QLabel, pix: QPixmap, axis: str) -> None:
        lbl.setPixmap(self._draw_triangulation_cross(pix, axis))

    def _on_triangulation_cross_toggled(self, checked: bool) -> None:
        self._triangulation_cross_enabled = bool(checked)
        if hasattr(self, "_settings"):
            self._settings.setValue("global/triangulation_cross", bool(checked))
        self._status.setText("Cruz de triangulación activada." if checked else "Cruz de triangulación desactivada.")
        self._render_selected_view()

    def _on_localization_cross_toggled(self, checked: bool) -> None:
        self._localization_cross_enabled = bool(checked)
        # Apagar solo OCULTA la cruz; A/B y las VOIs colocadas se conservan
        # para no invalidar el flujo de máscara + HMR (antes se borraban).
        if hasattr(self, "_settings"):
            self._settings.setValue("global/localization_cross", bool(checked))
        self._status.setText(
            "Localización activada: Ctrl+clic/arrastre = CT, Shift+clic/arrastre = SPECT."
            if checked else "Localización oculta (los puntos A/B se conservan)."
        )
        self._render_selected_view()

    def _spect_spacing_or_default(self) -> tuple[float, float, float]:
        sp = getattr(self, "_spect_spacing_zyx", None)
        if sp and len(sp) == 3 and all(float(v) > 0 for v in sp):
            return (float(sp[0]), float(sp[1]), float(sp[2]))
        return (6.8, 6.8, 6.8)

    def _localization_distance_mm(self) -> float | None:
        """Distancia euclídea 3D en mm entre el ancla y el punto actual."""
        a = getattr(self, "_localization_anchor_zyx", None)
        b = getattr(self, "_localization_point_zyx", None)
        if a is None or b is None:
            return None
        sz, sy, sx = self._spect_spacing_or_default()
        dz = (float(b[0]) - float(a[0])) * sz
        dy = (float(b[1]) - float(a[1])) * sy
        dx = (float(b[2]) - float(a[2])) * sx
        return float(np.sqrt(dz * dz + dy * dy + dx * dx))

    def _update_localization_distance(self) -> None:
        dist = self._localization_distance_mm()
        if dist is None:
            return
        a = self._localization_anchor_zyx
        b = self._localization_point_zyx
        self._status.setText(
            f"Distancia LOC: {dist:.1f} mm · "
            f"A(Z/Y/X)={a[0] + 1}/{a[1] + 1}/{a[2] + 1} → "
            f"B(Z/Y/X)={b[0] + 1}/{b[1] + 1}/{b[2] + 1}"
        )
        self._metrics.append(
            f"\n--- Medida localización ---\n"
            f"- A (ancla) Z/Y/X: {a[0] + 1}/{a[1] + 1}/{a[2] + 1}\n"
            f"- B (punto) Z/Y/X: {b[0] + 1}/{b[1] + 1}/{b[2] + 1}\n"
            f"- distancia: {dist:.1f} mm (spacing "
            f"{self._spect_spacing_or_default()[0]:.2f}/"
            f"{self._spect_spacing_or_default()[1]:.2f}/"
            f"{self._spect_spacing_or_default()[2]:.2f} mm)"
        )

    def _on_set_localization_anchor(self) -> None:
        """Fija el punto actual como ancla para medir distancia y crea VOI corazón temporal."""
        pt = getattr(self, "_localization_point_zyx", None)
        if pt is None:
            self._status.setText("Primero depositá una cruz de localización (Ctrl/Shift+clic).")
            return
        self._localization_anchor_zyx = (int(pt[0]), int(pt[1]), int(pt[2]))
        
        # Crear VOI corazón temporal para visualización en vivo
        # Usar valor del spin si existe, sino default 30mm
        heart_radius = 30.0
        if hasattr(self, "_heart_radius_spin"):
            heart_radius = float(self._heart_radius_spin.value())
        self._temp_voi_heart = VOISphere(
            cz=int(pt[0]), cy=int(pt[1]), cx=int(pt[2]),
            radius_mm=heart_radius
        )
        
        self._status.setText(
            f"Ancla fijada en Z/Y/X = {pt[0] + 1}/{pt[1] + 1}/{pt[2] + 1}. "
            "Depositá un segundo punto para medir la distancia."
        )
        self._render_selected_view()  # Redibujar para mostrar VOI

    def _on_clear_localization_anchor(self) -> None:
        self._localization_anchor_zyx = None
        self._temp_voi_heart = None
        self._temp_voi_mediastinum = None
        self._status.setText("Ancla de medición limpiada.")
        self._render_selected_view()

    def get_localization_points(self) -> list[dict]:
        """Exporta ancla y punto de localización para informes."""
        out: list[dict] = []
        a = getattr(self, "_localization_anchor_zyx", None)
        b = getattr(self, "_localization_point_zyx", None)
        if a is not None:
            out.append({"label": "A (ancla)", "zyx": [int(v) + 1 for v in a]})
        if b is not None:
            out.append({"label": "B (punto)", "zyx": [int(v) + 1 for v in b]})
        dist = self._localization_distance_mm()
        if dist is not None:
            out.append({"label": "Distancia A→B", "value_mm": round(dist, 1)})
        return out

    # ============================================================
    # Sistema S/VD (ratio corazón / vértebra / aorta)
    # ============================================================

    def _on_svd_roi_changed(self) -> None:
        """Cambia la ROI activa para depositar puntos S/V/D."""
        roi = str(self._svd_roi_combo.currentData())
        self._svd_active_roi = roi
        labels = {"S": "Corazón", "V": "Vértebra", "D": "Aorta"}
        self._status.setText(f"ROI activa: {labels.get(roi, roi)}. Depositá un punto y click 'Depositar punto'.")

    def _on_svd_deposit(self) -> None:
        """Deposita la cruz actual en la ROI activa S/V/D."""
        pt = getattr(self, "_localization_point_zyx", None)
        if pt is None:
            self._status.setText("Primero depositá una cruz de localización (Ctrl/Shift+clic).")
            return
        roi = self._svd_active_roi
        self._svd_points[roi] = (int(pt[0]), int(pt[1]), int(pt[2]))
        labels = {"S": "Corazón", "V": "Vértebra", "D": "Aorta"}
        self._status.setText(
            f"ROI {labels.get(roi, roi)} fijada en Z/Y/X = {pt[0]+1}/{pt[1]+1}/{pt[2]+1}. "
            f"Puntos S/V/D: "
            f"{'✓' if self._svd_points['S'] else '✗'}/"
            f"{'✓' if self._svd_points['V'] else '✗'}/"
            f"{'✓' if self._svd_points['D'] else '✗'}"
        )
        self._render_selected_view()

    def _on_svd_clear(self) -> None:
        """Borra los 3 puntos S/V/D."""
        self._svd_points = {"S": None, "V": None, "D": None}
        self._svd_result = None
        self._lbl_svd_result.setText("S/VD = N/D")
        self._lbl_svd_result.setStyleSheet(
            "font-size:14px; font-weight:700; color:#ffffff; background:#1e1b4b; padding:6px 12px;"
        )
        self._status.setText("Puntos S/V/D limpiados.")
        self._render_selected_view()

    def _calculate_svd_ratio(self) -> None:
        """Calcula el ratio S/VD usando los 3 puntos depositados."""
        try:
            if self._current_volume is None:
                QMessageBox.warning(self, "SINCRO", "Primero cargue un volumen SPECT.")
                return

            s_pt = self._svd_points.get("S")
            v_pt = self._svd_points.get("V")
            d_pt = self._svd_points.get("D")

            missing = []
            if s_pt is None:
                missing.append("S (corazón)")
            if v_pt is None:
                missing.append("V (vértebra)")
            if d_pt is None:
                missing.append("D (aorta)")
            if missing:
                QMessageBox.warning(
                    self, "SINCRO",
                    f"Faltan puntos: {', '.join(missing)}.\n\n"
                    "1. Active 'Localización'\n"
                    "2. Seleccione ROI activa en el combo S/V/D\n"
                    "3. Ctrl/Shift+clic en la estructura\n"
                    "4. Click 'Depositar punto'\n"
                    "5. Repita para S, V y D"
                )
                return

            spacing = self._spect_spacing_or_default()
            heart_radius = float(self._heart_radius_spin.value())
            vertebra_radius = float(self._svd_vertebra_spin.value())
            aorta_radius = float(self._svd_aorta_spin.value())

            voi_heart = VOISphere(
                cz=s_pt[0], cy=s_pt[1], cx=s_pt[2], radius_mm=heart_radius
            )
            voi_vertebra = VOISphere(
                cz=v_pt[0], cy=v_pt[1], cx=v_pt[2], radius_mm=vertebra_radius
            )
            voi_aorta = VOISphere(
                cz=d_pt[0], cy=d_pt[1], cx=d_pt[2], radius_mm=aorta_radius
            )

            vol_raw = getattr(self, "_unfiltered_volume", None)

            analysis_volume = self._spect_display_volume(self._current_volume)
            analysis_volume_raw = self._spect_display_volume(vol_raw) if vol_raw is not None else None
            result = compute_spect_ratio(
                volume=analysis_volume,
                spacing_zyx=spacing,
                voi_heart=voi_heart,
                voi_vertebra=voi_vertebra,
                voi_aorta=voi_aorta,
                volume_raw=analysis_volume_raw,
            )
            self._svd_result = result

            # Mostrar resultado
            self._lbl_svd_result.setText(result.s_vd_text)
            cls = result.classification
            color = {"POSITIVO": "#ef4444", "EQUIVOCO": "#f59e0b", "NEGATIVO": "#22c55e"}.get(cls, "#ffffff")
            self._lbl_svd_result.setStyleSheet(
                f"font-size:14px; font-weight:700; color:{color}; background:#1e1b4b; padding:6px 12px;"
            )

            self._status.setText(
                f"S/VD={result.s_vd:.2f} · S/V={result.s_v:.2f} · S/D={result.s_d:.2f} · V/D={result.v_d:.2f} ({cls})"
            )
            self._metrics.append(
                f"\n--- Ratio S/VD (SPECT 3D) ---\n"
                f"- S (corazón)  Z/Y/X: {s_pt[0]+1}/{s_pt[1]+1}/{s_pt[2]+1} · r={heart_radius:.0f}mm\n"
                f"- V (vértebra) Z/Y/X: {v_pt[0]+1}/{v_pt[1]+1}/{v_pt[2]+1} · r={vertebra_radius:.0f}mm\n"
                f"- D (aorta)    Z/Y/X: {d_pt[0]+1}/{d_pt[1]+1}/{d_pt[2]+1} · r={aorta_radius:.0f}mm\n"
                f"- S/VD = {result.s_vd:.3f} ({cls})\n"
                f"- S/V  = {result.s_v:.3f}\n"
                f"- S/D  = {result.s_d:.3f}\n"
                f"- V/D  = {result.v_d:.3f}\n"
                f"- Medias: S={result.s_mean:.3f} V={result.v_mean:.3f} D={result.d_mean:.3f} cts/voxel\n"
                f"- Cuentas: S={result.s_counts:.0f} V={result.v_counts:.0f} D={result.d_counts:.0f}\n"
                f"- Voxels:  S={result.s_voxels} V={result.v_voxels} D={result.d_voxels}\n"
                f"- Spacing: {spacing[0]:.2f}/{spacing[1]:.2f}/{spacing[2]:.2f} mm"
            )
            self._render_selected_view()

        except Exception as exc:
            self._svd_result = None
            self._lbl_svd_result.setText("S/VD = N/D · revisar VOI")
            self._lbl_svd_result.setStyleSheet(
                "font-size:14px; font-weight:700; color:#f59e0b; "
                "background:#1e1b4b; padding:6px 12px;"
            )
            self._status.setText(f"Error calculando S/VD: {exc}")
            QMessageBox.critical(self, "SINCRO", f"Error calculando S/VD:\n{exc}")

    def _show_svd_info_dialog(self) -> None:
        """Abre un diálogo con la guía completa de interpretación del ratio S/VD."""
        dlg = QDialog(self)
        dlg.setWindowTitle("ℹ️ Interpretación del Ratio S/VD")
        dlg.setMinimumSize(620, 580)
        dlg.resize(640, 600)

        layout = QVBoxLayout(dlg)

        # QTextBrowser soporta HTML con scroll nativo
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setStyleSheet(
            "QTextBrowser { font-size:13px; background:#0f172a; color:#e2e8f0; "
            "border:1px solid #334155; border-radius:6px; padding:8px; }"
        )
        browser.setHtml("""
<h2 style="color:#38bdf8; margin-top:0;">📐 Ratio S/VD — Guía de Interpretación</h2>

<h3 style="color:#f59e0b;">¿Qué es el ratio S/VD?</h3>
<p>El <b>ratio S/VD</b> es un índice semicuantitativo derivado de imágenes <b>SPECT 3D</b>
que compara la captación miocárdica del radiofármaco (PYP/Tc-99m pirofosfato)
contra dos referencias anatómicas simultáneas:</p>
<ul>
<li><b style="color:#ef4444;">S</b> = <b>Corazón</b> (miocardio ventricular izquierdo)</li>
<li><b style="color:#22c55e;">V</b> = <b>Vértebra</b> (cuerpo vertebral torácico, usualmente T8-T10)</li>
<li><b style="color:#a855f7;">D</b> = <b>Aorta</b> (pared aórtica descendente, referencia de sangre)</li>
</ul>

<h3 style="color:#f59e0b;">Fórmula</h3>
<div style="background:#1e293b; padding:12px; border-radius:6px; text-align:center;
font-size:18px; font-weight:bold; letter-spacing:1px; margin:8px 0;">
S/√(V × D)
</div>
<p>El numerador <b>S</b> es la cuenta media normalizada por voxel en el VOI cardíaco.
El denominador usa la <b>media geométrica</b> de las cuentas de vértebra (V) y aorta (D):
√(V × D). Esto estabiliza la referencia frente a variaciones individuales
(como osteoporosis que alteraría V sola, o anemia que alteraría D sola).</p>

<h3 style="color:#f59e0b;">Valores de corte (puntos de corte orientativos)</h3>
<table style="width:100%; border-collapse:collapse; margin:8px 0;">
<tr style="background:#1e293b;">
<th style="padding:8px; border:1px solid #334155; text-align:left;">Ratio S/VD</th>
<th style="padding:8px; border:1px solid #334155; text-align:left;">Clasificación</th>
<th style="padding:8px; border:1px solid #334155; text-align:left;">Color</th>
<th style="padding:8px; border:1px solid #334155; text-align:left;">Significado clínico</th>
</tr>
<tr>
<td style="padding:8px; border:1px solid #334155;"><b>≥ 2.20</b></td>
<td style="padding:8px; border:1px solid #334155; color:#ef4444; font-weight:bold;">POSITIVO</td>
<td style="padding:8px; border:1px solid #334155;">🔴 Rojo</td>
<td style="padding:8px; border:1px solid #334155;">Captación miocárdica elevada, consistente con amiloidosis por ATTR (confirmar con contexto clínico)</td>
</tr>
<tr>
<td style="padding:8px; border:1px solid #334155;"><b>1.80 – 2.19</b></td>
<td style="padding:8px; border:1px solid #334155; color:#f59e0b; font-weight:bold;">EQUIVOCO</td>
<td style="padding:8px; border:1px solid #334155;">🟡 Amarillo</td>
<td style="padding:8px; border:1px solid #334155;">Zona gris. Requiere correlación con HMR planar, ecocardiograma, biomarcadores y cuadro clínico</td>
</tr>
<tr>
<td style="padding:8px; border:1px solid #334155;"><b>&lt; 1.80</b></td>
<td style="padding:8px; border:1px solid #334155; color:#22c55e; font-weight:bold;">NEGATIVO</td>
<td style="padding:8px; border:1px solid #334155;">🟢 Verde</td>
<td style="padding:8px; border:1px solid #334155;">Captación miocárdica dentro de rangos normales. Hace poco probable amiloidosis ATTR significativa</td>
</tr>
</table>

<h3 style="color:#f59e0b;">Sub-ratios complementarios</h3>
<table style="width:100%; border-collapse:collapse; margin:8px 0;">
<tr style="background:#1e293b;">
<th style="padding:6px; border:1px solid #334155;">Ratio</th>
<th style="padding:6px; border:1px solid #334155;">Fórmula</th>
<th style="padding:6px; border:1px solid #334155;">Uso</th>
</tr>
<tr><td style="padding:6px; border:1px solid #334155;"><b>S/V</b></td>
<td style="padding:6px; border:1px solid #334155;">Corazón / Vértebra</td>
<td style="padding:6px; border:1px solid #334155;">Comparación directa contra hueso (similar al HMR planar clásico)</td></tr>
<tr><td style="padding:6px; border:1px solid #334155;"><b>S/D</b></td>
<td style="padding:6px; border:1px solid #334155;">Corazón / Aorta</td>
<td style="padding:6px; border:1px solid #334155;">Comparación contra pool sanguíneo (sensible a perfusión)</td></tr>
<tr><td style="padding:6px; border:1px solid #334155;"><b>V/D</b></td>
<td style="padding:6px; border:1px solid #334155;">Vértebra / Aorta</td>
<td style="padding:6px; border:1px solid #334155;">Control de calidad: relación hueso/sangre esperada ~1.0–2.5</td></tr>
</table>

<h3 style="color:#f59e0b;">¿Por qué √(V×D) en vez de una sola referencia?</h3>
<p>Usar <b>dos referencias</b> (hueso + sangre) en media geométrica tiene ventajas:</p>
<ol>
<li><b>Robustez:</b> si un paciente tiene osteoporosis (V bajo), D compensa. Si tiene anemia o bajo hematocrito (D atípico), V compensa.</li>
<li><b>Consistencia:</b> reduce la varianza inter-paciente comparado con usar solo V (como hace el HMR planar).</li>
<li><b>Detección de artefactos:</b> si V/D está fuera de rango 0.5–4.0, sugiere problema de posicionamiento o ROI mal colocada.</li>
</ol>

<h3 style="color:#f59e0b;">Diferencias con HMR planar (Perugini)</h3>
<table style="width:100%; border-collapse:collapse; margin:8px 0;">
<tr style="background:#1e293b;">
<th style="padding:6px; border:1px solid #334155;">Aspecto</th>
<th style="padding:6px; border:1px solid #334155;">HMR Planar</th>
<th style="padding:6px; border:1px solid #334155;">S/VD SPECT 3D</th>
</tr>
<tr><td style="padding:6px; border:1px solid #334155;">Imagen</td>
<td style="padding:6px; border:1px solid #334155;">Proyección planar torácica anterior</td>
<td style="padding:6px; border:1px solid #334155;">Volumen SPECT reconstruido (64³ típico)</td></tr>
<tr><td style="padding:6px; border:1px solid #334155;">Referencia</td>
<td style="padding:6px; border:1px solid #334155;">Costillas contralaterales</td>
<td style="padding:6px; border:1px solid #334155;">Vértebra + Aorta (√(V×D))</td></tr>
<tr><td style="padding:6px; border:1px solid #334155;">Superposición</td>
<td style="padding:6px; border:1px solid #334155;">Posible (corazón + costillas en misma línea)</td>
<td style="padding:6px; border:1px solid #334155;">Mínima (VOIs 3D separados espacialmente)</td></tr>
<tr><td style="padding:6px; border:1px solid #334155;">Corte positivo</td>
<td style="padding:6px; border:1px solid #334155;">≥ 1.5 (Perugini)</td>
<td style="padding:6px; border:1px solid #334155;">≥ 2.2 (orientativo, pendiente de validación)</td></tr>
</table>

<h3 style="color:#f59e0b;">Flujo diagnóstico sugerido</h3>
<div style="background:#1e293b; padding:12px; border-radius:6px; font-family:monospace; font-size:12px; line-height:1.8;">
<pre style="margin:0; color:#e2e8f0;">
┌─────────────────────┐
│  S/VD ≥ 2.2 POS     │ ──→ Consistente con ATTR+
│                      │     Corroborar con:
│                      │     • Ecocardiograma (strain)
│                      │     • Biomarcadores (NT-proBNP, troponina)
│                      │     • Clínica (IC, neuropatía)
├─────────────────────┤
│  S/VD 1.8–2.2 EQ    │ ──→ Zona gris
│                      │     • Comparar con HMR planar
│                      │     • Revisar posicionamiento VOI
│                      │     • Evaluar sub-ratios S/V y S/D
├─────────────────────┤
│  S/VD < 1.8 NEG     │ ──→ Poco probable ATTR
│                      │     Buscar otras causas de IC
└─────────────────────┘
</pre>
</div>

<h3 style="color:#f59e0b;">⚠️ Limitaciones y advertencias</h3>
<ul>
<li><b>Los puntos de corte son ORIENTATIVOS.</b> No sustituyen el juicio clínico del médico nuclear.</li>
<li><b>Pendiente de validación clínica prospectiva.</b> Los valores de corte pueden ajustarse con más datos.</li>
<li><b>Sensible al posicionamiento de VOIs.</b> Verifique que cada esfera esté centrada en la estructura correcta usando las vistas MPR.</li>
<li><b>Depende de la calidad de reconstrucción SPECT.</b> Artefactos de atenuación, scatter o movimiento afectan los valores.</li>
<li><b>No válido para otros radiofármacos.</b> Este ratio fue diseñado para <b>Tc-99m PYP</b> (pirofosfato).</li>
</ul>

<hr style="border-color:#334155; margin:16px 0;">
<p style="color:#94a3ab; font-size:11px; text-align:center;">
Módulo SINCRO — Ratio S/VD · Versión experimental fase 2<br>
Basado en metodología publicada (Emory University / Huttlin et al.)<br>
Los valores de corte deben validarse localmente antes de uso diagnóstico rutinario.
</p>
""")
        layout.addWidget(browser)

        # Botón cerrar
        btn_box = QHBoxLayout()
        btn_box.addStretch(1)
        close_btn = QPushButton("Cerrar")
        close_btn.setStyleSheet(
            "background:#3b82f6; color:white; font-weight:bold; "
            "padding:8px 24px; border-radius:4px; border:none;"
        )
        close_btn.clicked.connect(dlg.close)
        btn_box.addWidget(close_btn)
        layout.addLayout(btn_box)

        dlg.exec()

    def get_svd_result(self) -> SvdRatioResult | None:
        """Retorna el resultado S/VD calculado (para informes)."""
        return self._svd_result

    def get_hmr_spect_result(self) -> HmrSpectResult | None:
        """Retorna el resultado HMR-SPECT calculado (para informes)."""
        return self._hmr_result

    def _on_ct_anatomical_mode_toggled(self, checked: bool) -> None:
        """Aísla la vía CT experimental sin alterar el flujo manual estable.
        
        NOTA: A partir de v2.5, los datos CT se PRESERVAN al desactivar.
        Solo se ocultan los controles de edición manual. Use 'Reiniciar CT'
        para borrar realmente los datos.
        """
        enabled = bool(checked)
        self._btn_paint_erase.setEnabled(enabled)
        self._brush_radius_spin.setEnabled(enabled)

        if not enabled:
            # === PRESERVAR datos (no borrar!) ===
            if self._mask_edit_active:
                self._btn_toggle_mask_edit.setChecked(False)
            
            # Deshabilitar controles de edición pero NO borrar datos
            self._btn_toggle_mask_edit.setEnabled(False)
            self._btn_undo_mask.setEnabled(False)
            self._btn_reset_mask.setEnabled(False)
            self._btn_apply_mask_edit.setEnabled(False)
            if hasattr(self, '_btn_export_mask_nifti'):
                self._btn_export_mask_nifti.setEnabled(False)
            # Botones de persistencia: guardar/reiniciar desactivados, cargar siempre activo
            if hasattr(self, '_btn_save_ct_state'):
                self._btn_save_ct_state.setEnabled(False)
            if hasattr(self, '_btn_restart_ct'):
                self._btn_restart_ct.setEnabled(False)
            
            # Mostrar estado de VOIs manuales (si existen)
            ct_seg = getattr(self, '_ct_segmentation', None)
            if ct_seg is not None:
                voxel_count = int(ct_seg.mask_3d.sum())
                spacing = self._spect_spacing_or_default()
                voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
                volume_ml = (voxel_count * voxel_vol_mm3) / 1000.0
                self._mask_edit_status.setText(
                    f"💾 CT preservado | {voxel_count} vox | ❤️ {volume_ml:.1f} mL "
                    f"(reactive checkbox para editar)"
                )
            else:
                self._mask_edit_status.setText("Modo manual estable: VOIs esféricas ancladas en A/B")
            
            anchor = getattr(self, "_localization_anchor_zyx", None)
            point = getattr(self, "_localization_point_zyx", None)
            if anchor is not None:
                self._temp_voi_heart = VOISphere(
                    cz=int(anchor[0]), cy=int(anchor[1]), cx=int(anchor[2]),
                    radius_mm=float(self._heart_radius_spin.value()),
                )
            if point is not None:
                self._temp_voi_mediastinum = VOISphere(
                    cz=int(point[0]), cy=int(point[1]), cx=int(point[2]),
                    radius_mm=float(self._mediastinum_radius_spin.value()),
                )
            
            # NO borrar hmr_result ni ct_segmentation — solo cambiar texto si no hay datos
            if self._hmr_result is None:
                self._lbl_hmr_result.setText("HMR-SPECT = N/D · recalcular")
                self._lbl_hmr_result.setStyleSheet(
                    "font-size:14px; font-weight:700; color:#ffffff; "
                    "background:#000000; padding:6px 12px;"
                )
            self._status.setText(
                "CT anatómica desactivada (datos preservados). Reactive para continuar editando."
            )
        else:
            self._btn_toggle_mask_edit.setEnabled(False)
            
            # Restaurar controles si hay datos CT
            ct_seg = getattr(self, '_ct_segmentation', None)
            if ct_seg is not None:
                self._btn_toggle_mask_edit.setEnabled(True)
                if hasattr(self, '_btn_export_mask_nifti'):
                    self._btn_export_mask_nifti.setEnabled(True)
                # Habilitar botones de persistencia
                if hasattr(self, '_btn_save_ct_state'):
                    self._btn_save_ct_state.setEnabled(True)
                if hasattr(self, '_btn_restart_ct'):
                    self._btn_restart_ct.setEnabled(True)
                voxel_count = int(ct_seg.mask_3d.sum())
                spacing = self._spect_spacing_or_default()
                voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
                volume_ml = (voxel_count * voxel_vol_mm3) / 1000.0
                self._mask_edit_status.setText(
                    f"🟢 CT restaurado | {voxel_count} vox | ❤️ {volume_ml:.1f} mL"
                )
            else:
                self._mask_edit_status.setText(
                    "CT experimental activado. Calcule HMR para generar la máscara."
                )
            self._status.setText(
                "CT anatómica/PVE experimental activado. Las VOIs pueden diferir de A/B."
            )
        self._render_selected_view()

    def _calculate_hmr_spect(self) -> None:
        """Calcula HMR-SPECT usando VOIs esféricas desde puntos de localización."""
        try:
            # Verificar volumen SPECT
            if self._current_volume is None:
                QMessageBox.warning(self, "SINCRO", "Primero cargue un volumen SPECT.")
                return
            
            # Verificar puntos de localización
            anchor = getattr(self, "_localization_anchor_zyx", None)
            point = getattr(self, "_localization_point_zyx", None)
            # B estable: usar el centro del círculo azul dibujado (fijado en el
            # momento del clic). La cruz viva navega con los cortes y deriva,
            # lo que muestreaba el mediastino en otra posición.
            tv_med = getattr(self, "_temp_voi_mediastinum", None)
            if tv_med is not None:
                point = (int(round(tv_med.cz)), int(round(tv_med.cy)), int(round(tv_med.cx)))
            
            if anchor is None:
                QMessageBox.warning(
                    self, "SINCRO",
                    "Falta el punto Ancla A (corazón).\n\n"
                    "1. Active 'Localización'\n"
                    "2. Ctrl+clic en el centro del corazón\n"
                    "3. Click 'Fijar ancla A'"
                )
                return
            
            if point is None:
                QMessageBox.warning(
                    self, "SINCRO",
                    "Falta el punto B (mediastino).\n\n"
                    "Active 'Localización' y Ctrl+clic en mediastino superior."
                )
                return
            
            # Obtener spacing
            spacing = getattr(self, "_spect_spacing_zyx", None)
            if spacing is None:
                spacing = (4.0, 4.0, 4.0)  # Default aproximado
            
            # Obtener método y radios
            method_str = self._hmr_method_combo.currentData()
            method = HmrSpectMethod(method_str)
            heart_radius = float(self._heart_radius_spin.value())
            mediastinum_radius = float(self._mediastinum_radius_spin.value())
            
            # Crear VOIs desde puntos de localización (base esférica)
            voi_heart, voi_mediastinum = create_voi_from_localization(
                anchor_zyx=anchor,
                point_zyx=point,
                heart_radius_mm=heart_radius,
                mediastinum_radius_mm=mediastinum_radius
            )
            
            # ── Fase 2: Mejorar VOIs con anatomía CT si disponible ─────
            voi_type_used = "esférica"
            ct_seg = None
            
            use_ct_anatomical = bool(
                hasattr(self, "_ct_anatomical_check")
                and self._ct_anatomical_check.isChecked()
            )

            if (
                use_ct_anatomical
                and getattr(self, "_ct_volume", None) is not None
                and getattr(self, "_ct_spacing_zyx", None) is not None
            ):
                try:
                    from core.amyloid_spect import (
                        segment_myocardium_from_ct,
                        create_anatomical_heart_voi,
                        create_bone_safe_mediastinum_voi,
                    )
                    
                    # ── LÓGICA DE PRESERVACIÓN DE MÁSCARA ─────────────
                    # Calcular ct_transformed y spacing SIEMPRE (lo usa el
                    # mediastino y, si no reutilizamos, la segmentación).
                    _ct_reg = getattr(self, "_ct_registered", None)
                    _ct_vol = _ct_reg if _ct_reg is not None else self._ct_volume
                    if _ct_reg is not None:
                        ct_spacing_used = spacing  # grilla SPECT
                    else:
                        ct_spacing_used = self._ct_spacing_zyx
                    if _ct_reg is not None:
                        ct_transformed = self._ct_registered_visual_transform(
                            np.asarray(_ct_vol, dtype=np.float64)
                        )
                    else:
                        ct_transformed = self._ct_transform_3d(np.asarray(_ct_vol, dtype=np.float64))
                    
                    # 3 fuentes de reutilización (OR lógico):
                    #   a) El usuario acaba de aplicar edición manual (_reuse_edited_segmentation)
                    #   b) El checkbox "🔒 Preservar máscara" está activado
                    #   c) Hay segmentación previa que fue editada manualmente
                    _reuse = bool(getattr(self, "_reuse_edited_segmentation", False))
                    _have_prev_seg = getattr(self, "_ct_segmentation", None) is not None
                    _preserve_checked = (
                        hasattr(self, "_preserve_mask_check")
                        and self._preserve_mask_check.isChecked()
                        and _have_prev_seg
                    )
                    
                    # Detectar si la máscara previa fue editada manualmente
                    # (tiene flag _mask_was_manually_editado o tiene voxels != auto)
                    _mask_was_edited = bool(
                        getattr(self, "_mask_was_manually_edited", False)
                    )
                    
                    # Reutilizar si:
                    #   a) Apply reciente (_reuse)
                    #   b) Preserve checked Y hay segmentación previa (cualquiera, no solo editada)
                    #      Si el checkbox está activado, SIEMPRE preserva la máscara existente.
                    #   c) Preserve checked Y máscara fue editada manualmente (redundante con b)
                    if (_reuse or _preserve_checked) and _have_prev_seg:
                        # Reutilizar la máscara existente tal cual
                        ct_seg = self._ct_segmentation
                        self._metrics.append("🔒 Máscara PRESERVADA (no re-segmentada)")
                    else:
                        ct_seg = segment_myocardium_from_ct(
                            ct_transformed,
                            ct_spacing_used,
                            seed_zyx=anchor,
                            seed_radius_mm=max(heart_radius * 1.5, 50.0),
                        )
                        
                    self._ct_segmentation = ct_seg
                    
                    # Resetear cache del cubo auto (nueva segmentación)
                    self._auto_cube_bbox_cached = None
                    # NO resetear _mask_was_manually_edited aquí — se resetea solo
                    # al hacer nueva segmentación (rama else) o al reiniciar CT
                    
                    # === F2.4: Habilitar edición manual de máscara ===
                    self._mask_edit_original = ct_seg.mask_3d.copy()
                    self._mask_edit_undo_stack.clear()
                    self._mask_edit_has_changes = False
                    if hasattr(self, '_btn_toggle_mask_edit'):
                        self._btn_toggle_mask_edit.setEnabled(True)
                        self._btn_reset_mask.setEnabled(False)
                        self._btn_apply_mask_edit.setEnabled(False)
                        self._btn_undo_mask.setEnabled(False)
                        if hasattr(self, '_btn_export_mask_nifti'):
                            self._btn_export_mask_nifti.setEnabled(True)
                        if hasattr(self, '_btn_save_ct_state'):
                            self._btn_save_ct_state.setEnabled(True)
                        if hasattr(self, '_btn_restart_ct'):
                            self._btn_restart_ct.setEnabled(True)
                        self._mask_edit_status.setText(
                            f"✅ Segmentación CT lista ({int(ct_seg.mask_3d.sum())} voxels). "
                            "Pulsa 'Editar Máscara' para refinar."
                        )
                    
                    # Crear VOI anatómica del corazón (reemplaza la esfera)
                    # ct_spacing_used = spacing SPECT cuando el CT ya está
                    # registrado a la grilla SPECT (evita deformar/desplazar).
                    voi_heart_anat = create_anatomical_heart_voi(
                        ct_segmentation=ct_seg,
                        spect_shape=self._current_volume.shape,
                        spect_spacing=spacing,
                        ct_spacing=ct_spacing_used,
                    )
                    
                    # Crear VOI de mediastino que evita hueso
                    # El punto B (mediastino) está en coordenadas de la grilla
                    # SPECT/display, por eso el ct_volume y su spacing deben
                    # corresponder a esa misma grilla (ct_transformed registrado).
                    voi_med_anat = create_bone_safe_mediastinum_voi(
                        ct_volume=ct_transformed,
                        ct_spacing=ct_spacing_used,
                        mediastinum_center_zyx=(float(point[0]), float(point[1]), float(point[2])),
                        mediastinum_radius_mm=mediastinum_radius,
                        spect_shape=self._current_volume.shape,
                        spect_spacing=spacing,
                    )
                    
                    # Verificar que las VOI anatómicas tengan suficiente contenido
                    min_pixels_heart = 50  # mínimo píxeles para VOI corazón válida
                    heart_mask = voi_heart_anat.mask_3d()
                    heart_pixel_count = int(np.sum(heart_mask)) if isinstance(heart_mask, np.ndarray) else 0
                    
                    if heart_pixel_count >= min_pixels_heart:
                        voi_heart = voi_heart_anat
                        voi_type_used = "anatómica-CT"
                        
                        med_mask = voi_med_anat.mask_3d()
                        med_pixel_count = int(np.sum(med_mask)) if isinstance(med_mask, np.ndarray) else 0
                        if med_pixel_count >= 10:
                            voi_mediastinum = voi_med_anat
                            voi_type_used = "anatómica-CT (hueso-safe)"
                    else:
                        self._metrics.append(f"[Fase2] VOI anatómica muy pequeña ({heart_pixel_count} px), usando esfera")
                        
                    self._metrics.append(f"[Fase2] VOI tipo: {voi_type_used}")
                    
                except Exception as exc_f2:
                    self._metrics.append(f"[Fase2] No se pudo crear VOI anatómica: {exc_f2}")
            
            # Buscar volumen raw (sin filtrar) si está disponible
            # Usar _base_spect_volume como raw (volumen base sin procesamiento)
            volume_raw = getattr(self, "_base_spect_volume", None)
            if volume_raw is None:
                # Si no hay base, usar el mismo volumen (no habrá HMR raw)
                volume_raw = None
            elif np.array_equal(volume_raw, self._current_volume):
                # Si son iguales, no hay volumen raw separado
                volume_raw = None
            
            # Calcular HMR-SPECT
            analysis_volume = self._spect_display_volume(self._current_volume)
            analysis_volume_raw = self._spect_display_volume(volume_raw) if volume_raw is not None else None
            result = compute_hmr_spect(
                volume=analysis_volume,
                spacing_zyx=spacing,
                voi_heart=voi_heart,
                voi_mediastinum=voi_mediastinum,
                method=method,
                volume_raw=analysis_volume_raw,
            )
            
            self._hmr_result = result
            
            # ── Corrección PVE (si hay CT disponible) ──────────────────
            pve_result = None
            # ct_seg ya puede existir del bloque Fase 2 anterior
            if (
                use_ct_anatomical
                and ct_seg is None
                and getattr(self, "_ct_volume", None) is not None
                and getattr(self, "_ct_spacing_zyx", None) is not None
            ):
                try:
                    from core.amyloid_spect import (
                        segment_myocardium_from_ct,
                        apply_pve_correction_to_hmr,
                    )
                    _ct_reg = getattr(self, "_ct_registered", None)
                    ct_vol = _ct_reg if _ct_reg is not None else self._ct_volume
                    if _ct_reg is not None:
                        ct_transformed = self._ct_registered_visual_transform(
                            np.asarray(ct_vol, dtype=np.float64)
                        )
                    else:
                        ct_transformed = self._ct_transform_3d(np.asarray(ct_vol, dtype=np.float64))
                    
                    ct_seg = segment_myocardium_from_ct(
                        ct_transformed,
                        self._ct_spacing_zyx,
                        seed_zyx=anchor,
                        seed_radius_mm=max(heart_radius * 1.5, 50.0),
                    )
                    self._ct_segmentation = ct_seg
                    # Habilitar F2.4 también desde bloque PVE
                    self._mask_edit_original = ct_seg.mask_3d.copy()
                    self._mask_edit_undo_stack.clear()
                    self._mask_edit_has_changes = False
                    if hasattr(self, '_btn_toggle_mask_edit'):
                        self._btn_toggle_mask_edit.setEnabled(True)
                        if hasattr(self, '_btn_export_mask_nifti'):
                            self._btn_export_mask_nifti.setEnabled(True)
                except Exception as exc_seg:
                    self._metrics.append(f"[PVE] No se pudo segmentar CT: {exc_seg}")
            
            if ct_seg is not None:
                try:
                    from core.amyloid_spect import apply_pve_correction_to_hmr
                    pve_result = apply_pve_correction_to_hmr(
                        result,
                        ct_segmentation=ct_seg,
                        fwhm_mm=12.0,  # Típico para SPECT cardíaco con colimador LEHR
                    )
                    self._pve_result = pve_result
                except Exception as exc:
                    self._metrics.append(f"[PVE] No se pudo aplicar corrección PVE: {exc}")
            
            # Actualizar UI con resultado (mostrar ambos HMR si están disponibles)
            if pve_result is not None:
                hmr_text = (
                    f"HMR original = {pve_result.hmr_original:.2f} ({pve_result.classification_original})\n"
                    f"HMR corregido PVE = {pve_result.hmr_pve_corrected:.2f} ({pve_result.classification_corrected})"
                )
            elif result.hmr_raw is not None:
                hmr_text = f"HMR (filtrado) = {result.hmr:.2f}\nHMR (raw) = {result.hmr_raw:.2f}"
            else:
                hmr_text = f"HMR-SPECT = {result.hmr:.2f}"
            
            self._lbl_hmr_result.setText(hmr_text)
            
            # ── Actualizar volumen VISIBLE (no solo en consola oculta) ──
            if hasattr(self, '_lbl_volume_display'):
                heart_vol = getattr(result, 'heart_volume_ml', None)
                med_vol = getattr(result, 'mediastinum_volume_ml', None)
                if heart_vol is not None:
                    vol_text = f"❤️ {heart_vol:.1f} mL"
                    if med_vol is not None:
                        vol_text += f" | 🩻 {med_vol:.1f} mL"
                    self._lbl_volume_display.setText(vol_text)
                # También mostrar volumen de máscara CT si existe
                if ct_seg is not None:
                    ct_vol_ml = getattr(ct_seg, 'volume_mm3', 0) / 1000.0
                    if ct_vol_ml > 0:
                        current = self._lbl_volume_display.text()
                        self._lbl_volume_display.setText(f"{current} | 📦 CT {ct_vol_ml:.1f}mL")
            
            # Color según clasificación del HMR corregido (si existe) o raw/filtrado
            if pve_result is not None:
                hmr_clinical = pve_result.hmr_pve_corrected
                classification = pve_result.classification_corrected
            else:
                hmr_clinical = result.hmr_raw if result.hmr_raw is not None else result.hmr
                if hmr_clinical >= 1.6:
                    classification = "POSITIVO"
                elif hmr_clinical >= 1.5:
                    classification = "EQUIVOCO"
                else:
                    classification = "NEGATIVO"

            color = "#22c55e" if classification == "NEGATIVO" else "#f59e0b" if classification == "EQUIVOCO" else "#ef4444"

            self._lbl_hmr_result.setStyleSheet(
                f"font-size:14px; font-weight:700; color:{color}; "
                f"background:#000000; padding:6px 12px;"
            )
            
            # Mostrar detalles
            details = f"HMR-SPECT\n{'='*30}\n"
            if result.hmr_raw is not None:
                details += f"HMR (raw): {result.hmr_raw:.2f} ← valor clínico\n"
                details += f"HMR (filtrado): {result.hmr:.2f}\n"
            else:
                details += f"HMR: {result.hmr:.2f}\n"
            details += f"Clasificación: {classification}\n\n"
            details += f"Método: {result.method}\n"
            details += f"Tipo VOI: {voi_type_used}\n"
            details += f"A usado (Z/Y/X): {anchor[0]+1}/{anchor[1]+1}/{anchor[2]+1}\n"
            details += f"B usado (Z/Y/X): {point[0]+1}/{point[1]+1}/{point[2]+1}\n"
            details += f"Media corazón: {result.heart_mean:.2f} cts/píxel"
            details += f"\nMedia mediastino: {result.mediastinum_mean:.2f} cts/píxel"
            details += f"\nCuentas totales corazón: {result.heart_counts:.0f}"
            if result.heart_counts_raw > 0:
                details += f" (raw: {result.heart_counts_raw:.0f})"
            details += f"\nCuentas totales mediastino: {result.mediastinum_counts:.0f}"
            if result.mediastinum_counts_raw > 0:
                details += f" (raw: {result.mediastinum_counts_raw:.0f})"
            details += f"\nVolumen corazón: {result.heart_volume_ml:.1f} mL"
            details += f"\nVolumen mediastino: {result.mediastinum_volume_ml:.1f} mL"
            
            # === Volumen cardíaco desde máscara CT manual (si existe) ===
            ct_seg = getattr(self, '_ct_segmentation', None)
            if ct_seg is not None and hasattr(ct_seg, 'mask_3d'):
                mask_voxels = int(ct_seg.mask_3d.sum())
                if mask_voxels > 0:
                    spacing = self._spect_spacing_or_default()
                    voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
                    mask_volume_ml = (mask_voxels * voxel_vol_mm3) / 1000.0
                    details += f"\n\n=== Máscara CT Manual ==="
                    details += f"\nVoxels máscara: {mask_voxels}"
                    details += f"\n❤️ Volumen máscara: {mask_volume_ml:.1f} mL ({mask_voxels * voxel_vol_mm3:.0f} mm³)"
                    # Guardar para referencia futura
                    ct_seg.volume_mm3 = mask_voxels * voxel_vol_mm3
            
            if result.slice_idx is not None:
                details += f"\nSlice axial: {result.slice_idx}"
            
            # Información de diagnóstico
            details += f"\n\n{'='*30}\nDIAGNÓSTICO VOIs:\n"
            details += f"Píxeles corazón: {result.heart_pixels}\n"
            details += f"Píxeles mediastino: {result.mediastinum_pixels}\n"
            details += f"Ratio de medias usado: {result.hmr:.2f}"
            
            # Advertencia si el mediastino tiene muy pocas cuentas
            if result.mediastinum_mean < 1.0:
                details += "\n\n⚠️ ADVERTENCIA: El mediastino tiene\nmuy bajas cuentas (<1 cts/píxel).\n¿Está el VOI en una zona vacía?"
            
            # ── Información PVE si disponible ────────────────────────
            if pve_result is not None:
                details += f"\n\n{'='*40}\n🔬 CORRECCIÓN PVE (Efecto Volumen Parcial)\n{'='*40}"
                details += f"\nGrosor pared (CT): {pve_result.wall_thickness_mm:.1f} mm"
                details += f"\nFWHM sistema: {pve_result.fwhm_mm:.1f} mm"
                details += f"\nCoeficiente RC: {pve_result.rc_heart:.3f}"
                details += f"\nFactor corrección: {pve_result.pve_factor:.3f}"
                details += f"\nΔ HMR: {pve_result.delta_pct:+.1f}%"
                
                if ct_seg is not None:
                    details += f"\n\nSegmentación CT:"
                    details += f"\n  Volumen miocardio: {ct_seg.volume_mm3/1000:.1f} mL"
                    details += f"\n  Slices con tejido: {ct_seg.n_slices}"
                    if ct_seg.wall_thickness_mm:
                        details += "\n  Grosor por segmento:"
                        for seg, thick in list(ct_seg.wall_thickness_mm.items())[:4]:
                            details += f"\n    {seg}: {thick:.1f} mm"
                
                if pve_result.classification_original != pve_result.classification_corrected:
                    details += f"\n\n⚠️ ¡LA PVE CAMBIÓ LA CLASIFICACIÓN!"
                    details += f"\n   {pve_result.classification_original} → {pve_result.classification_corrected}"
                    details += f"\n   Revisar calidad de la segmentación CT."
                
                # Notas del proceso PVE
                for note in pve_result.notes[-5:]:
                    details += f"\n{note}"
            
            QMessageBox.information(self, "SINCRO — HMR-SPECT", details)
            
            # Re-renderizar para mostrar los VOIs
            self._render_selected_view()
            
        except Exception as exc:
            self._hmr_result = None
            self._pve_result = None
            self._lbl_hmr_result.setText("HMR-SPECT = N/D · revisar VOI")
            self._lbl_hmr_result.setStyleSheet(
                "font-size:14px; font-weight:700; color:#f59e0b; "
                "background:#000000; padding:6px 12px;"
            )
            self._status.setText(f"Error calculando HMR-SPECT: {exc}")
            QMessageBox.critical(self, "SINCRO", f"Error calculando HMR-SPECT:\n{exc}")

    @staticmethod
    def _blend_mask_over_rgb(rgb: np.ndarray, mask2d: np.ndarray | None, alpha: float) -> np.ndarray:
        arr = np.asarray(rgb, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("RGB inválido para blend de máscara")
        if mask2d is None:
            return np.clip(arr, 0, 255).astype(np.uint8)
        m = np.asarray(mask2d, dtype=bool)
        a = float(np.clip(alpha, 0.0, 1.0))
        if np.any(m) and a > 0.0:
            arr[m, 0] = (1.0 - a) * arr[m, 0] + a * 255.0
            arr[m, 1] = (1.0 - a) * arr[m, 1] + a * 64.0
            arr[m, 2] = (1.0 - a) * arr[m, 2] + a * 64.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _render_preview(self, volume: np.ndarray):
        pv = self._slices_preview_at(volume)
        # Aplicar colormap seleccionado al combo antes de convertir a pixmap
        ax_data = pv["axial"]
        co_data = pv["coronal"]
        sa_data = pv["sagittal"]
        
        # Aplicar colormap si los datos están normalizados (0-1)
        ax_rgb = (self._apply_cmap(ax_data) * 255.0).astype(np.uint8)
        co_rgb = (self._apply_cmap(co_data) * 255.0).astype(np.uint8)
        sa_rgb = (self._apply_cmap(sa_data) * 255.0).astype(np.uint8)
        
        ax = self._rgb_to_pixmap(ax_rgb).scaled(
            self._axial_lbl.width(), self._axial_lbl.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        co = self._rgb_to_pixmap(co_rgb).scaled(
            self._cor_lbl.width(), self._cor_lbl.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        sa = self._rgb_to_pixmap(sa_rgb).scaled(
            self._sag_lbl.width(), self._sag_lbl.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._set_axis_pixmap_with_cross(self._axial_lbl, ax, "axial")
        self._set_axis_pixmap_with_cross(self._cor_lbl, co, "coronal")
        self._set_axis_pixmap_with_cross(self._sag_lbl, sa, "sagittal")

    def _update_slice_controls(self):
        if self._current_volume is None:
            return
        shape = tuple(int(v) for v in np.asarray(self._current_volume).shape[:3])
        sliders = ((self._slice_z, "axial", shape[0], self._slice_z_lbl, "z"),
                   (self._slice_y, "coronal", shape[1], self._slice_y_lbl, "y"),
                   (self._slice_x, "sagittal", shape[2], self._slice_x_lbl, "x"))
        for slider, key, n, lbl, name in sliders:
            slider.blockSignals(True)
            slider.setRange(0, max(0, n - 1))
            val = self._slice_idx.get(key, n // 2)
            if val <= 0 or val >= n:
                val = n // 2
            self._slice_idx[key] = int(val)
            slider.setValue(int(val))
            slider.setEnabled(n > 1)
            lbl.setText(f"{name} {int(val) + 1}/{n}")
            slider.blockSignals(False)

    def _on_slice_slider_changed(self):
        self._slice_idx["axial"] = int(self._slice_z.value())
        self._slice_idx["coronal"] = int(self._slice_y.value())
        self._slice_idx["sagittal"] = int(self._slice_x.value())
        self._move_localization_point_to_current_slices(None)
        if self._current_volume is not None:
            shape = tuple(int(v) for v in np.asarray(self._current_volume).shape[:3])
            self._slice_z_lbl.setText(f"z {self._slice_idx['axial'] + 1}/{shape[0]}")
            self._slice_y_lbl.setText(f"y {self._slice_idx['coronal'] + 1}/{shape[1]}")
            self._slice_x_lbl.setText(f"x {self._slice_idx['sagittal'] + 1}/{shape[2]}")
        self._render_current_with_overlay()

    def _on_link_zoom_changed(self, state: int):
        """Handler cuando se ancla/desancalan los zooms."""
        is_linked = (state == Qt.CheckState.Checked.value())
        # Si se acaba de anclar, sincronizar el valor del CT al SPECT
        if is_linked:
            spect_val = int(self._spect_zoom_spin.value())
            self._ct_zoom_spin.blockSignals(True)
            self._ct_zoom_spin.setValue(spect_val)
            self._ct_zoom_spin.blockSignals(False)
            self._ct_zoom_pct = spect_val
    
    def _on_zoom_changed(self):
        """Handler cuando cambia algún spinbox de zoom."""
        new_spect = int(self._spect_zoom_spin.value())
        new_ct = int(self._ct_zoom_spin.value())
        
        # Si están anclados, sincronizar
        is_linked = getattr(self, '_link_zoom_check', None) is not None and self._link_zoom_check.isChecked()
        if is_linked:
            sender = self.sender()
            if sender == self._spect_zoom_spin:
                self._ct_zoom_spin.blockSignals(True)
                self._ct_zoom_spin.setValue(new_spect)
                self._ct_zoom_spin.blockSignals(False)
                new_ct = new_spect
            elif sender == self._ct_zoom_spin:
                self._spect_zoom_spin.blockSignals(True)
                self._spect_zoom_spin.setValue(new_ct)
                self._spect_zoom_spin.blockSignals(False)
                new_spect = new_ct
        
        self._spect_zoom_pct = new_spect
        self._ct_zoom_pct = new_ct
        self._render_current_with_overlay()

    def _on_spect_range_changed(self, low: int, high: int):
        self._spect_win_low = int(low)
        self._spect_win_high = int(high)
        self._spect_range_lbl.setText(f"Base {self._spect_win_low}% · Top {self._spect_win_high}%")
        self._render_current_with_overlay()

    def _set_spect_range(self, low: int, high: int):
        self._spect_range_slider.set_values(int(low), int(high))

    def _on_ct_window_changed(self):
        self._ct_window = str(self._ct_window_combo.currentData() or "bone")
        self._render_current_with_overlay()

    def _on_spect_display_smooth_changed(self, value: float):
        self._spect_display_sigma = max(0.0, float(value))
        if hasattr(self, "_settings"):
            self._settings.setValue("global/spect_display_sigma", self._spect_display_sigma)
        self._render_selected_view()

    def _smooth_display_2d(self, img: np.ndarray) -> np.ndarray:
        sigma = float(getattr(self, "_spect_display_sigma", 0.0))
        if sigma < 0.05:
            return img
        return ndi.gaussian_filter(np.asarray(img, dtype=np.float64), sigma=sigma, mode="nearest")

    def _on_ct_trial_toggled(self, checked: bool):
        self._ct_visual_trial_mode = bool(checked)
        if checked:
            self._status.setText("CT nítida activada (realce visual del MPR de CT).")
        else:
            self._status.setText("CT nítida desactivada.")
        if hasattr(self, "_settings"):
            self._settings.setValue("global/ct_sharp", bool(checked))
        self._render_current_with_overlay()

    @staticmethod
    def _map_idx_between_grids(idx_ref: int, ref_n: int, tgt_n: int) -> int:
        if ref_n <= 1 or tgt_n <= 1:
            return 0
        r = float(np.clip(idx_ref, 0, ref_n - 1)) / float(ref_n - 1)
        return int(np.clip(round(r * float(tgt_n - 1)), 0, tgt_n - 1))

    def _invalidate_ct_grid_trial_cache(self):
        self._trial_cache_signature = None
        self._trial_spect_on_ct = None
        self._trial_ct_native = None
        self._trial_ct_native_spacing = None
        self._trial_ref_shape = None

    def _ensure_ct_grid_trial_cache(self) -> bool:
        if self._ct_volume is None or self._current_volume is None:
            self._invalidate_ct_grid_trial_cache()
            return False

        sig = (
            id(self._current_volume),
            id(self._ct_volume),
            bool(getattr(self, "_spect_flip_x_test", False)),
            bool(getattr(self, "_spect_flip_y_test", False)),
            bool(getattr(self, "_spect_flip_z_test", False)),
            bool(getattr(self, "_ct_flip_x_test", False)),
            bool(getattr(self, "_ct_flip_y_test", False)),
            bool(getattr(self, "_ct_flip_z_test", False)),
            tuple(np.asarray(self._spect_affine_ijk_to_lps).ravel()) if self._spect_affine_ijk_to_lps is not None else None,
            tuple(np.asarray(self._ct_affine_ijk_to_lps).ravel()) if self._ct_affine_ijk_to_lps is not None else None,
            tuple(self._spect_spacing_zyx) if self._spect_spacing_zyx is not None else None,
            tuple(self._ct_spacing_zyx) if self._ct_spacing_zyx is not None else None,
            tuple(getattr(self, '_ct_total_shift_zyx', (0.0, 0.0, 0.0))),  # Invalidar si cambia el registro/nudge
            float(self._rot_z.value()) if hasattr(self, '_rot_z') else 0.0,  # rot Z nudge
            float(self._rot_y.value()) if hasattr(self, '_rot_y') else 0.0,  # rot Y nudge
            float(self._rot_x.value()) if hasattr(self, '_rot_x') else 0.0,  # rot X nudge
        )
        if sig == self._trial_cache_signature and self._trial_spect_on_ct is not None and self._trial_ct_native is not None:
            return True

        try:
            spect_tx = self._spect_transform_3d(np.asarray(self._current_volume, dtype=np.float64))
            ct_tx = self._ct_transform_3d(np.asarray(self._ct_volume, dtype=np.float64))
            
            # === Estrategia: CT nativo recortado al FOV del SPECT ===
            # En vez de expandir el SPECT al FOV completo del CT (que deja el corazón
            # pequeño y al borde), recortamos el CT al FOV del SPECT pero a mayor
            # resolución (2x o 3x la grilla SPECT original).
            # Esto da CT nítido + SPECT bien dimensionado.
            spect_sp = self._spect_spacing_zyx or (6.8, 6.8, 6.8)
            ct_sp = self._ct_spacing_zyx or (1.0, 1.0, 1.0)
            
            # Grilla objetivo: 2x la resolución SPECT en cada eje (128³ si SPECT es 64³)
            target_shape = tuple(int(spect_tx.shape[i] * 2) for i in range(3))
            target_spacing = tuple(float(spect_sp[i]) / 2.0 for i in range(3))
            
            # Crear affine del SPECT escalada 2x (mismo FOV, spacing/2)
            spect_affine_2x = None
            if self._spect_affine_ijk_to_lps is not None:
                sa = np.asarray(self._spect_affine_ijk_to_lps, dtype=np.float64).copy()
                # Dividir spacing (elementos diagonales) por 2, mantener origen
                sa[0, 0] /= 2.0
                sa[1, 1] /= 2.0
                sa[2, 2] /= 2.0
                spect_affine_2x = sa
            
            # Remuestrear CT a la grilla objetivo (mayor resolución, mismo FOV que SPECT)
            ct_native, ct_notes = resample_volume_to_spect_grid(
                ct_tx,
                np.zeros(target_shape),  # solo usa el shape
                source_spacing_zyx=ct_sp,
                spect_spacing_zyx=target_spacing,
                source_affine_ijk_to_lps=self._ct_affine_ijk_to_lps,
                spect_affine_ijk_to_lps=spect_affine_2x,
                fill_value=-1024.0,
                order=1,
            )
            ct_native_spacing = target_spacing
            
            # === Aplicar rotaciones del nudge (igual que _apply_ct_nudge) ===
            # Las rotaciones están en grados y se aplican sobre la grilla 2x.
            rot_z = float(self._rot_z.value()) if hasattr(self, '_rot_z') else 0.0
            rot_y = float(self._rot_y.value()) if hasattr(self, '_rot_y') else 0.0
            rot_x = float(self._rot_x.value()) if hasattr(self, '_rot_x') else 0.0
            if abs(rot_z) > 1e-6:
                ct_native = ndi.rotate(ct_native, angle=rot_z, axes=(1, 2), reshape=False, order=1, mode='nearest')
            if abs(rot_y) > 1e-6:
                ct_native = ndi.rotate(ct_native, angle=rot_y, axes=(0, 2), reshape=False, order=1, mode='nearest')
            if abs(rot_x) > 1e-6:
                ct_native = ndi.rotate(ct_native, angle=rot_x, axes=(0, 1), reshape=False, order=1, mode='nearest')
            if hasattr(self, '_metrics') and (abs(rot_z) > 1e-6 or abs(rot_y) > 1e-6 or abs(rot_x) > 1e-6):
                self._metrics.append(
                    f"[CT-NATIVE] Rotaciones nudge aplicadas: "
                    f"rot(z,y,x)=({rot_z:.1f},{rot_y:.1f},{rot_x:.1f})°"
                )
            
            # === Aplicar shift de registro (alineación CT↔SPECT) ===
            # El shift total está en píxeles de la grilla SPECT original (64³).
            # Convertir a píxeles de la grilla objetivo (128³): multiplicar por 2.
            total_shift = getattr(self, '_ct_total_shift_zyx', (0.0, 0.0, 0.0))
            if any(abs(s) > 0.01 for s in total_shift):
                shift_target = (
                    float(total_shift[0]) * 2.0,
                    float(total_shift[1]) * 2.0,
                    float(total_shift[2]) * 2.0,
                )
                ct_native = ndi.shift(ct_native, shift=shift_target, order=1, mode='nearest')
                if hasattr(self, '_metrics'):
                    self._metrics.append(
                        f"[CT-NATIVE] Shift registro aplicado: "
                        f"Δ(z,y,x)=({shift_target[0]:.1f},{shift_target[1]:.1f},{shift_target[2]:.1f}) px (grid 2x)"
                    )
            
            # SPECT → misma grilla objetivo (2x upsampling, suave)
            spect_on_ct, notes = resample_volume_to_spect_grid(
                spect_tx,
                ct_native,
                source_spacing_zyx=spect_sp,
                spect_spacing_zyx=target_spacing,
                source_affine_ijk_to_lps=self._spect_affine_ijk_to_lps,
                spect_affine_ijk_to_lps=spect_affine_2x,
                fill_value=float(np.min(spect_tx)) if spect_tx.size else 0.0,
                order=1,  # bilineal: SPECT se ve suave al hacer upsampling
            )
            # Diagnóstico de registro
            if hasattr(self, '_metrics'):
                for n in notes:
                    self._metrics.append(f"[CT-NATIVE] {n}")
                self._metrics.append(
                    f"[CT-NATIVE] SPECT {spect_tx.shape} → grid {spect_on_ct.shape} | "
                    f"CT {ct_tx.shape} → {ct_native.shape} (2x SPECT res)"
                )
                # Detectar si el corazón está al borde
                sp_nonzero = np.argwhere(spect_on_ct > float(np.percentile(spect_on_ct, 90)))
                if len(sp_nonzero) > 0:
                    z_min, y_min, x_min = sp_nonzero.min(axis=0)
                    z_max, y_max, x_max = sp_nonzero.max(axis=0)
                    self._metrics.append(
                        f"[CT-NATIVE] SPECT 90% percentile bbox: "
                        f"z=[{z_min}-{z_max}/{spect_on_ct.shape[0]}] "
                        f"y=[{y_min}-{y_max}/{spect_on_ct.shape[1]}] "
                        f"x=[{x_min}-{x_max}/{spect_on_ct.shape[2]}]"
                    )
            self._trial_spect_on_ct = np.ascontiguousarray(np.asarray(spect_on_ct, dtype=np.float32))  # float32 ahorra memoria
            self._trial_ct_native = np.ascontiguousarray(np.asarray(ct_native, dtype=np.float32))
            self._trial_ct_native_spacing = ct_native_spacing
            self._trial_ref_shape = tuple(int(v) for v in np.asarray(self._current_volume).shape[:3])
            self._trial_cache_signature = sig
            return True
        except Exception:
            self._invalidate_ct_grid_trial_cache()
            return False

    def _on_ct_grid_trial_toggled(self, checked: bool):
        self._ct_grid_trial_mode = bool(checked)
        self._invalidate_ct_grid_trial_cache()
        if checked:
            self._status.setText("[PRUEBA/BETA] CT nativa + SPECT en grilla CT activado. Desmarcar para rollback inmediato.")
            if hasattr(self, "_metrics"):
                self._metrics.append("[PRUEBA/BETA] CT nativa + SPECT escalado a CT activado (solo visual, reversible).")
        else:
            self._status.setText("Modo PRUEBA/BETA grilla CT desactivado. Rollback al estado actual aplicado.")
            if hasattr(self, "_metrics"):
                self._metrics.append("[PRUEBA/BETA] CT nativa + SPECT escalado a CT desactivado (rollback aplicado).")
        self._render_current_with_overlay()

    # ─────────────────────────────────────────────
    # Diálogo: Rol del CT en amiloidosis cardíaca
    # ─────────────────────────────────────────────
    def _show_ct_role_info(self):
        from PyQt6.QtWidgets import QTextBrowser, QDialogButtonBox

        dlg = QDialog(self)
        dlg.setWindowTitle("ℹ️ Rol del CT en Amiloidosis Cardíaca SPECT/CT")
        dlg.setMinimumWidth(580)
        dlg.setMinimumHeight(520)

        vbox = QVBoxLayout(dlg)

        text = QTextBrowser()
        text.setOpenExternalLinks(True)
        text.setStyleSheet(
            "font-size:11px; line-height:1.45; padding:8px; "
            "background:#f8fafc; border-radius:6px;"
        )
        text.setHtml(
            "<h2 style='color:#1e40af;'>📋 Rol del CT en Amiloidosis Cardíaca SPECT/CT</h2>"

            "<h3 style='color:#2563eb;margin-top:14px;'>1. Corrección de Atenuación (AC)</h3>"
            "<p>El CT se usa para generar un <b>μ-map</b> (mapa de atenuación) que corrige la atenuación fotónica del SPECT. Sin esta corrección:</p>"
            "<ul>"
            "<li>La pared <b>anterior</b> aparece falsamente hipocaptante (esternón, costillas)"
            "<li>La pared <b>inferior</b> puede sobreestimarse"
            "<li>Las cuantificaciones de <b>SUV</b> o <b>Uptake Ratio</b> se sesgan"
            "</ul>"

            "<h3 style='color:#2563eb;margin-top:12px;'>2. Localización Anatómica (Fusión)</h3>"
            "<p>Permite confirmar que la captación del trazador:</p>"
            "<ul>"
            "<li>Es <b>miocárdica</b> (pared del VI) y no sanguínea, costal, esternal o mamaria"
            "<li>Distingue captura <b>cardíaca</b> vs <b>vascular</b> (aorta, arterias)"
            "<li>Localiza captación extracardíca relevante (hígado, bazo, músculo)"
            "</ul>"

            "<h3 style='color:#2563eb;margin-top:12px;'>3. Cálculo de Relaciones H/C (Heart-to-Contralateral)</h3>"
            "<p>El estándar de Perkins (JNC 2005, actualizado por Perugini) requiere medir ROI en:</p>"
            "<ul>"
            "<li><b>Heart</b> (miocardio) — localizado con precisión gracias al CT"
            "<li><b>Contralateral</b> (tejido circulante, típicamente tórax derecho) — evitando costillas/grasa"
            "</ul>"
            "<p>La fusión SPECT/CT mejora la reproducibilidad del placement de ROIs.</p>"

            "<h3 style='color:#2563eb;margin-top:12px;'>4. Detección de Patrón de Captación</h3>"
            "<p>Con CT se distingue visualmente:</p>"
            "<table style='border-collapse:collapse;width:100%;margin:6px 0;'>"
            "<tr style='background:#dbeafe;'><th style='padding:6px;text-align:left;border:1px solid #bfdbfe;'>Patrón</th><th style='padding:6px;text-align:left;border:1px solid #bfdbfe;'>Significado</th></tr>"
            "<tr><td style='padding:5px;border:1px solid #e5e7eb;'><b>Difuso miocárdico</b></td><td style='padding:5px;border:1px solid #e5e7eb;'>Sugestivo de amiloidosis (score 2-3)</td></tr>"
            "<tr><td style='padding:5px;border:1px solid #e5e7eb;'><b>Focal/subendocárdico</b></td><td style='padding:5px;border:1px solid #e5e7eb;'>Más sugestivo de isquemia/infarto</td></tr>"
            "<tr><td style='padding:5px;border:1px solid #e5e7eb;'><b>Costal/óseo</b></td><td style='padding:5px;border:1px solid #e5e7eb;'>Degenerativa, no cardíaca</td></tr>"
            "</table>"

            "<h3 style='color:#2563eb;margin-top:12px;'>5. Exclusión de Miméticos</h3>"
            "<p>El CT ayuda a descartar:</p>"
            "<ul>"
            "<li><b>Derrame pericárdico</b> (atenúa el SPECT)"
            "<li><b>Calcificaciones masivas</b> (pueden confundirse)"
            "<li><b>Artefactos de movimiento</b> (correlacionando anatomía)"
            "<li><b>Hipertrofia asimétrica</b> (HOCM vs amiloidosis)"
            "</ul>"

            "<hr style='border:none;border-top:1px dashed #cbd5e1;margin:14px 0;'/>"

            "<div style='background:#fef3c7;padding:10px;border-radius:6px;border-left:4px solid #f59e0b;'>"
            "<b style='color:#92400e;'>⚠️ Limitación Importante</b>"
            "<p style='margin:4px 0 0 0;color:#78350f;font-size:10.5px;'>"
            "El CT que llega es frecuentemente un <b>CT de baja dosis</b> (del SPECT/CT híbrido) — no un CT diagnóstico cardíaco contrastado.<br/>"
            "Su resolución es limitada (~1–2 mm vs 0.5 mm de un CT cardíaco), pero suficiente para:<br/>"
            "✅ Corrección de atenuación &nbsp;|&nbsp; ✅ Fusión anatómica &nbsp;|&nbsp; ✅ Corrección PVE &nbsp;|&nbsp; ❌ No para evaluar LVE ni realce tardío"
            "</p>"
            "</div>"

            "<hr style='border:none;border-top:1px dashed #cbd5e1;margin:14px 0;'/>"

            "<div style='background:#dbeafe;padding:10px;border-radius:6px;border-left:4px solid #3b82f6;'>"
            "<b style='color:#1e40af;'>🔬 Corrección PVE Implementada (v1.60.0+)</b>"
            "<p style='margin:4px 0 0 0;color:#1e3a8a;font-size:10.5px;'>"
            "<b>Efecto de Volumen Parcial (PVE):</b> La resolución SPECT (~12mm FWHM) es similar al grosor de la pared miocárdica (~10-14mm).<br/>"
            "Cada voxel mezcla actividad del miocardio y la cavidad, <b>subestimando el HMR hasta 20-40%</b>.<br/><br/>"
            "<b>Solución implementada:</b><br/>"
            "1️⃣ Segmentación automática del miocardio desde CT<br/>"
            "2️⃣ Medición de grosor parietal por segmento (AHA)<br/>"
            "3️⃣ Cálculo de Coeficiente de Recuperación (RC) analítico<br/>"
            "4️⃣ Aplicación al HMR: HMR_corr = HMR_orig ÷ RC<br/><br/>"
            "La corrección se aplica automáticamente cuando hay CT cargado y se calcula HMR."
            "</p>"
            "</div>"
        )
        vbox.addWidget(text)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(dlg.close)
        vbox.addWidget(btns)

        dlg.exec()

    def _trial_slices_on_ct_grid(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
        if not self._ensure_ct_grid_trial_cache():
            return None
        sp = self._trial_spect_on_ct  # ya float32 contiguo
        ct = self._trial_ct_native
        ref = self._trial_ref_shape or tuple(int(v) for v in sp.shape)

        off_z = float(self._spect_view_offset.get("axial", 0.0))
        off_y = float(self._spect_view_offset.get("coronal", 0.0))
        off_x = float(self._spect_view_offset.get("sagittal", 0.0))
        if abs(off_z) > 1e-6 or abs(off_y) > 1e-6 or abs(off_x) > 1e-6:
            sp = ndi.shift(sp, shift=(off_z, off_y, off_x), order=1, mode='nearest')

        z_ref = int(np.clip(self._slice_idx.get("axial", max(0, ref[0] // 2)), 0, max(0, ref[0] - 1)))
        y_ref = int(np.clip(self._slice_idx.get("coronal", max(0, ref[1] // 2)), 0, max(0, ref[1] - 1)))
        x_ref = int(np.clip(self._slice_idx.get("sagittal", max(0, ref[2] // 2)), 0, max(0, ref[2] - 1)))

        z = self._map_idx_between_grids(z_ref, ref[0], sp.shape[0])
        y = self._map_idx_between_grids(y_ref, ref[1], sp.shape[1])
        x = self._map_idx_between_grids(x_ref, ref[2], sp.shape[2])

        # Extraer slices 2D (operación O(1) en numpy)
        sp_ax = self._window_spect(sp[z])
        sp_co = self._window_spect(sp[:, y, :])
        sp_sa = self._window_spect(sp[:, :, x])

        ct_ax = self._window_ct(ct[z])
        ct_co = self._window_ct(ct[:, y, :])
        ct_sa = self._window_ct(ct[:, :, x])

        # Corrección de aspecto físico para cortes no-axiales.
        # Usar el spacing del CT nativo reducido (no el original) para que el aspecto sea correcto.
        ct_sp = getattr(self, "_trial_ct_native_spacing", None) or getattr(self, "_ct_spacing_zyx", None)
        if ct_sp is not None and len(ct_sp) == 3:
            z_mm = max(1e-6, float(ct_sp[0]))
            y_mm = max(1e-6, float(ct_sp[1]))
            x_mm = max(1e-6, float(ct_sp[2]))
            # coronal: eje vertical=z, horizontal=x → repetir z para mantener aspecto
            ratio_co = z_mm / x_mm
            ratio_sa = z_mm / y_mm
            sp_co, ct_co = self._aspect_correct_2d(sp_co, ct_co, ratio_co)
            sp_sa, ct_sa = self._aspect_correct_2d(sp_sa, ct_sa, ratio_sa)

        if bool(getattr(self, "_ct_visual_trial_mode", False)):
            ct_ax = self._enhance_ct_trial(ct_ax)
            ct_co = self._enhance_ct_trial(ct_co)
            ct_sa = self._enhance_ct_trial(ct_sa)

        # Zoom/pan: solo si no es 100% (evita ndi.zoom innecesario)
        sp_prev = {
            "axial": self._apply_zoom_pan_2d(sp_ax, self._spect_zoom_pct, self._spect_pan_px["axial"], order=1),
            "coronal": self._apply_zoom_pan_2d(sp_co, self._spect_zoom_pct, self._spect_pan_px["coronal"], order=1),
            "sagittal": self._apply_zoom_pan_2d(sp_sa, self._spect_zoom_pct, self._spect_pan_px["sagittal"], order=1),
        }
        ct_prev = {
            "axial": self._apply_zoom_pan_2d(ct_ax, self._ct_zoom_pct, self._ct_pan_px["axial"], order=1),
            "coronal": self._apply_zoom_pan_2d(ct_co, self._ct_zoom_pct, self._ct_pan_px["coronal"], order=1),
            "sagittal": self._apply_zoom_pan_2d(ct_sa, self._ct_zoom_pct, self._ct_pan_px["sagittal"], order=1),
        }
        return sp_prev, ct_prev

    @staticmethod
    def _aspect_correct_2d(img_a: np.ndarray, img_b: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
        """Corrige aspecto físico de dos imágenes 2D simultáneamente.
        
        Si ratio ≈ entero, usa np.repeat (rapidísimo).
        Si no, usa ndi.zoom con prefilter=False.
        """
        if abs(ratio - 1.0) < 0.05:
            return img_a, img_b
        # Buscar factor entero cercano
        n_int = max(1, int(round(ratio)))
        if abs(ratio - n_int) < 0.15 and n_int >= 1:
            # np.repeat es O(N) sin interpolación — instantáneo
            return np.repeat(img_a, n_int, axis=0), np.repeat(img_b, n_int, axis=0)
        # Fallback: ndi.zoom con prefilter=False (evita cómputo extra)
        za = ndi.zoom(img_a, (ratio, 1.0), order=1, prefilter=False)
        zb = ndi.zoom(img_b, (ratio, 1.0), order=1, prefilter=False)
        return za, zb

    @staticmethod
    def _apply_zoom_pan_2d(img: np.ndarray, zoom_pct: int, pan_yx: list[int] | tuple[int, int], order: int = 1) -> np.ndarray:
        """Aplica zoom y pan a imagen 2D. Evita ndi.zoom si zoom=100% y pan=0."""
        arr = np.asarray(img, dtype=np.float64)
        dy, dx = int(pan_yx[0]), int(pan_yx[1])
        z = max(0.05, float(zoom_pct) / 100.0)
        
        if abs(z - 1.0) < 1e-6 and dy == 0 and dx == 0:
            return arr  # caso común: sin zoom ni pan → retorno directo
        
        if abs(z - 1.0) < 1e-6:
            # Solo pan: usar roll (O(N), sin interpolación)
            return np.roll(arr, shift=(dy, dx), axis=(0, 1))
        
        # Zoom + pan: ndi.zoom con prefilter=False
        out_shape = arr.shape
        scaled = ndi.zoom(arr, z, order=order, prefilter=False)
        result = np.zeros(out_shape, dtype=np.float64)
        src_slices = []
        dst_slices = []
        for src_len, dst_len in zip(scaled.shape, out_shape):
            if src_len <= dst_len:
                src0 = 0
                src1 = src_len
                dst0 = (dst_len - src_len) // 2
                dst1 = dst0 + src_len
            else:
                src0 = (src_len - dst_len) // 2
                src1 = src0 + dst_len
                dst0 = 0
                dst1 = dst_len
            src_slices.append(slice(src0, src1))
            dst_slices.append(slice(dst0, dst1))
        result[tuple(dst_slices)] = scaled[tuple(src_slices)]
        
        if dy != 0 or dx != 0:
            result = np.roll(result, shift=(dy, dx), axis=(0, 1))
        return result

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)

        if ctrl and key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._ct_zoom_spin.setValue(int(np.clip(self._ct_zoom_spin.value() + 5, 50, 200)))
            self._status.setText(f"Ctrl +: zoom CT {self._ct_zoom_spin.value()}%")
            event.accept()
            return
        if ctrl and key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self._ct_zoom_spin.setValue(int(np.clip(self._ct_zoom_spin.value() - 5, 50, 200)))
            self._status.setText(f"Ctrl -: zoom CT {self._ct_zoom_spin.value()}%")
            event.accept()
            return
        if shift and key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._spect_zoom_spin.setValue(int(np.clip(self._spect_zoom_spin.value() + 5, 50, 200)))
            self._status.setText(f"Shift +: zoom SPECT {self._spect_zoom_spin.value()}%")
            event.accept()
            return
        if shift and key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self._spect_zoom_spin.setValue(int(np.clip(self._spect_zoom_spin.value() - 5, 50, 200)))
            self._status.setText(f"Shift -: zoom SPECT {self._spect_zoom_spin.value()}%")
            event.accept()
            return

        super().keyPressEvent(event)

    def _on_image_wheel(self, event, axis: str):
        delta = int(event.angleDelta().y())
        if delta == 0:
            event.accept()
            return
        step = 1 if delta > 0 else -1
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        alt = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        if ctrl:
            self._rot_z.setValue(float(np.clip(self._rot_z.value() + step * 0.5, -45.0, 45.0)))
            self._status.setText(f"Ctrl+rueda: rotación CT eje z = {self._rot_z.value():.1f}°")
            self._render_current_with_overlay()
            event.accept()
            return
        if shift:
            self._rot_y.setValue(float(np.clip(self._rot_y.value() + step * 0.5, -45.0, 45.0)))
            self._status.setText(f"Shift+rueda: cabeceo CT eje y = {self._rot_y.value():.1f}°")
            self._render_current_with_overlay()
            event.accept()
            return
        if alt:
            self._rot_x.setValue(float(np.clip(self._rot_x.value() + step * 0.5, -45.0, 45.0)))
            self._status.setText(f"Alt+rueda: rotación CT eje x = {self._rot_x.value():.1f}°")
            self._render_current_with_overlay()
            event.accept()
            return
        slider = {"axial": self._slice_z, "coronal": self._slice_y, "sagittal": self._slice_x}.get(axis)
        if slider is not None and slider.isEnabled():
            slider.setValue(int(np.clip(slider.value() + step, slider.minimum(), slider.maximum())))
            self._move_localization_point_to_current_slices(axis)
        event.accept()

    def _move_localization_point_to_current_slices(self, changed_axis: str | None = None) -> None:
        """Hace que la cruz depositada navegue junto con los cortes activos."""
        if not bool(getattr(self, "_localization_cross_enabled", False)):
            return
        if getattr(self, "_localization_point_zyx", None) is None:
            return
        z, y, x = (int(v) for v in self._localization_point_zyx)
        if changed_axis in (None, "axial"):
            z = int(self._slice_idx.get("axial", z))
        if changed_axis in (None, "coronal"):
            y = int(self._slice_idx.get("coronal", y))
        if changed_axis in (None, "sagittal"):
            x = int(self._slice_idx.get("sagittal", x))
        self._localization_point_zyx = (z, y, x)

    def _on_image_mouse_press(self, event, axis: str):
        # === F2.4: Si modo edición activo, interceptar click en CUALQUIER plano ===
        if self._mask_edit_active:
            if event.button() == Qt.MouseButton.LeftButton:
                # Botón IZQUIERDO = PINTAR
                self._mask_edit_paint_mode = True
                self._apply_brush_stroke(event, axis)
                event.accept()
                return
            elif event.button() == Qt.MouseButton.RightButton:
                # Botón DERECHO = BORRAR
                self._mask_edit_paint_mode = False
                self._apply_brush_stroke(event, axis)
                event.accept()
                return
        
        if event.button() != Qt.MouseButton.LeftButton:
            return
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if not (ctrl or shift):
            return
        self._drag_state = {
            "axis": axis,
            "target": "ct" if ctrl else "spect",
            "pos": event.position().toPoint(),
            "moved": False,
        }
        event.accept()

    def _on_image_mouse_move(self, event, axis: str):
        # === F2.4: Si modo edición activo y botón presionado, pintar/borrar en CUALQUIER plano ===
        if self._mask_edit_active:
            if event.buttons() & Qt.MouseButton.LeftButton:
                self._mask_edit_paint_mode = True
                self._apply_brush_stroke(event, axis)
                event.accept()
                return
            elif event.buttons() & Qt.MouseButton.RightButton:
                self._mask_edit_paint_mode = False
                self._apply_brush_stroke(event, axis)
                event.accept()
                return
        
        if not self._drag_state or self._drag_state.get("axis") != axis:
            return
        pos = event.position().toPoint()
        old = self._drag_state["pos"]
        dx = int(pos.x() - old.x())
        dy = int(pos.y() - old.y())
        if dx == 0 and dy == 0:
            return
        self._drag_state["moved"] = True
        target = self._drag_state.get("target")
        if target == "ct" and self._ct_auto_registered is not None:
            dz, dyw, dxw = self._drag_delta_to_world_zyx(axis, dx, dy)
            self._nudge_z.setValue(float(np.clip(self._nudge_z.value() + dz, -64.0, 64.0)))
            self._nudge_y.setValue(float(np.clip(self._nudge_y.value() + dyw, -64.0, 64.0)))
            self._nudge_x.setValue(float(np.clip(self._nudge_x.value() + dxw, -64.0, 64.0)))
            self._status.setText(
                "Ctrl+arrastre (triangulado): "
                f"CT Δ(z,y,x)=({self._nudge_z.value():.1f},{self._nudge_y.value():.1f},{self._nudge_x.value():.1f}) px"
            )
        else:
            # Shift+arrastre en SPECT: igual que Ctrl+arrastre en CT,
            # aplicar desplazamiento global 3D para que todas las vistas
            # se ajusten juntas a la nueva posición.
            dz, dyw, dxw = self._drag_delta_to_world_zyx(axis, dx, dy)
            self._spect_view_offset["axial"] = float(np.clip(self._spect_view_offset.get("axial", 0.0) + dz, -64.0, 64.0))
            self._spect_view_offset["coronal"] = float(np.clip(self._spect_view_offset.get("coronal", 0.0) + dyw, -64.0, 64.0))
            self._spect_view_offset["sagittal"] = float(np.clip(self._spect_view_offset.get("sagittal", 0.0) + dxw, -64.0, 64.0))
            self._status.setText(
                "Shift+arrastre (triangulado): "
                f"SPECT Δ(z,y,x)=({self._spect_view_offset['axial']:.1f},"
                f"{self._spect_view_offset['coronal']:.1f},"
                f"{self._spect_view_offset['sagittal']:.1f}) px"
            )
            self._render_current_with_overlay()
        self._drag_state["pos"] = pos
        event.accept()

    def _mouse_pos_to_volume_zyx(self, event, axis: str, target: str) -> tuple[int, int, int] | None:
        vol = self._ct_registered if target == "ct" and self._ct_registered is not None else self._current_volume
        if vol is None:
            return None
        shape = tuple(int(v) for v in np.asarray(vol).shape[:3])
        if len(shape) != 3 or min(shape) <= 1:
            return None
        lbl = self._axis_label(axis)
        pm = lbl.pixmap()
        if pm is None or pm.isNull():
            return None
        pos = event.position().toPoint()
        lbl_w, lbl_h = max(1, int(lbl.width())), max(1, int(lbl.height()))
        pm_w, pm_h = max(1, int(pm.width())), max(1, int(pm.height()))
        x0 = max(0, (lbl_w - pm_w) // 2)
        y0 = max(0, (lbl_h - pm_h) // 2)
        px = int(np.clip(pos.x() - x0, 0, pm_w - 1))
        py = int(np.clip(pos.y() - y0, 0, pm_h - 1))

        z = int(np.clip(self._slice_idx.get("axial", shape[0] // 2), 0, shape[0] - 1))
        y = int(np.clip(self._slice_idx.get("coronal", shape[1] // 2), 0, shape[1] - 1))
        x = int(np.clip(self._slice_idx.get("sagittal", shape[2] // 2), 0, shape[2] - 1))

        if axis == "axial":
            x = int(round(px / max(1, pm_w - 1) * (shape[2] - 1)))
            y = int(round(py / max(1, pm_h - 1) * (shape[1] - 1)))
        elif axis == "coronal":
            x = int(round(px / max(1, pm_w - 1) * (shape[2] - 1)))
            z = int(round(py / max(1, pm_h - 1) * (shape[0] - 1)))
        else:
            y = int(round(px / max(1, pm_w - 1) * (shape[1] - 1)))
            z = int(round(py / max(1, pm_h - 1) * (shape[0] - 1)))
        return (int(np.clip(z, 0, shape[0] - 1)), int(np.clip(y, 0, shape[1] - 1)), int(np.clip(x, 0, shape[2] - 1)))

    def _localize_from_view_click(self, event, axis: str, target: str) -> bool:
        zyx = self._mouse_pos_to_volume_zyx(event, axis, target)
        if zyx is None:
            return False
        z, y, x = zyx
        self._localization_point_zyx = (int(z), int(y), int(x))
        
        # Crear VOI mediastino temporal si ya existe el ancla (corazón)
        anchor = getattr(self, "_localization_anchor_zyx", None)
        if anchor is not None:
            # Usar valor del spin si existe, sino default 25mm
            mediastinum_radius = 25.0
            if hasattr(self, "_mediastinum_radius_spin"):
                mediastinum_radius = float(self._mediastinum_radius_spin.value())
            self._temp_voi_mediastinum = VOISphere(
                cz=int(z), cy=int(y), cx=int(x),
                radius_mm=mediastinum_radius
            )
        
        self._update_localization_distance()
        for slider, value in ((self._slice_z, z), (self._slice_y, y), (self._slice_x, x)):
            slider.blockSignals(True)
            slider.setValue(int(np.clip(value, slider.minimum(), slider.maximum())))
            slider.blockSignals(False)
        self._slice_idx["axial"] = int(z)
        self._slice_idx["coronal"] = int(y)
        self._slice_idx["sagittal"] = int(x)
        shape = tuple(int(v) for v in np.asarray(self._current_volume).shape[:3]) if self._current_volume is not None else (0, 0, 0)
        if len(shape) == 3 and min(shape) > 0:
            self._slice_z_lbl.setText(f"z {z + 1}/{shape[0]}")
            self._slice_y_lbl.setText(f"y {y + 1}/{shape[1]}")
            self._slice_x_lbl.setText(f"x {x + 1}/{shape[2]}")
        self._status.setText(f"Cruz depositada ({target.upper()}) desde {axis}: Z/Y/X = {z + 1}/{y + 1}/{x + 1}")
        self._render_current_with_overlay()
        return True

    def _on_image_mouse_release(self, event, axis: str):
        if self._drag_state and self._drag_state.get("axis") == axis:
            moved = bool(self._drag_state.get("moved", False))
            target = str(self._drag_state.get("target") or "spect")
            if (not moved) and bool(getattr(self, "_localization_cross_enabled", False)):
                self._localize_from_view_click(event, axis, target=target)
        self._drag_state = None
        event.accept()

    # ============================================================
    # F2.4: Edición manual de máscara CT (brush/erase)
    # ============================================================
    
    def _on_mask_edit_toggled(self, checked: bool):
        """Activa/desactiva el modo de edición de máscara CT."""
        from PyQt6.QtGui import QCursor
        from PyQt6.QtCore import Qt
        
        self._mask_edit_active = checked
        
        if checked:
            # Verificar que exista segmentación CT
            ct_seg = getattr(self, '_ct_segmentation', None)
            if ct_seg is None:
                self._btn_toggle_mask_edit.setChecked(False)
                self._status.setText("❌ F2.4: No hay segmentación CT disponible. Cargar CT y calcular HMR primero.")
                return
            
            # Guardar máscara original si es la primera vez
            if self._mask_edit_original is None:
                self._mask_edit_original = ct_seg.mask_3d.copy()
            
            # Cambiar cursor
            self._axial_lbl.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self._btn_toggle_mask_edit.setText("✏️ Editando...")
            self._btn_toggle_mask_edit.setStyleSheet(
                "background-color:#7c3aed; color:white; font-weight:bold; padding:6px 12px;"
            )
            # Calcular y mostrar volumen inicial
            voxel_count = int(ct_seg.mask_3d.sum())
            spacing = self._spect_spacing_or_default()
            voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
            volume_mm3 = voxel_count * voxel_vol_mm3
            volume_ml = volume_mm3 / 1000.0
            
            self._mask_edit_status.setText(
                f"🟢 Modo edición ACTIVO | 🖌️ Izq=PINTAR 🧹 Der=BORRAR | "
                f"❤️ Volumen: {volume_ml:.1f} mL ({voxel_count} voxels)"
            )
            self._mask_edit_status.setStyleSheet("color:#a78bfa; font-style:normal; font-weight:600;")
            self._status.setText("✏️ F2.4: Modo edición activo. Izq=pintar, Der=borrar en CUALQUIER vista MPR.")
        else:
            # Restaurar cursor
            self._axial_lbl.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            self._btn_toggle_mask_edit.setText("✏️ Editar Máscara")
            self._btn_toggle_mask_edit.setStyleSheet("")
            self._mask_edit_status.setText("Modo edición: INACTIVO")
            self._mask_edit_status.setStyleSheet("color:#9ca3af; font-style:italic; font-size:11px;")
            if self._mask_edit_has_changes:
                self._status.setText("F2.4: Edición pausada. Hay cambios pendientes → 'Aplicar → Recalcular'")
            else:
                self._status.setText("F2.4: Modo edición desactivado.")
    
    def _on_paint_erase_toggled(self, checked: bool):
        """Toggle entre modo pintar y borrar.
        
        NOTA: Ahora el modo se selecciona automáticamente con el mouse:
        - Botón IZQUIERDO = Pintar
        - Botón DERECHO = Borrar
        Este botón sigue funcionando como override manual si se necesita.
        """
        self._mask_edit_paint_mode = checked  # True = pintar, False = borrar
        if checked:
            self._btn_paint_erase.setText("🖌️ Pintar (izq)")
            self._btn_paint_erase.setStyleSheet(
                "background-color:#2563eb; color:white; font-weight:bold;"
            )
        else:
            self._btn_paint_erase.setText("🧹 Borrar (der)")
            self._btn_paint_erase.setStyleSheet(
                "background-color:#dc2626; color:white; font-weight:bold;"
            )
    
    def _apply_brush_stroke(self, event, axis: str):
        """Aplica un stroke del pincel en la posición del mouse.
        
        Funciona en los 3 planos MPR:
        - axial: modifica mask_3d[z], mapea (px,py) -> (y,x)
        - coronal: modifica mask_3d[:,y,:], mapea (px,py) -> (z,x)
        - sagittal: modifica mask_3d[:,:,x], mapea (px,py) -> (z,y)
        """
        ct_seg = getattr(self, '_ct_segmentation', None)
        if ct_seg is None:
            return
        
        lbl = self._axis_label(axis)
        pm = lbl.pixmap()
        if pm is None or pm.isNull():
            return
        
        # Posición del mouse relativa al pixmap
        pos = event.position().toPoint()
        lbl_w, lbl_h = max(1, int(lbl.width())), max(1, int(lbl.height()))
        pm_w, pm_h = max(1, int(pm.width())), max(1, int(pm.height()))
        x0 = max(0, (lbl_w - pm_w) // 2)
        y0 = max(0, (lbl_h - pm_h) // 2)
        
        px = int(np.clip(pos.x() - x0, 0, pm_w - 1))
        py = int(np.clip(pos.y() - y0, 0, pm_h - 1))
        
        # Convertir a coordenadas del volumen según el plano
        mask_shape = ct_seg.mask_3d.shape  # (nz, ny, nx)
        
        # Variables compartidas para undo y volumen
        slice_idx = -1
        current_slice = None
        
        try:
            if axis == "axial":
                z_idx = int(np.clip(self._slice_idx.get("axial", mask_shape[0] // 2), 0, mask_shape[0] - 1))
                slice_idx = z_idx
                v0 = int(round(py / max(1, pm_h - 1) * (mask_shape[1] - 1)))  # Y
                v1 = int(round(px / max(1, pm_w - 1) * (mask_shape[2] - 1)))  # X
                v0 = int(np.clip(v0, 0, mask_shape[1] - 1))
                v1 = int(np.clip(v1, 0, mask_shape[2] - 1))
                r0 = max(1, int(round(int(self._brush_radius_spin.value()) / max(1, pm_h - 1) * (mask_shape[1] - 1))))
                r1 = max(1, int(round(int(self._brush_radius_spin.value()) / max(1, pm_w - 1) * (mask_shape[2] - 1))))
                # Guardar slice para undo
                current_slice = ct_seg.mask_3d[z_idx].copy()
                
                # Crear brush y aplicar
                yy, xx = np.meshgrid(np.arange(-r0, r0 + 1), np.arange(-r1, r1 + 1), indexing='ij')
                dist = np.sqrt((yy / max(r0, 1))**2 + (xx / max(r1, 1))**2)
                brush_mask = dist <= 1.0
                
                y_start = max(0, v0 - r0); y_end = min(mask_shape[1], v0 + r0 + 1)
                x_start = max(0, v1 - r1); x_end = min(mask_shape[2], v1 + r1 + 1)
                by_start = max(0, r0 - v0); by_end = by_start + (y_end - y_start)
                bx_start = max(0, r1 - v1); bx_end = bx_start + (x_end - x_start)
                
                if self._mask_edit_paint_mode:
                    ct_seg.mask_3d[z_idx, y_start:y_end, x_start:x_end] |= brush_mask[by_start:by_end, bx_start:bx_end]
                else:
                    ct_seg.mask_3d[z_idx, y_start:y_end, x_start:x_end] &= ~brush_mask[by_start:by_end, bx_start:bx_end]
                    
            elif axis == "coronal":
                y_idx = int(np.clip(self._slice_idx.get("coronal", mask_shape[1] // 2), 0, mask_shape[1] - 1))
                slice_idx = y_idx
                # En coronal: pantalla (px=X, py=Z) -> volumen (z, x)
                v0 = int(round(py / max(1, pm_h - 1) * (mask_shape[0] - 1)))  # Z
                v1 = int(round(px / max(1, pm_w - 1) * (mask_shape[2] - 1)))  # X
                v0 = int(np.clip(v0, 0, mask_shape[0] - 1))
                v1 = int(np.clip(v1, 0, mask_shape[2] - 1))
                r0 = max(1, int(round(int(self._brush_radius_spin.value()) / max(1, pm_h - 1) * (mask_shape[0] - 1))))
                r1 = max(1, int(round(int(self._brush_radius_spin.value()) / max(1, pm_w - 1) * (mask_shape[2] - 1))))
                # Guardar slice coronal para undo (copiar el slice 2D completo)
                current_slice = ct_seg.mask_3d[:, y_idx, :].copy()
                
                # Crear brush y aplicar sobre slice coronal [:, y_idx, :]
                zz, xx = np.meshgrid(np.arange(-r0, r0 + 1), np.arange(-r1, r1 + 1), indexing='ij')
                dist = np.sqrt((zz / max(r0, 1))**2 + (xx / max(r1, 1))**2)
                brush_mask = dist <= 1.0
                
                z_start = max(0, v0 - r0); z_end = min(mask_shape[0], v0 + r0 + 1)
                x_start = max(0, v1 - r1); x_end = min(mask_shape[2], v1 + r1 + 1)
                bz_start = max(0, r0 - v0); bz_end = bz_start + (z_end - z_start)
                bx_start = max(0, r1 - v1); bx_end = bx_start + (x_end - x_start)
                
                # Validar shapes antes de asignar
                target_shape = (z_end - z_start, x_end - x_start)
                source_shape = (bz_end - bz_start, bx_end - bx_start)
                if target_shape == source_shape:
                    if self._mask_edit_paint_mode:
                        ct_seg.mask_3d[z_start:z_end, y_idx, x_start:x_end] |= brush_mask[bz_start:bz_end, bx_start:bx_end]
                    else:
                        ct_seg.mask_3d[z_start:z_end, y_idx, x_start:x_end] &= ~brush_mask[bz_start:bz_end, bx_start:bx_end]
                else:
                    print(f'[BRUSH-WARN] Shape mismatch coronal: target={target_shape} source={source_shape}')
                    
            else:  # sagittal
                x_idx = int(np.clip(self._slice_idx.get("sagittal", mask_shape[2] // 2), 0, mask_shape[2] - 1))
                slice_idx = x_idx
                # En sagital: pantalla (px=Y, py=Z) -> volumen (z, y)
                v0 = int(round(py / max(1, pm_h - 1) * (mask_shape[0] - 1)))  # Z
                v1 = int(round(px / max(1, pm_w - 1) * (mask_shape[1] - 1)))  # Y
                v0 = int(np.clip(v0, 0, mask_shape[0] - 1))
                v1 = int(np.clip(v1, 0, mask_shape[1] - 1))
                r0 = max(1, int(round(int(self._brush_radius_spin.value()) / max(1, pm_h - 1) * (mask_shape[0] - 1))))
                r1 = max(1, int(round(int(self._brush_radius_spin.value()) / max(1, pm_w - 1) * (mask_shape[1] - 1))))
                # Guardar slice sagital para undo
                current_slice = ct_seg.mask_3d[:, :, x_idx].copy()
                
                # Crear brush y aplicar sobre slice sagital [:, :, x_idx]
                zz, yy = np.meshgrid(np.arange(-r0, r0 + 1), np.arange(-r1, r1 + 1), indexing='ij')
                dist = np.sqrt((zz / max(r0, 1))**2 + (yy / max(r1, 1))**2)
                brush_mask = dist <= 1.0
                
                z_start = max(0, v0 - r0); z_end = min(mask_shape[0], v0 + r0 + 1)
                y_start = max(0, v1 - r1); y_end = min(mask_shape[1], v1 + r1 + 1)
                bz_start = max(0, r0 - v0); bz_end = bz_start + (z_end - z_start)
                by_start = max(0, r1 - v1); by_end = by_start + (y_end - y_start)
                
                # Validar shapes antes de asignar
                target_shape = (z_end - z_start, y_end - y_start)
                source_shape = (bz_end - bz_start, by_end - by_start)
                if target_shape == source_shape:
                    if self._mask_edit_paint_mode:
                        ct_seg.mask_3d[z_start:z_end, y_start:y_end, x_idx] |= brush_mask[bz_start:bz_end, by_start:by_end]
                    else:
                        ct_seg.mask_3d[z_start:z_end, y_start:y_end, x_idx] &= ~brush_mask[bz_start:bz_end, by_start:by_end]
                else:
                    print(f'[BRUSH-WARN] Shape mismatch sagittal: target={target_shape} source={source_shape}')
                    
        except Exception as e:
            print(f'[BRUSH-ERROR] {axis}: {type(e).__name__}: {e}')
            return  # No guardar en undo si falló
        
        # Guardar en stack de undo con formato correcto: (axis, idx, prev_slice)
        if current_slice is not None and slice_idx >= 0:
            self._mask_edit_undo_stack.append((axis, slice_idx, current_slice.copy()))
            if len(self._mask_edit_undo_stack) > 20:
                self._mask_edit_undo_stack.pop(0)
        
        # Marcar cambios pendientes
        self._mask_edit_has_changes = True
        self._btn_undo_mask.setEnabled(True)
        self._btn_reset_mask.setEnabled(True)
        self._btn_apply_mask_edit.setEnabled(True)
        
        # Calcular volumen cardíaco en tiempo real
        voxel_count = int(ct_seg.mask_3d.sum())
        spacing = self._spect_spacing_or_default()
        voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
        volume_mm3 = voxel_count * voxel_vol_mm3
        volume_ml = volume_mm3 / 1000.0
        
        # Nombre del slice según plano
        if axis == "axial":
            slice_name = f"Z={slice_idx+1}/{mask_shape[0]}"
            slice_voxels = int(ct_seg.mask_3d[slice_idx].sum())
        elif axis == "coronal":
            slice_name = f"Y={slice_idx+1}/{mask_shape[1]}"
            slice_voxels = int(ct_seg.mask_3d[:, slice_idx, :].sum())
        else:
            slice_name = f"X={slice_idx+1}/{mask_shape[2]}"
            slice_voxels = int(ct_seg.mask_3d[:, :, slice_idx].sum())
        
        # Actualizar status con volumen cardíaco
        brush_radius_px = int(self._brush_radius_spin.value())
        self._mask_edit_status.setText(
            f"🟢 {axis.capitalize()} {slice_name} | "
            f"Slice: {slice_voxels} vox | Total: {voxel_count} vox | "
            f"❤️ Volumen: {volume_ml:.1f} mL ({volume_mm3:.0f} mm³) | "
            f"{'🖌️ PINTAR' if self._mask_edit_paint_mode else '🧹 BORRAR'} r={brush_radius_px}px"
        )
        
        # Guardar volumen calculado en el objeto VOIAnatomical para reportes
        ct_seg.volume_mm3 = volume_mm3
        
        # Re-renderizar para mostrar cambios en vivo
        self._render_current_with_overlay()
    
    def _undo_mask_edit(self):
        """Deshace la última acción de pincel (funciona en los 3 planos)."""
        if not self._mask_edit_undo_stack:
            return
        
        ct_seg = getattr(self, '_ct_segmentation', None)
        if ct_seg is None:
            return
        
        mask_shape = ct_seg.mask_3d.shape
        entry = self._mask_edit_undo_stack.pop()
        
        # Formato nuevo: (axis, slice_idx, prev_slice)
        if len(entry) == 3:
            plane, idx, prev_slice = entry
            if plane == "axial":
                ct_seg.mask_3d[idx] = prev_slice
                label_txt = f"↩️ Deshecho axial Z={idx+1}"
            elif plane == "coronal":
                ct_seg.mask_3d[:, idx, :] = prev_slice
                label_txt = f"↩️ Deshecho coronal Y={idx+1}"
            else:  # sagittal
                ct_seg.mask_3d[:, :, idx] = prev_slice
                label_txt = f"↩️ Deshecho sagital X={idx+1}"
        else:
            # Formato viejo por compatibilidad: (z_idx, prev_slice)
            z_idx, prev_slice = entry
            ct_seg.mask_3d[z_idx] = prev_slice
            label_txt = f"↩️ Deshecho slice {z_idx+1}"
        
        if not self._mask_edit_undo_stack:
            self._btn_undo_mask.setEnabled(False)
        
        # Recalcular volumen después de deshacer
        voxel_count = int(ct_seg.mask_3d.sum())
        spacing = self._spect_spacing_or_default()
        voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
        volume_mm3 = voxel_count * voxel_vol_mm3
        volume_ml = volume_mm3 / 1000.0
        ct_seg.volume_mm3 = volume_mm3
        
        self._mask_edit_status.setText(
            f"{label_txt} | Voxels: {voxel_count} | "
            f"❤️ Volumen: {volume_ml:.1f} mL ({volume_mm3:.0f} mm³)"
        )
        self._render_current_with_overlay()
    
    def _reset_mask_to_original(self):
        """Restaura la máscara CT a su estado original (pre-edición)."""
        ct_seg = getattr(self, '_ct_segmentation', None)
        if ct_seg is None or self._mask_edit_original is None:
            return
        
        ct_seg.mask_3d[:] = self._mask_edit_original[:]
        self._mask_edit_undo_stack.clear()
        self._mask_edit_has_changes = False
        
        self._btn_undo_mask.setEnabled(False)
        self._btn_reset_mask.setEnabled(False)
        self._btn_apply_mask_edit.setEnabled(False)
        
        # Recalcular volumen después de restaurar
        voxel_count = int(ct_seg.mask_3d.sum())
        spacing = self._spect_spacing_or_default()
        voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
        volume_mm3 = voxel_count * voxel_vol_mm3
        volume_ml = volume_mm3 / 1000.0
        ct_seg.volume_mm3 = volume_mm3
        
        self._mask_edit_status.setText(
            f"🔄 Máscara restaurada | Voxels: {voxel_count} | "
            f"❤️ Volumen: {volume_ml:.1f} mL ({volume_mm3:.0f} mm³)"
        )
        self._status.setText("F2.4: Máscara restaurada a su estado original (segmentación automática).")
        self._render_current_with_overlay()
    
    def _apply_mask_edit_and_recalc(self):
        """Aplica los cambios de edición manual y recalcula HMR."""
        ct_seg = getattr(self, '_ct_segmentation', None)
        if ct_seg is None:
            return
        
        # Confirmar cambios: actualizar original
        self._mask_edit_original = ct_seg.mask_3d.copy()
        self._mask_edit_has_changes = False
        self._mask_edit_undo_stack.clear()
        
        # Actualizar botones
        self._btn_reset_mask.setEnabled(False)
        self._btn_apply_mask_edit.setEnabled(False)
        self._btn_undo_mask.setEnabled(False)
        
        voxel_count = int(ct_seg.mask_3d.sum())
        spacing = self._spect_spacing_or_default()
        voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
        volume_mm3 = voxel_count * voxel_vol_mm3
        volume_ml = volume_mm3 / 1000.0
        ct_seg.volume_mm3 = volume_mm3
        
        # Recalcular centroide de la máscara editada (crítico para posición correcta en MIP)
        from scipy.ndimage import center_of_mass as _com
        mask_arr = np.asarray(ct_seg.mask_3d, dtype=bool)
        if mask_arr.sum() > 0:
            try:
                new_centroid = _com(mask_arr)
                ct_seg.centroid_zyx = (float(new_centroid[0]), float(new_centroid[1]), float(new_centroid[2]))
            except Exception:
                pass  # Mantener centroide anterior si falla
        
        self._mask_edit_status.setText(
            f"✅ Cambios aplicados | Voxels: {voxel_count} | "
            f"❤️ Volumen: {volume_ml:.1f} mL ({volume_mm3:.0f} mm³)"
        )
        self._status.setText(f"✅ F2.4: Máscara editada aplicada ({voxel_count} voxels). Recalculando HMR...")
        
        # Marcar que esta máscara fue editada manualmente (para preservación futura)
        self._mask_was_manually_edited = True
        
        # Desactivar modo edición
        if self._mask_edit_active:
            self._btn_toggle_mask_edit.setChecked(False)
        
        # Re-calcular HMR usando la máscara editada (no re-segmentar)
        self._reuse_edited_segmentation = True
        try:
            self._calculate_hmr_spect()
        finally:
            self._reuse_edited_segmentation = False

    def _export_mask_nifti(self):
        """Exporta la máscara CT segmentada como NIfTI .nii.gz para 3D Slicer."""
        ct_seg = getattr(self, "_ct_segmentation", None)
        if ct_seg is None:
            QMessageBox.warning(self, "SINCRO", "No hay segmentación CT para exportar.")
            return

        try:
            import nibabel as nib
        except ImportError:
            QMessageBox.warning(
                self, "SINCRO",
                "Falta 'nibabel'. Instalar con:\n  pip install nibabel"
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar máscara NIfTI",
            "mascara_miocardio_ct.nii.gz",
            "NIfTI (*.nii.gz *.nii)",
        )
        if not path:
            return

        try:
            mask = np.asarray(ct_seg.mask_3d, dtype=np.uint8)
            spacing = self._spect_spacing_or_default()
            sz, sy, sx = (float(v) for v in spacing)
            affine = np.diag([sx, sy, sz, 1.0])
            img = nib.Nifti1Image(mask, affine)
            img.header.set_zooms((sx, sy, sz))
            nib.save(img, path)
            self._status.setText(f"💾 Máscara exportada: {path}")
            self._mask_edit_status.setText(
                f"💾 Exportada: {path} ({int(mask.sum())} voxels)"
            )
        except Exception as exc:
            QMessageBox.critical(self, "SINCRO", f"Error al exportar:\n{exc}")

    # ============================================================
    # Persistencia de estado CT (guardar/cargar/reiniciar)
    # ============================================================
    
    def _save_ct_state(self):
        """Guarda el estado completo de la segmentación CT en archivo JSON."""
        ct_seg = getattr(self, '_ct_segmentation', None)
        if ct_seg is None:
            QMessageBox.warning(self, "SINCRO", "No hay segmentación CT para guardar.")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar Estado CT",
            f"ct_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON (*.json)",
        )
        if not path:
            return
        
        try:
            import json
            
            state = {
                'version': '2.5',
                'timestamp': datetime.now().isoformat(),
                'mask_3d': ct_seg.mask_3d.tolist(),
                'mask_shape': list(ct_seg.mask_3d.shape),
                'volume_mm3': getattr(ct_seg, 'volume_mm3', 0.0),
                'centroid_zyx': list(ct_seg.centroid_zyx) if hasattr(ct_seg, 'centroid_zyx') else [32, 32, 32],
                'spect_spacing': list(self._spect_spacing_or_default()),
                # Guardar HMR si existe
                'hmr_result': None,
                'pve_result': None,
            }
            
            if self._hmr_result is not None:
                hr = self._hmr_result
                state['hmr_result'] = {
                    'hmr': hr.hmr,
                    'hmr_raw': hr.hmr_raw,
                    'classification': hr.classification,
                    'heart_counts': float(hr.heart_counts),
                    'mediastinum_counts': float(hr.mediastinum_counts),
                    'heart_volume_ml': hr.heart_volume_ml,
                    'method': hr.method,
                }
            
            if self._pve_result is not None:
                pr = self._pve_result
                state['pve_result'] = {
                    'hmr_original': pr.hmr_original,
                    'hmr_pve_corrected': pr.hmr_pve_corrected,
                    'classification_original': pr.classification_original,
                    'classification_corrected': pr.classification_corrected,
                }
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            
            voxel_count = int(ct_seg.mask_3d.sum())
            self._status.setText(f"💾 Estado CT guardado: {path} ({voxel_count} voxels)")
            self._metrics.append(f"[CT-STATE] Guardado en: {path}")
            
        except Exception as exc:
            QMessageBox.critical(self, "SINCRO", f"Error al guardar estado:\n{exc}")
            import traceback
            traceback.print_exc()
    
    def _load_ct_state(self):
        """Carga un estado de segmentación CT guardado previamente."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Cargar Estado CT",
            "",
            "JSON (*.json)",
        )
        if not path:
            return
        
        try:
            import json
            
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Validar versión
            version = state.get('version', '1.0')
            self._metrics.append(f"[CT-STATE] Cargando v{version} desde: {path}")
            
            # Reconstruir máscara 3D
            mask_data = np.array(state['mask_3d'], dtype=bool)
            centroid = tuple(state.get('centroid_zyx', [32, 32, 32]))
            volume_mm3 = state.get('volume_mm3', 0.0)
            
            from mod_SINCRO.core.amyloid_spect import VOIAnatomical
            
            new_ct_seg = VOIAnatomical(
                mask_3d_data=mask_data,
                centroid_zyx=centroid,
                source="ct_segmentation_loaded",
                volume_mm3=volume_mm3,
            )
            
            # Aplicar estado cargado
            self._ct_segmentation = new_ct_seg
            self._mask_edit_original = mask_data.copy()
            self._mask_edit_undo_stack.clear()
            self._mask_edit_has_changes = False
            
            # Restaurar HMR si existe
            if state.get('hmr_result'):
                hr_data = state['hmr_result']
                from mod_SINCRO.core.amyloid_spect import HmrSpectResult
                # Crear un resultado HMR simplificado
                self._lbl_hmr_result.setText(f"HMR = {hr_data['hmr']:.2f} (cargado)")
                self._metrics.append(f"[CT-STATE] HMR restaurado: {hr_data['hmr']:.2f}")
            
            # Habilitar controles
            self._btn_toggle_mask_edit.setEnabled(True)
            self._btn_export_mask_nifti.setEnabled(True)
            self._btn_restart_ct.setEnabled(True)
            self._btn_save_ct_state.setEnabled(True)
            
            # Activar checkbox CT anatómico si no lo está
            if not self._ct_anatomical_check.isChecked():
                self._ct_anatomical_check.setChecked(True)
            
            # Mostrar volumen
            voxel_count = int(mask_data.sum())
            spacing = tuple(state.get('spect_spacing', [6.8, 6.8, 6.8]))
            voxel_vol_mm3 = spacing[0] * spacing[1] * spacing[2]
            volume_ml = (voxel_count * voxel_vol_mm3) / 1000.0
            
            self._mask_edit_status.setText(
                f"📂 CT cargado | {voxel_count} vox | ❤️ {volume_ml:.1f} mL"
            )
            self._status.setText(f"📂 Estado CT cargado desde: {path}")
            self._render_selected_view()
            
        except Exception as exc:
            QMessageBox.critical(self, "SINCRO", f"Error al cargar estado:\n{exc}")
            import traceback
            traceback.print_exc()
    
    def _restart_ct_state(self):
        """Reinicia completamente la segmentación CT, volviendo al estado inicial."""
        ct_seg = getattr(self, '_ct_segmentation', None)
        if ct_seg is None:
            QMessageBox.information(self, "SINCRO", "No hay segmentación CT para reiniciar.")
            return
        
        reply = QMessageBox.question(
            self, "Reiniciar CT",
            "¿Está seguro de BORRAR toda la segmentación CT?\n\n"
            "Esto eliminará:\n"
            "• Máscara editada manualmente\n"
            "• Resultados HMR/PVE\n"
            "• Historial de undo\n\n"
            "No se puede deshacer.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Borrar TODO el estado CT
        self._ct_segmentation = None
        self._hmr_result = None
        self._pve_result = None
        self._mask_edit_original = None
        self._mask_edit_undo_stack.clear()
        self._mask_edit_has_changes = False
        self._reuse_edited_segmentation = False
        self._mask_was_manually_edited = False
        self._auto_cube_bbox_cached = None  # resetear cache del cubo auto
        
        # Deshabilitar controles
        self._btn_toggle_mask_edit.setEnabled(False)
        self._btn_undo_mask.setEnabled(False)
        self._btn_reset_mask.setEnabled(False)
        self._btn_apply_mask_edit.setEnabled(False)
        self._btn_export_mask_nifti.setEnabled(False)
        self._btn_save_ct_state.setEnabled(False)
        self._btn_restart_ct.setEnabled(False)
        
        # Resetear UI
        self._mask_edit_status.setText("Modo manual estable: VOIs esféricas ancladas en A/B")
        self._lbl_hmr_result.setText("HMR-SPECT = N/D · recalcular")
        self._lbl_hmr_result.setStyleSheet(
            "font-size:14px; font-weight:700; color:#ffffff; "
            "background:#000000; padding:6px 12px;"
        )
        
        # Desactivar checkbox
        self._ct_anatomical_check.setChecked(False)
        
        self._status.setText("🗑️ Segmentación CT reiniciada completamente. Recalcule HMR para empezar de nuevo.")
        self._metrics.append("[CT-STATE] Reinicio completo de segmentación CT.")
        self._render_selected_view()

    def _axis_label(self, axis: str) -> QLabel:
        if axis == "axial":
            return self._axial_lbl
        if axis == "coronal":
            return self._cor_lbl
        return self._sag_lbl

    def _drag_delta_to_world_zyx(self, axis: str, dx_px: int, dy_px: int) -> tuple[float, float, float]:
        """Convierte delta de mouse en una vista 2D a delta global z/y/x (px volumen)."""
        vol = self._base_spect_volume if self._base_spect_volume is not None else self._current_volume
        if vol is None:
            return (0.0, 0.0, 0.0)
        shape = tuple(int(v) for v in np.asarray(vol).shape)
        if len(shape) != 3:
            return (0.0, 0.0, 0.0)

        lbl = self._axis_label(axis)
        pm = lbl.pixmap()
        w = int(pm.width()) if pm is not None else max(1, int(lbl.width()))
        h = int(pm.height()) if pm is not None else max(1, int(lbl.height()))
        w = max(1, w)
        h = max(1, h)

        sens = max(0.05, float(getattr(self, "_ct_drag_sensitivity", 1.0)))

        if axis == "axial":
            sx = float(shape[2]) / float(w)
            sy = float(shape[1]) / float(h)
            return (0.0, float(dy_px) * sy * sens, float(dx_px) * sx * sens)
        if axis == "coronal":
            sx = float(shape[2]) / float(w)
            sz = float(shape[0]) / float(h)
            return (float(dy_px) * sz * sens, 0.0, float(dx_px) * sx * sens)
        # sagittal: horizontal=y, vertical=z
        sy = float(shape[1]) / float(w)
        sz = float(shape[0]) / float(h)
        return (float(dy_px) * sz * sens, float(dx_px) * sy * sens, 0.0)

    def _slices_preview_at(self, volume: np.ndarray) -> dict[str, np.ndarray]:
        vol = self._spect_display_volume(np.asarray(volume, dtype=np.float64))
        if vol.ndim != 3:
            return central_slices_preview(vol)
        z = int(np.clip(self._slice_idx.get("axial", vol.shape[0] // 2), 0, vol.shape[0] - 1))
        y = int(np.clip(self._slice_idx.get("coronal", vol.shape[1] // 2), 0, vol.shape[1] - 1))
        x = int(np.clip(self._slice_idx.get("sagittal", vol.shape[2] // 2), 0, vol.shape[2] - 1))
        return {
            "axial": self._apply_zoom_pan_2d(self._window_spect(self._smooth_display_2d(vol[z])), self._spect_zoom_pct, self._spect_pan_px["axial"], order=1),
            "coronal": self._apply_zoom_pan_2d(self._window_spect(self._smooth_display_2d(vol[:, y, :])), self._spect_zoom_pct, self._spect_pan_px["coronal"], order=1),
            "sagittal": self._apply_zoom_pan_2d(self._window_spect(self._smooth_display_2d(vol[:, :, x])), self._spect_zoom_pct, self._spect_pan_px["sagittal"], order=1),
        }

    def _reset_view_offsets(self):
        self._spect_view_offset = {"axial": 0, "coronal": 0, "sagittal": 0}
        self._ct_view_offset = {"axial": 0, "coronal": 0, "sagittal": 0}
        self._spect_pan_px = {"axial": [0, 0], "coronal": [0, 0], "sagittal": [0, 0]}
        self._ct_pan_px = {"axial": [0, 0], "coronal": [0, 0], "sagittal": [0, 0]}
        self._spect_zoom_spin.blockSignals(True)
        self._ct_zoom_spin.blockSignals(True)
        self._spect_zoom_spin.setValue(100)
        self._ct_zoom_spin.setValue(100)
        self._spect_zoom_spin.blockSignals(False)
        self._ct_zoom_spin.blockSignals(False)
        self._spect_range_slider.set_values(0, 100)
        self._spect_zoom_pct = 100
        self._ct_zoom_pct = 100
        self._spect_win_low = 0
        self._status.setText("Offsets, pan, zoom y rango visual SPECT/CT reseteados.")
        self._render_current_with_overlay()

    def _ct_slices_preview_at(self, volume: np.ndarray, *, already_transformed: bool = False) -> dict[str, np.ndarray]:
        vol = np.asarray(volume, dtype=np.float64)
        if already_transformed:
            vol = self._ct_registered_visual_transform(vol)
        else:
            vol = self._ct_transform_3d(vol)
        if vol.ndim != 3:
            return self._slices_preview_at(vol)
        z = int(np.clip(self._slice_idx.get("axial", vol.shape[0] // 2) + self._ct_view_offset.get("axial", 0), 0, vol.shape[0] - 1))
        y = int(np.clip(self._slice_idx.get("coronal", vol.shape[1] // 2) + self._ct_view_offset.get("coronal", 0), 0, vol.shape[1] - 1))
        x = int(np.clip(self._slice_idx.get("sagittal", vol.shape[2] // 2) + self._ct_view_offset.get("sagittal", 0), 0, vol.shape[2] - 1))
        axial = self._window_ct(vol[z])
        coronal = self._window_ct(vol[:, y, :])
        sagittal = self._window_ct(vol[:, :, x])

        if bool(getattr(self, "_ct_visual_trial_mode", False)):
            axial = self._enhance_ct_trial(axial)
            coronal = self._enhance_ct_trial(coronal)
            sagittal = self._enhance_ct_trial(sagittal)

        return {
            "axial": self._apply_zoom_pan_2d(axial, self._ct_zoom_pct, self._ct_pan_px["axial"], order=1),
            "coronal": self._apply_zoom_pan_2d(coronal, self._ct_zoom_pct, self._ct_pan_px["coronal"], order=1),
            "sagittal": self._apply_zoom_pan_2d(sagittal, self._ct_zoom_pct, self._ct_pan_px["sagittal"], order=1),
        }

    def _render_triplet(self, left_title: str, left_arr: np.ndarray, mid_title: str, mid_arr: np.ndarray, right_title: str, right_arr: np.ndarray):
        for lbl, title, arr in (
            (self._axial_lbl, left_title, left_arr),
            (self._cor_lbl, mid_title, mid_arr),
            (self._sag_lbl, right_title, right_arr),
        ):
            pix = self._arr_to_pixmap(self._as_display_2d(arr)).scaled(
                lbl.width(), lbl.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            axis = "axial" if lbl is self._axial_lbl else ("coronal" if lbl is self._cor_lbl else "sagittal")
            self._set_axis_pixmap_with_cross(lbl, pix, axis)
            lbl.setToolTip(title)

    @staticmethod
    def _as_display_2d(arr: np.ndarray) -> np.ndarray:
        img = np.asarray(arr, dtype=np.float64)
        if img.ndim == 4:
            img = np.mean(img, axis=0)
        while img.ndim > 2:
            img = img[img.shape[0] // 2]
        if img.ndim != 2:
            raise ValueError(f"Imagen inválida para display: {img.shape}")
        return img

    @classmethod
    def _montage_2d(cls, arr: np.ndarray, max_slices: int = 12) -> np.ndarray:
        vol = np.asarray(arr, dtype=np.float64)
        if vol.ndim == 4:
            vol = np.mean(vol, axis=0)
        if vol.ndim == 2:
            return vol
        while vol.ndim > 3:
            vol = vol[vol.shape[0] // 2]
        if vol.ndim != 3:
            return cls._as_display_2d(vol)
        n = int(vol.shape[0])
        count = min(max_slices, n)
        idxs = np.linspace(0, n - 1, count).round().astype(int) if count > 1 else np.array([n // 2])
        tiles = [cls._safe_pad_2d(cls._normalize(vol[int(i)])) for i in idxs]
        cols = min(4, len(tiles))
        rows = int(np.ceil(len(tiles) / cols))
        h = max(t.shape[0] for t in tiles)
        w = max(t.shape[1] for t in tiles)
        canvas = np.zeros((rows * h, cols * w), dtype=np.float64)
        for k, tile in enumerate(tiles):
            r = k // cols
            c = k % cols
            canvas[r * h:r * h + tile.shape[0], c * w:c * w + tile.shape[1]] = tile
        return canvas

    @staticmethod
    def _safe_pad_2d(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Tile inválido: {arr.shape}")
        return arr

    def _render_selected_view(self):
        mode = str(self._view_combo.currentData() or "mpr") if hasattr(self, "_view_combo") else "mpr"
        qc_mode = str(self._qc_mode.currentData() or "off") if hasattr(self, "_qc_mode") else "off"
        if self._ct_registered is not None and qc_mode != "off":
            self._render_current_with_overlay()
            return
        if mode == "tomo" and self._recon_bundle is not None and self._recon_bundle.tomo_cuts:
            if self._current_volume is not None:
                self._render_preview(self._current_volume)
            else:
                cuts = self._recon_bundle.tomo_cuts
                self._render_triplet("Axial tomo", cuts["axial"], "Coronal tomo", cuts["coronal"], "Sagital tomo", cuts["sagittal"])
            return
        if mode == "cardiac" and self._recon_bundle is not None and self._recon_bundle.cardiac_axes:
            axes = self._recon_bundle.cardiac_axes
            self._render_triplet(
                "SA montage",
                self._montage_2d(axes["SA"]),
                "HLA montage",
                self._montage_2d(axes["HLA"]),
                "VLA montage",
                self._montage_2d(axes["VLA"]),
            )
            return
        self._render_current_with_overlay()

    def _render_current_with_overlay(self):
        if self._current_volume is None:
            return
        alpha = float(self._blend_slider.value()) / 100.0
        self._blend_lbl.setText(f"{self._blend_slider.value()}%")
        self._qc_split_lbl.setText(f"{self._qc_split_slider.value()}%")
        self._fusion_pct = int(self._fusion_slider.value()) if hasattr(self, "_fusion_slider") else int(getattr(self, "_fusion_pct", 55))
        if hasattr(self, "_fusion_lbl"):
            self._fusion_lbl.setText(f"{self._fusion_pct}%")

        mode = str(self._qc_mode.currentData() or "off")
        pv = None
        ct_prev = None

        if bool(getattr(self, "_ct_grid_trial_mode", False)) and self._ct_volume is not None and mode != "off":
            trial_pair = self._trial_slices_on_ct_grid()
            if trial_pair is not None:
                pv, ct_prev = trial_pair
            else:
                pv = self._slices_preview_at(self._current_volume)
                if self._ct_registered is not None:
                    ct_prev = self._ct_slices_preview_at(np.asarray(self._ct_registered, dtype=np.float64), already_transformed=True)
        else:
            pv = self._slices_preview_at(self._current_volume)
            if self._ct_registered is not None and mode != "off":
                ct_prev = self._ct_slices_preview_at(np.asarray(self._ct_registered, dtype=np.float64), already_transformed=True)

        if ct_prev is not None:
            ax_rgb = self._make_qc_rgb(pv["axial"], ct_prev["axial"], mode, self._qc_split_slider.value())
            co_rgb = self._make_qc_rgb(pv["coronal"], ct_prev["coronal"], mode, self._qc_split_slider.value())
            sa_rgb = self._make_qc_rgb(pv["sagittal"], ct_prev["sagittal"], mode, self._qc_split_slider.value())
            if self._bone_mask is not None and alpha > 0.0:
                bm = self._slices_preview_at(np.asarray(self._bone_mask, dtype=np.float64))
                ax_rgb = self._blend_mask_over_rgb(ax_rgb, bm.get("axial"), alpha)
                co_rgb = self._blend_mask_over_rgb(co_rgb, bm.get("coronal"), alpha)
                sa_rgb = self._blend_mask_over_rgb(sa_rgb, bm.get("sagittal"), alpha)
        elif self._bone_mask is not None:
            bm = self._slices_preview_at(np.asarray(self._bone_mask, dtype=np.float64))
            ax_rgb = self._make_overlay_rgb(pv["axial"], bm.get("axial"), alpha)
            co_rgb = self._make_overlay_rgb(pv["coronal"], bm.get("coronal"), alpha)
            sa_rgb = self._make_overlay_rgb(pv["sagittal"], bm.get("sagittal"), alpha)
        else:
            self._render_preview(self._current_volume)
            # NOTA: No hacer return aquí — seguir para renderizar MIP

        # Actualizar cortes con overlay solo si se generaron RGBs arriba
        if 'ax_rgb' in dir() and 'co_rgb' in dir():
            ax = self._rgb_to_pixmap(ax_rgb).scaled(
                self._axial_lbl.width(), self._axial_lbl.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            co = self._rgb_to_pixmap(co_rgb).scaled(
                self._cor_lbl.width(), self._cor_lbl.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            sa = self._rgb_to_pixmap(sa_rgb).scaled(
                self._sag_lbl.width(), self._sag_lbl.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._set_axis_pixmap_with_cross(self._axial_lbl, ax, "axial")
            self._set_axis_pixmap_with_cross(self._cor_lbl, co, "coronal")
            self._set_axis_pixmap_with_cross(self._sag_lbl, sa, "sagittal")

        # Generar MIP con VOIs (SIEMPRE, incluso sin CT)
        self._render_mip_with_vois()

    def _render_mip_with_vois(self):
        """Configura el widget MIP rotatorio con el volumen y VOIs actuales."""
        if not hasattr(self, "_mip_widget"):
            return
        
        # === Optimización: cachear el volumen transformado para no recalcular en cada scroll ===
        vol_id = id(self._current_volume)
        flip_sig = (
            bool(getattr(self, "_spect_flip_x_test", False)),
            bool(getattr(self, "_spect_flip_y_test", False)),
            bool(getattr(self, "_spect_flip_z_test", False)),
        )
        cache_sig = (vol_id, flip_sig)
        if cache_sig != getattr(self, "_mip_vol_cache_sig", None):
            # Solo recalcular si cambió el volumen o los flips
            if self._current_volume is not None:
                spacing = getattr(self, "_voxel_spacing_mm", (4.0, 4.0, 4.0))
                vol_tx = self._spect_transform_3d(np.asarray(self._current_volume, dtype=np.float64))
                self._mip_widget.set_volume(vol_tx, spacing)
            else:
                self._mip_widget.set_volume(None)
            # Volumen sin filtro
            _uf = getattr(self, "_unfiltered_volume", None)
            if _uf is not None:
                _uf_tx = self._spect_transform_3d(np.asarray(_uf, dtype=np.float64))
                self._mip_widget.set_volume_unfiltered(_uf_tx)
            else:
                self._mip_widget.set_volume_unfiltered(None)
            self._mip_vol_cache_sig = cache_sig
        
        # Pasar colormap (siempre, por si cambió)
        self._mip_widget.set_colormap(self._apply_cmap)
        
        # Pasar VOIs si existen
        voi_heart = None
        voi_med = None
        if self._hmr_result is not None:
            voi_heart = getattr(self._hmr_result, "voi_heart", None)
            voi_med = getattr(self._hmr_result, "voi_mediastinum", None)
        # Fallback: VOIs temporales (mientras se colocan, antes de calcular H/M)
        if voi_heart is None:
            voi_heart = getattr(self, "_temp_voi_heart", None)
        if voi_med is None:
            voi_med = getattr(self, "_temp_voi_mediastinum", None)
        self._mip_widget.set_vois(voi_heart, voi_med)
        
        # === Pasar máscara CT y cubo automático al MIP ===
        ct_seg = getattr(self, "_ct_segmentation", None)
        if ct_seg is not None:
            mask_3d = getattr(ct_seg, "mask_3d", None)
            if mask_3d is not None:
                # La máscara ya vive en la grilla de display (se segmentó sobre
                # la CT registrada transformada): pasar tal cual. Aplicarle
                # _spect_transform_3d otra vez la espejaba en el MIP.
                self._mip_widget.set_mask_3d(np.asarray(mask_3d, dtype=bool))
            else:
                self._mip_widget.set_mask_3d(None)
            
            # Cubo automático: guardar bbox la primera vez que se segmenta
            # (antes de edición manual)
            auto_bbox = getattr(self, "_auto_cube_bbox_cached", None)
            if auto_bbox is None and not getattr(self, "_mask_was_manually_edited", False):
                # Calcular bbox de la máscara actual (es la auto, no editada aún)
                if mask_3d is not None:
                    coords = np.argwhere(np.asarray(mask_3d, dtype=bool))
                    if coords.size > 0:
                        auto_bbox = (
                            int(coords[:, 0].min()), int(coords[:, 0].max()),
                            int(coords[:, 1].min()), int(coords[:, 1].max()),
                            int(coords[:, 2].min()), int(coords[:, 2].max()),
                        )
                        self._auto_cube_bbox_cached = auto_bbox
            self._mip_widget.set_auto_cube_bbox(auto_bbox)
        else:
            self._mip_widget.set_mask_3d(None)
            self._mip_widget.set_auto_cube_bbox(None)

    def _show_fusion_report_layout(self):
        try:
            if self._base_spect_volume is None and self._current_volume is None:
                self._status.setText("Cargar primero un SPECT.")
                return
            spect_vol = np.asarray(
                self._base_spect_volume if self._base_spect_volume is not None else self._current_volume,
                dtype=np.float64,
            )
            spect_vol = self._spect_transform_3d(spect_vol)
            if spect_vol.ndim != 3:
                raise ValueError(f"SPECT inválido para informe: {spect_vol.shape}")

            ct_vol = None
            # Informe final: priorizar resolución nativa de CT (no CT degradada a grilla SPECT).
            if self._ct_volume is not None:
                tmp = self._ct_transform_3d(np.asarray(self._ct_volume, dtype=np.float64))
                ct_vol = tmp if tmp.ndim == 3 else None
            elif self._ct_registered is not None:
                ct_vol = self._ct_transform_3d(np.asarray(self._ct_registered, dtype=np.float64))

            loc_points = self.get_localization_points()
            dlg = FusionReportLayoutDialog(
                self,
                spect_vol=spect_vol,
                ct_vol=ct_vol,
                fusion_pct=int(self._fusion_slider.value()) if hasattr(self, "_fusion_slider") else int(getattr(self, "_fusion_pct", 55)),
                spect_window_fn=self._window_spect,
                ct_window_fn=self._window_ct,
                cmap_fn=self._apply_cmap,
                slice_idx=dict(self._slice_idx),
                localization_points=loc_points,
                display_spacing_zyx=(
                    tuple(self._ct_spacing_zyx)
                    if (ct_vol is not None and self._ct_spacing_zyx is not None and len(self._ct_spacing_zyx) == 3)
                    else (tuple(self._spect_spacing_zyx) if (self._spect_spacing_zyx is not None and len(self._spect_spacing_zyx) == 3) else None)
                ),
                hmr_result=getattr(self, "_hmr_result", None),
                svd_result=getattr(self, "_svd_result", None),
            )
            if hasattr(self, "_metrics") and ct_vol is not None:
                self._metrics.append(
                    "[Informe fusión] CT en resolución nativa para salida final "
                    f"(shape={tuple(int(v) for v in np.asarray(ct_vol).shape)})."
                )
            dlg.exec()
        except Exception as exc:
            self._status.setText(f"Error abriendo vista informe fusión: {exc}")
            QMessageBox.critical(self, "SINCRO", f"No se pudo abrir la vista informe fusión:\n{exc}")

    def _clear_bone_overlay(self):
        self._bone_mask = None
        if self._pre_bone_volume is not None:
            self._current_volume = np.asarray(self._pre_bone_volume, dtype=np.float64)
        self._pre_bone_volume = None
        self._blend_slider.setEnabled(False)
        self._btn_bone.setText("5. Sustracción ósea visual")
        self._status.setText("Sustracción ósea desactivada.")
        self._render_current_with_overlay()

    def _load_spect(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar DICOM SPECT",
            self._last_dir(),
            "DICOM (*.dcm *.DCM);;Todos (*)",
        )
        if not path:
            return
        self._remember_path(path)

        try:
            self._task_progress_start("Cargando SPECT...")
            # Carga inicial siempre rápida en FBP para feedback inmediato en UI.
            method = "fbp"
            selected_method = str(self._recon_combo.currentData() or "fbp")
            self._analysis = run_amyloid_spect_analysis(path, recon_method=method)
            self._task_progress_step(55, "Preparando volumen base...")
            self._dicom_profile_info = self._read_dicom_profile_info(path)
            self._workflow_tag = self._infer_workflow_tag(self._dicom_profile_info)
            self._study_is_gated = int(getattr(self._analysis, "n_gates", 1) or 1) >= 2
            self._spect_spacing_zyx = getattr(self._analysis, "spacing_zyx", None)
            self._spect_affine_ijk_to_lps = getattr(self._analysis, "affine_ijk_to_lps", None)
            self._current_spect_path = path
            self._restore_study_ui_state()
            self._current_volume = np.asarray(self._analysis.volume, dtype=np.float64)
            self._base_spect_volume = np.asarray(self._analysis.volume, dtype=np.float64)
            self._bone_mask = None
            self._pre_bone_volume = None
            self._btn_bone.setText("5. Sustracción ósea visual")
            self._qc_mode.setEnabled(False)
            self._qc_split_slider.setEnabled(False)
            self._fusion_slider.setEnabled(False)
            self._update_slice_controls()
            self._render_preview(self._current_volume)
            self._btn_bone.setEnabled(True)
            self._btn_recon_pipeline.setEnabled(True)
            self._btn_load_ct.setEnabled(True)
            self._btn_load_ct_dir.setEnabled(True)
            self._btn_load_att.setEnabled(True)
            self._btn_register.setEnabled(self._ct_volume is not None)
            self._btn_save_cam_preset.setEnabled(True)
            self._btn_apply_cam_preset.setEnabled(True)
            self._btn_apply_ac.setEnabled(self._att_map_volume is not None)
            self._btn_export_axes_dcm.setEnabled(False)
            self._blend_slider.setEnabled(False)
            self._refresh_gated_controls()
            self._task_progress_step(88, "Renderizando vistas...")
            self._status.setText(
                f"SPECT cargado: {'crudo' if self._analysis.was_raw else 'reconstruido'} · "
                f"shape {self._current_volume.shape} · "
                f"{'gatillado' if self._study_is_gated else 'no gatillado'}"
            )
            self._write_metrics(self._analysis.metrics, self._analysis.notes)
            if selected_method != "fbp":
                self._metrics.append(
                    f"- Carga inicial forzada a FBP rápido (selección actual: {selected_method.upper()}). "
                    "Para método final usar 'Recon + cortes'."
                )
            if self._ct_path:
                self._metrics.append(f"\nCT asociado guardado: {self._ct_path}")
                self._metrics.append(self._matrix_recommendation(self._current_volume.shape))
            self._metrics.append("\n--- Perfil DICOM SPECT ---")
            self._metrics.append(f"- workflow detectado: {self._workflow_tag}")
            self._metrics.append(f"- perfil: {self._profile_compact_text()}")
            if self._workflow_tag == "perf_spect_ct":
                self._apply_camera_profile_preset(auto=True)
            self._persist_ui_state()
            self._task_progress_done("SPECT listo")
        except Exception as exc:
            self._progress.setFormat("Error")
            self._status.setText(f"Error cargando SPECT: {exc}")
            self._metrics.setPlainText(f"Error:\n{exc}")

    def _reconstruct_with_perf_pipeline(self):
        if self._analysis is None:
            self._status.setText("Cargar primero un DICOM SPECT.")
            return
        if self._recon_thread is not None:
            self._status.setText("Reconstrucción en curso...")
            return
        try:
            cfg = self._build_recon_config()
            if not bool(getattr(self._analysis, "was_raw", False)):
                self._metrics.append(
                    "\n[WARN] El estudio cargado ya está reconstruido: los filtros de proyección "
                    "(none/lowpass/butterworth/wiener) no se pueden reaplicar sobre proyecciones crudas."
                )
            self._metrics.append(
                "\n--- Solicitud recon actual ---\n"
                f"- método ungated: {str(cfg.reconstruction_method).upper()}\n"
                f"- filtro ungated: {cfg.ungated_filter.kind} (cutoff={float(cfg.ungated_filter.cutoff):.2f}, orden={int(cfg.ungated_filter.order)})\n"
                f"- método gated: {str(cfg.gated_method or cfg.reconstruction_method).upper()}\n"
                f"- filtro gated: {cfg.gated_filter.kind} (cutoff={float(cfg.gated_filter.cutoff):.2f}, orden={int(cfg.gated_filter.order)})"
            )
            cuts_mode = str(self._cuts_mode_combo.currentData() or "mixed")
            if not self._study_is_gated and cuts_mode == "cardiac":
                cuts_mode = "tomo"
                self._set_combo_by_data(self._cuts_mode_combo, "tomo")
                self._metrics.append(
                    "\nAMYLO no gatillado: modo 'Cardíacos' no disponible; se usan cortes tomográficos."
                )
            self._pending_recon_cfg = cfg
            self._pending_recon_preset_label = self._current_preset_label()
            self._pending_recon_cuts_mode = cuts_mode
            ac_iter_enabled = bool(getattr(cfg, "attenuation_correction", False))
            self._pending_recon_ac_iter = ac_iter_enabled
            self._status.setText(
                "Recon solicitada: "
                f"{str(cfg.reconstruction_method).upper()} · "
                f"filtro={cfg.ungated_filter.kind}({float(cfg.ungated_filter.cutoff):.2f}/{int(cfg.ungated_filter.order)})"
            )

            att_rs_for_iter = None
            att_px_cm = None
            if ac_iter_enabled:
                if self._att_map_volume is None:
                    self._status.setText("AC iterativa: cargar primero ATT MAP o desactivar el toggle.")
                    return
                att_rs_for_iter, notes_rs = resample_volume_to_spect_grid(
                    self._att_map_volume,
                    np.asarray(self._analysis.volume, dtype=np.float64),
                    source_spacing_zyx=self._att_spacing_zyx,
                    spect_spacing_zyx=self._spect_spacing_zyx,
                    source_affine_ijk_to_lps=self._att_affine_ijk_to_lps,
                    spect_affine_ijk_to_lps=self._spect_affine_ijk_to_lps,
                    fill_value=0.0,
                )
                self._att_map_registered = np.asarray(att_rs_for_iter, dtype=np.float64)
                px_mm = float(self._spect_spacing_zyx[2]) if self._spect_spacing_zyx is not None else 6.8
                att_px_cm = max(1e-4, px_mm / 10.0)
                self._metrics.append("\n--- AC iterativa (pre-recon) ---")
                for n in notes_rs:
                    self._metrics.append(f"- {n}")
                self._metrics.append(
                    f"- ATT MAP para iterativa listo: shape={self._att_map_registered.shape}, px={px_mm:.3f} mm, μ-scale={float(getattr(cfg, 'attenuation_mu_scale', 1.0)):.2f}"
                )

            self._set_recon_busy(True)
            self._recon_thread = QThread(self)
            self._recon_worker = _AmyloidReconWorker(
                self._analysis.source_path,
                cfg,
                cuts_mode,
                attenuation_mu_map=att_rs_for_iter,
                attenuation_pixel_size_cm=att_px_cm,
            )
            self._recon_worker.moveToThread(self._recon_thread)
            self._recon_thread.started.connect(self._recon_worker.run)
            self._recon_worker.progress.connect(self._on_recon_progress)
            self._recon_worker.finished.connect(self._on_recon_finished)
            self._recon_worker.failed.connect(self._on_recon_failed)
            self._recon_worker.finished.connect(self._recon_thread.quit)
            self._recon_worker.failed.connect(self._recon_thread.quit)
            self._recon_thread.finished.connect(self._cleanup_recon_thread)
            self._recon_thread.start()
        except Exception as exc:
            self._set_recon_busy(False)
            self._status.setText(f"Error iniciando recon+cortes: {exc}")

    def _cancel_reconstruction(self):
        if self._recon_worker is not None:
            self._recon_worker.cancel()
            self._status.setText("Cancelando reconstrucción...")
            self._progress.setFormat("Cancelando...")

    def _on_recon_progress(self, value: int, message: str):
        self._progress.setValue(int(value))
        self._progress.setFormat(f"{int(value)}% · {message}")
        self._status.setText(message)

    def _on_recon_finished(self, bundle):
        try:
            self._recon_bundle = bundle
            self._base_spect_volume = np.asarray(self._recon_bundle.ungated_volume, dtype=np.float64)
            self._current_volume = np.asarray(self._recon_bundle.ungated_volume, dtype=np.float64)
            # Guardar volumen sin post-filtro para toggle en MIP
            _uf = getattr(self._recon_bundle, "ungated_volume_unfiltered", None)
            if _uf is not None:
                self._unfiltered_volume = np.asarray(_uf, dtype=np.float64)
            else:
                self._unfiltered_volume = None
            self._spect_spacing_zyx = getattr(self._recon_bundle, "spacing_zyx", self._spect_spacing_zyx)
            self._spect_affine_ijk_to_lps = getattr(self._recon_bundle, "affine_ijk_to_lps", self._spect_affine_ijk_to_lps)
            self._bone_mask = None
            self._pre_bone_volume = None
            self._btn_bone.setText("5. Sustracción ósea visual")
            self._update_slice_controls()
            self._btn_export_axes_dcm.setEnabled(bool(self._recon_bundle.cardiac_axes))
            self._view_combo.setEnabled(bool(self._recon_bundle.tomo_cuts or self._recon_bundle.cardiac_axes))
            if self._recon_bundle.cardiac_axes:
                self._set_combo_by_data(self._view_combo, "cardiac")
            elif self._recon_bundle.tomo_cuts:
                self._set_combo_by_data(self._view_combo, "tomo")
            else:
                self._set_combo_by_data(self._view_combo, "mpr")
            self._render_selected_view()

            notes = list(self._recon_bundle.notes)
            sa_shape = self._recon_bundle.cardiac_axes.get("SA", np.zeros((0,))).shape if self._recon_bundle.cardiac_axes else ()
            tomo_keys = ", ".join(self._recon_bundle.tomo_cuts.keys()) if self._recon_bundle.tomo_cuts else "-"
            self._status.setText(
                f"Pipeline perfusión listo · {self._pending_recon_preset_label} · modo={self._pending_recon_cuts_mode} · "
                f"ungated {self._current_volume.shape} · SA {sa_shape}"
            )
            self._metrics.append("\n--- Recon + cortes (perfusión) ---")
            self._metrics.append(f"- preset: {self._pending_recon_preset_label}")
            cfg = self._pending_recon_cfg
            self._metrics.append(
                f"- config: método={cfg.reconstruction_method.upper()}, "
                f"gated={str(cfg.gated_method or cfg.reconstruction_method).upper()}, "
                f"iter={int(cfg.iterative_iterations)}, subsets={int(cfg.osem_subsets)}"
            )
            self._metrics.append(
                f"- AC iterativa: {'ON' if self._pending_recon_ac_iter else 'OFF'}"
            )
            self._metrics.append(f"- modo cortes: {self._pending_recon_cuts_mode}")
            self._metrics.append(f"- tomográficos: {tomo_keys}")
            self._metrics.append(f"- {self._matrix_recommendation(self._current_volume.shape)}")
            self._append_grid_report()
            for n in notes:
                self._metrics.append(f"- {n}")
            self._persist_ui_state()
            self._progress.setValue(100)
            self._progress.setFormat("100% · Reconstrucción y cortes listos")
            self._set_recon_busy(False)
        except Exception as exc:
            self._on_recon_failed(str(exc))

    def _on_recon_failed(self, message: str):
        self._set_recon_busy(False)
        if "cancelada" in str(message).lower():
            self._status.setText("Reconstrucción cancelada.")
            self._progress.setFormat("Cancelada")
        else:
            self._status.setText(f"Error en recon+cortes: {message}")
            self._progress.setFormat("Error")

    def _cleanup_recon_thread(self):
        if self._recon_worker is not None:
            self._recon_worker.deleteLater()
        if self._recon_thread is not None:
            self._recon_thread.deleteLater()
        self._recon_worker = None
        self._recon_thread = None

    def _set_recon_busy(self, busy: bool):
        for widget in (
            self._btn_load,
            self._preset_combo,
            self._recon_combo,
            self._cuts_mode_combo,
            self._btn_load_ct,
            self._btn_load_ct_dir,
            self._btn_register,
            self._btn_save_cam_preset,
            self._btn_apply_cam_preset,
            self._btn_load_att,
            self._btn_apply_ac,
            self._btn_bone,
            self._btn_recon_pipeline,
            self._btn_fusion_layout,
            self._btn_export_axes_dcm,
        ):
            widget.setEnabled(not busy and widget.isEnabled())
        if busy:
            self._progress.setValue(0)
            self._progress.setFormat("0% · Preparando reconstrucción...")
            self._btn_cancel_recon.setEnabled(True)
        else:
            self._btn_load.setEnabled(True)
            self._preset_combo.setEnabled(True)
            self._cuts_mode_combo.setEnabled(True)
            self._on_preset_changed()
            self._btn_load_ct.setEnabled(self._analysis is not None)
            self._btn_load_ct_dir.setEnabled(self._analysis is not None)
            self._btn_load_att.setEnabled(self._analysis is not None)
            self._btn_register.setEnabled(self._ct_volume is not None and self._base_spect_volume is not None)
            self._btn_save_cam_preset.setEnabled(self._analysis is not None)
            self._btn_apply_cam_preset.setEnabled(self._analysis is not None)
            self._btn_apply_ac.setEnabled(self._att_map_volume is not None and self._base_spect_volume is not None)
            self._btn_bone.setEnabled(self._current_volume is not None)
            self._btn_recon_pipeline.setEnabled(self._analysis is not None)
            self._btn_fusion_layout.setEnabled(self._current_volume is not None)
            self._btn_export_axes_dcm.setEnabled(self._recon_bundle is not None and bool(self._recon_bundle.cardiac_axes))
            self._btn_cancel_recon.setEnabled(False)

    def _export_axes_dicom(self):
        if self._recon_bundle is None or not self._recon_bundle.cardiac_axes:
            self._status.setText("No hay ejes cardíacos para exportar.")
            return
        out_dir = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta para ejes DICOM",
            self._last_dir(),
        )
        if not out_dir:
            return
        self._remember_path(out_dir)
        try:
            exported = export_amyloid_cardiac_axes_dicom(
                self._recon_bundle,
                out_dir,
                base_name="AMYLO_SPECT",
            )
            if exported:
                self._metrics.append("\n--- Export DICOM ejes ---")
                for k, v in exported.items():
                    self._metrics.append(f"- {k}: {v}")
                self._status.setText(f"Ejes DICOM exportados ({len(exported)}).")
            else:
                self._status.setText("No se exportó: no hay ejes cardíacos disponibles.")
        except Exception as exc:
            self._status.setText(f"Error exportando ejes DICOM: {exc}")

    def _load_ct(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar DICOM CT",
            self._last_dir(),
            "DICOM (*.dcm *.DCM);;Todos (*)",
        )
        if not path:
            return
        self._remember_path(path)
        self._load_ct_path(path)

    def _load_ct_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Seleccionar carpeta con serie CT",
            self._last_dir(os.path.dirname(self._ct_path) if self._ct_path else ""),
        )
        if not path:
            return
        self._remember_path(path)
        self._load_ct_path(path)

    def _load_ct_path(self, path: str):
        try:
            self._task_progress_start("Cargando CT...")
            ct = load_ct_volume_from_path(path)
            self._task_progress_step(60, "Remapeando CT y actualizando estado...")
            self._ct_volume = np.asarray(ct.volume, dtype=np.float64)
            self._ct_spacing_zyx = getattr(ct, "spacing_zyx", None)
            self._ct_affine_ijk_to_lps = getattr(ct, "affine_ijk_to_lps", None)
            self._ct_path = ct.source_path
            self._ct_registered = None
            self._ct_auto_registered = None
            self._pending_camera_profile_adjust = None
            self._reset_ct_nudge(update_view=False)
            self._qc_mode.setEnabled(False)
            self._qc_split_slider.setEnabled(False)
            self._fusion_slider.setEnabled(False)
            self._btn_register.setEnabled(self._base_spect_volume is not None)
            self._status.setText(f"CT cargado · {ct.series_description} · shape {self._ct_volume.shape}")
            self._metrics.append("\n--- CT cargado ---")
            for note in ct.notes:
                self._metrics.append(f"- {note}")
            self._append_grid_report()
            self._metrics.append("Ejecutar 'Registrar CT↔SPECT'.")
            self._persist_ui_state()
            self._task_progress_done("CT listo")
        except Exception as exc:
            self._progress.setFormat("Error")
            self._status.setText(f"Error cargando CT: {exc}")

    def _load_att_map(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar DICOM ATT MAP",
            self._last_dir(os.path.dirname(self._att_path) if self._att_path else ""),
            "DICOM (*.dcm *.DCM);;Todos (*)",
        )
        if not path:
            return
        self._remember_path(path)
        try:
            self._task_progress_start("Cargando ATT MAP...")
            att = load_attenuation_map_from_path(path)
            self._task_progress_step(70, "Actualizando ATT MAP en estado...")
            self._att_map_volume = np.asarray(att.volume, dtype=np.float64)
            self._att_map_registered = None
            self._att_spacing_zyx = getattr(att, "spacing_zyx", None)
            self._att_affine_ijk_to_lps = getattr(att, "affine_ijk_to_lps", None)
            self._att_path = att.source_path
            self._btn_apply_ac.setEnabled(self._base_spect_volume is not None)
            # Autoaplicar el mu-scale sugerido: tejido blando debe quedar en
            # 0.154 cm^-1 (Tc-99m); sin esto la AC con mapas escalados destruye la imagen.
            nz = self._att_map_volume[self._att_map_volume > 0]
            if nz.size > 100:
                med = float(np.median(nz))
                if med > 1e-9:
                    suggested = 0.154 / med
                    if abs(suggested - float(self._ac_mu_scale_spin.value())) / max(suggested, 1e-9) > 0.05:
                        self._ac_mu_scale_spin.setValue(round(suggested, 4))
                        self._metrics.append(
                            f"- µ-scale autoaplicado: {suggested:.4g} (mediana mapa {med:.4g} → 0.154 cm⁻¹). "
                            "Ajustar manualmente si el mapa no es de tejido blando dominante."
                        )
            self._status.setText(f"ATT MAP cargado · {att.series_description} · shape {self._att_map_volume.shape}")
            self._metrics.append("\n--- ATT MAP cargado ---")
            for note in att.notes:
                self._metrics.append(f"- {note}")
            self._task_progress_done("ATT MAP listo")
        except Exception as exc:
            self._progress.setFormat("Error")
            self._status.setText(f"Error cargando ATT MAP: {exc}")

    def _apply_ac_prototype(self):
        if self._base_spect_volume is None:
            self._status.setText("Cargar/reconstruir primero un SPECT.")
            return
        if self._att_map_volume is None:
            self._status.setText("Cargar primero ATT MAP.")
            return
        try:
            self._task_progress_start("Aplicando AC (prototipo/Chang)...")
            att_rs, notes_rs = resample_volume_to_spect_grid(
                self._att_map_volume,
                self._base_spect_volume,
                source_spacing_zyx=self._att_spacing_zyx,
                spect_spacing_zyx=self._spect_spacing_zyx,
                source_affine_ijk_to_lps=self._att_affine_ijk_to_lps,
                spect_affine_ijk_to_lps=self._spect_affine_ijk_to_lps,
                fill_value=0.0,
            )
            self._task_progress_step(45, "Calculando AC prototipo...")
            self._att_map_registered = np.asarray(att_rs, dtype=np.float64)
            corrected_proto, notes_proto = apply_attenuation_correction_prototype(
                self._base_spect_volume,
                self._att_map_registered,
                mu_scale=0.12,
            )
            self._task_progress_step(70, "Calculando AC Chang...")
            corrected, notes_ac = apply_attenuation_correction_chang(
                self._base_spect_volume,
                self._att_map_registered,
                spect_spacing_zyx=self._spect_spacing_zyx,
                mu_scale=1.0,
                n_angles=36,
            )
            self._task_progress_step(90, "Renderizando volumen corregido...")
            self._current_volume = np.asarray(corrected, dtype=np.float64)
            self._bone_mask = None
            self._pre_bone_volume = None
            self._btn_bone.setText("5. Sustracción ósea visual")
            self._update_slice_controls()
            self._render_current_with_overlay()
            self._status.setText("AC Chang aplicada con ATT MAP (experimental).")
            self._metrics.append("\n--- Corrección por atenuación (prototipo) ---")
            for n in notes_rs:
                self._metrics.append(f"- {n}")
            for n in notes_proto:
                self._metrics.append(f"- [AC prototipo] {n}")
            for n in notes_ac:
                self._metrics.append(f"- [AC Chang] {n}")
            self._task_progress_done("AC aplicada")
        except Exception as exc:
            self._progress.setFormat("Error")
            self._status.setText(f"Error aplicando AC prototipo: {exc}")

    def _register_ct_to_spect(self):
        if self._ct_volume is None or self._base_spect_volume is None:
            self._status.setText("Cargar primero SPECT y CT.")
            return
        try:
            spect_ref = self._spect_transform_3d(self._base_spect_volume)
            ct_ref = self._ct_transform_3d(self._ct_volume)
            self._task_progress_start("Registrando CT↔SPECT...")
            ct_reg, shift_zyx, notes = register_ct_to_spect_rigid(
                ct_ref,
                spect_ref,
                ct_spacing_zyx=self._ct_spacing_zyx,
                spect_spacing_zyx=self._spect_spacing_zyx,
                ct_affine_ijk_to_lps=self._ct_affine_ijk_to_lps,
                spect_affine_ijk_to_lps=self._spect_affine_ijk_to_lps,
                refine_ncc=True,
                ncc_search_radius_zyx=(2, 4, 4),
            )
            self._task_progress_step(70, "Refinando traslación fina...")
            ct_reg, fine_shift_zyx, fine_notes = refine_ct_to_spect_translation(
                ct_reg,
                spect_ref,
                search_radius_zyx=(3, 8, 8),
                ct_bone_hu_threshold=200.0,
                spect_focus_percentile=85.0,
            )
            self._task_progress_step(90, "Renderizando registro...")
            self._ct_auto_registered = np.asarray(ct_reg, dtype=np.float64)
            self._ct_registered = np.asarray(ct_reg, dtype=np.float64)
            self._ct_registered_flip_signature = (
                bool(self._ct_flip_x_test),
                bool(self._ct_flip_y_test),
                bool(self._ct_flip_z_test),
            )
            # Guardar shift total de registro (en píxeles de grilla SPECT)
            self._ct_registration_shift_zyx = (
                float(shift_zyx[0]) + float(fine_shift_zyx[0]),
                float(shift_zyx[1]) + float(fine_shift_zyx[1]),
                float(shift_zyx[2]) + float(fine_shift_zyx[2]),
            )
            self._ct_total_shift_zyx = self._ct_registration_shift_zyx
            # Invalidar caché del modo CT nativa para que use el nuevo shift
            self._invalidate_ct_grid_trial_cache()
            for spin in (self._nudge_z, self._nudge_y, self._nudge_x, self._rot_z, self._rot_y, self._rot_x):
                spin.blockSignals(True)
                spin.setValue(0.0)
                spin.blockSignals(False)
                spin.setEnabled(True)
            self._btn_reset_nudge.setEnabled(True)
            self._btn_reset_rot.setEnabled(True)
            self._qc_mode.setEnabled(True)
            self._qc_split_slider.setEnabled(True)
            self._fusion_slider.setEnabled(True)
            self._set_combo_by_data(self._qc_mode, "fusion")
            self._status.setText(
                "Registro CT↔SPECT listo "
                f"Δ(z,y,x)=({shift_zyx[0]:.1f},{shift_zyx[1]:.1f},{shift_zyx[2]:.1f})"
            )
            self._metrics.append("\n--- Registro CT↔SPECT ---")
            for n in notes:
                self._metrics.append(n)
            for n in fine_notes:
                self._metrics.append(n)
            self._metrics.append("Auto-orientación CT: OFF (se respeta orientación nativa de la TC).")
            self._metrics.append(
                "Refinamiento fino incremental Δ(z,y,x)="
                f"({float(fine_shift_zyx[0]):.1f},{float(fine_shift_zyx[1]):.1f},{float(fine_shift_zyx[2]):.1f}) px."
            )
            self._metrics.append(
                "Prueba orientación SPECT: "
                f"flip X={'ON' if self._spect_flip_x_test else 'OFF'} · "
                f"flip Y={'ON' if self._spect_flip_y_test else 'OFF'} · "
                f"flip Z={'ON' if self._spect_flip_z_test else 'OFF'}"
            )
            self._metrics.append(
                "Prueba orientación CT: "
                f"flip X={'ON' if self._ct_flip_x_test else 'OFF'} · "
                f"flip Y={'ON' if self._ct_flip_y_test else 'OFF'} · "
                f"flip Z={'ON' if self._ct_flip_z_test else 'OFF'}"
            )
            self._append_grid_report()
            if self._pending_camera_profile_adjust:
                self._apply_adjustments_to_ui(self._pending_camera_profile_adjust, auto=True)
            self._render_current_with_overlay()
            self._task_progress_done("Registro CT↔SPECT listo")
        except Exception as exc:
            self._progress.setFormat("Error")
            self._status.setText(f"Error en registro CT↔SPECT: {exc}")

    def _apply_ct_nudge(self):
        if self._ct_auto_registered is None:
            return
        shift = (float(self._nudge_z.value()), float(self._nudge_y.value()), float(self._nudge_x.value()))
        rot = (float(self._rot_z.value()), float(self._rot_y.value()), float(self._rot_x.value()))
        ct = np.asarray(self._ct_auto_registered, dtype=np.float64)
        # Invalidar caché del modo CT nativa (el nudge cambia la alineación)
        self._invalidate_ct_grid_trial_cache()
        if abs(rot[0]) > 1e-6:
            ct = ndi.rotate(ct, angle=rot[0], axes=(1, 2), reshape=False, order=1, mode="nearest")
        if abs(rot[1]) > 1e-6:
            ct = ndi.rotate(ct, angle=rot[1], axes=(0, 2), reshape=False, order=1, mode="nearest")
        if abs(rot[2]) > 1e-6:
            ct = ndi.rotate(ct, angle=rot[2], axes=(0, 1), reshape=False, order=1, mode="nearest")
        self._ct_registered = ndi.shift(ct, shift=shift, order=1, mode="nearest")
        # Guardar shift total (registro + nudge) para modo CT nativa
        self._ct_total_shift_zyx = (
            self._ct_registration_shift_zyx[0] + shift[0],
            self._ct_registration_shift_zyx[1] + shift[1],
            self._ct_registration_shift_zyx[2] + shift[2],
        )
        self._status.setText(
            f"Ajuste CT manual Δ(z,y,x)=({shift[0]:.1f},{shift[1]:.1f},{shift[2]:.1f}) px · "
            f"rot(z,y,x)=({rot[0]:.1f},{rot[1]:.1f},{rot[2]:.1f})°"
        )
        self._render_current_with_overlay()
        self._persist_ui_state()

    def _reset_ct_nudge(self, update_view: bool = True):
        for spin in (self._nudge_z, self._nudge_y, self._nudge_x, self._rot_z, self._rot_y, self._rot_x):
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
        if self._ct_auto_registered is not None:
            self._ct_registered = np.asarray(self._ct_auto_registered, dtype=np.float64)
            # Resetear shift total al shift de registro solo
            self._ct_total_shift_zyx = self._ct_registration_shift_zyx
        self._invalidate_ct_grid_trial_cache()
        if update_view:
            self._render_current_with_overlay()
            self._persist_ui_state()

    def _reset_ct_rotation(self):
        for spin in (self._rot_z, self._rot_y, self._rot_x):
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
        self._apply_ct_nudge()

    def _apply_bone_suppression(self):
        if self._current_volume is None:
            return
        if self._bone_mask is not None:
            self._clear_bone_overlay()
            return
        try:
            self._task_progress_start("Aplicando sustracción ósea...")
            self._pre_bone_volume = np.asarray(self._current_volume, dtype=np.float64)
            ct_vol = None
            if self._ct_check.isChecked():
                ct_vol = self._ct_registered if self._ct_registered is not None else self._ct_volume
            res = apply_visual_bone_suppression(
                self._base_spect_volume if self._base_spect_volume is not None else self._current_volume,
                ct_volume=ct_vol,
            )
            self._task_progress_step(80, "Renderizando sustracción ósea...")
            self._current_volume = np.asarray(res.enhanced_volume, dtype=np.float64)
            self._bone_mask = np.asarray(res.bone_mask, dtype=np.uint8)
            self._blend_slider.setEnabled(True)
            self._btn_bone.setText("5. Quitar sustracción ósea")
            self._render_current_with_overlay()
            self._status.setText(f"Sustracción ósea visual aplicada ({res.method}).")
            self._metrics.append("\n--- Sustracción ósea ---")
            self._task_progress_done("Sustracción ósea lista")
            self._metrics.append("\n".join(res.notes))
        except Exception as exc:
            self._status.setText(f"Error en sustracción ósea: {exc}")

    def _write_metrics(self, metrics: dict, notes: list[str]):
        txt = ["Métricas 3D proxy (experimental):"]
        for k in ("lv_mean", "lv_peak", "bg_mean", "ratio_lv_bg", "heterogeneity_cv", "p80_threshold"):
            if k in metrics:
                txt.append(f"- {k}: {float(metrics[k]):.4f}")
        txt.append("")
        txt.append("Notas:")
        for n in notes:
            txt.append(f"- {n}")
        self._metrics.setPlainText("\n".join(txt))

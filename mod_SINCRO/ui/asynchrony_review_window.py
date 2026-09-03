"""SINCRO - ui.asynchrony_review_window.

Vista de inspección asincrónica lado a lado:
- Izquierda: ROI del flujo principal.
- Derecha: contornos irregulares de medición de pared (ECTb) por gate/slice.
"""
from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (
	QCheckBox,
	QComboBox,
	QDoubleSpinBox,
	QFormLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QPushButton,
	QScrollArea,
	QSpinBox,
	QSplitter,
	QVBoxLayout,
	QWidget,
)

from core.ectb_lv import analyze_lv_ectb
from ui.cine_widget import CineWidget


class AsynchronyReviewWindow(QWidget):
	"""Comparador visual principal vs pared/ECTb."""

	def __init__(self, main_window=None, parent=None):
		super().__init__(parent)
		self._main = main_window
		self._wall_result = None
		self._wall_reason = ""
		self._wall_px_mm = 1.0
		self._syncing = False
		self._show_center_contour = True
		self._show_endo_contour = True
		self._show_epi_contour = True
		self.setWindowTitle("Vista asincronía — comparación principal vs pared")
		self.resize(1300, 760)
		self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

		self.help_label = QLabel(
			"Ambos paneles usan la misma escala de colores y la misma ventana. "
			"Izquierda: ROI del flujo principal.  Derecha: contornos irregulares ECTb (centro/endo/epi) por gate."
		)
		self.help_label.setWordWrap(True)
		self.help_label.setStyleSheet("color:#5b6470; font-size:10pt;")

		self.center_check = QCheckBox("Centro")
		self.center_check.setChecked(True)
		self.endo_check = QCheckBox("Endocardio")
		self.endo_check.setChecked(True)
		self.epi_check = QCheckBox("Epicardio")
		self.epi_check.setChecked(True)
		self.center_check.toggled.connect(self._on_contour_visibility_changed)
		self.endo_check.toggled.connect(self._on_contour_visibility_changed)
		self.epi_check.toggled.connect(self._on_contour_visibility_changed)
		self.goto_valid_btn = QPushButton("Ir a corte con contornos")
		self.goto_valid_btn.setToolTip(
			"El ECTb solo calcula contornos en los cortes con miocardio suficiente. "
			"Este botón salta al corte válido más cercano."
		)
		self.goto_valid_btn.clicked.connect(self._goto_nearest_valid_slice)
		legend_row = QHBoxLayout()
		legend_row.setContentsMargins(0, 0, 0, 4)
		legend_row.addWidget(QLabel("Mostrar contornos:"))
		legend_row.addWidget(self.center_check)
		legend_row.addWidget(self.endo_check)
		legend_row.addWidget(self.epi_check)
		legend_row.addSpacing(12)
		legend_row.addWidget(self.goto_valid_btn)
		legend_row.addStretch(1)

		self.status_label = QLabel("—")
		self.status_label.setWordWrap(True)
		self.status_label.setStyleSheet("color:#a06000;")

		# Decisión: qué geometría alimenta realmente el análisis de fase.
		self.apply_combo = QComboBox()
		self.apply_combo.addItem("Anillo (ROI clásica) — panel izquierdo", "ring")
		self.apply_combo.addItem("Pared irregular según bordes detectados — panel derecho", "ectb_wall")
		self.apply_btn = QPushButton("Aplicar y reprocesar")
		self.apply_btn.setToolTip(
			"Usa la geometría elegida como ROI del análisis de fase y vuelve a procesar el estudio."
		)
		self.apply_btn.clicked.connect(self._apply_selected_roi_source)
		self.applied_label = QLabel("—")
		self.applied_label.setStyleSheet("color:#5b6470;")
		apply_row = QHBoxLayout()
		apply_row.setContentsMargins(0, 0, 0, 4)
		apply_row.addWidget(QLabel("ROI que usa el análisis:"))
		apply_row.addWidget(self.apply_combo, 1)
		apply_row.addWidget(self.apply_btn)
		apply_row.addWidget(self.applied_label)
		apply_row.addStretch(1)

		# Centrado del VI, acá mismo: es donde se ve el problema, así que es donde
		# tiene que poder prenderse y apagarse para comparar sin cambiar de ventana.
		self.cavity_center_check = QCheckBox("Centrar ROI en la cavidad")
		self.cavity_center_check.setToolTip(
			"Apagado: el centro es el centroide de la máscara de miocardio, que se corre "
			"hacia el sector que más capta.\n"
			"Encendido: el centro se recalcula sobre la cavidad.\n"
			"Reprocesa el estudio y actualiza ambos paneles."
		)
		self.cavity_center_check.toggled.connect(self._on_cavity_center_toggled)
		center_row = QHBoxLayout()
		center_row.setContentsMargins(0, 0, 0, 4)
		center_row.addWidget(QLabel("Centro del VI:"))
		center_row.addWidget(self.cavity_center_check)
		center_row.addStretch(1)

		# Centro manual acá mismo: se clickea sobre el panel izquierdo y el resto
		# (radios, ángulo AHA, muestreo polar) se recalcula desde ese punto.
		self.manual_center_check = QCheckBox("Centro manual (clic en cavidad)")
		self.manual_center_check.setToolTip(
			"Clic izquierdo en el panel izquierdo fija el centro de la cavidad; clic derecho lo borra.\n"
			"Mueve el muestreo polar/AHA y los radios (no recorta la máscara: FEVI no cambia por esto)."
		)
		self.manual_center_check.toggled.connect(self._on_manual_center_mode_toggled)
		self.manual_center_all_check = QCheckBox("Aplicar el clic a todos los cortes")
		self.manual_center_all_check.toggled.connect(self._on_manual_center_all_toggled)
		self.manual_center_clear_btn = QPushButton("Limpiar centros")
		self.manual_center_clear_btn.clicked.connect(self._on_manual_center_clear)
		manual_center_row = QHBoxLayout()
		manual_center_row.setContentsMargins(0, 0, 0, 4)
		manual_center_row.addWidget(self.manual_center_check)
		manual_center_row.addWidget(self.manual_center_all_check)
		manual_center_row.addWidget(self.manual_center_clear_btn)
		manual_center_row.addStretch(1)

		self.cine_main = CineWidget(self)
		self.cine_main.setToolTip("Flujo principal con ROIs actuales.")
		self.cine_main.set_controls_visible(True)

		self.cine_wall = CineWidget(self)
		self.cine_wall.setToolTip("Vista pared/ECTb con contornos irregulares por gate.")
		self.cine_wall.set_controls_visible(False)

		splitter = QSplitter(Qt.Orientation.Horizontal, self)
		splitter.addWidget(self.cine_main)
		splitter.addWidget(self.cine_wall)
		splitter.setStretchFactor(0, 1)
		splitter.setStretchFactor(1, 1)
		splitter.setSizes([650, 650])
		self.splitter = splitter

		# Mapa de fase coloreado (resultado real de asincronía): se muestra el PNG
		# que la main renderiza (polar_clinico.png = bullseye + histograma + métricas).
		# Es el "mapa de fase" que el usuario quiere ver acá, no en la principal.
		self.phase_map_view = QLabel("Sin mapa de fase: procesá un estudio.")
		self.phase_map_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.phase_map_view.setStyleSheet("color:#5b6470; background:#0b1220;")
		self._phase_map_pixmap = None
		phase_map_scroll = QScrollArea(self)
		phase_map_scroll.setWidgetResizable(True)
		phase_map_scroll.setWidget(self.phase_map_view)
		phase_map_scroll.setMinimumHeight(160)
		self.phase_map_scroll = phase_map_scroll

		viewers_split = QSplitter(Qt.Orientation.Vertical, self)
		viewers_split.addWidget(splitter)
		viewers_split.addWidget(phase_map_scroll)
		viewers_split.setStretchFactor(0, 3)
		viewers_split.setStretchFactor(1, 2)
		viewers_split.setSizes([460, 300])
		self.viewers_split = viewers_split

		content = QWidget(self)
		content_layout = QVBoxLayout(content)
		content_layout.setContentsMargins(6, 6, 6, 6)
		content_layout.addWidget(self.help_label)
		content_layout.addLayout(legend_row)
		content_layout.addLayout(apply_row)
		content_layout.addLayout(center_row)
		content_layout.addLayout(manual_center_row)
		content_layout.addWidget(self.status_label)
		content_layout.addWidget(viewers_split)

		sidebar = self._build_sync_sidebar()

		root = QHBoxLayout(self)
		root.setContentsMargins(0, 0, 0, 0)
		root.addWidget(content, 1)
		root.addWidget(sidebar, 0)

		self._wire_sync()
		self.cine_main.centerPicked.connect(self._on_center_picked)
		self.sync_from_main()

	def _refresh_applied_label(self):
		"""Muestra qué geometría está usando hoy el análisis principal."""
		main = self._main
		source = str(main.roi_source()) if main is not None and hasattr(main, "roi_source") else "ring"
		index = self.apply_combo.findData(source)
		if index >= 0 and index != self.apply_combo.currentIndex():
			self.apply_combo.setCurrentIndex(index)
		name = "pared ECTb" if source == "ectb_wall" else "anillo clásico"
		self.applied_label.setText(f"activa: {name}")
		self._sync_cavity_center_check()

	def _sync_cavity_center_check(self):
		"""Refleja el estado real del centrado sin disparar un reproceso."""
		main = self._main
		if main is None or not hasattr(main, "cavity_center_enabled"):
			return
		enabled = bool(main.cavity_center_enabled())
		if bool(self.cavity_center_check.isChecked()) == enabled:
			return
		self.cavity_center_check.blockSignals(True)
		self.cavity_center_check.setChecked(enabled)
		self.cavity_center_check.blockSignals(False)

	def _on_cavity_center_toggled(self, checked: bool):
		"""Cambia el criterio de centrado desde esta ventana y reprocesa."""
		main = self._main
		if main is None or not hasattr(main, "set_cavity_center_enabled"):
			self._set_status("No hay ventana principal para cambiar el centrado.")
			return
		# set_cavity_center_enabled dispara el toggled de la principal, que ya
		# reprocesa; si el valor no cambió no hay nada que hacer.
		if not main.set_cavity_center_enabled(bool(checked)):
			return
		modo = "cavidad" if bool(checked) else "centroide de miocardio"
		if getattr(main, "study", None) is None:
			self._set_status(f"Centro del VI: {modo}. Cargá un estudio para procesarlo.")
			return
		self._set_status(f"Centro del VI: {modo}. Reprocesando...", ok=True)

	def _on_manual_center_mode_toggled(self, checked: bool):
		"""Prende/apaga el modo de fijar centro por clic en el panel izquierdo."""
		self.cine_main.set_center_pick_mode(bool(checked))
		if checked:
			self._set_status(
				"Centro manual activo: clic izquierdo en la cavidad (clic derecho borra).", ok=True
			)
		else:
			self._set_status("Centro manual desactivado.")

	def _on_manual_center_all_toggled(self, checked: bool):
		"""Refleja en la ventana principal si el clic aplica a todos los cortes."""
		main = self._main
		chk = getattr(main, "manual_center_all_check", None) if main is not None else None
		if chk is not None and bool(chk.isChecked()) != bool(checked):
			chk.setChecked(bool(checked))

	def _on_manual_center_clear(self):
		"""Borra todos los centros manuales vía la ventana principal."""
		main = self._main
		if main is None or not hasattr(main, "_clear_manual_centers"):
			self._set_status("No hay ventana principal para limpiar centros.")
			return
		main._clear_manual_centers()
		self._push_manual_centers()
		self._set_status("Centros manuales borrados; vuelve el centro automático.", ok=True)

	def _on_center_picked(self, slice_index, center):
		"""Reenvía el clic de centro a la ventana principal y refresca el marcador."""
		main = self._main
		if main is None or not hasattr(main, "_on_center_picked"):
			self._set_status("No hay ventana principal para fijar el centro.")
			return
		main._on_center_picked(slice_index, center)
		self._push_manual_centers()

	def _push_manual_centers(self):
		"""Dibuja en el panel izquierdo los centros manuales de la principal."""
		main = self._main
		centers = getattr(main, "manual_center_per_slice", None) if main is not None else None
		self.cine_main.set_manual_centers(centers)

	def _sync_manual_center_checks(self):
		"""Refleja el 'aplicar a todos' de la principal sin disparar señales."""
		main = self._main
		chk = getattr(main, "manual_center_all_check", None) if main is not None else None
		if chk is None:
			return
		enabled = bool(chk.isChecked())
		if bool(self.manual_center_all_check.isChecked()) == enabled:
			return
		self.manual_center_all_check.blockSignals(True)
		self.manual_center_all_check.setChecked(enabled)
		self.manual_center_all_check.blockSignals(False)

	def _build_sync_sidebar(self) -> QWidget:
		"""Sidebar propio de sincronía: controles de fase y readout de métricas.

		Los controles reflejan los de la ventana principal (que sigue siendo el
		motor). Al tocar 'Reprocesar' se vuelcan los valores al main y se corre
		process_current; el readout se refresca desde main.metrics y la FEVI.
		"""
		panel = QWidget(self)
		panel.setMaximumWidth(280)
		v = QVBoxLayout(panel)
		v.setContentsMargins(6, 6, 6, 6)
		v.setSpacing(8)

		phase_box = QGroupBox("Parámetros de fase")
		form = QFormLayout(phase_box)

		self.seg_method_combo = QComboBox()
		self.seg_method_combo.addItems(["auto", "threshold", "manual"])
		self.threshold_spin = QDoubleSpinBox()
		self.threshold_spin.setRange(0.01, 0.90)
		self.threshold_spin.setSingleStep(0.01)
		self.sigma_spin = QDoubleSpinBox()
		self.sigma_spin.setRange(0.0, 6.0)
		self.sigma_spin.setSingleStep(0.1)
		self.harmonics_spin = QSpinBox()
		self.harmonics_spin.setRange(1, 4)
		self.amp_filter_spin = QDoubleSpinBox()
		self.amp_filter_spin.setRange(0.01, 0.80)
		self.amp_filter_spin.setSingleStep(0.01)
		self.cmap_combo = QComboBox()
		main = self._main
		cmaps = list(getattr(main, "_all_cmaps", []) or [])
		if not cmaps and main is not None and hasattr(main, "cmap_combo"):
			cmaps = [main.cmap_combo.itemText(i) for i in range(main.cmap_combo.count())]
		self.cmap_combo.addItems(cmaps or ["french"])

		form.addRow("Segmentación", self.seg_method_combo)
		form.addRow("Umbral", self.threshold_spin)
		form.addRow("Sigma", self.sigma_spin)
		form.addRow("Armónicos", self.harmonics_spin)
		form.addRow("Filtro amplitud", self.amp_filter_spin)
		form.addRow("Colormap", self.cmap_combo)
		v.addWidget(phase_box)

		self.reprocess_btn = QPushButton("Reprocesar")
		self.reprocess_btn.setToolTip("Vuelca estos parámetros a la ventana principal y reprocesa el estudio.")
		self.reprocess_btn.clicked.connect(self._apply_phase_controls_and_reprocess)
		v.addWidget(self.reprocess_btn)

		# Colormap es barato: se aplica en vivo sin reprocesar.
		self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)

		readout_box = QGroupBox("Resultados en vivo")
		readout_layout = QVBoxLayout(readout_box)
		self.metrics_readout = QLabel("—")
		self.metrics_readout.setWordWrap(True)
		self.metrics_readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
		self.metrics_readout.setStyleSheet("font-size:11pt; color:#1f2937;")
		readout_layout.addWidget(self.metrics_readout)
		v.addWidget(readout_box)

		v.addStretch(1)
		return panel

	def _apply_phase_controls_and_reprocess(self):
		"""Vuelca los parámetros del sidebar a la ventana principal y reprocesa."""
		main = self._main
		if main is None:
			self._set_status("No hay ventana principal para reprocesar.")
			return
		self._write_phase_controls_to_main()
		if getattr(main, "study", None) is None:
			self._set_status("Parámetros aplicados. Cargá un estudio en la ventana principal.")
			return
		try:
			main.process_current()
		except Exception as exc:
			self._set_status(f"No se pudo reprocesar: {exc}")
			return
		self._set_status("Reprocesado con los parámetros de fase actuales.", ok=True)

	def _write_phase_controls_to_main(self):
		"""Copia los valores del sidebar a los widgets del motor (sin reprocesar)."""
		main = self._main
		if main is None:
			return
		combo = getattr(main, "seg_method", None)
		if combo is not None:
			idx = combo.findText(self.seg_method_combo.currentText())
			if idx >= 0:
				combo.setCurrentIndex(idx)
		for src, dst_name in (
			(self.threshold_spin, "threshold_spin"),
			(self.sigma_spin, "sigma_spin"),
			(self.harmonics_spin, "harmonics_spin"),
			(self.amp_filter_spin, "phase_threshold_spin"),
		):
			dst = getattr(main, dst_name, None)
			if dst is not None:
				dst.setValue(src.value())
		# El colormap del sidebar manda el color del mapa de fase renderizado.
		cmap_text = self.cmap_combo.currentText()
		for attr in ("cmap_combo", "report_cmap_phase", "report_cmap_polar_clinico"):
			combo = getattr(main, attr, None)
			if combo is None:
				continue
			idx = combo.findText(cmap_text)
			if idx >= 0 and idx != combo.currentIndex():
				combo.setCurrentIndex(idx)

	def _on_cmap_changed(self, text: str):
		"""Recolorea EN VIVO los visores de esta ventana y deja listo el mapa de fase.

		- SPECT de fondo (cine_main/cine_wall): recolor inmediato acá, no en la main.
		- Mapa de fase (polar_clinico.png): se recolorea al Reprocesar; para eso se
		  vuelca el colormap a los cmaps de reporte de fase de la main.
		"""
		text = str(text)
		# 1) Recolor inmediato del SPECT en esta ventana (dispara _sync_visual_from_main).
		combo_local = getattr(self.cine_main, "cmap_combo", None)
		if combo_local is not None:
			idx = combo_local.findText(text)
			if idx >= 0 and idx != combo_local.currentIndex():
				combo_local.setCurrentIndex(idx)
		# 2) Volcar a la main: cine + colormaps de reporte de fase (recolor al reprocesar).
		main = self._main
		if main is None:
			return
		for attr in ("cmap_combo", "report_cmap_phase", "report_cmap_polar_clinico"):
			combo = getattr(main, attr, None)
			if combo is None:
				continue
			idx = combo.findText(text)
			if idx >= 0 and idx != combo.currentIndex():
				combo.setCurrentIndex(idx)

	def _sync_phase_controls_from_main(self):
		"""Refleja en el sidebar los valores actuales del motor, sin disparar señales."""
		main = self._main
		if main is None:
			return
		pairs = [
			(self.seg_method_combo, getattr(main, "seg_method", None), "combo"),
			(self.threshold_spin, getattr(main, "threshold_spin", None), "spin"),
			(self.sigma_spin, getattr(main, "sigma_spin", None), "spin"),
			(self.harmonics_spin, getattr(main, "harmonics_spin", None), "spin"),
			(self.amp_filter_spin, getattr(main, "phase_threshold_spin", None), "spin"),
			(self.cmap_combo, getattr(main, "cmap_combo", None), "combo"),
		]
		for widget, src, kind in pairs:
			if src is None:
				continue
			widget.blockSignals(True)
			if kind == "spin":
				widget.setValue(src.value())
			else:
				idx = widget.findText(src.currentText())
				if idx >= 0:
					widget.setCurrentIndex(idx)
			widget.blockSignals(False)

	def _refresh_metrics_readout(self):
		"""Muestra PSD / BW / Entropy / FEVI del estudio ya procesado."""
		main = self._main
		metrics = getattr(main, "metrics", None) if main is not None else None
		if not metrics:
			self.metrics_readout.setText("Sin resultados: procesá un estudio.")
			return

		def _fmt(key, unit=""):
			val = metrics.get(key)
			try:
				return f"{float(val):.1f}{unit}"
			except (TypeError, ValueError):
				return "N/D"

		fevi_txt = "N/D"
		try:
			ef = main._estimate_lv_ef()
			if ef and ef.get("available"):
				fevi_txt = f"{float(ef.get('ef_pct')):.1f}%"
		except Exception:
			fevi_txt = "N/D"

		cls = metrics.get("technical_classification", metrics.get("classification", "N/D"))
		self.metrics_readout.setText(
			f"Phase SD: {_fmt('phase_sd', '°')}\n"
			f"Bandwidth: {_fmt('bandwidth', '°')}\n"
			f"Entropy: {_fmt('entropy_normalized_pct', '%')}\n"
			f"FEVI: {fevi_txt}\n"
			f"PSD técnico: {cls} (no dx)"
		)

	def _refresh_phase_map_view(self):
		"""Muestra el mapa de fase coloreado que renderiza la main (polar_clinico.png)."""
		main = self._main
		out_dir = getattr(main, "output_dir", None) if main is not None else None
		path = os.path.join(out_dir, "polar_clinico.png") if out_dir else ""
		if not path or not os.path.exists(path):
			self._phase_map_pixmap = None
			self.phase_map_view.setText("Sin mapa de fase: procesá un estudio.")
			return
		pix = QPixmap(path)
		if pix.isNull():
			self._phase_map_pixmap = None
			self.phase_map_view.setText("No se pudo leer el mapa de fase.")
			return
		self._phase_map_pixmap = pix
		self._apply_phase_map_scale()

	def _apply_phase_map_scale(self):
		"""Escala el mapa de fase al ancho disponible manteniendo proporción."""
		pix = self._phase_map_pixmap
		if pix is None:
			return
		vp = self.phase_map_scroll.viewport().width() if hasattr(self, "phase_map_scroll") else 0
		target_w = max(320, vp - 4) if vp > 0 else 800
		self.phase_map_view.setPixmap(pix.scaledToWidth(int(target_w), Qt.TransformationMode.SmoothTransformation))

	def resizeEvent(self, event):
		super().resizeEvent(event)
		self._apply_phase_map_scale()

	def _apply_selected_roi_source(self):
		"""Fija la geometría elegida en la ventana principal y reprocesa."""
		main = self._main
		if main is None or not hasattr(main, "set_roi_source"):
			self._set_status("No hay ventana principal para aplicar la ROI.")
			return
		source = str(self.apply_combo.currentData() or "ring")
		main.set_roi_source(source)
		self._refresh_applied_label()
		if getattr(main, "study", None) is None:
			self._set_status("ROI seleccionada. Cargá un estudio en la ventana principal para procesarlo.")
			return
		try:
			main.process_current()
		except Exception as exc:
			self._set_status(f"No se pudo reprocesar: {exc}")
			return
		name = "pared ECTb" if source == "ectb_wall" else "anillo clásico"
		self._set_status(f"Análisis reprocesado con {name}.", ok=True)

	def _wire_sync(self):
		self.cine_main.gate_slider.valueChanged.connect(self._sync_gate_slice_from_main)
		self.cine_main.slice_slider.valueChanged.connect(self._sync_gate_slice_from_main)
		self.cine_main.cmap_combo.currentTextChanged.connect(self._sync_visual_from_main)
		self.cine_main.invert_cmap_check.toggled.connect(self._sync_visual_from_main)
		self.cine_main.smooth_slider.valueChanged.connect(self._sync_visual_from_main)
		self.cine_main.interp_combo.currentTextChanged.connect(self._sync_visual_from_main)
		self.cine_main.window_low_slider.valueChanged.connect(self._sync_visual_from_main)
		self.cine_main.window_high_slider.valueChanged.connect(self._sync_visual_from_main)

	def sync_from_main(self):
		"""Actualiza ambos paneles con el estudio activo."""
		if self._main is None:
			self.cine_main.set_cube(None)
			self.cine_wall.set_cube(None)
			return

		self._refresh_applied_label()
		study = getattr(self._main, "study", None)
		cube = getattr(study, "cube", None) if study is not None else None
		self.cine_main.set_cube(cube)
		self.cine_wall.set_cube(cube)
		if cube is not None:
			try:
				rois = self._main._parse_manual_rois()
			except Exception:
				rois = {}
			self.cine_main.set_manual_rois(rois)
			self.cine_wall.set_manual_rois({})
			self._ensure_wall_result()
			self._sync_visual_from_main()
			self._sync_gate_slice_from_main()
		else:
			self.cine_main.set_manual_rois(None)
			self.cine_wall.set_manual_rois(None)
			self.cine_wall.preview.set_overlay_contours([])

		self._push_manual_centers()
		self._sync_manual_center_checks()
		self._sync_phase_controls_from_main()
		self._refresh_metrics_readout()
		self._refresh_phase_map_view()
		self._apply_ui_preferences()

	def _apply_ui_preferences(self):
		if self._main is None:
			return
		show_helpers = bool(getattr(self._main, "_ui_show_helpers", True))
		enable_tooltips = bool(getattr(self._main, "_ui_enable_tooltips", True))
		compact_controls = bool(getattr(self._main, "_ui_compact_controls", False))
		self.cine_main.set_ui_preferences(
			show_helpers=show_helpers,
			enable_tooltips=enable_tooltips,
			compact_controls=compact_controls,
		)
		self.cine_wall.set_ui_preferences(
			show_helpers=False,
			enable_tooltips=enable_tooltips,
			compact_controls=True,
		)

	def _sync_visual_from_main(self, *_args):
		"""Copia la escala de colores y la ventana del panel izquierdo al derecho.

		Así las dos imágenes se ven con el mismo tratamiento y la única diferencia
		visible entre paneles es el ROI/contorno dibujado encima.
		"""
		if self._syncing:
			return
		self._syncing = True
		try:
			self.cine_wall.cmap_combo.blockSignals(True)
			self.cine_wall.cmap_combo.setCurrentText(self.cine_main.cmap_combo.currentText())
			self.cine_wall.cmap_combo.blockSignals(False)

			self.cine_wall.invert_cmap_check.blockSignals(True)
			self.cine_wall.invert_cmap_check.setChecked(self.cine_main.invert_cmap_check.isChecked())
			self.cine_wall.invert_cmap_check.blockSignals(False)

			for src, dst in (
				(self.cine_main.window_low_slider, self.cine_wall.window_low_slider),
				(self.cine_main.window_high_slider, self.cine_wall.window_high_slider),
				(self.cine_main.smooth_slider, self.cine_wall.smooth_slider),
			):
				dst.blockSignals(True)
				dst.setValue(int(src.value()))
				dst.blockSignals(False)

			self.cine_wall._window_low = float(self.cine_main._window_low)
			self.cine_wall._window_high = float(self.cine_main._window_high)
			self.cine_wall._smooth_sigma = float(self.cine_main._smooth_sigma)
			self.cine_wall.set_display_interp(self.cine_main.display_interp())
			self.cine_wall._update_view()
		finally:
			self._syncing = False
		self._refresh_wall_overlay()

	def _on_contour_visibility_changed(self, *_args):
		self._show_center_contour = bool(self.center_check.isChecked())
		self._show_endo_contour = bool(self.endo_check.isChecked())
		self._show_epi_contour = bool(self.epi_check.isChecked())
		self._refresh_wall_overlay()

	def _sync_gate_slice_from_main(self, *_args):
		if self._syncing:
			return
		self._syncing = True
		try:
			g = int(self.cine_main.current_gate_index())
			s = int(self.cine_main.current_slice_index())
			self.cine_wall.gate_slider.setValue(g)
			self.cine_wall.slice_slider.setValue(s)
			self._refresh_wall_overlay(gate_index=g, slice_index=s)
		finally:
			self._syncing = False

	def _ectb_seed_segmentation(self):
		"""Segmentación anular que el ECTb usa como semilla.

		Si el análisis ya está corriendo con la pared ECTb aplicada, `main.seg`
		dejó de ser un anillo y pasó a ser la propia pared irregular. Pasarle eso
		al ECTb lo deja sin cavidad desde donde tirar los rayos y no devuelve
		contornos. Por eso se prefiere siempre la copia anular.
		"""
		main = self._main
		if main is None:
			return None
		base = getattr(main, "seg_ring_base", None)
		if base is not None and str(getattr(base, "method", "")) != "ectb_wall":
			return base
		seg = getattr(main, "seg", None)
		if seg is not None and str(getattr(seg, "method", "")) == "ectb_wall":
			return None
		return seg

	def _ensure_wall_result(self):
		"""Calcula contornos ECTb para overlay irregular por gate/slice."""
		main = self._main
		self._wall_reason = ""
		if main is None:
			self._wall_result = None
			self._wall_reason = "No hay ventana principal asociada."
			return
		study = getattr(main, "study", None)
		seg = self._ectb_seed_segmentation()
		if study is None or seg is None:
			self._wall_result = None
			self._wall_reason = "Falta estudio o segmentación anular: procesá primero en la ventana principal."
			return
		pixel_spacing = getattr(study, "pixel_spacing", None)
		slice_mm = getattr(study, "z_spacing_mm", None)
		cube = getattr(study, "cube", None)
		if not pixel_spacing or slice_mm is None or cube is None:
			self._wall_result = None
			self._wall_reason = "El estudio no trae spacing o cubo gated válido."
			return

		self._wall_px_mm = max(1e-6, float(np.mean([abs(float(pixel_spacing[0])), abs(float(pixel_spacing[1]))])))
		try:
			if hasattr(main, "_apply_gate_dropout_correction") and bool(getattr(main, "gate_dropout_check", None) and main.gate_dropout_check.isChecked()):
				cube, _ = main._apply_gate_dropout_correction(cube, log=False)
			cfg = main.ectb_config() if hasattr(main, "ectb_config") else None
			result = analyze_lv_ectb(
				cube,
				seg,
				(float(pixel_spacing[0]), float(pixel_spacing[1])),
				float(slice_mm),
				cfg,
			)
			if getattr(result, "available", False):
				self._wall_result = result
			else:
				self._wall_result = None
				self._wall_reason = str(getattr(result, "reason", "") or "El ECTb no pudo cuantificar.")
		except Exception as exc:
			self._wall_result = None
			self._wall_reason = f"Error al calcular contornos ECTb: {exc}"

	def _refresh_wall_overlay(self, *, gate_index: int | None = None, slice_index: int | None = None):
		result = self._wall_result
		main = self._main
		if result is None or main is None:
			self.cine_wall.preview.set_overlay_contours([])
			self._set_status(
				str(getattr(self, "_wall_reason", "") or "Sin contornos ECTb: procesá el estudio en la ventana principal.")
			)
			return
		seg = self._ectb_seed_segmentation()
		if seg is None:
			self.cine_wall.preview.set_overlay_contours([])
			self._set_status("Sin segmentación anular cargada: no hay contornos para dibujar.")
			return

		g = int(self.cine_wall.current_gate_index() if gate_index is None else gate_index)
		s = int(self.cine_wall.current_slice_index() if slice_index is None else slice_index)
		contours = self._contours_for_gate_slice(result, seg, s, g)
		self.cine_wall.preview.set_overlay_contours(contours)

		valid = tuple(int(v) for v in getattr(result, "valid_slices", ()) or ())
		if s not in valid:
			self._set_status(
				f"El corte {s + 1} no tiene contornos ECTb. Cortes con contorno: "
				f"{', '.join(str(v + 1) for v in valid) if valid else 'ninguno'}."
			)
		elif not contours:
			self._set_status("Contornos ocultos: activá centro, endocardio o epicardio.")
		else:
			self._set_status(
				f"Contornos ECTb en corte {s + 1}, gate {g + 1} ({len(valid)} cortes con contorno).",
				ok=True,
			)

	def _set_status(self, message: str, *, ok: bool = False):
		self.status_label.setText(str(message))
		self.status_label.setStyleSheet("color:#2e7d32;" if ok else "color:#a06000;")

	def _goto_nearest_valid_slice(self):
		"""Lleva ambos paneles al corte más cercano que tenga contornos ECTb."""
		result = self._wall_result
		if result is None:
			self._set_status("Todavía no hay contornos ECTb calculados.")
			return
		valid = [int(v) for v in getattr(result, "valid_slices", ()) or ()]
		if not valid:
			self._set_status("El ECTb no encontró cortes válidos en este estudio.")
			return
		current = int(self.cine_main.current_slice_index())
		target = min(valid, key=lambda v: (abs(v - current), v))
		self.cine_main.slice_slider.setValue(target)

	def _contours_for_gate_slice(self, result, seg, slice_index: int, gate_index: int):
		if slice_index not in result.valid_slices:
			return []
		row = result.valid_slices.index(slice_index)
		centers = np.asarray(getattr(seg, "center_per_slice", np.empty((0, 2))), dtype=np.float64)
		if slice_index >= centers.shape[0]:
			return []
		cy, cx = float(centers[slice_index, 0]), float(centers[slice_index, 1])
		px_mm = max(float(self._wall_px_mm), 1e-6)
		n_ang = int(result.center_radii_mm.shape[-1])
		angles = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
		if gate_index < 0 or gate_index >= int(result.center_radii_mm.shape[0]):
			return []

		out = []
		for key, color, radii in (
			("center", QColor(255, 210, 63, 180), result.center_radii_mm[gate_index, row]),
			("endo", QColor(80, 160, 255), result.endo_radii_mm[gate_index, row]),
			("epi", QColor(80, 220, 120), result.epi_radii_mm[gate_index, row]),
		):
			if key == "center" and not bool(getattr(self, "_show_center_contour", True)):
				continue
			if key == "endo" and not bool(getattr(self, "_show_endo_contour", True)):
				continue
			if key == "epi" and not bool(getattr(self, "_show_epi_contour", True)):
				continue
			r_px = np.asarray(radii, dtype=np.float64) / px_mm
			xs = cx + r_px * np.cos(angles)
			ys = cy + r_px * np.sin(angles)
			out.append((color, np.stack([ys, xs], axis=1)))
		return out

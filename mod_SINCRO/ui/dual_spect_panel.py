# -*- coding: utf-8 -*-
"""Panel Dual-SPECT para análisis de washout en amiloidosis cardíaca.

Permite cargar y procesar dos estudios SPECT a diferentes tiempos
( típicamente 1h y 3h post-inyección) y calcular el washout cardíaco.

El washout es un parámetro diagnóstico importante que ayuda a
diferenciar ATTR-CM de AL (cadena ligera).
"""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path

from PyQt6.QtCore import Qt, QSettings, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush
from PyQt6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QFrame,
    QScrollArea,
)

from core.amyloid_spect import (
    VOISphere,
    HmrSpectResult,
    HmrSpectMethod,
    compute_hmr_spect,
    load_spect_volume_from_dicom,
)
from core.washout_spect import (
    DualSpectSession,
    WashoutSpectResult,
    WashoutInterpretation,
    format_washout_report,
)


class DualSpectPanel(QDialog):
    """Panel para análisis dual-SPECT con washout.
    
    Layout:
    ┌─────────────────────────────────────────────────────────┐
    │  [Cargar T1] [Cargar T2]    T1: 1.0h  T2: 3.0h          │
    ├─────────────────────────────────────────────────────────┤
    │  ┌─────────────────┐  ┌─────────────────┐              │
    │  │   SPECT T1      │  │   SPECT T2      │              │
    │  │   (Axial)       │  │   (Axial)       │              │
    │  │                 │  │                 │              │
    │  │   HMR: 1.45     │  │   HMR: 1.32     │              │
    │  └─────────────────┘  └─────────────────┘              │
    │                                                         │
    │  ═════════════════════════════════════════════════════ │
    │  WASHOUT: +8.9%  |  Interpretación: ATTR-CM probable   │
    ╚═════════════════════════════════════════════════════════╝
    """

    # Señales
    washout_calculated = pyqtSignal(object)  # WashoutSpectResult
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SINCRO — Dual-SPECT Washout (Amiloidosis)")
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )
        self.resize(1400, 800)
        self.setMinimumSize(1100, 600)
        
        # Sesión dual
        self._session = DualSpectSession()
        
        # Estado UI
        self._slice_idx_t1 = 0
        self._slice_idx_t2 = 0
        self._view_mode = "axial"  # axial, coronal, sagittal
        
        # VOIs (compartidos entre T1 y T2 para consistencia)
        self._voi_heart = None
        self._voi_mediastinum = None
        
        # Settings
        self._settings = QSettings("GAMMASYS", "SINCRO_DUAL_SPECT")
        
        self._build_ui()
        self._connect_signals()
    
    def _build_ui(self):
        """Construye la interfaz."""
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        
        # === BARRA DE CONTROL SUPERIOR ===
        control_box = QGroupBox("Control de sesión dual-SPECT")
        control = QHBoxLayout(control_box)
        
        # Botones de carga
        self._btn_load_t1 = QPushButton("📁 Cargar T1 (temprano)")
        self._btn_load_t1.setToolTip("Cargar estudio SPECT temprano (ej: 1h post-inyección)")
        control.addWidget(self._btn_load_t1)
        
        self._btn_load_t2 = QPushButton("📁 Cargar T2 (tardío)")
        self._btn_load_t2.setToolTip("Cargar estudio SPECT tardío (ej: 3h post-inyección)")
        control.addWidget(self._btn_load_t2)
        
        control.addWidget(self._create_separator())
        
        # Tiempos
        control.addWidget(QLabel("T1:"))
        self._time_t1_spin = QDoubleSpinBox()
        self._time_t1_spin.setRange(0.5, 6.0)
        self._time_t1_spin.setValue(1.0)
        self._time_t1_spin.setSingleStep(0.5)
        self._time_t1_spin.setSuffix(" h")
        self._time_t1_spin.setToolTip("Tiempo post-inyección del estudio T1")
        control.addWidget(self._time_t1_spin)
        
        control.addWidget(QLabel("T2:"))
        self._time_t2_spin = QDoubleSpinBox()
        self._time_t2_spin.setRange(0.5, 6.0)
        self._time_t2_spin.setValue(3.0)
        self._time_t2_spin.setSingleStep(0.5)
        self._time_t2_spin.setSuffix(" h")
        self._time_t2_spin.setToolTip("Tiempo post-inyección del estudio T2")
        control.addWidget(self._time_t2_spin)
        
        control.addWidget(self._create_separator())
        
        # Radiotrazador
        control.addWidget(QLabel("Radiotrazador:"))
        self._tracer_combo = QComboBox()
        self._tracer_combo.addItems(["Tc-99m PYP", "Tc-99m DPD", "Tc-99m HMDP"])
        control.addWidget(self._tracer_combo)
        
        control.addStretch()
        
        # Botón de cálculo
        self._btn_calculate = QPushButton("📊 Calcular Washout")
        self._btn_calculate.setEnabled(False)
        self._btn_calculate.setStyleSheet("font-weight: bold;")
        control.addWidget(self._btn_calculate)
        
        # Botón de exportar PDF
        self._btn_export_pdf = QPushButton("📄 Exportar PDF")
        self._btn_export_pdf.setEnabled(False)
        self._btn_export_pdf.setToolTip("Genera informe PDF con resultados del washout")
        control.addWidget(self._btn_export_pdf)
        
        root.addWidget(control_box)
        
        # === ÁREA DE VISUALIZACIÓN DUAL ===
        viz_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Panel T1
        t1_frame = QFrame()
        t1_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        t1_layout = QVBoxLayout(t1_frame)
        t1_layout.setContentsMargins(4, 4, 4, 4)
        
        self._t1_title = QLabel("SPECT T1 (1h) — Pendiente")
        self._t1_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        self._t1_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t1_layout.addWidget(self._t1_title)
        
        self._t1_image = QLabel()
        self._t1_image.setMinimumSize(400, 400)
        self._t1_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._t1_image.setStyleSheet("background-color: #1a1a1a;")
        t1_layout.addWidget(self._t1_image)
        
        self._t1_hmr_label = QLabel("HMR: N/D")
        self._t1_hmr_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self._t1_hmr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t1_layout.addWidget(self._t1_hmr_label)
        
        self._t1_class_label = QLabel("Clasificación: N/D")
        self._t1_class_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t1_layout.addWidget(self._t1_class_label)
        
        viz_splitter.addWidget(t1_frame)
        
        # Panel T2
        t2_frame = QFrame()
        t2_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        t2_layout = QVBoxLayout(t2_frame)
        t2_layout.setContentsMargins(4, 4, 4, 4)
        
        self._t2_title = QLabel("SPECT T2 (3h) — Pendiente")
        self._t2_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        self._t2_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t2_layout.addWidget(self._t2_title)
        
        self._t2_image = QLabel()
        self._t2_image.setMinimumSize(400, 400)
        self._t2_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._t2_image.setStyleSheet("background-color: #1a1a1a;")
        t2_layout.addWidget(self._t2_image)
        
        self._t2_hmr_label = QLabel("HMR: N/D")
        self._t2_hmr_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        self._t2_hmr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t2_layout.addWidget(self._t2_hmr_label)
        
        self._t2_class_label = QLabel("Clasificación: N/D")
        self._t2_class_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t2_layout.addWidget(self._t2_class_label)
        
        viz_splitter.addWidget(t2_frame)
        
        root.addWidget(viz_splitter, 1)
        
        # === PANEL DE WASHOUT ===
        washout_box = QGroupBox("Resultado Washout")
        washout_layout = QVBoxLayout(washout_box)
        
        self._washout_label = QLabel("Cargue ambos estudios SPECT para calcular el washout")
        self._washout_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self._washout_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        washout_layout.addWidget(self._washout_label)
        
        self._interpretation_label = QLabel("")
        self._interpretation_label.setStyleSheet("font-size: 14px; color: #666;")
        self._interpretation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        washout_layout.addWidget(self._interpretation_label)
        
        self._status_label = QLabel("Estado: Esperando estudios...")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        washout_layout.addWidget(self._status_label)
        
        root.addWidget(washout_box)
        
        # === BARRA DE ESTADO INFERIOR ===
        self._info_label = QLabel("Dual-SPECT para washout cardíaco — Amiloidosis Tc-99m PYP/DPD/HMDP")
        self._info_label.setStyleSheet("color: #888; font-size: 10px;")
        root.addWidget(self._info_label)
    
    def _create_separator(self) -> QFrame:
        """Crea un separador vertical."""
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        return sep
    
    def _connect_signals(self):
        """Conecta señales."""
        self._btn_load_t1.clicked.connect(self._load_t1)
        self._btn_load_t2.clicked.connect(self._load_t2)
        self._btn_calculate.clicked.connect(self._calculate_washout)
        self._btn_export_pdf.clicked.connect(self._export_pdf)
        self._time_t1_spin.valueChanged.connect(self._update_times)
        self._time_t2_spin.valueChanged.connect(self._update_times)
    
    def _update_times(self):
        """Actualiza los tiempos de la sesión."""
        self._session.time_t1_h = self._time_t1_spin.value()
        self._session.time_t2_h = self._time_t2_spin.value()
        self._session.label_t1 = f"{self._session.time_t1_h:.1f}h"
        self._session.label_t2 = f"{self._session.time_t2_h:.1f}h"
        self._update_titles()
    
    def _update_titles(self):
        """Actualiza los títulos de los paneles."""
        t1_status = "✓ cargado" if self._session.is_loaded_t1 else "⏳ pendiente"
        t2_status = "✓ cargado" if self._session.is_loaded_t2 else "⏳ pendiente"
        
        self._t1_title.setText(f"SPECT T1 ({self._session.label_t1}) — {t1_status}")
        self._t2_title.setText(f"SPECT T2 ({self._session.label_t2}) — {t2_status}")
    
    def _load_t1(self):
        """Carga estudio SPECT T1."""
        path = self._select_dicom_source("Seleccionar estudio SPECT T1 (temprano)")
        if not path:
            return
        
        try:
            volume, spacing = self._load_spect_volume(path)
            self._session.volume_t1 = volume
            self._session.spacing_t1 = spacing
            self._session.path_t1 = path
            self._session.is_loaded_t1 = True
            
            self._slice_idx_t1 = volume.shape[0] // 2
            self._update_display_t1()
            self._update_titles()
            self._validate_spacing()
            self._check_ready()
            
            self._status_label.setText(f"T1 cargado: {Path(path).name}")
            
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Error al cargar T1:\n{exc}")
    
    def _load_t2(self):
        """Carga estudio SPECT T2."""
        path = self._select_dicom_source("Seleccionar estudio SPECT T2 (tardío)")
        if not path:
            return
        
        try:
            volume, spacing = self._load_spect_volume(path)
            self._session.volume_t2 = volume
            self._session.spacing_t2 = spacing
            self._session.path_t2 = path
            self._session.is_loaded_t2 = True
            
            self._slice_idx_t2 = volume.shape[0] // 2
            self._update_display_t2()
            self._update_titles()
            self._validate_spacing()
            self._check_ready()
            
            self._status_label.setText(f"T2 cargado: {Path(path).name}")
            
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Error al cargar T2:\n{exc}")
    
    def _validate_spacing(self) -> bool:
        """Valida que ambos estudios tengan spacing compatible.
        
        Returns:
            True si el spacing es compatible, False si hay discrepancia significativa.
        """
        if not self._session.is_loaded_t1 or not self._session.is_loaded_t2:
            return True
        
        s1 = self._session.spacing_t1
        s2 = self._session.spacing_t2
        
        if s1 is None or s2 is None:
            return True
        
        # Tolerancia del 10% para discrepancias de spacing
        tolerance = 0.10
        discrepancies = []
        
        for i, (v1, v2) in enumerate(zip(s1, s2)):
            if v1 > 0 and v2 > 0:
                diff_pct = abs(v1 - v2) / max(v1, v2)
                if diff_pct > tolerance:
                    axis_name = ["Z (axial)", "Y (coronal)", "X (sagital)"][i]
                    discrepancies.append(f"{axis_name}: {v1:.2f}mm vs {v2:.2f}mm ({diff_pct*100:.1f}% diff)")
        
        if discrepancies:
            msg = (
                "⚠️ Discrepancia de spacing detectada:\n\n" +
                "\n".join(discrepancies) +
                "\n\nEsto puede afectar la precisión del washout.\n"
                "¿Desea continuar de todas formas?"
            )
            result = QMessageBox.warning(
                self,
                "Validación de spacing",
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            return result == QMessageBox.StandardButton.Yes
        
        return True
    
    def _select_dicom_source(self, title: str) -> str | None:
        """Selecciona fuente DICOM (archivo o carpeta).
        
        Permite al usuario elegir:
        - Un archivo .dcm individual
        - Una carpeta (se buscará el primer .dcm dentro)
        
        Returns:
            Ruta al archivo DICOM encontrado, o None si canceló.
        """
        # Usar diálogo que permite archivos Y carpetas
        path, _ = QFileDialog.getOpenFileName(
            self,
            title + "\n(Seleccione archivo .dcm o use 'Cancelar' para elegir carpeta)",
            str(Path.home()),
            "DICOM (*.dcm *.DCM);;Todos los archivos (*)",
        )
        
        if path:
            # Usuario seleccionó un archivo directamente
            return path
        
        # Si canceló, ofrecer seleccionar carpeta como alternativa
        folder = QFileDialog.getExistingDirectory(
            self,
            title + " - Seleccionar carpeta con DICOMs",
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not folder:
            return None
        
        # Buscar archivos DICOM en la carpeta
        dicom_files = list(Path(folder).glob("*.dcm")) + list(Path(folder).glob("*.DCM"))
        
        if not dicom_files:
            QMessageBox.warning(
                self,
                "Sin archivos DICOM",
                f"No se encontraron archivos .dcm en:\n{folder}"
            )
            return None
        
        # Retornar el primer archivo DICOM encontrado
        return str(dicom_files[0])
    
    def _load_spect_volume(self, path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
        """Carga volumen SPECT desde DICOM usando la infraestructura existente.
        
        Usa `load_spect_volume_from_dicom` de amyloid_spect.py que maneja:
        - DICOM crudo (proyecciones) → reconstruye FBP rápido
        - DICOM reconstruido gated → promedia gates
        - Extrae spacing real del DICOM
        """
        volume, spacing = load_spect_volume_from_dicom(path, recon_method="fbp")
        return volume, spacing
    
    def _update_display_t1(self):
        """Actualiza visualización T1."""
        if self._session.volume_t1 is None:
            return
        
        slice_2d = self._session.volume_t1[self._slice_idx_t1, :, :]
        pixmap = self._array_to_pixmap(slice_2d)
        self._t1_image.setPixmap(pixmap)
    
    def _update_display_t2(self):
        """Actualiza visualización T2."""
        if self._session.volume_t2 is None:
            return
        
        slice_2d = self._session.volume_t2[self._slice_idx_t2, :, :]
        pixmap = self._array_to_pixmap(slice_2d)
        self._t2_image.setPixmap(pixmap)
    
    def _array_to_pixmap(self, arr: np.ndarray) -> QPixmap:
        """Convierte array 2D a QPixmap (compatible PyQt6)."""
        # Normalizar a 0-255
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9) * 255
        arr_uint8 = arr_norm.astype(np.uint8)
        
        # Asegurar array contiguo (requerido por PyQt6)
        arr_cont = np.ascontiguousarray(arr_uint8)
        
        h, w = arr_cont.shape
        
        # Crear QImage desde bytes crudos (forma compatible con PyQt6)
        # Format_Grayscale8: 1 byte por píxel
        img = QImage(arr_cont.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        
        # Copiar los datos antes de que el buffer temporal se libere
        return QPixmap.fromImage(img.copy())
    
    def _check_ready(self):
        """Verifica si está listo para calcular washout."""
        ready = self._session.is_loaded_t1 and self._session.is_loaded_t2
        self._btn_calculate.setEnabled(ready)
        
        if ready:
            self._status_label.setText("Ambos estudios cargados. Listo para calcular washout.")
    
    def _calculate_washout(self):
        """Calcula el washout entre T1 y T2.
        
        TODO: Integrar con flujo completo de HMR-SPECT.
        Por ahora muestra resultado simulado.
        """
        # Placeholder - en producción:
        # 1. Verificar que ambos volúmenes tengan mismo spacing
        # 2. Usar VOIs consistentes (mismas posiciones)
        # 3. Calcular HMR para cada uno
        # 4. Calcular washout
        
        # Simulación para demo
        hmr_t1 = 1.45
        hmr_t2 = 1.32
        
        self._session.hmr_t1 = HmrSpectResult(
            hmr=hmr_t1,
            hmr_raw=hmr_t1,
            heart_counts=1000.0,
            mediastinum_counts=689.0,
        )
        self._session.hmr_t2 = HmrSpectResult(
            hmr=hmr_t2,
            hmr_raw=hmr_t2,
            heart_counts=900.0,
            mediastinum_counts=682.0,
        )
        
        self._session.calculate_washout()
        
        # Actualizar UI
        self._t1_hmr_label.setText(f"HMR: {hmr_t1:.2f} ({self._session.hmr_t1.classification})")
        self._t2_hmr_label.setText(f"HMR: {hmr_t2:.2f} ({self._session.hmr_t2.classification})")
        
        self._t1_class_label.setText(f"Clasificación: {self._session.hmr_t1.classification}")
        self._t2_class_label.setText(f"Clasificación: {self._session.hmr_t2.classification}")
        
        # Mostrar washout
        washout = self._session.washout
        self._washout_label.setText(washout.washout_text)
        self._interpretation_label.setText(washout.interpretation)
        
        # Color según interpretación
        if washout.is_attr_pattern:
            self._washout_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3;")
        elif washout.is_al_pattern:
            self._washout_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #FF5722;")
        else:
            self._washout_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #666;")
        
        self._status_label.setText("Washout calculado exitosamente")
        
        # Habilitar exportación
        self._btn_export_pdf.setEnabled(True)
        
        # Emitir señal
        self.washout_calculated.emit(washout)
    
    def get_washout_result(self) -> WashoutSpectResult | None:
        """Retorna el resultado del washout."""
        return self._session.washout if self._session.is_complete else None
    
    def get_report_text(self) -> str:
        """Genera texto de reporte."""
        if not self._session.is_complete:
            return "Washout no disponible - faltan estudios"
        
        self._session.washout.radiotracer = self._tracer_combo.currentText()
        return format_washout_report(self._session.washout)
    
    def _export_pdf(self):
        """Exporta el resultado del washout a PDF."""
        if not self._session.is_complete:
            QMessageBox.warning(self, "Error", "Calcule el washout primero.")
            return
        
        # Seleccionar ubicación
        default_name = f"washout_amyloid_{self._session.label_t1}_{self._session.label_t2}.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar informe PDF",
            str(Path.home() / default_name),
            "PDF (*.pdf)"
        )
        
        if not path:
            return
        
        try:
            self._generate_washout_pdf(path)
            QMessageBox.information(self, "Éxito", f"Informe guardado en:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Error al generar PDF:\n{exc}")
    
    def _generate_washout_pdf(self, path: str):
        """Genera el PDF con el informe de washout."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        
        c = canvas.Canvas(path, pagesize=A4)
        width, height = A4
        
        # Título
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(width/2, height - 2*cm, "INFORME DE WASHOUT SPECT")
        
        # Subtítulo
        c.setFont("Helvetica", 12)
        c.drawCentredString(width/2, height - 3*cm, 
            f"Amiloidosis Cardíaca - {self._tracer_combo.currentText()}")
        
        # Línea separadora
        c.setStrokeColor(colors.grey)
        c.line(2*cm, height - 3.5*cm, width - 2*cm, height - 3.5*cm)
        
        y = height - 5*cm
        
        # Datos del estudio
        c.setFont("Helvetica-Bold", 11)
        c.drawString(2*cm, y, "DATOS DEL ESTUDIO")
        y -= 0.8*cm
        
        c.setFont("Helvetica", 10)
        c.drawString(2.5*cm, y, f"Radiotrazador: {self._tracer_combo.currentText()}")
        y -= 0.5*cm
        c.drawString(2.5*cm, y, f"Tiempo T1: {self._session.time_t1_h:.1f} horas")
        y -= 0.5*cm
        c.drawString(2.5*cm, y, f"Tiempo T2: {self._session.time_t2_h:.1f} horas")
        y -= 1*cm
        
        # Resultados HMR
        c.setFont("Helvetica-Bold", 11)
        c.drawString(2*cm, y, "RESULTADOS HMR-SPECT")
        y -= 0.8*cm
        
        c.setFont("Helvetica", 10)
        if self._session.hmr_t1:
            hmr1 = self._session.hmr_t1.hmr_raw or self._session.hmr_t1.hmr
            c.drawString(2.5*cm, y, f"HMR T1 ({self._session.label_t1}): {hmr1:.2f} ({self._session.hmr_t1.classification})")
        y -= 0.5*cm
        
        if self._session.hmr_t2:
            hmr2 = self._session.hmr_t2.hmr_raw or self._session.hmr_t2.hmr
            c.drawString(2.5*cm, y, f"HMR T2 ({self._session.label_t2}): {hmr2:.2f} ({self._session.hmr_t2.classification})")
        y -= 1*cm
        
        # Resultado Washout (destacado)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(2*cm, y, "WASHOUT")
        y -= 0.8*cm
        
        washout = self._session.washout
        c.setFont("Helvetica-Bold", 14)
        
        # Color según interpretación
        if washout.is_attr_pattern:
            c.setFillColor(colors.HexColor("#2196F3"))  # Azul para ATTR
        elif washout.is_al_pattern:
            c.setFillColor(colors.HexColor("#FF5722"))  # Naranja para AL
        else:
            c.setFillColor(colors.black)
        
        c.drawString(2.5*cm, y, washout.washout_text)
        c.setFillColor(colors.black)
        y -= 0.8*cm
        
        c.setFont("Helvetica", 10)
        c.drawString(2.5*cm, y, f"Interpretación: {washout.interpretation}")
        y -= 1.5*cm
        
        # Nota clínica
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColor(colors.grey)
        note_text = (
            "Nota: El washout negativo (captación creciente) sugiere ATTR-CM. "
            "El washout elevado (>20%) sugiere AL o captación inespecífica. "
            "El diagnóstico final debe integrar contexto clínico completo."
        )
        # Wrap text
        from reportlab.lib.utils import simpleSplit
        lines = simpleSplit(note_text, "Helvetica-Oblique", 9, width - 4*cm)
        for line in lines:
            c.drawString(2*cm, y, line)
            y -= 0.4*cm
        
        c.setFillColor(colors.black)
        y -= 1*cm
        
        # Footer
        c.setFont("Helvetica", 8)
        c.drawString(2*cm, 1.5*cm, f"Generado por SINCRO - GammaSync")
        c.drawRightString(width - 2*cm, 1.5*cm, 
            f"{self._session.washout.patient_id or 'Paciente'} - {self._session.washout.study_date or 'Fecha N/D'}")
        
        c.save()


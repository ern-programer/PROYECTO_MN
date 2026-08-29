"""SINCRO — Diálogo de informe AMYLO unificado (plantillas + HTML/PDF)."""

from __future__ import annotations

import os
import tempfile

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from report.amylo_spect_report import (
    ALL_SECTIONS,
    BUILTIN_TEMPLATES,
    generate_amylo_html,
    generate_amylo_pdf,
    load_custom_templates,
    save_custom_templates,
)


class AmyloReportDialog(QDialog):
    """Elegir/editar plantilla, secciones y salida del informe AMYLO."""

    def __init__(self, panel):
        super().__init__(panel)
        self._panel = panel
        self._settings = QSettings("GAMMASYS", "SINCRO_AMYLO_SPECT")
        self._custom = load_custom_templates(self._settings)
        self._accent = "#38bdf8"

        self.setWindowTitle("SINCRO — Informe AMYLO (HTML / PDF)")
        self.resize(560, 620)

        root = QVBoxLayout(self)
        root.setSpacing(8)

        # --- Plantilla ---
        tpl_box = QGroupBox("Plantilla")
        tpl_lay = QVBoxLayout(tpl_box)
        row = QHBoxLayout()
        row.addWidget(QLabel("Plantilla:"))
        self._tpl_combo = QComboBox()
        self._reload_template_combo()
        self._tpl_combo.currentIndexChanged.connect(self._on_template_selected)
        row.addWidget(self._tpl_combo, 1)
        tpl_lay.addLayout(row)

        row2 = QHBoxLayout()
        self._btn_accent = QPushButton("Color de acento")
        self._btn_accent.clicked.connect(self._pick_accent)
        row2.addWidget(self._btn_accent)
        self._dark_check = QCheckBox("Tema oscuro (HTML)")
        self._dark_check.setChecked(True)
        row2.addWidget(self._dark_check)
        row2.addStretch(1)
        tpl_lay.addLayout(row2)

        row3 = QHBoxLayout()
        btn_save_tpl = QPushButton("💾 Guardar como plantilla…")
        btn_save_tpl.setToolTip("Guarda la combinación actual de secciones/color/tema con un nombre propio.")
        btn_save_tpl.clicked.connect(self._save_template)
        row3.addWidget(btn_save_tpl)
        self._btn_del_tpl = QPushButton("🗑 Eliminar plantilla")
        self._btn_del_tpl.setToolTip("Solo plantillas personalizadas.")
        self._btn_del_tpl.clicked.connect(self._delete_template)
        row3.addWidget(self._btn_del_tpl)
        row3.addStretch(1)
        tpl_lay.addLayout(row3)
        root.addWidget(tpl_box)

        # --- Secciones ---
        sec_box = QGroupBox("Secciones del informe")
        sec_lay = QVBoxLayout(sec_box)
        self._section_checks: dict[str, QCheckBox] = {}
        for key, label in ALL_SECTIONS:
            chk = QCheckBox(label)
            chk.setChecked(True)
            self._section_checks[key] = chk
            sec_lay.addWidget(chk)
        planar_available = self._panel._read_planar_bridge() is not None
        if not planar_available:
            for key in ("planar", "comparativa"):
                self._section_checks[key].setChecked(False)
                self._section_checks[key].setEnabled(False)
                self._section_checks[key].setToolTip(
                    "Sin métricas planares publicadas. Generá primero el informe del módulo "
                    "Amyloidosis Planar para habilitar el informe integrado."
                )
        root.addWidget(sec_box)

        # --- Animaciones / salida ---
        out_box = QGroupBox("Contenido dinámico y salida")
        out_lay = QVBoxLayout(out_box)
        row4 = QHBoxLayout()
        self._gif_check = QCheckBox("Incluir GIFs (MIP rotatorio + barrido axial)")
        self._gif_check.setChecked(True)
        self._gif_check.setToolTip("Los GIFs solo animan en la salida HTML; el PDF incluye un frame estático.")
        row4.addWidget(self._gif_check)
        row4.addWidget(QLabel("Frames:"))
        self._gif_frames_spin = QSpinBox()
        self._gif_frames_spin.setRange(8, 60)
        self._gif_frames_spin.setValue(24)
        row4.addWidget(self._gif_frames_spin)
        row4.addStretch(1)
        out_lay.addLayout(row4)

        row5 = QHBoxLayout()
        row5.addWidget(QLabel("Salida:"))
        self._out_combo = QComboBox()
        self._out_combo.addItem("HTML + PDF", "both")
        self._out_combo.addItem("Solo HTML", "html")
        self._out_combo.addItem("Solo PDF", "pdf")
        row5.addWidget(self._out_combo)
        self._open_check = QCheckBox("Abrir al terminar")
        self._open_check.setChecked(True)
        row5.addWidget(self._open_check)
        row5.addStretch(1)
        out_lay.addLayout(row5)
        root.addWidget(out_box)

        # --- Acciones ---
        btns = QHBoxLayout()
        btn_preview = QPushButton("👁 Vista previa composición fusión")
        btn_preview.clicked.connect(self._panel._show_fusion_report_layout)
        btns.addWidget(btn_preview)
        btns.addStretch(1)
        self._btn_generate = QPushButton("📄 Generar informe")
        self._btn_generate.setStyleSheet(
            "background-color:#16a34a; color:white; font-weight:bold; padding:8px 18px; border-radius:4px;"
        )
        self._btn_generate.clicked.connect(self._generate)
        btns.addWidget(self._btn_generate)
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.reject)
        btns.addWidget(btn_close)
        root.addLayout(btns)

        self._status = QLabel("")
        self._status.setStyleSheet("color:#64748b; font-size:11px;")
        root.addWidget(self._status)

        last_tpl = str(self._settings.value("amylo_report/last_template", "") or "")
        idx = self._tpl_combo.findText(last_tpl)
        self._tpl_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._on_template_selected()

    # ------------------------------------------------------------------
    def _all_templates(self) -> dict[str, dict]:
        merged = dict(BUILTIN_TEMPLATES)
        merged.update(self._custom)
        return merged

    def _reload_template_combo(self):
        current = self._tpl_combo.currentText() if self._tpl_combo.count() else ""
        self._tpl_combo.blockSignals(True)
        self._tpl_combo.clear()
        self._tpl_combo.addItems(list(self._all_templates().keys()))
        idx = self._tpl_combo.findText(current)
        if idx >= 0:
            self._tpl_combo.setCurrentIndex(idx)
        self._tpl_combo.blockSignals(False)

    def _on_template_selected(self):
        name = self._tpl_combo.currentText()
        tpl = self._all_templates().get(name)
        if not tpl:
            return
        self._accent = str(tpl.get("accent", "#38bdf8"))
        self._apply_accent_button()
        self._dark_check.setChecked(bool(tpl.get("dark", True)))
        wanted = set(tpl.get("sections", []))
        for key, chk in self._section_checks.items():
            if chk.isEnabled():
                chk.setChecked(key in wanted)
        self._btn_del_tpl.setEnabled(name in self._custom)

    def _apply_accent_button(self):
        self._btn_accent.setStyleSheet(
            f"background:{self._accent}; color:white; font-weight:bold; padding:4px 12px; border-radius:4px;"
        )

    def _pick_accent(self):
        from PyQt6.QtGui import QColor
        color = QColorDialog.getColor(QColor(self._accent), self, "Color de acento del informe")
        if color.isValid():
            self._accent = color.name()
            self._apply_accent_button()

    def _current_template(self) -> dict:
        return {
            "accent": self._accent,
            "dark": bool(self._dark_check.isChecked()),
            "sections": [k for k, chk in self._section_checks.items() if chk.isChecked()],
        }

    def _save_template(self):
        name, ok = QInputDialog.getText(self, "Guardar plantilla", "Nombre de la plantilla:")
        name = str(name or "").strip()
        if not ok or not name:
            return
        if name in BUILTIN_TEMPLATES:
            QMessageBox.warning(self, "SINCRO", "Ese nombre está reservado para una plantilla builtin.")
            return
        self._custom[name] = self._current_template()
        save_custom_templates(self._settings, self._custom)
        self._reload_template_combo()
        idx = self._tpl_combo.findText(name)
        if idx >= 0:
            self._tpl_combo.setCurrentIndex(idx)
        self._status.setText(f"Plantilla «{name}» guardada.")

    def _delete_template(self):
        name = self._tpl_combo.currentText()
        if name not in self._custom:
            return
        if QMessageBox.question(self, "SINCRO", f"¿Eliminar la plantilla «{name}»?") != QMessageBox.StandardButton.Yes:
            return
        self._custom.pop(name, None)
        save_custom_templates(self._settings, self._custom)
        self._reload_template_combo()
        self._tpl_combo.setCurrentIndex(0)
        self._on_template_selected()

    # ------------------------------------------------------------------
    def _generate(self):
        out_dir = QFileDialog.getExistingDirectory(
            self, "Carpeta de salida del informe", self._panel._last_dir()
        )
        if not out_dir:
            return
        tpl_name = self._tpl_combo.currentText()
        tpl = self._current_template()
        self._settings.setValue("amylo_report/last_template", tpl_name)
        mode = str(self._out_combo.currentData() or "both")

        self._btn_generate.setEnabled(False)
        self._status.setText("Capturando imágenes y animaciones…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            assets_dir = tempfile.mkdtemp(prefix="amylo_report_")
            images, gifs = self._panel._collect_report_assets(
                assets_dir,
                include_gifs=bool(self._gif_check.isChecked() and "gifs" in tpl["sections"]),
                gif_frames=int(self._gif_frames_spin.value()),
            )
            data = self._panel._build_amylo_report_data()
            data.images = images
            data.gifs = gifs
            data.template = tpl
            data.template_name = tpl_name

            pid = str((data.patient or {}).get("id", "") or "paciente").replace(" ", "_")
            outputs: list[str] = []
            if mode in ("both", "html"):
                self._status.setText("Generando HTML…")
                QApplication.processEvents()
                html_path = os.path.join(out_dir, f"informe_amylo_{pid}.html")
                generate_amylo_html(data, html_path)
                outputs.append(html_path)
            if mode in ("both", "pdf"):
                self._status.setText("Generando PDF…")
                QApplication.processEvents()
                pdf_path = os.path.join(out_dir, f"informe_amylo_{pid}.pdf")
                generate_amylo_pdf(data, pdf_path)
                outputs.append(pdf_path)

            self._status.setText("Listo: " + " · ".join(os.path.basename(p) for p in outputs))
            if self._open_check.isChecked():
                for p in outputs:
                    try:
                        os.startfile(p)  # noqa: S606 — apertura local solicitada por el usuario
                    except Exception:
                        pass
            QMessageBox.information(self, "SINCRO", "Informe generado:\n" + "\n".join(outputs))
        except Exception as exc:
            QMessageBox.critical(self, "SINCRO", f"Error generando el informe:\n{exc}")
            self._status.setText(f"Error: {exc}")
        finally:
            QApplication.restoreOverrideCursor()
            self._btn_generate.setEnabled(True)

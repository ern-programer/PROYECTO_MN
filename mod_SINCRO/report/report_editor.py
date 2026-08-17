# -*- coding: utf-8 -*-
"""SINCRO — Editor de informe clínico con formato rico.

Diálogo con QTextEdit + toolbar de formato para que el médico escriba
su interpretación. Se pre-carga con el resumen ejecutivo y se guarda
como HTML para embeber en el informe final.

Uso:
    from report.report_editor import ReportEditorDialog
    dlg = ReportEditorDialog(parent=self, exec_summary=summary, patient_name="...")
    if dlg.exec():
        html = dlg.get_html()
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QTextCharFormat, QTextCursor, QAction, QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QToolBar, QTextEdit,
    QComboBox, QPushButton, QLabel, QFontComboBox, QSpinBox,
    QSizePolicy, QFileDialog, QMessageBox,
)


class ReportEditorDialog(QDialog):
    """Editor de texto clínico con herramientas de formato."""

    def __init__(
        self,
        parent=None,
        *,
        exec_summary: str = "",
        patient_name: str = "",
        study_desc: str = "",
        phase_label: str = "Estudio",
    ):
        super().__init__(parent)
        self.setWindowTitle("Editor de informe clínico")
        self.resize(900, 700)
        self._exec_summary = exec_summary
        self._patient_name = patient_name
        self._study_desc = study_desc
        self._phase_label = phase_label

        self._build_ui()
        self._load_initial_content()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Toolbar de formato.
        toolbar = QToolBar()
        toolbar.setMovable(False)
        root.addWidget(toolbar)

        # Fuente.
        self._font_combo = QFontComboBox()
        self._font_combo.setCurrentFont(QFont("Segoe UI"))
        self._font_combo.setMaximumWidth(180)
        self._font_combo.currentFontChanged.connect(self._on_font_changed)
        toolbar.addWidget(self._font_combo)

        # Tamaño.
        self._size_spin = QSpinBox()
        self._size_spin.setRange(8, 36)
        self._size_spin.setValue(11)
        self._size_spin.setSuffix(" pt")
        self._size_spin.setFixedWidth(65)
        self._size_spin.valueChanged.connect(self._on_size_changed)
        toolbar.addWidget(self._size_spin)

        toolbar.addSeparator()

        # Negrita, cursiva, subrayado.
        for text, shortcut, slot in [
            ("N", "Ctrl+B", self._toggle_bold),
            ("K", "Ctrl+I", self._toggle_italic),
            ("S", "Ctrl+U", self._toggle_underline),
        ]:
            btn = QPushButton(text)
            btn.setFixedWidth(28)
            btn.setCheckable(True)
            btn.setShortcut(shortcut)
            btn.clicked.connect(slot)
            toolbar.addWidget(btn)

        toolbar.addSeparator()

        # Títulos.
        self._heading_combo = QComboBox()
        self._heading_combo.addItems(["Párrafo", "Título 1", "Título 2", "Título 3", "Resumen ejecutivo"])
        self._heading_combo.setFixedWidth(140)
        self._heading_combo.currentIndexChanged.connect(self._on_heading_changed)
        toolbar.addWidget(QLabel(" Estilo:"))
        toolbar.addWidget(self._heading_combo)

        toolbar.addSeparator()

        # Color del texto.
        for color, label, tooltip in [
            ("#e2e8f0", "Blanco", "Texto blanco (default)"),
            ("#38bdf8", "Azul", "Texto azul (títulos)"),
            ("#fbbf24", "Amarillo", "Texto amarillo (destacado)"),
            ("#f87171", "Rojo", "Texto rojo (alerta)"),
            ("#4ade80", "Verde", "Texto verde (positivo)"),
        ]:
            btn = QPushButton()
            btn.setFixedSize(22, 22)
            btn.setStyleSheet(f"background:{color}; border:1px solid #475569; border-radius:3px;")
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda _=False, c=color: self._set_text_color(c))
            toolbar.addWidget(btn)

        # Editor.
        self._editor = QTextEdit()
        self._editor.setAcceptRichText(True)
        self._editor.setStyleSheet("""
            QTextEdit {
                background: #1e293b;
                color: #e2e8f0;
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 11pt;
                padding: 16px;
                border: 1px solid #475569;
                border-radius: 8px;
                selection-background-color: #38bdf8;
            }
        """)
        self._editor.textChanged.connect(self._update_preview_label)
        root.addWidget(self._editor, 1)

        # Info + botones.
        bottom = QHBoxLayout()
        self._lbl_info = QLabel("Listo")
        self._lbl_info.setStyleSheet("color: #94a3b8; font-size: 9pt;")
        bottom.addWidget(self._lbl_info)
        bottom.addStretch(1)
        btn_preview = QPushButton("Vista previa")
        btn_preview.setToolTip("Abre el HTML en el navegador para previsualizar.")
        btn_preview.clicked.connect(self._preview_in_browser)
        bottom.addWidget(btn_preview)
        btn_save = QPushButton("Guardar HTML...")
        btn_save.clicked.connect(self._save_html_file)
        bottom.addWidget(btn_save)
        btn_ok = QPushButton("Aceptar")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        bottom.addWidget(btn_ok)
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(btn_cancel)
        root.addLayout(bottom)

    def _load_initial_content(self):
        """Carga el resumen ejecutivo como contenido inicial."""
        html_parts = []
        html_parts.append(
            '<h1 style="color:#38bdf8; font-size:18pt; margin-bottom:4px;">'
            f'{self._patient_name or "Paciente"}</h1>'
        )
        html_parts.append(
            f'<p style="color:#94a3b8; font-size:10pt; margin-top:0;">'
            f'{self._study_desc} — {self._phase_label}</p>'
        )
        html_parts.append('<hr style="border:1px solid #475569;">')

        if self._exec_summary:
            html_parts.append(
                '<h2 style="color:#38bdf8; font-size:14pt;">Resumen ejecutivo</h2>'
            )
            html_parts.append(self._exec_summary)
            html_parts.append('<hr style="border:1px solid #475569;">')

        html_parts.append(
            '<h2 style="color:#38bdf8; font-size:14pt;">Interpretación clínica</h2>'
        )
        html_parts.append(
            '<p style="color:#e2e8f0;">Escriba aquí su interpretación de los resultados...</p>'
        )

        self._editor.setHtml("".join(html_parts))
        # Mover cursor al final.
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._editor.setTextCursor(cursor)

    def _current_format(self) -> QTextCharFormat:
        return self._editor.textCursor().charFormat()

    def _apply_format(self, fmt: QTextCharFormat):
        cursor = self._editor.textCursor()
        if not cursor.hasSelection():
            cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        cursor.mergeCharFormat(fmt)
        self._editor.mergeCurrentCharFormat(fmt)

    def _on_font_changed(self, font: QFont):
        fmt = QTextCharFormat()
        fmt.setFont(font)
        self._apply_format(fmt)

    def _on_size_changed(self, size: int):
        fmt = QTextCharFormat()
        fmt.setFontPointSize(float(size))
        self._apply_format(fmt)

    def _toggle_bold(self):
        fmt = QTextCharFormat()
        fmt.setFontWeight(QFont.Weight.Bold if self._editor.fontWeight() < QFont.Weight.Bold else QFont.Weight.Normal)
        self._apply_format(fmt)

    def _toggle_italic(self):
        fmt = QTextCharFormat()
        fmt.setFontItalic(not self._editor.fontItalic())
        self._apply_format(fmt)

    def _toggle_underline(self):
        fmt = QTextCharFormat()
        fmt.setFontUnderline(not self._editor.fontUnderline())
        self._apply_format(fmt)

    def _set_text_color(self, color_hex: str):
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color_hex))
        self._apply_format(fmt)

    def _on_heading_changed(self, index: int):
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock)
        cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
        if index == 0:
            # Párrafo normal.
            fmt = QTextCharFormat()
            fmt.setFontPointSize(11)
            fmt.setForeground(QColor("#e2e8f0"))
            cursor.mergeCharFormat(fmt)
        elif index in (1, 2, 3):
            sizes = {1: 18, 2: 14, 3: 12}
            fmt = QTextCharFormat()
            fmt.setFontPointSize(sizes[index])
            fmt.setFontWeight(QFont.Weight.Bold)
            fmt.setForeground(QColor("#38bdf8"))
            cursor.mergeCharFormat(fmt)
        elif index == 4:
            fmt = QTextCharFormat()
            fmt.setFontPointSize(10)
            fmt.setForeground(QColor("#94a3b8"))
            fmt.setFontItalic(True)
            cursor.mergeCharFormat(fmt)

    def _update_preview_label(self):
        text = self._editor.toPlainText()
        n_chars = len(text)
        self._lbl_info.setText(f"{n_chars} caracteres")

    def get_html(self) -> str:
        """Retorna el contenido del editor como HTML."""
        return self._editor.toHtml()

    def get_plain_text(self) -> str:
        """Retorna el contenido del editor como texto plano."""
        return self._editor.toPlainText()

    def _preview_in_browser(self):
        """Abre una vista previa del HTML en el navegador."""
        import tempfile, os
        html = self._wrap_html(self.get_html())
        path = os.path.join(tempfile.gettempdir(), "sincro_preview_informe.html")
        with open(path, "wb") as f:
            f.write(html.encode("utf-8"))
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _save_html_file(self):
        """Guarda el contenido como archivo HTML."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar informe HTML...", "informe_clinico.html",
            "Archivos HTML (*.html);;Todos (*.*)",
        )
        if not path:
            return
        html = self._wrap_html(self.get_html())
        try:
            with open(path, "wb") as f:
                f.write(html.encode("utf-8"))
            QMessageBox.information(self, "SINCRO", f"Guardado en: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "SINCRO", f"Error al guardar:\n{exc}")

    def _wrap_html(self, body_html: str) -> str:
        """Envuelve el HTML del editor en una página completa con el mismo estilo del informe."""
        return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SINCRO — Informe clínico</title>
<style>
  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    line-height: 1.6;
    max-width: 900px;
    margin: 0 auto;
    padding: 32px 24px;
  }}
  h1 {{ color: #38bdf8; font-size: 1.6rem; letter-spacing: 1px; margin-bottom: 4px; }}
  h2 {{ color: #38bdf8; font-size: 1.2rem; margin-top: 24px; margin-bottom: 8px; }}
  h3 {{ color: #38bdf8; font-size: 1.05rem; margin-top: 16px; }}
  p {{ margin-bottom: 8px; }}
  hr {{ border: 1px solid #475569; margin: 16px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  th {{ background: #1a3a5c; color: white; padding: 8px; text-align: left; font-size: 0.85rem; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #475569; font-size: 0.9rem; }}
  .footer {{ text-align: center; padding: 16px; margin-top: 32px; border-top: 1px solid #475569; color: #94a3b8; font-size: 0.8rem; }}
</style>
</head>
<body>
{body_html}
<div class="footer">
  Informe generado por SINCRO · Clínica GammaSync<br>
  Resultados orientativos para apoyo clínico y auditoría técnica.
</div>
</body>
</html>"""

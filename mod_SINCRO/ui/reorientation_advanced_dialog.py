"""Ventana de Controles Avanzados de orientación (Rec/Ref).

Se abre desde el diálogo de reorientación con el botón "Controles avanzados".
Agrupa las herramientas geométricas peligrosas (rotar/espejar/swap) y la
gestión de presets (fábrica + usuario), separándolas del flujo normal para
evitar cambios accidentales de orientación durante el uso clínico diario.

No contiene lógica de imagen: delega en el diálogo padre a través de callbacks,
de modo que el volumen reorientado y las vistas siguen viviendo en un solo lugar.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ReorientationAdvancedDialog(QDialog):
    """Panel no modal de controles avanzados + presets de orientación."""

    def __init__(self, parent, store, on_apply_op, on_reset_ops, on_apply_preset,
                 current_state_getter):
        super().__init__(parent)
        self.setWindowTitle("Controles avanzados · Orientación")
        self.setModal(False)
        self.resize(430, 430)
        self._store = store
        self._on_apply_op = on_apply_op          # callable(op:str)
        self._on_reset_ops = on_reset_ops        # callable()
        self._on_apply_preset = on_apply_preset  # callable(preset:dict)
        self._get_state = current_state_getter   # callable()->dict

        root = QVBoxLayout(self)

        warn = QLabel(
            "⚠ <b>Controles avanzados de orientación</b>. Modifican cómo se "
            "muestran SA/HLA/VLA. Usalos solo para casos especiales "
            "(dextrocardia, situs inversus, equipos con orientación no estándar)."
        )
        warn.setWordWrap(True)
        # Mismo tono cálido, pero más oscuro para mejor legibilidad.
        warn.setStyleSheet("color:#b88e36;")
        root.addWidget(warn)

        # --- Presets -------------------------------------------------------
        gp = QGroupBox("Presets de orientación")
        gpl = QVBoxLayout(gp)
        row = QHBoxLayout()
        row.addWidget(QLabel("Preset"))
        self.preset_combo = QComboBox()
        self._reload_presets()
        row.addWidget(self.preset_combo, 1)
        self.btn_apply_preset = QPushButton("Aplicar")
        self.btn_apply_preset.clicked.connect(self._apply_selected_preset)
        row.addWidget(self.btn_apply_preset)
        gpl.addLayout(row)

        self.lbl_desc = QLabel("")
        self.lbl_desc.setWordWrap(True)
        # Mismo azul informativo, más oscuro para contraste en fondo claro.
        self.lbl_desc.setStyleSheet("color:#5f84b8;font-size:11px;")
        gpl.addWidget(self.lbl_desc)
        self.preset_combo.currentTextChanged.connect(self._update_desc)
        self._update_desc(self.preset_combo.currentText())

        save_row = QHBoxLayout()
        self.user_name_edit = QLineEdit()
        self.user_name_edit.setPlaceholderText("Nombre del preset de usuario")
        save_row.addWidget(self.user_name_edit, 1)
        self.btn_save = QPushButton("Guardar actual")
        self.btn_save.setToolTip("Guarda el estado de orientación actual como preset de usuario.")
        self.btn_save.clicked.connect(self._save_user_preset)
        save_row.addWidget(self.btn_save)
        self.btn_del = QPushButton("Borrar")
        self.btn_del.setToolTip("Borra el preset de usuario seleccionado (los de fábrica no se borran).")
        self.btn_del.clicked.connect(self._delete_user_preset)
        save_row.addWidget(self.btn_del)
        gpl.addLayout(save_row)
        root.addWidget(gp)

        # --- Herramientas geométricas -------------------------------------
        gt = QGroupBox("Herramientas geométricas (aplican al instante)")
        grid = QGridLayout(gt)
        tools = [
            ("Rotar +90°", "rot+90"),
            ("Rotar -90°", "rot-90"),
            ("Espejar L/R", "flip-lr"),
            ("Espejar A/P", "flip-ap"),
            ("Swap HLA/VLA", "swap-hla-vla"),
        ]
        for idx, (label, op) in enumerate(tools):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, o=op: self._on_apply_op(o))
            grid.addWidget(b, idx // 2, idx % 2)
        btn_reset = QPushButton("Reset orientación")
        btn_reset.setStyleSheet("font-weight:bold;")
        btn_reset.clicked.connect(self._reset)
        grid.addWidget(btn_reset, (len(tools) + 1) // 2, 0, 1, 2)
        root.addWidget(gt)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        btn_close = QPushButton("Cerrar")
        btn_close.clicked.connect(self.close)
        close_row.addWidget(btn_close)
        root.addLayout(close_row)

    # ------------------------------------------------------------- helpers
    def _reload_presets(self, select: str | None = None):
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(self._store.names())
        if select and select in self._store.names():
            self.preset_combo.setCurrentText(select)
        self.preset_combo.blockSignals(False)

    def _update_desc(self, name):
        p = self._store.get(str(name))
        if not p:
            self.lbl_desc.setText("")
            return
        tag = "fábrica" if self._store.is_factory(str(name)) else "usuario"
        ops = ", ".join(p.get("post_ops", [])) or "—"
        self.lbl_desc.setText(
            f"[{tag}] {p.get('description', '')}\n"
            f"flip A/P={p.get('flip_ap')} · invertir lateral={p.get('invert_ll_handles')} · ops: {ops}"
        )

    def _apply_selected_preset(self):
        name = self.preset_combo.currentText().strip()
        p = self._store.get(name)
        if not p:
            QMessageBox.information(self, "SINCRO", "No hay preset seleccionado.")
            return
        self._on_apply_preset(p)

    def _save_user_preset(self):
        name = self.user_name_edit.text().strip()
        if not name:
            QMessageBox.information(self, "SINCRO", "Ingresá un nombre para el preset de usuario.")
            return
        st = self._get_state()
        try:
            self._store.save_user(
                name,
                flip_ap=bool(st.get("flip_ap", False)),
                invert_ll_handles=bool(st.get("invert_ll_handles", False)),
                post_ops=list(st.get("post_ops", [])),
                description=st.get("description", "Preset de usuario"),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "SINCRO", str(exc))
            return
        self._reload_presets(select=name)
        QMessageBox.information(self, "SINCRO", f"Preset de usuario '{name}' guardado.")

    def _delete_user_preset(self):
        name = self.preset_combo.currentText().strip()
        if self._store.is_factory(name):
            QMessageBox.information(self, "SINCRO", "Los presets de fábrica no se pueden borrar.")
            return
        if self._store.delete_user(name):
            self._reload_presets()
            QMessageBox.information(self, "SINCRO", f"Preset '{name}' borrado.")
        else:
            QMessageBox.information(self, "SINCRO", "No se encontró el preset para borrar.")

    def _reset(self):
        self._on_reset_ops()

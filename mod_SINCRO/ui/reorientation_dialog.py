"""Dialogo interactivo de reorientacion cardiaca (estilo Rec/Ref Xeleris).

Muestra las dos vistas de referencia anatomicas que se calculan desde la
geometria de adquisicion DICOM (posicion del paciente, angulo inicial, sentido
de giro, arco): vista **anterior (AP)** y **lateral izquierda**, obtenidas por
reproyeccion del volumen reconstruido al angulo de detector correspondiente.

El usuario traza el eje largo del VI en cada una de esas vistas ortogonales; de
ambas lineas se arma el vector 3D del eje largo y se reslicea a eje corto (SA),
derivando HLA y VLA. Markers Base/Apex y espesor completan el Rec/Ref.

Al aceptar, expone el volumen reorientado (SA-alineado) gated y ungated, junto
con los limites Base/Apex y el espesor elegidos, para que la ventana principal
genere los cortes anatomicamente correctos.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

from core.cardiac_reorientation import (
    auto_orient_lv,
    default_center,
    hla_slice,
    reslice_from_vector,
    reslice_from_vector_gated,
    sa_slice,
    vla_slice,
)
from core.col_registry import available_colormaps
from core.reorientation_presets import DEFAULT_PRESET_NAME, ReorientationPresetStore
from core.spect_geometry import SpectGeometry, reference_views, reproject_view
from ui.reorientation_advanced_dialog import ReorientationAdvancedDialog

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.patches import Ellipse, Rectangle
    _MPL_OK = True
except Exception:  # pragma: no cover
    _MPL_OK = False


def _norm(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    if arr.size == 0:
        return arr
    p99 = float(np.percentile(arr, 99.5))
    p2 = float(np.percentile(arr, 2.0))
    return np.clip((arr - p2) / max(p99 - p2, 1e-8), 0.0, 1.0)


class _Handle:
    __slots__ = ("name", "axes", "x", "y", "horizontal")

    def __init__(self, name, axes, x, y, horizontal=False):
        self.name = name
        self.axes = axes
        self.x = float(x)
        self.y = float(y)
        self.horizontal = horizontal  # limit line: solo importa y


class CardiacReorientationDialog(QDialog):
    """Reorientacion oblicua interactiva del VI."""

    def __init__(self, ungated_volume, gated_volume=None, source_label="", geometry=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reorientar corazón · Rec/Ref")
        self.setModal(True)
        self.resize(1180, 820)

        self._ung = np.asarray(ungated_volume, dtype=np.float64)
        self._gated = None if gated_volume is None else np.asarray(gated_volume, dtype=np.float64)
        self._geometry = geometry if isinstance(geometry, SpectGeometry) else None
        self.reoriented_ungated = None
        self.reoriented_gated = None
        self.result_long_axis = None
        self.result_center = None
        self.base_k = 0
        self.apex_k = 0
        self.thickness = 1

        n = int(max(self._ung.shape))
        self._N = n
        cz, cy, cx = default_center(self._ung)
        self._center = [cz, cy, cx]

        # VOI elíptica (heart box) estilo Xeleris: centro (z,y,x) + semiejes en
        # vóxeles. Al recortar el corazón y excluir hígado/tórax, el reslice se
        # centra y ventana sobre el VI y el anillo SA sale limpio.
        self._voi_cz = float(cz)
        self._voi_cy = float(cy)
        self._voi_cx = float(cx)
        r0 = max(6.0, n * 0.22)
        self._voi_rz = r0
        self._voi_ry = r0
        self._voi_rx = r0
        self._out = n

        # Auto-orientación del VI usando el movimiento del gated SPECT: aísla el
        # VI del hígado por variabilidad temporal y estima centro + eje largo.
        self._auto = None
        try:
            self._auto = auto_orient_lv(self._gated, self._ung)
        except Exception:
            self._auto = None
        if self._auto is not None:
            acz, acy, acx = self._auto["center"]
            self._voi_cz, self._voi_cy, self._voi_cx = float(acz), float(acy), float(acx)
            arz, ary, arx = self._auto["semiaxes"]
            self._voi_rz, self._voi_ry, self._voi_rx = float(arz), float(ary), float(arx)
            self._center = [self._voi_cz, self._voi_cy, self._voi_cx]
            cz, cy, cx = self._voi_cz, self._voi_cy, self._voi_cx

        # Colormap y ventaneo en vivo para los cortes SA/HLA/VLA.
        self._cmap = "odyssey_cool"
        self._win_lo = 0.0   # fracción 0..1 del máximo robusto (piso)
        self._win_hi = 1.0   # fracción 0..1 del máximo robusto (techo)
        self._disp_max = 1.0
        # Convención de columnas de vistas de referencia (display):
        # AP sin espejo, lateral izquierda espejada para que "mire a la izquierda".
        self._ap_col_sign = +1
        self._ll_col_sign = -1
        # Store de presets de orientación (fábrica + usuario en disco).
        self._preset_store = ReorientationPresetStore()
        # Estado de orientación (se completa aplicando el preset default).
        # - _preset_flip_ap: espejar A/P en los cortes reorientados.
        # - _preset_invert_ll_handles: invertir base↔ápex SOLO en lateral izq.
        # - _post_ops: ajustes geométricos manuales EXTRA (rotar/espejar/swap).
        #   OJO: "flip-ap" del preset NO se guarda en _post_ops; se aplica una
        #   sola vez a través de _effective_post_ops() para evitar doble flip.
        self._preset_flip_ap = True
        self._preset_invert_ll_handles = True
        self._post_ops = []
        # Ajuste fino manual por plano (grados): SA/HLA/VLA.
        self._fine_rot_sa = 0
        self._fine_rot_hla = 0
        self._fine_rot_vla = 0
        _default = self._preset_store.get(DEFAULT_PRESET_NAME)
        if _default is not None:
            self._preset_flip_ap = bool(_default.get("flip_ap", True))
            self._preset_invert_ll_handles = bool(_default.get("invert_ll_handles", True))
            self._post_ops = [
                op for op in list(_default.get("post_ops", [])) if op != "flip-ap"
            ]

        # Vistas de referencia: cortes tomográficos (slab-MIP) ortogonales
        # centrados en el VI. A diferencia de las proyecciones completas, el
        # slab excluye el hígado (a otra profundidad) y evita superposiciones.
        # Orientación cruda (fila=z, col=detector) para que los handles mapeen
        # directo a (z, x) [anterior] y (z, y) [lateral] del volumen.
        self._ap_view, self._ll_view, self._ap_angle, self._ll_angle, self._geo_note = self._build_reference_views()
        nz_ap, wx = self._ap_view.shape
        nz_ll, wy = self._ll_view.shape

        # Estado geometrico inicial: eje largo casi vertical (a lo largo de z).
        r = max(8.0, nz_ap * 0.30)
        # Vista anterior (AP): fila=z, col=x. P1=base (arriba), P2=apex (abajo).
        self.h_tra1 = _Handle("ap1", None, self._ap_disp_from_vol_x(cx), cz - r)
        self.h_tra2 = _Handle("ap2", None, self._ap_disp_from_vol_x(cx), cz + r)
        # Vista lateral izquierda: fila=z, col=y.
        self.h_cor1 = _Handle("ll1", None, self._ll_disp_from_vol_y(cy), cz - r)
        self.h_cor2 = _Handle("ll2", None, self._ll_disp_from_vol_y(cy), cz + r)
        # Si hubo auto-orientación, coloca la línea del eje largo según el PCA.
        if self._auto is not None:
            self._set_handles_from_long_axis(self._auto["long_axis"], self._auto["half_length"])
        if self._preset_invert_ll_handles:
            self._flip_lateral_handles(redraw=False)
        # Limites base/apex en HLA (fila=k del volumen reorientado). Se ajustan
        # al tamaño reorientado real en el primer _recompute_and_draw.
        self._base_k = int(n * 0.30)
        self._apex_k = int(n * 0.70)
        self._limits_init = False

        if not _MPL_OK:
            lay = QVBoxLayout(self)
            lay.addWidget(QLabel("matplotlib con backend Qt no disponible; no se puede reorientar."))
            btn = QPushButton("Cerrar")
            btn.clicked.connect(self.reject)
            lay.addWidget(btn)
            return

        self._drag: _Handle | None = None
        self._fine_drag_start: tuple[float, float, int, int, int] | None = None
        self._build_ui(source_label)
        self._recompute_and_draw(full=True)

    # ------------------------------------------------- vistas de referencia
    def _slab_views(self):
        """Cortes tomográficos ortogonales (slab-MIP) centrados en la VOI.

        - ``ap`` (z, x): MIP coronal sobre un slab en y alrededor del centro →
          equivalente a la vista **anterior**, sin hígado (a otra profundidad y).
        - ``ll`` (z, y): MIP sagital sobre un slab en x alrededor del centro →
          equivalente a la vista **lateral izquierda**.
        """
        vol = self._ung
        Z, Y, X = vol.shape
        cy = int(np.clip(round(self._voi_cy), 0, Y - 1))
        cx = int(np.clip(round(self._voi_cx), 0, X - 1))
        hy = int(max(3, round(0.6 * self._voi_ry)))
        hx = int(max(3, round(0.6 * self._voi_rx)))
        y0, y1 = max(0, cy - hy), min(Y, cy + hy + 1)
        x0, x1 = max(0, cx - hx), min(X, cx + hx + 1)
        ap = vol[:, y0:y1, :].max(axis=1)   # (z, x) slab coronal
        ll = vol[:, :, x0:x1].max(axis=2)   # (z, y) slab sagital
        if self._ap_col_sign < 0:
            ap = ap[:, ::-1]
        if self._ll_col_sign < 0:
            ll = ll[:, ::-1]
        return np.asarray(ap, dtype=np.float64), np.asarray(ll, dtype=np.float64)

    def _map_disp_to_vol_col(self, x_disp, width, sign):
        x = float(np.clip(x_disp, 0.0, max(0.0, float(width - 1))))
        return x if sign >= 0 else float((width - 1) - x)

    def _map_vol_to_disp_col(self, x_vol, width, sign):
        x = float(np.clip(x_vol, 0.0, max(0.0, float(width - 1))))
        return x if sign >= 0 else float((width - 1) - x)

    def _ap_vol_from_disp_x(self, x_disp):
        return self._map_disp_to_vol_col(x_disp, self._ap_view.shape[1], self._ap_col_sign)

    def _ap_disp_from_vol_x(self, x_vol):
        return self._map_vol_to_disp_col(x_vol, self._ap_view.shape[1], self._ap_col_sign)

    def _ll_vol_from_disp_y(self, y_disp):
        return self._map_disp_to_vol_col(y_disp, self._ll_view.shape[1], self._ll_col_sign)

    def _ll_disp_from_vol_y(self, y_vol):
        return self._map_vol_to_disp_col(y_vol, self._ll_view.shape[1], self._ll_col_sign)

    def _set_handles_from_long_axis(self, u, half):
        """Coloca la línea del eje largo (base→ápex) en ambas vistas desde ``u``."""
        u = np.asarray(u, dtype=np.float64)
        nrm = float(np.linalg.norm(u))
        u = u / (nrm if nrm > 0 else 1.0)
        cz, cy, cx = self._voi_cz, self._voi_cy, self._voi_cx
        bz, by, bx = cz - half * u[0], cy - half * u[1], cx - half * u[2]  # base
        az, ay, ax = cz + half * u[0], cy + half * u[1], cx + half * u[2]  # ápex
        self.h_tra1.x, self.h_tra1.y = self._ap_disp_from_vol_x(bx), float(bz)   # base (anterior)
        self.h_tra2.x, self.h_tra2.y = self._ap_disp_from_vol_x(ax), float(az)   # ápex (anterior)
        self.h_cor1.x, self.h_cor1.y = self._ll_disp_from_vol_y(by), float(bz)   # base (lateral)
        self.h_cor2.x, self.h_cor2.y = self._ll_disp_from_vol_y(ay), float(az)   # ápex (lateral)

    def _effective_post_ops(self):
        """Ops geométricas reales a aplicar: preset (flip-ap) + ajustes manuales.

        El ``flip-ap`` del preset se materializa aquí UNA sola vez y no vive en
        ``self._post_ops`` para evitar duplicarlo. Se filtra cualquier ``flip-ap``
        remanente de ``_post_ops`` por seguridad.
        """
        ops = []
        if self._preset_flip_ap:
            ops.append("flip-ap")
        ops.extend(op for op in self._post_ops if op != "flip-ap")
        return ops

    def _ops_note(self):
        ops = self._effective_post_ops()
        if not ops:
            return "sin ajustes manuales"
        counts = {}
        for op in ops:
            counts[op] = counts.get(op, 0) + 1
        order = ["flip-ap", "flip-lr", "rot+90", "rot-90", "swap-hla-vla"]
        parts = []
        for op in order:
            c = counts.get(op, 0)
            if c > 0:
                parts.append(f"{op}×{c}" if c > 1 else op)
        return " · ".join(parts)

    def _fine_rot_note(self):
        a = int(self._fine_rot_sa)
        b = int(self._fine_rot_hla)
        c = int(self._fine_rot_vla)
        if a == 0 and b == 0 and c == 0:
            return "fine: off"
        return f"fine SA={a:+d}° HLA={b:+d}° VLA={c:+d}°"

    def _apply_fine_rotations_3d(self, vol3d):
        out = np.asarray(vol3d, dtype=np.float64)
        if ndi is None:
            return out
        if self._fine_rot_sa:
            out = ndi.rotate(out, angle=float(self._fine_rot_sa), axes=(1, 2), reshape=False, order=1, mode="nearest")
        if self._fine_rot_hla:
            out = ndi.rotate(out, angle=float(self._fine_rot_hla), axes=(0, 2), reshape=False, order=1, mode="nearest")
        if self._fine_rot_vla:
            out = ndi.rotate(out, angle=float(self._fine_rot_vla), axes=(0, 1), reshape=False, order=1, mode="nearest")
        return np.ascontiguousarray(out)

    def _apply_fine_rotations_4d(self, cube4d):
        out = np.asarray(cube4d, dtype=np.float64)
        if ndi is None:
            return out
        if not (self._fine_rot_sa or self._fine_rot_hla or self._fine_rot_vla):
            return out
        gates = []
        for g in range(out.shape[0]):
            gates.append(self._apply_fine_rotations_3d(out[g]))
        return np.ascontiguousarray(np.stack(gates, axis=0))

    def _apply_post_ops_3d(self, vol3d):
        out = np.asarray(vol3d, dtype=np.float64)
        for op in self._effective_post_ops():
            if op == "rot+90":
                out = np.rot90(out, k=1, axes=(1, 2))
            elif op == "rot-90":
                out = np.rot90(out, k=-1, axes=(1, 2))
            elif op == "flip-lr":
                out = np.flip(out, axis=2)
            elif op == "flip-ap":
                out = np.flip(out, axis=1)
            elif op == "swap-hla-vla":
                out = np.transpose(out, (0, 2, 1))
        out = self._apply_fine_rotations_3d(out)
        return np.ascontiguousarray(out)

    def _apply_post_ops_4d(self, cube4d):
        out = np.asarray(cube4d, dtype=np.float64)
        for op in self._effective_post_ops():
            if op == "rot+90":
                out = np.rot90(out, k=1, axes=(2, 3))
            elif op == "rot-90":
                out = np.rot90(out, k=-1, axes=(2, 3))
            elif op == "flip-lr":
                out = np.flip(out, axis=3)
            elif op == "flip-ap":
                out = np.flip(out, axis=2)
            elif op == "swap-hla-vla":
                out = np.transpose(out, (0, 1, 3, 2))
        out = self._apply_fine_rotations_4d(out)
        return np.ascontiguousarray(out)

    def _add_post_op(self, op):
        op = str(op)
        # "flip-ap" se maneja como toggle del preset (no se acumula en _post_ops)
        # para no chocar con el flip A/P del preset y evitar dobles espejos.
        if op == "flip-ap":
            self._preset_flip_ap = not self._preset_flip_ap
        else:
            self._post_ops.append(op)
        self._recompute_and_draw()

    def _reset_post_ops(self):
        self._post_ops = []
        self._preset_flip_ap = False
        self._recompute_and_draw()

    # ------------------------------------------------- controles avanzados
    def _open_advanced(self):
        """Abre el panel no modal de controles avanzados + presets."""
        dlg = ReorientationAdvancedDialog(
            self,
            store=self._preset_store,
            on_apply_op=self._add_post_op,
            on_reset_ops=self._reset_post_ops,
            on_apply_preset=self._apply_preset,
            current_state_getter=self._current_orientation_state,
        )
        dlg.show()

    def _apply_preset(self, p):
        """Aplica un preset de orientación (dict flip_ap/invert_ll/post_ops).

        El estado de handles laterales depende de ``invert_ll_handles``; si
        cambia respecto al actual, se re-aplica la inversión lateral. El
        ``flip-ap`` del preset se refleja solo en ``_preset_flip_ap`` (no en
        ``_post_ops``) para no duplicar el espejo A/P.
        """
        if not p:
            return
        want_invert_ll = bool(p.get("invert_ll_handles", False))
        if want_invert_ll != self._preset_invert_ll_handles:
            self._flip_lateral_handles(redraw=False)
            self._preset_invert_ll_handles = want_invert_ll
        self._preset_flip_ap = bool(p.get("flip_ap", False))
        self._post_ops = [
            op for op in list(p.get("post_ops", [])) if op != "flip-ap"
        ]
        self._recompute_and_draw()
        self.canvas.draw_idle()

    def _current_orientation_state(self):
        """Estado de orientación actual como dict (para guardar como preset)."""
        return {
            "flip_ap": bool(self._preset_flip_ap),
            "invert_ll_handles": bool(self._preset_invert_ll_handles),
            "post_ops": list(self._post_ops),
            "description": "Preset de usuario (estado actual de orientación)",
        }

    def _flip_base_apex(self):
        """Intercambia base y ápex del eje largo en ambas vistas."""
        self.h_tra1.x, self.h_tra2.x = self.h_tra2.x, self.h_tra1.x
        self.h_tra1.y, self.h_tra2.y = self.h_tra2.y, self.h_tra1.y
        self.h_cor1.x, self.h_cor2.x = self.h_cor2.x, self.h_cor1.x
        self.h_cor1.y, self.h_cor2.y = self.h_cor2.y, self.h_cor1.y
        self._recompute_and_draw()
        self.canvas.draw_idle()

    def _flip_lateral_handles(self, redraw=True):
        """Intercambia base/ápex SOLO en la vista lateral izquierda."""
        self.h_cor1.x, self.h_cor2.x = self.h_cor2.x, self.h_cor1.x
        self.h_cor1.y, self.h_cor2.y = self.h_cor2.y, self.h_cor1.y
        if redraw:
            self._recompute_and_draw()
            self.canvas.draw_idle()

    def _auto_orient(self):
        """Re-ejecuta la auto-orientación del VI (movimiento gated + PCA)."""
        try:
            auto = auto_orient_lv(self._gated, self._ung)
        except Exception:
            auto = None
        if not auto:
            return
        self._auto = auto
        acz, acy, acx = auto["center"]
        self._voi_cz, self._voi_cy, self._voi_cx = float(acz), float(acy), float(acx)
        arz, ary, arx = auto["semiaxes"]
        self._voi_rz, self._voi_ry, self._voi_rx = float(arz), float(ary), float(arx)
        self._set_handles_from_long_axis(auto["long_axis"], auto["half_length"])
        if self._preset_invert_ll_handles:
            self._flip_lateral_handles(redraw=False)
        self._limits_init = False
        self._recompute_and_draw()
        self.canvas.draw_idle()

    def _build_reference_views(self):
        """Genera las vistas anterior/lateral como cortes tomográficos (slab-MIP).

        Devuelve (ap_view, ll_view, ap_angle, ll_angle, nota). Los ángulos de
        geometría se conservan solo para la nota informativa; las vistas son
        cortes del volumen reconstruido centrados en el VI (no proyecciones).
        """
        geo = self._geometry
        ap_angle = geo.anterior_angle() if geo is not None else None
        ll_angle = geo.left_lateral_angle() if geo is not None else None
        ap, ll = self._slab_views()
        if geo is not None:
            note = (
                f"Cortes tomográficos (slab-MIP) centrados en VI · "
                f"PP={geo.patient_position or '?'} start={geo.start_angle:.0f}° "
                f"dir={geo.rotation_direction or '?'} arco={geo.scan_arc}° · "
                f"AP_sign={self._ap_col_sign:+d} LL_sign={self._ll_col_sign:+d}"
            )
        else:
            note = "Cortes tomográficos (slab-MIP) ortogonales centrados en el VI."
        return ap, ll, ap_angle, ll_angle, note

    # ------------------------------------------------------------------ UI
    def _build_ui(self, source_label):
        root = QVBoxLayout(self)
        info = QLabel(
            "El VI se detecta y orienta <b>automáticamente</b> con el movimiento del gated "
            "(elipse amarilla + eje largo verde ya colocados sobre <b>cortes tomográficos</b>). "
            "Ajustá la <b>elipse (VOI)</b> y el <b>eje largo (base→ápex)</b> en <b>Anterior</b> y "
            "<b>Lateral izq.</b> si hace falta; usá <b>Invertir base↔ápex</b> si quedaron al revés. "
            "SA/HLA/VLA se actualizan al instante."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        self.fig = Figure(figsize=(11.4, 6.6), facecolor="#0b1220")
        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas, 1)

        gs = self.fig.add_gridspec(2, 3, hspace=0.22, wspace=0.12)
        self.ax_tra = self.fig.add_subplot(gs[0, 0])
        self.ax_cor = self.fig.add_subplot(gs[0, 1])
        self.ax_hla = self.fig.add_subplot(gs[0, 2])
        self.ax_sa = self.fig.add_subplot(gs[1, 0])
        self.ax_vla = self.fig.add_subplot(gs[1, 1])
        self.ax_sa_stack = self.fig.add_subplot(gs[1, 2])
        for ax in (self.ax_tra, self.ax_cor, self.ax_hla, self.ax_sa, self.ax_vla, self.ax_sa_stack):
            ax.set_facecolor("#020611")
            ax.set_xticks([])
            ax.set_yticks([])

        self.h_tra1.axes = self.ax_tra
        self.h_tra2.axes = self.ax_tra
        self.h_cor1.axes = self.ax_cor
        self.h_cor2.axes = self.ax_cor

        # Controles inferiores.
        ctrl = QWidget()
        crow = QHBoxLayout(ctrl)
        crow.setContentsMargins(0, 0, 0, 0)
        self.lbl_angles = QLabel("Azimut 0° · Elevación 0°")
        self.lbl_angles.setStyleSheet("color:#7cf29a;font-weight:bold;")
        crow.addWidget(self.lbl_angles)
        crow.addStretch(1)
        crow.addWidget(QLabel("Base"))
        self.spin_base = QSpinBox()
        self.spin_base.setRange(0, self._N - 1)
        self.spin_base.setValue(self._base_k)
        self.spin_base.valueChanged.connect(self._on_base_spin)
        crow.addWidget(self.spin_base)
        crow.addWidget(QLabel("Ápex"))
        self.spin_apex = QSpinBox()
        self.spin_apex.setRange(0, self._N - 1)
        self.spin_apex.setValue(self._apex_k)
        self.spin_apex.valueChanged.connect(self._on_apex_spin)
        crow.addWidget(self.spin_apex)
        crow.addWidget(QLabel("Esp"))
        self.spin_thick = QSpinBox()
        self.spin_thick.setRange(1, 9)
        self.spin_thick.setValue(1)
        crow.addWidget(self.spin_thick)
        root.addWidget(ctrl)

        # Fila de visualización: colormap + ventaneo (min/max) en vivo.
        viz = QWidget()
        vrow = QHBoxLayout(viz)
        vrow.setContentsMargins(0, 0, 0, 0)
        vrow.addWidget(QLabel("Escala"))
        self.cmap_combo = QComboBox()
        try:
            cmaps = available_colormaps()
        except Exception:
            cmaps = ["odyssey_cool", "gray", "hot", "turbo"]
        self.cmap_combo.addItems(cmaps)
        if self._cmap in cmaps:
            self.cmap_combo.setCurrentText(self._cmap)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        vrow.addWidget(self.cmap_combo)
        vrow.addSpacing(16)
        vrow.addWidget(QLabel("Ventana mín"))
        self.slider_lo = QSlider(Qt.Orientation.Horizontal)
        self.slider_lo.setRange(0, 100)
        self.slider_lo.setValue(0)
        self.slider_lo.setFixedWidth(160)
        self.slider_lo.valueChanged.connect(self._on_window_changed)
        vrow.addWidget(self.slider_lo)
        vrow.addWidget(QLabel("máx"))
        self.slider_hi = QSlider(Qt.Orientation.Horizontal)
        self.slider_hi.setRange(0, 100)
        self.slider_hi.setValue(100)
        self.slider_hi.setFixedWidth(160)
        self.slider_hi.valueChanged.connect(self._on_window_changed)
        vrow.addWidget(self.slider_hi)
        self.lbl_window = QLabel("0–100%")
        self.lbl_window.setStyleSheet("color:#8fd3ff;")
        vrow.addWidget(self.lbl_window)
        vrow.addStretch(1)
        root.addWidget(viz)

        # Fila de ajuste fino manual (por plano): SA/HLA/VLA.
        fine = QWidget()
        frow = QHBoxLayout(fine)
        frow.setContentsMargins(0, 0, 0, 0)
        frow.addWidget(QLabel("Ajuste fino"))

        frow.addWidget(QLabel("SA"))
        self.spin_fine_sa = QSpinBox()
        self.spin_fine_sa.setRange(-45, 45)
        self.spin_fine_sa.setSingleStep(1)
        self.spin_fine_sa.setValue(0)
        self.spin_fine_sa.setToolTip("Rota finamente en plano SA (ejes y/x).")
        self.spin_fine_sa.valueChanged.connect(self._on_fine_rot_changed)
        frow.addWidget(self.spin_fine_sa)

        frow.addWidget(QLabel("HLA"))
        self.spin_fine_hla = QSpinBox()
        self.spin_fine_hla.setRange(-45, 45)
        self.spin_fine_hla.setSingleStep(1)
        self.spin_fine_hla.setValue(0)
        self.spin_fine_hla.setToolTip("Rota finamente en plano HLA (ejes z/x).")
        self.spin_fine_hla.valueChanged.connect(self._on_fine_rot_changed)
        frow.addWidget(self.spin_fine_hla)

        frow.addWidget(QLabel("VLA"))
        self.spin_fine_vla = QSpinBox()
        self.spin_fine_vla.setRange(-45, 45)
        self.spin_fine_vla.setSingleStep(1)
        self.spin_fine_vla.setValue(0)
        self.spin_fine_vla.setToolTip("Rota finamente en plano VLA (ejes z/y).")
        self.spin_fine_vla.valueChanged.connect(self._on_fine_rot_changed)
        frow.addWidget(self.spin_fine_vla)

        btn_fine_reset = QPushButton("Reset fino")
        btn_fine_reset.setToolTip("Vuelve SA/HLA/VLA fino a 0°.")
        btn_fine_reset.clicked.connect(self._reset_fine_rot)
        frow.addWidget(btn_fine_reset)
        frow.addStretch(1)
        root.addWidget(fine)

        brow = QHBoxLayout()
        btn_auto = QPushButton("Auto-orientar VI")
        btn_auto.setToolTip("Detecta el VI por el movimiento del gated y coloca elipse + eje largo.")
        btn_auto.clicked.connect(self._auto_orient)
        brow.addWidget(btn_auto)
        btn_flip = QPushButton("Invertir base↔ápex")
        btn_flip.setToolTip("Intercambia los extremos base y ápex del eje largo.")
        btn_flip.clicked.connect(self._flip_base_apex)
        brow.addWidget(btn_flip)
        btn_adv = QPushButton("Controles avanzados")
        btn_adv.setToolTip(
            "Abre presets de orientación (Xeleris/Odyssey, dextrocardia…) y "
            "herramientas geométricas (rotar/espejar/swap) para casos especiales."
        )
        btn_adv.clicked.connect(self._open_advanced)
        brow.addWidget(btn_adv)
        brow.addStretch(1)
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        brow.addWidget(btn_cancel)
        btn_ok = QPushButton("Aplicar y generar cortes")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self._accept)
        brow.addWidget(btn_ok)
        root.addLayout(brow)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        # Evita expansión horizontal por textos largos en status (ops acumuladas).
        try:
            self.lbl_angles.setWordWrap(False)
            self.lbl_angles.setSizePolicy(self.lbl_angles.sizePolicy().horizontalPolicy(), self.lbl_angles.sizePolicy().verticalPolicy())
            self.lbl_angles.setMinimumWidth(300)
            self.lbl_angles.setMaximumWidth(880)
        except Exception:
            pass

    # ----------------------------------------------------------- geometria
    def _long_axis_vector(self) -> np.ndarray:
        """Vector de eje largo (z, y, x) desde las dos vistas ortogonales.

        Vista AP: fila=z, col=x → aporta (dz_ap, dx). Vista lateral: fila=z,
        col=y → aporta (dz_ll, dy). Se promedia la componente z de ambas.
        """
        dz_ap = self.h_tra2.y - self.h_tra1.y
        dx = self._ap_vol_from_disp_x(self.h_tra2.x) - self._ap_vol_from_disp_x(self.h_tra1.x)
        dz_ll = self.h_cor2.y - self.h_cor1.y
        dy = self._ll_vol_from_disp_y(self.h_cor2.x) - self._ll_vol_from_disp_y(self.h_cor1.x)
        uz = 0.5 * (dz_ap + dz_ll)
        u = np.array([uz, dy, dx], dtype=np.float64)
        nrm = float(np.linalg.norm(u))
        return u / nrm if nrm > 0 else np.array([1.0, 0.0, 0.0])

    def _current_center(self):
        cx = 0.5 * (self.h_tra1.x + self.h_tra2.x)
        cy = 0.5 * (self.h_cor1.x + self.h_cor2.x)
        cz = 0.25 * (self.h_tra1.y + self.h_tra2.y + self.h_cor1.y + self.h_cor2.y)
        return [float(cz), float(cy), float(cx)]

    # ------------------------------------------------------------- VOI/ROI
    def _voi_semiaxes(self):
        return max(self._voi_rz, 3.0), max(self._voi_ry, 3.0), max(self._voi_rx, 3.0)

    def _voi_mask(self, shape):
        """Máscara elipsoidal 3D (z, y, x) de la VOI cardíaca."""
        Z, Y, X = shape[-3:]
        rz, ry, rx = self._voi_semiaxes()
        zz = (np.arange(Z)[:, None, None] - self._voi_cz) / rz
        yy = (np.arange(Y)[None, :, None] - self._voi_cy) / ry
        xx = (np.arange(X)[None, None, :] - self._voi_cx) / rx
        return (zz * zz + yy * yy + xx * xx) <= 1.0

    def _apply_voi(self, vol):
        m = self._voi_mask(vol.shape)
        return np.asarray(vol, dtype=np.float64) * m

    def _apply_voi_gated(self, cube):
        m = self._voi_mask(cube.shape)
        return np.asarray(cube, dtype=np.float64) * m[None, ...]

    def _voi_out_size(self):
        rz, ry, rx = self._voi_semiaxes()
        return int(np.clip(round(2.0 * max(rz, ry, rx) * 1.25), 16, self._N))

    def _voi_center(self):
        return (float(self._voi_cz), float(self._voi_cy), float(self._voi_cx))

    # ------------------------------------------------- visualización en vivo
    def _on_cmap_changed(self, name):
        self._cmap = str(name)
        self._draw_previews()
        self.canvas.draw_idle()

    def _on_window_changed(self, _=None):
        lo = self.slider_lo.value() / 100.0
        hi = self.slider_hi.value() / 100.0
        if hi <= lo:
            hi = min(1.0, lo + 0.02)
            self.slider_hi.blockSignals(True)
            self.slider_hi.setValue(int(round(hi * 100)))
            self.slider_hi.blockSignals(False)
        self._win_lo = lo
        self._win_hi = hi
        self.lbl_window.setText(f"{int(lo * 100)}–{int(hi * 100)}%")
        self._draw_previews()
        self.canvas.draw_idle()

    def _on_fine_rot_changed(self, _=None):
        self._set_fine_rot_values(
            int(self.spin_fine_sa.value()),
            int(self.spin_fine_hla.value()),
            int(self.spin_fine_vla.value()),
            recompute=True,
        )

    def _reset_fine_rot(self):
        self._set_fine_rot_values(0, 0, 0, recompute=True)

    def _set_fine_rot_values(self, sa_deg: int, hla_deg: int, vla_deg: int, recompute=True):
        sa = int(np.clip(int(sa_deg), -45, 45))
        hla = int(np.clip(int(hla_deg), -45, 45))
        vla = int(np.clip(int(vla_deg), -45, 45))
        self._fine_rot_sa = sa
        self._fine_rot_hla = hla
        self._fine_rot_vla = vla
        if hasattr(self, "spin_fine_sa"):
            self.spin_fine_sa.blockSignals(True); self.spin_fine_sa.setValue(sa); self.spin_fine_sa.blockSignals(False)
            self.spin_fine_hla.blockSignals(True); self.spin_fine_hla.setValue(hla); self.spin_fine_hla.blockSignals(False)
            self.spin_fine_vla.blockSignals(True); self.spin_fine_vla.setValue(vla); self.spin_fine_vla.blockSignals(False)
        if recompute:
            self._recompute_and_draw()

    def _hla_row_to_k(self, row):
        """Convierte fila mostrada en HLA (apex arriba) a índice k interno."""
        n = int(self._reo.shape[0]) if hasattr(self, "_reo") else int(self._N)
        return int(np.clip((n - 1) - int(round(float(row))), 0, n - 1))

    def _k_to_hla_row(self, k):
        """Convierte índice k interno a fila mostrada en HLA (apex arriba)."""
        n = int(self._reo.shape[0]) if hasattr(self, "_reo") else int(self._N)
        kk = int(np.clip(int(round(float(k))), 0, n - 1))
        return int((n - 1) - kk)

    def _win_bounds(self):
        vmin = self._win_lo * self._disp_max
        vmax = self._win_hi * self._disp_max
        if vmax <= vmin:
            vmax = vmin + 1e-6
        return vmin, vmax

    # ------------------------------------------------------- recompute/draw
    def _recompute_and_draw(self, full=False):
        u = self._long_axis_vector()
        center = self._voi_center()
        # Refresca los cortes tomográficos de referencia para que el slab siga
        # el centro de la VOI (excluye hígado a otra profundidad).
        self._ap_view, self._ll_view = self._slab_views()
        vol_voi = self._apply_voi(self._ung)
        out = self._voi_out_size()
        self._out = out
        try:
            reo0 = reslice_from_vector(vol_voi, center, u, out, order=1)
            self._reo = self._apply_post_ops_3d(reo0)
        except Exception:
            self._reo = np.zeros((out, out, out))
        # Rango de display robusto sobre el volumen reorientado (heart-box).
        self._disp_max = float(np.percentile(self._reo, 99.5)) or 1.0
        # Límites Base/Ápex: en el primer cálculo se fijan relativos al tamaño
        # reorientado real; luego se respeta lo que ajustó el usuario (con clamp).
        if not self._limits_init:
            self._base_k = int(round(out * 0.15))
            self._apex_k = int(round(out * 0.85))
            self._limits_init = True
        self._base_k = int(np.clip(self._base_k, 0, out - 1))
        self._apex_k = int(np.clip(self._apex_k, 0, out - 1))
        if hasattr(self, "spin_base"):
            self.spin_base.blockSignals(True); self.spin_base.setRange(0, out - 1); self.spin_base.setValue(self._base_k); self.spin_base.blockSignals(False)
            self.spin_apex.blockSignals(True); self.spin_apex.setRange(0, out - 1); self.spin_apex.setValue(self._apex_k); self.spin_apex.blockSignals(False)
        self._draw_orient_views()
        self._draw_previews()
        # Inclinacion del eje largo respecto de la vertical (z).
        tilt = np.degrees(np.arctan2(float(np.hypot(u[1], u[2])), abs(float(u[0])) + 1e-9))
        brief = f"Inclinación {tilt:+.0f}° · Ajustes: {self._ops_note()} · {self._fine_rot_note()}"
        self.lbl_angles.setText(brief)
        self.lbl_angles.setToolTip(self._geo_note)
        self.canvas.draw_idle()

    def _draw_orient_views(self):
        # Vista Anterior (AP): fila=z, col=x. Linea de eje largo base->apex.
        self.ax_tra.clear()
        self.ax_tra.set_facecolor("#020611")
        self.ax_tra.imshow(_norm(self._ap_view), cmap="gray", vmin=0, vmax=1,
                           interpolation="bicubic", aspect="auto")
        self.ax_tra.plot(
            [self.h_tra1.x, self.h_tra2.x], [self.h_tra1.y, self.h_tra2.y],
            "-", color="#33ff66", lw=1.6,
        )
        self.ax_tra.plot(self.h_tra1.x, self.h_tra1.y, "o", color="#ffe14d", ms=7)
        self.ax_tra.plot(self.h_tra2.x, self.h_tra2.y, "o", color="#ff5a5a", ms=7)
        # VOI elíptica (heart box): AP define semiejes en x (ancho) y z (alto).
        ap_cx = self._ap_disp_from_vol_x(self._voi_cx)
        ap_rx_x = self._ap_disp_from_vol_x(self._voi_cx + self._voi_rx)
        ap_rx = abs(ap_rx_x - ap_cx)
        self.ax_tra.add_patch(Ellipse(
            (ap_cx, self._voi_cz), width=2.0 * ap_rx, height=2.0 * self._voi_rz,
            fill=False, edgecolor="#ffd21a", lw=1.6, ls="-"))
        self.ax_tra.plot(ap_cx, self._voi_cz, "s", color="#ffd21a", ms=7)
        self.ax_tra.plot(ap_rx_x, self._voi_cz - self._voi_rz, "s", color="#ffd21a", ms=7)
        ttl_ap = "Anterior (AP)" if self._ap_angle is not None else "Anterior (aprox.)"
        self.ax_tra.set_title(f"{ttl_ap} · corte tomográfico + eje largo", color="white", fontsize=9, fontweight="bold")
        self.ax_tra.set_xticks([]); self.ax_tra.set_yticks([])

        # Vista Lateral izquierda: fila=z, col=y.
        self.ax_cor.clear()
        self.ax_cor.set_facecolor("#020611")
        self.ax_cor.imshow(_norm(self._ll_view), cmap="gray", vmin=0, vmax=1,
                           interpolation="bicubic", aspect="auto")
        self.ax_cor.plot(
            [self.h_cor1.x, self.h_cor2.x], [self.h_cor1.y, self.h_cor2.y],
            "-", color="#33ff66", lw=1.6,
        )
        self.ax_cor.plot(self.h_cor1.x, self.h_cor1.y, "o", color="#ffe14d", ms=7)
        self.ax_cor.plot(self.h_cor2.x, self.h_cor2.y, "o", color="#ff5a5a", ms=7)
        # VOI en lateral: define semiejes en y (ancho) y z (alto).
        ll_cy = self._ll_disp_from_vol_y(self._voi_cy)
        ll_ry_x = self._ll_disp_from_vol_y(self._voi_cy + self._voi_ry)
        ll_ry = abs(ll_ry_x - ll_cy)
        self.ax_cor.add_patch(Ellipse(
            (ll_cy, self._voi_cz), width=2.0 * ll_ry, height=2.0 * self._voi_rz,
            fill=False, edgecolor="#ffd21a", lw=1.6, ls="-"))
        self.ax_cor.plot(ll_cy, self._voi_cz, "s", color="#ffd21a", ms=7)
        self.ax_cor.plot(ll_ry_x, self._voi_cz - self._voi_rz, "s", color="#ffd21a", ms=7)
        ttl_ll = "Lateral izq." if self._ll_angle is not None else "Lateral (aprox.)"
        self.ax_cor.set_title(f"{ttl_ll} · eje largo + VOI", color="white", fontsize=9, fontweight="bold")
        self.ax_cor.set_xticks([]); self.ax_cor.set_yticks([])

    def _draw_previews(self):
        reo = self._reo
        n = reo.shape[0]
        cmap = self._cmap
        vmin, vmax = self._win_bounds()
        kmid = int(np.clip((self._base_k + self._apex_k) // 2, 0, n - 1))
        jmid = n // 2
        imid = n // 2

        sa = sa_slice(reo, kmid)
        hla = hla_slice(reo, jmid)
        vla = vla_slice(reo, imid)

        self.ax_sa.clear(); self.ax_sa.set_facecolor("#020611")
        self.ax_sa.imshow(sa, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bicubic")
        self.ax_sa.set_title(f"SA (k={kmid}) · ANT↑ SEP←", color="white", fontsize=9, fontweight="bold")
        # Caja guía (estilo Odyssey) para ajuste fino visual del corte SA.
        h_sa, w_sa = sa.shape[:2]
        bw = max(8.0, 0.48 * w_sa)
        bh = max(8.0, 0.48 * h_sa)
        bx0 = 0.5 * (w_sa - bw)
        by0 = 0.5 * (h_sa - bh)
        self.ax_sa.add_patch(Rectangle((bx0, by0), bw, bh, fill=False, edgecolor="#6ee7ff", lw=1.0, ls="--", alpha=0.85))
        cx_sa, cy_sa = 0.5 * (w_sa - 1), 0.5 * (h_sa - 1)
        self.ax_sa.plot([cx_sa - 2.0, cx_sa + 2.0], [cy_sa, cy_sa], color="#6ee7ff", lw=0.9)
        self.ax_sa.plot([cx_sa, cx_sa], [cy_sa - 2.0, cy_sa + 2.0], color="#6ee7ff", lw=0.9)
        # Handle de ajuste fino mouse-driven.
        self._fine_sa_handle = (bx0 + bw, by0)
        self.ax_sa.plot(self._fine_sa_handle[0], self._fine_sa_handle[1], "s", color="#6ee7ff", ms=5)
        self.ax_sa.text(0.03, 0.06, "Drag □: fine SA", transform=self.ax_sa.transAxes,
                color="#9cdcff", fontsize=7, fontweight="bold")
        self.ax_sa.set_xticks([]); self.ax_sa.set_yticks([])

        # HLA se dibuja con APEX arriba/BASE abajo (fila k invertida). Las líneas
        # de límite deben usar la coordenada de fila invertida: k' = (n-1) - k.
        self.ax_hla.clear(); self.ax_hla.set_facecolor("#020611")
        self.ax_hla.imshow(hla, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bicubic", aspect="auto")
        base_row = (n - 1) - self._base_k
        apex_row = (n - 1) - self._apex_k
        self.ax_hla.axhline(base_row, color="#ff3333", lw=1.6)
        self.ax_hla.axhline(apex_row, color="#ff3333", lw=1.6)
        self.ax_hla.axhline((n - 1) - kmid, color="#40ff5a", lw=1.0, ls="--")
        self.ax_hla.text(0.03, 0.05, "Ápex ▲ / Base ▼", transform=self.ax_hla.transAxes,
                         color="#7cf29a", fontsize=8, fontweight="bold")
        self._fine_hla_handle = (0.95 * max(1, hla.shape[1] - 1), 0.10 * max(1, hla.shape[0] - 1))
        self.ax_hla.plot(self._fine_hla_handle[0], self._fine_hla_handle[1], "s", color="#6ee7ff", ms=5)
        self.ax_hla.text(0.64, 0.06, "Drag □: fine HLA", transform=self.ax_hla.transAxes,
                 color="#9cdcff", fontsize=7, fontweight="bold")
        self.ax_hla.set_title("HLA · APEX↑ SEP← · límites", color="white", fontsize=9, fontweight="bold")
        self.ax_hla.set_xticks([]); self.ax_hla.set_yticks([])

        # VLA: BASE izq / APEX der (eje largo horizontal). Las líneas base/ápex
        # son verticales (columnas = k).
        self.ax_vla.clear(); self.ax_vla.set_facecolor("#020611")
        self.ax_vla.imshow(vla, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bicubic", aspect="auto")
        self.ax_vla.axvline(self._base_k, color="#ff3333", lw=1.2)
        self.ax_vla.axvline(self._apex_k, color="#ff3333", lw=1.2)
        self._fine_vla_handle = (0.95 * max(1, vla.shape[1] - 1), 0.10 * max(1, vla.shape[0] - 1))
        self.ax_vla.plot(self._fine_vla_handle[0], self._fine_vla_handle[1], "s", color="#6ee7ff", ms=5)
        self.ax_vla.text(0.64, 0.06, "Drag □: fine VLA", transform=self.ax_vla.transAxes,
                 color="#9cdcff", fontsize=7, fontweight="bold")
        self.ax_vla.set_title("VLA · ANT↑ BASE←", color="white", fontsize=9, fontweight="bold")
        self.ax_vla.set_xticks([]); self.ax_vla.set_yticks([])

        # Tira SA base/medio/apex (orientación anatómica fija).
        self.ax_sa_stack.clear(); self.ax_sa_stack.set_facecolor("#020611")
        kb = int(np.clip(self._base_k, 0, n - 1))
        ka = int(np.clip(self._apex_k, 0, n - 1))
        strip = np.concatenate([sa_slice(reo, kb), sa_slice(reo, kmid), sa_slice(reo, ka)], axis=1)
        self.ax_sa_stack.imshow(strip, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bicubic", aspect="auto")
        self.ax_sa_stack.set_title("SA Base · Medio · Ápex", color="white", fontsize=9, fontweight="bold")
        self.ax_sa_stack.set_xticks([]); self.ax_sa_stack.set_yticks([])

    # --------------------------------------------------------- interaccion
    def _pick(self, event) -> _Handle | None:
        if event.inaxes is None:
            return None
        cands = []
        if event.inaxes is self.ax_tra:
            cands = [
                self.h_tra1, self.h_tra2,
                _Handle("voi_ap_c", self.ax_tra, self._ap_disp_from_vol_x(self._voi_cx), self._voi_cz),
                _Handle("voi_ap_r", self.ax_tra, self._ap_disp_from_vol_x(self._voi_cx + self._voi_rx), self._voi_cz - self._voi_rz),
            ]
        elif event.inaxes is self.ax_cor:
            cands = [
                self.h_cor1, self.h_cor2,
                _Handle("voi_ll_c", self.ax_cor, self._ll_disp_from_vol_y(self._voi_cy), self._voi_cz),
                _Handle("voi_ll_r", self.ax_cor, self._ll_disp_from_vol_y(self._voi_cy + self._voi_ry), self._voi_cz - self._voi_rz),
            ]
        elif event.inaxes is self.ax_hla:
            # limites base/apex: elegir la linea mas cercana en k (data y).
            if event.ydata is None:
                return None
            if hasattr(self, "_fine_hla_handle"):
                fhx, fhy = self._fine_hla_handle
                tx, ty = self.ax_hla.transData.transform((fhx, fhy))
                if np.hypot(tx - event.x, ty - event.y) <= 16:
                    return _Handle("fine_hla", self.ax_hla, fhx, fhy)
            base_row = self._k_to_hla_row(self._base_k)
            apex_row = self._k_to_hla_row(self._apex_k)
            db = abs(event.ydata - base_row)
            da = abs(event.ydata - apex_row)
            return _Handle("apex_line" if da <= db else "base_line", self.ax_hla, event.xdata or 0, event.ydata, horizontal=True)
        elif event.inaxes is self.ax_vla:
            if event.xdata is None:
                return None
            if hasattr(self, "_fine_vla_handle"):
                fhx, fhy = self._fine_vla_handle
                tx, ty = self.ax_vla.transData.transform((fhx, fhy))
                if np.hypot(tx - event.x, ty - event.y) <= 16:
                    return _Handle("fine_vla", self.ax_vla, fhx, fhy)
            db = abs(event.xdata - self._base_k)
            da = abs(event.xdata - self._apex_k)
            return _Handle("apex_line_vla" if da <= db else "base_line_vla", self.ax_vla, event.xdata, event.ydata or 0, horizontal=True)
        elif event.inaxes is self.ax_sa:
            if hasattr(self, "_fine_sa_handle"):
                fhx, fhy = self._fine_sa_handle
                tx, ty = self.ax_sa.transData.transform((fhx, fhy))
                if np.hypot(tx - event.x, ty - event.y) <= 16:
                    return _Handle("fine_sa", self.ax_sa, fhx, fhy)
            return None
        best, bestd = None, 1e9
        trans = event.inaxes.transData
        for h in cands:
            hx, hy = trans.transform((h.x, h.y))
            d = float(np.hypot(hx - event.x, hy - event.y))
            if d < bestd:
                best, bestd = h, d
        return best if bestd <= 22 else None

    def _on_press(self, event):
        if event.button != 1:
            return
        self._drag = self._pick(event)
        self._fine_drag_start = None
        if self._drag is not None and self._drag.name in {"fine_sa", "fine_hla", "fine_vla"}:
            self._fine_drag_start = (
                float(event.x),
                float(event.y),
                int(self._fine_rot_sa),
                int(self._fine_rot_hla),
                int(self._fine_rot_vla),
            )

    def _on_motion(self, event):
        if self._drag is None or event.inaxes is None:
            return
        h = self._drag
        if h.name in {"fine_sa", "fine_hla", "fine_vla"}:
            if self._fine_drag_start is None:
                return
            x0, y0, sa0, hla0, vla0 = self._fine_drag_start
            dx = float(event.x) - x0
            dy = float(event.y) - y0
            if h.name == "fine_sa":
                self._set_fine_rot_values(sa0 + int(round(dx / 6.0)), hla0, vla0, recompute=True)
            elif h.name == "fine_hla":
                self._set_fine_rot_values(sa0, hla0 + int(round(-dy / 6.0)), vla0, recompute=True)
            else:  # fine_vla
                self._set_fine_rot_values(sa0, hla0, vla0 + int(round(-dy / 6.0)), recompute=True)
            return
        if event.xdata is None:
            return
        if h.name == "base_line":
            self._base_k = self._hla_row_to_k(event.ydata)
            self.spin_base.blockSignals(True); self.spin_base.setValue(self._base_k); self.spin_base.blockSignals(False)
            self._draw_previews(); self.canvas.draw_idle()
            return
        if h.name == "apex_line":
            self._apex_k = self._hla_row_to_k(event.ydata)
            self.spin_apex.blockSignals(True); self.spin_apex.setValue(self._apex_k); self.spin_apex.blockSignals(False)
            self._draw_previews(); self.canvas.draw_idle()
            return
        if h.name == "base_line_vla":
            self._base_k = int(np.clip(round(float(event.xdata)), 0, self._N - 1))
            self.spin_base.blockSignals(True); self.spin_base.setValue(self._base_k); self.spin_base.blockSignals(False)
            self._draw_previews(); self.canvas.draw_idle()
            return
        if h.name == "apex_line_vla":
            self._apex_k = int(np.clip(round(float(event.xdata)), 0, self._N - 1))
            self.spin_apex.blockSignals(True); self.spin_apex.setValue(self._apex_k); self.spin_apex.blockSignals(False)
            self._draw_previews(); self.canvas.draw_idle()
            return
        if h.name == "voi_ap_c":
            self._voi_cx = self._ap_vol_from_disp_x(float(event.xdata)); self._voi_cz = float(event.ydata)
            self._recompute_and_draw(); return
        if h.name == "voi_ap_r":
            ex = self._ap_vol_from_disp_x(float(event.xdata))
            self._voi_rx = max(3.0, abs(ex - self._voi_cx))
            self._voi_rz = max(3.0, abs(float(event.ydata) - self._voi_cz))
            self._recompute_and_draw(); return
        if h.name == "voi_ll_c":
            self._voi_cy = self._ll_vol_from_disp_y(float(event.xdata)); self._voi_cz = float(event.ydata)
            self._recompute_and_draw(); return
        if h.name == "voi_ll_r":
            ey = self._ll_vol_from_disp_y(float(event.xdata))
            self._voi_ry = max(3.0, abs(ey - self._voi_cy))
            self._voi_rz = max(3.0, abs(float(event.ydata) - self._voi_cz))
            self._recompute_and_draw(); return
        if event.inaxes is not h.axes:
            return
        h.x = float(event.xdata)
        h.y = float(event.ydata)
        self._recompute_and_draw()

    def _on_release(self, event):
        self._drag = None
        self._fine_drag_start = None

    def _on_base_spin(self, v):
        self._base_k = int(v)
        self._draw_previews(); self.canvas.draw_idle()

    def _on_apex_spin(self, v):
        self._apex_k = int(v)
        self._draw_previews(); self.canvas.draw_idle()

    # -------------------------------------------------------------- result
    def _accept(self):
        u = self._long_axis_vector()
        center = self._voi_center()
        out = self._voi_out_size()
        self.result_long_axis = u
        self.result_center = center
        self.thickness = int(self.spin_thick.value())
        self.base_k = int(min(self._base_k, self._apex_k))
        self.apex_k = int(max(self._base_k, self._apex_k))
        try:
            vol_voi = self._apply_voi(self._ung)
            reo_ung = reslice_from_vector(vol_voi, center, u, out, order=1)
            self.reoriented_ungated = self._apply_post_ops_3d(reo_ung)
            if self._gated is not None:
                cube_voi = self._apply_voi_gated(self._gated)
                reo_g = reslice_from_vector_gated(cube_voi, center, u, out, order=1)
                self.reoriented_gated = self._apply_post_ops_4d(reo_g)
        except Exception:
            self.reoriented_ungated = self._reo
        self.accept()

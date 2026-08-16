# -*- coding: utf-8 -*-
"""Panel 3D del VI: miocardio sólido + malla alambre ED/ES, con cine sincronizado.

Dos paneles PyVista lado a lado, en una ventana flotante:

- **Izquierdo (miocardio):** isosuperficie del volumen SA reorientado del gate
  actual (o la suma de gates si se quiere estática). Texturizada con el colormap
  del preview de la app. Rotación/zoom con el mouse (interactor VTK).
- **Derecho (malla):** "fantasma de alambre" del VI calculado por ECTb. La malla
  de ED queda fija (gris tenue, referencia del molde telediastólico) y la malla
  del gate actual se contrae/expande en naranja, sincronizada con el cine.

El cine es compartido: un QTimer avanza el gate en ambos paneles a la vez.

Export: captura frame a frame del cine a GIF (imageio) o AVI (imageio-ffmpeg),
para embeber en el informe HTML planeado.
"""
from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QSpinBox,
    QCheckBox, QComboBox, QFileDialog, QMessageBox, QWidget,
)
from ui.cine_widget import RangeSlider, VerticalColorStrip


class LV3DDialog(QDialog):
    """Ventana 3D del VI: miocardio sólido + malla ECTb, con cine."""

    def __init__(
        self,
        parent=None,
        *,
        lv_meshes: dict,
        myo_volume: np.ndarray | None = None,
        spacing_mm: tuple = (1.0, 1.0, 1.0),
        cmap_name: str = "odyssey_cool",
        interval_ms: int = 120,
        ectb_result=None,
        seg=None,
        pixel_mm: tuple = (1.0, 1.0),
        polar_map: np.ndarray | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("SINCRO — VI 3D")
        self.resize(1100, 620)

        self._lv = lv_meshes
        self._n_gates = int(lv_meshes["n_gates"])
        self._ed_gate = int(lv_meshes["ed_gate"])
        self._gate = self._ed_gate
        self._playing = False
        self._interval_ms = int(interval_ms)

        # Radios ECTb para la cáscara del miocardio (endo/epi por gate) y las
        # posiciones axiales. La cáscara late con el cine (no es estática).
        self._endo_radii = lv_meshes.get("endo_radii_mm")
        self._epi_radii = lv_meshes.get("epi_radii_mm")
        self._z_positions = lv_meshes.get("z_positions_mm")
        self._center_offsets_mm = lv_meshes.get("center_offsets_mm")
        self._origin_mm = lv_meshes.get("origin_mm", (0.0, 0.0, 0.0))

        # Volumen de actividad para mapear sobre la cáscara (defectos hipocaptantes
        # se ven como zonas oscuras/frías). Si es None, la cáscara va con color fijo.
        self._myo_volume = myo_volume
        self._spacing_mm = spacing_mm
        available_3d_cmaps = ("french", "turbo", "hot", "odyssey_cool", "coolwarm", "viridis")
        self._cmap_name = str(cmap_name) if str(cmap_name) in available_3d_cmaps else "french"
        try:
            from viz.colormaps import get_phase_cmap
            self._cmap = get_phase_cmap(self._cmap_name)
        except Exception:
            self._cmap = "turbo"
        self._color_low = 0.0
        self._color_high = 1.0

        # Datos para la Reconstrucción Dinámica 3D (volumen desde segmentación).
        self._ectb_result = ectb_result
        self._seg = seg
        self._pixel_mm = pixel_mm

        # Mapa polar de perfusión para proyectar sobre la cáscara (textura).
        self._polar_map = polar_map

        # Referencias a actores para poder actualizarlos sin recrear la escena.
        self._actor_myo = None
        self._actor_ed = None
        self._actor_gate = None
        self._meshes = lv_meshes["meshes"]
        self._surface_status = "sin mapa polar"

        self._build_ui()

        # Escena 3D (PyVista/Qt). Se construye perezosamente para que el dialog
        # abra rápido aunque VTK tarde en inicializar.
        self._plotter = None
        self._init_3d(myo_volume, spacing_mm, cmap_name)

        self._timer = QTimer(self)
        self._timer.setInterval(self._interval_ms)
        self._timer.timeout.connect(self._advance_gate)

    # ------------------------------------------------------------ UI
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Controles superiores.
        bar = QHBoxLayout()
        self.btn_play = QPushButton("▶ Cine")
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._toggle_play)
        bar.addWidget(self.btn_play)

        bar.addWidget(QLabel("Gate"))
        self.spin_gate = QSpinBox()
        self.spin_gate.setRange(1, self._n_gates)
        self.spin_gate.setValue(self._ed_gate + 1)
        self.spin_gate.valueChanged.connect(self._on_gate_spin)
        bar.addWidget(self.spin_gate)

        bar.addWidget(QLabel("Velocidad"))
        self.spin_ms = QSpinBox()
        self.spin_ms.setRange(30, 1000)
        self.spin_ms.setSingleStep(20)
        self.spin_ms.setValue(self._interval_ms)
        self.spin_ms.setSuffix(" ms")
        self.spin_ms.valueChanged.connect(lambda v: self._timer.setInterval(int(v)))
        bar.addWidget(self.spin_ms)

        self.chk_link = QCheckBox("Cámaras vinculadas")
        self.chk_link.setChecked(True)
        bar.addWidget(self.chk_link)

        # Modo de superficie del miocardio: cáscara ECTb (geométrica, rápida),
        # borde por gradiente (física, más lenta pero más precisa), o
        # Reconstrucción Dinámica 3D (volumen desde segmentación ECTb por gate).
        bar.addSpacing(12)
        bar.addWidget(QLabel("Superficie:"))
        self.combo_surface = QComboBox()
        self.combo_surface.addItem("Cáscara ECTb (rápida)", "shell")
        self.combo_surface.addItem("Borde por gradiente (física)", "gradient")
        self.combo_surface.addItem("Reconstrucción Dinámica 3D", "dynamic")
        self.combo_surface.setCurrentIndex(0)
        self.combo_surface.setMinimumWidth(180)  # que se lea completo
        self.combo_surface.setToolTip(
            "Cáscara ECTb: geometría pura endo→epi, rápida, no muestra actividad real.\n"
            "Borde por gradiente: detecta el borde del miocardio por máximo gradiente "
            "(el 'borde blanco' de la escala french), más lento pero más preciso y "
            "muestra la actividad real del volumen.\n"
            "Reconstrucción Dinámica 3D: reconstruye el volumen del miocardio desde la "
            "segmentación ECTb por gate (sin fondo), apila los cortes en Z y genera la "
            "isosuperficie. Es la pared exacta del VI latiendo, sin fondo ni hígado."
        )
        self.combo_surface.currentIndexChanged.connect(self._on_surface_mode_changed)
        bar.addWidget(self.combo_surface)

        bar.addStretch(1)

        self.btn_export = QPushButton("Exportar cine (GIF/AVI)")
        self.btn_export.clicked.connect(self._export_cine)
        bar.addWidget(self.btn_export)

        root.addLayout(bar)

        # Motor de color lateral: el RangeSlider necesita recorrido vertical
        # real; en una barra horizontal de 54 px los dos handles se solapaban.
        color_side = QWidget()
        color_side.setFixedWidth(150)
        color_bar = QVBoxLayout(color_side)
        color_bar.setContentsMargins(4, 4, 4, 4)
        color_bar.addWidget(QLabel("Color 3D:"))
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(["french", "turbo", "hot", "odyssey_cool", "coolwarm", "viridis"])
        self.combo_cmap.setCurrentText(self._cmap_name)
        self.combo_cmap.currentTextChanged.connect(self._on_3d_cmap_changed)
        color_bar.addWidget(self.combo_cmap)
        color_bar.addWidget(QLabel("Base/Top"))
        range_row = QHBoxLayout()
        self.color_range = RangeSlider()
        self.color_range.setMinimumHeight(330)
        self.color_range.set_values(0, 100)
        self.color_range.valuesChanged.connect(self._on_3d_range_changed)
        range_row.addWidget(self.color_range, 1)
        self.color_strip = VerticalColorStrip(self._cmap_name)
        self.color_strip.setMinimumHeight(330)
        range_row.addWidget(self.color_strip)
        color_bar.addLayout(range_row, 1)
        self.lbl_color_range = QLabel("0–100%")
        self.lbl_color_range.setAlignment(Qt.AlignmentFlag.AlignCenter)
        color_bar.addWidget(self.lbl_color_range)

        # Contenedor de la escena 3D (lo llena _init_3d).
        self._3d_host = QWidget()
        self._3d_host.setMinimumHeight(480)
        self._3d_host.setStyleSheet("background: #0b1220;")
        content = QHBoxLayout()
        content.addWidget(color_side)
        content.addWidget(self._3d_host, 1)
        root.addLayout(content, 1)

        self.lbl_orientation = QLabel(
            "Orientación: ÁPEX = punta · BASE = anillo abierto · "
            "ANTERIOR arriba · INFERIOR abajo · SEPTAL izquierda · LATERAL derecha"
        )
        self.lbl_orientation.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_orientation.setStyleSheet("color: #334155; font-size: 11px; font-weight: 600;")
        root.addWidget(self.lbl_orientation)

        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color: #666666; font-size: 11px;")
        root.addWidget(self.lbl_info)

    # ------------------------------------------------------------ escena 3D
    def _init_3d(self, myo_volume, spacing_mm, cmap_name):
        from pyvistaqt import QtInteractor
        import pyvista as pv

        layout = QHBoxLayout(self._3d_host)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plotter = QtInteractor(self._3d_host, shape=(1, 2))
        layout.addWidget(self._plotter.interactor)

        # ---------------- Panel izquierdo: miocardio sólido ----------------
        self._plotter.subplot(0, 0)
        self._plotter.set_background("#0b1220")
        self._rebuild_myo_surface()
        # Ejes de referencia que giran solidarios con la escena.
        self._plotter.add_axes(line_width=2, labels_off=False)
        self._add_anatomical_labels("anat_left")

        # ---------------- Panel derecho: malla alambre ---------------------
        self._plotter.subplot(0, 1)
        self._plotter.set_background("#0b1220")
        mesh_ed = self._meshes[self._ed_gate]
        self._actor_ed = self._plotter.add_mesh(
            mesh_ed, style="wireframe", color="#4a6a8a", line_width=1.2,
            opacity=0.55, name="mesh_ed",
        )
        mesh_g = self._meshes[self._gate]
        self._actor_gate = self._plotter.add_mesh(
            mesh_g, style="wireframe", color="#ffb020", line_width=2.0,
            name="mesh_gate",
        )
        self._plotter.add_axes(line_width=2, labels_off=False)
        self._add_anatomical_labels("anat_right")

        # Vincular cámaras (rotás una, la otra sigue).
        if self.chk_link.isChecked():
            self._plotter.link_views()

        self._plotter.reset_camera()
        self._update_info()

    def _add_anatomical_labels(self, name: str):
        """Agrega referencias anatómicas solidarias con la malla.

        Convención SA validada por el módulo ECTb: anterior=-Y,
        inferior=+Y, septal=-X, lateral=+X; Z va de ápex a base.
        """
        if self._plotter is None or not self._meshes:
            return
        mesh = self._meshes[self._ed_gate]
        xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in mesh.bounds)
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)
        pad_xy = 0.12 * max(xmax - xmin, ymax - ymin, 1.0)
        pad_z = 0.10 * max(zmax - zmin, 1.0)
        points = np.asarray([
            [cx, cy, zmin - pad_z],
            [cx, cy, zmax + pad_z],
            [cx, ymin - pad_xy, cz],
            [cx, ymax + pad_xy, cz],
            [xmin - pad_xy, cy, cz],
            [xmax + pad_xy, cy, cz],
        ], dtype=np.float64)
        labels = ["ÁPEX", "BASE", "ANTERIOR", "INFERIOR", "SEPTAL", "LATERAL"]
        try:
            self._plotter.add_point_labels(
                points, labels, name=name, always_visible=True,
                show_points=False, shape=None, font_size=11,
                text_color="#f8fafc", render_points_as_spheres=False,
            )
        except Exception:
            pass

    # ------------------------------------------------------------ cine
    def _rebuild_myo_surface(self):
        """Construye la superficie del miocardio del gate actual.

        Dos modos (combo "Superficie"):
        - **Cáscara ECTb** (shell): geometría pura endo→epi por gate, rápida,
          no muestra actividad real. Late con el cine.
        - **Borde por gradiente** (gradient): detecta el borde del miocardio
          por máximo gradiente (el 'borde blanco' de la escala french), más
          lento pero más preciso y muestra la actividad real del volumen.
        """
        if self._plotter is None:
            return
        mode = str(self.combo_surface.currentData()) if hasattr(self, "combo_surface") else "shell"

        if mode == "gradient":
            # Borde por gradiente: superficie de máximo gradiente del volumen.
            if self._myo_volume is None or not np.asarray(self._myo_volume).size:
                return
            from core.lv_mesh import gradient_edge_mesh
            try:
                shell = gradient_edge_mesh(
                    self._myo_volume, self._spacing_mm,
                    smooth_sigma=1.0, grad_percentile=85.0, level_frac=0.5,
                )
            except Exception:
                return
            if shell is None or shell.n_points == 0:
                return
            # El borde por gradiente no tiene actividad sampleada (es una sola
            # superficie del borde, no la pared completa). Se renderiza con color
            # fijo y opacidad para ver la forma.
            self._plotter.subplot(0, 0)
            if self._actor_myo is None:
                self._actor_myo = self._plotter.add_mesh(
                    shell, color="#c0392b", smooth_shading=True, name="myo",
                    ambient=0.3, diffuse=0.7, specular=0.2, opacity=0.85,
                )
            else:
                self._actor_myo.mapper.SetInputData(shell)
            self._plotter.render()
            return

        if mode == "dynamic":
            # Reconstrucción Dinámica 3D: volumen del miocardio desde la
            # segmentación ECTb del gate actual (sin fondo), isosuperficie.
            if self._myo_volume is None or not np.asarray(self._myo_volume).size:
                return
            from core.lv_mesh import dynamic_volume_mesh
            try:
                shell = dynamic_volume_mesh(
                    self._ectb_result, self._seg, self._pixel_mm, self._spacing_mm[0],
                    gate_index=self._gate, level=0.5,
                )
            except Exception:
                return
            if shell is None or shell.n_points == 0:
                return
            self._plotter.subplot(0, 0)
            if self._actor_myo is None:
                self._actor_myo = self._plotter.add_mesh(
                    shell, color="#c0392b", smooth_shading=True, name="myo",
                    ambient=0.3, diffuse=0.7, specular=0.2,
                )
            else:
                self._actor_myo.mapper.SetInputData(shell)
            self._plotter.render()
            return

        # Modo cáscara ECTb (default).
        if self._endo_radii is None or self._epi_radii is None:
            return
        from core.lv_mesh import myocardium_shell_mesh, sample_volume_on_mesh
        try:
            shell = myocardium_shell_mesh(
                self._endo_radii[self._gate],
                self._epi_radii[self._gate],
                self._z_positions,
                centers_mm=self._center_offsets_mm,
                shape_index=0.0,      # preservar contornos medidos; no imponer elipsoide
                interp_z=3.0,         # loft suave entre cortes reales
                smooth_angular=1.0,   # quitar serrucho angular sin borrar asimetría
                smooth_iter=6,
                smooth_relax=0.04,
                apex_virtual_rings=5,
                apex_taper=0.32,
            )
        except Exception:
            return

        # Samplear la actividad del volumen sobre la cáscara (si hay volumen).
        # Si el sampleo falla (origen desalineado), la cáscara se colorea por
        # ESPESOR de pared (calculado de los radios ECTb, no del volumen).
        scalar_name = None
        polar_status = "sin mapa polar"
        if self._polar_map is not None and np.asarray(self._polar_map).size:
            # Proyectar el mapa polar de perfusión sobre la cáscara (textura).
            from core.lv_mesh import polar_texture_on_shell
            try:
                shell = polar_texture_on_shell(shell, self._polar_map, self._z_positions)
                perf = np.asarray(shell.point_data.get("perfusion", []), dtype=np.float64)
                if perf.size and np.isfinite(perf).any() and float(np.nanmax(perf) - np.nanmin(perf)) > 1e-6:
                    scalar_name = "perfusion"
                    polar_status = f"mapa polar {float(np.nanmin(perf)):.2f}-{float(np.nanmax(perf)):.2f}"
                else:
                    polar_status = "mapa polar sin rango útil"
            except Exception:
                polar_status = "error en mapa polar"
        # No caer al sampleo volumétrico si falta el mapa polar: esa vía
        # depende de una transformación física mesh↔volumen que todavía no
        # está validada y podía devolver todo cero, produciendo una malla negra.
        # El fallback seguro y clínicamente interpretable es thickness.

        # ``add_mesh(..., smooth_shading=True)`` calcula normales al crear el
        # actor, pero en el cine reemplazamos su dataset directamente. Cada
        # frame nuevo debe traer normales de vértice para no verse facetado.
        try:
            shell = shell.compute_normals(
                point_normals=True, cell_normals=False,
                split_vertices=False, consistent_normals=True,
            )
        except Exception:
            pass

        self._plotter.subplot(0, 0)
        if self._actor_myo is None:
            if scalar_name is not None and scalar_name in shell.point_data:
                vals = np.asarray(shell.point_data[scalar_name], dtype=np.float64)
                clim = self._perfusion_clim(vals)
                self._actor_myo = self._plotter.add_mesh(
                    shell, scalars=scalar_name, cmap=self._cmap, clim=clim,
                    smooth_shading=True, name="myo",
                    ambient=0.3, diffuse=0.7, specular=0.2,
                    show_scalar_bar=False, interpolate_before_map=True,
                )
            elif "thickness" in shell.point_data:
                # Colorear por espesor de pared (mm). Más grueso = más frío
                # (azul), más fino = más caliente (rojo). Es una medida física
                # real que no depende del volumen ni de la alineación.
                self._actor_myo = self._plotter.add_mesh(
                    shell, scalars="thickness", cmap="coolwarm",
                    smooth_shading=True, name="myo",
                    ambient=0.3, diffuse=0.7, specular=0.2,
                    show_scalar_bar=True,
                    scalar_bar_args={"title": "Espesor (mm)", "color": "white"},
                )
            else:
                self._actor_myo = self._plotter.add_mesh(
                    shell, color="#c0392b", smooth_shading=True, name="myo",
                    ambient=0.3, diffuse=0.7, specular=0.2,
                )
        else:
            self._actor_myo.mapper.SetInputData(shell)
            # Al cambiar de gate VTK pierde la selección explícita del array y
            # podía colorear toda la superficie con un único valor. Reafirmar
            # perfusión + rango en cada frame mantiene el mismo LUT del estático.
            if scalar_name is not None and scalar_name in shell.point_data:
                vals = np.asarray(shell.point_data[scalar_name], dtype=np.float64)
                clim = self._perfusion_clim(vals)
                mapper = self._actor_myo.mapper
                mapper.SetScalarModeToUsePointFieldData()
                mapper.SelectColorArray(scalar_name)
                mapper.SetScalarRange(*clim)
                mapper.ScalarVisibilityOn()
                mapper.InterpolateScalarsBeforeMappingOn()
            try:
                self._actor_myo.mapper.Update()
            except Exception:
                pass
        self._surface_status = polar_status
        self._update_info()
        self._plotter.render()

    def _perfusion_clim(self, fallback_values: np.ndarray) -> tuple[float, float]:
        """Rango de color fijo para todos los gates del cine."""
        source = np.asarray(self._polar_map, dtype=np.float64) if self._polar_map is not None else fallback_values
        finite = source[np.isfinite(source)]
        if finite.size == 0:
            finite = np.asarray(fallback_values, dtype=np.float64)
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        span = max(vmax - vmin, 1e-8)
        return (vmin + self._color_low * span, vmin + self._color_high * span)

    def _on_3d_cmap_changed(self, name: str):
        try:
            from viz.colormaps import get_phase_cmap
            self._cmap = get_phase_cmap(str(name))
        except Exception:
            self._cmap = "turbo"
        self.color_strip.set_cmap(str(name))
        self._recreate_myo_actor()

    def _on_3d_range_changed(self, low: int, high: int):
        self._color_low = float(low) / 100.0
        self._color_high = float(high) / 100.0
        self.lbl_color_range.setText(f"{low}–{high}%")
        self._recreate_myo_actor()

    def _recreate_myo_actor(self):
        if self._plotter is not None and self._actor_myo is not None:
            try:
                self._plotter.subplot(0, 0)
                self._plotter.remove_actor(self._actor_myo)
            except Exception:
                pass
        self._actor_myo = None
        self._rebuild_myo_surface()

    def _on_surface_mode_changed(self, _idx):
        # Al cambiar de modo, recrear la superficie (el actor puede tener
        # distintos datos: cáscara con actividad vs borde sin actividad).
        if self._plotter is not None and self._actor_myo is not None:
            try:
                self._plotter.subplot(0, 0)
                self._plotter.remove_actor(self._actor_myo)
            except Exception:
                pass
        self._actor_myo = None
        self._rebuild_myo_surface()

    def _toggle_play(self, on: bool):
        self._playing = bool(on)
        self.btn_play.setText("⏸ Pausa" if on else "▶ Cine")
        if on:
            self._timer.start()
        else:
            self._timer.stop()

    def _advance_gate(self):
        self._gate = (self._gate + 1) % self._n_gates
        self.spin_gate.blockSignals(True)
        self.spin_gate.setValue(self._gate + 1)
        self.spin_gate.blockSignals(False)
        self._update_meshes()

    def _on_gate_spin(self, v: int):
        self._gate = int(v) - 1
        self._update_meshes()

    def _update_meshes(self):
        if self._plotter is None:
            return
        # Actualizar la malla del gate actual sin recrear el actor (mismo n° de
        # puntos entre gates: la topología no cambia, solo las coordenadas).
        self._plotter.subplot(0, 1)
        new_mesh = self._meshes[self._gate]
        if self._actor_gate is not None:
            # Reemplazo del dataset del actor (rápido, sin re-mapear).
            self._actor_gate.mapper.SetInputData(new_mesh)
        # La cáscara del miocardio también late: recalcularla con el gate actual.
        self._rebuild_myo_surface()
        self._plotter.render()
        self._update_info()

    def _update_info(self):
        self.lbl_info.setText(
            f"Gate {self._gate + 1}/{self._n_gates}   ·   {self._surface_status}   ·   "
            f"ED = gate {self._ed_gate + 1} "
            f"(malla gris fija)   ·   Arrastrá para rotar, rueda para zoom."
        )

    # ------------------------------------------------------------ export
    def _export_cine(self):
        """Captura el cine frame a frame y exporta a GIF o AVI."""
        if self._plotter is None:
            return
        path, filt = QFileDialog.getSaveFileName(
            self, "Exportar cine 3D", "vi_3d_cine.gif",
            "GIF animado (*.gif);;Video AVI (*.avi)",
        )
        if not path:
            return
        try:
            import imageio.v2 as imageio
        except Exception:
            QMessageBox.warning(self, "SINCRO", "Falta imageio para exportar.")
            return

        was_playing = self._playing
        if was_playing:
            self._timer.stop()

        frames = []
        start_gate = self._gate
        for g in range(self._n_gates):
            self._gate = g
            self._update_meshes()
            self._plotter.render()
            img = self._plotter.screenshot(transparent_background=False)
            frames.append(np.asarray(img))
        self._gate = start_gate
        self._update_meshes()

        try:
            if path.lower().endswith(".avi"):
                imageio.mimsave(path, frames, fps=max(1, int(1000 / self._timer.interval())))
            else:
                imageio.mimsave(path, frames, duration=self._timer.interval() / 1000.0, loop=0)
            self._log_ok(f"Cine 3D exportado: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "SINCRO", f"No se pudo exportar: {exc}")

        if was_playing:
            self._timer.start()

    def _log_ok(self, msg: str):
        self.lbl_info.setText(msg)

    def closeEvent(self, ev):
        try:
            self._timer.stop()
            if self._plotter is not None:
                self._plotter.close()
        except Exception:
            pass
        super().closeEvent(ev)

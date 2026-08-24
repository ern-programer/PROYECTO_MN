# Wing: SINCRO

**Estado:** En desarrollo activo  
**Ubicación:** `d:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\`  
**Propósito:** Análisis de fase/disincronía cardíaca y amiloidosis SPECT

---

## Arquitectura

```
mod_SINCRO/
├── main.py                 # Entry point
├── ui/
│   ├── main_window.py      # Ventana principal
│   ├── amyloid_window.py   # Panel planar amiloidosis
│   ├── amyloid_spect_panel.py  # Panel SPECT/CT amiloidosis
│   ├── mip_rotator_widget.py   # Widget MIP 360° interactivo (NUEVO)
│   └── dual_spect_panel.py     # Panel dual-SPECT washout
├── core/
│   ├── amyloid_spect.py    # Lógica HMR-SPECT, VOISphere
│   └── washout_spect.py    # Lógica washout dual-SPECT
├── report/
│   └── report_generator.py
└── viz/
    └── polar_map.py
```

---

## Features implementados

### 1. HMR-SPECT (Amiloidosis 3D)
- **Archivo:** `core/amyloid_spect.py`
- **Clases:**
  - `VOISphere`: Esfera 3D con centro (cz, cy, cx) y radio en mm
  - `HmrSpectResult`: Resultado con HMR, counts, volúmenes, clasificación
  - `compute_hmr_spect()`: Cálculo HMR sobre volumen SPECT
- **Clasificación:**
  - HMR ≥ 1.6: POSITIVO (alta captación cardíaca)
  - HMR 1.5-1.6: EQUIVOCO
  - HMR < 1.5: NEGATIVO
- **VOIs en vivo:** Visualización temporal durante posicionamiento de localizadores

### 2. MIP Rotatorio 360° (NUEVO)
- **Archivo:** `ui/mip_rotator_widget.py`
- **Clase:** `MipRotatorWidget(QLabel)`
- **Interacción:**
  - Arrastrar horizontal: rotar azimut (0-360°)
  - Arrastrar vertical: rotar elevación (-60° a +60°)
  - Doble click: reset a vista AP
  - Rueda mouse: zoom
- **Features:**
  - Rotación 3D con `scipy.ndimage.rotate()`
  - Proyección MIP dinámica
  - VOIs proyectados como círculos con etiquetas
  - Sin overlay de HMR ni selector de dirección
  Washout % = (1 - HMR_t2 / HMR_t1) × 100
  ```
- **Interpretación:**
  - Washout < 0% (negativo): ATTR-CM probable
  - Washout > 20%: AL posible o captación inespecífica
  - Washout 0-20%: Indeterminado
- **Panel UI:** `ui/dual_spect_panel.py`
  - Carga T1 (ej: 1h) y T2 (ej: 3h)
  - Visualización lado a lado
  - Cálculo automático de washout

### 3. Integración con main_window
- Botón "Dual-SPECT Washout" agregado en sidebar
- Abre `DualSpectPanel` como diálogo independiente

---

## Dependencias

```
PyQt6
numpy
scipy
pydicom
pynetdicom
reportlab
```

---

## Próximos pasos

1. **Integrar carga real de DICOM** en `DualSpectPanel._load_spect_volume()`
2. **Sincronizar VOIs** entre T1 y T2 (mismas posiciones)
3. **Validar spacing** - ambos estudios deben tener mismo voxel size
4. **Exportar a PDF** - incluir washout en informe
5. **Testing** con estudios reales de 1h y 3h

---

## Referencias científicas

- Dorbala et al. ASNC/AHA/ASE/EANM/HFSA/ISA/SCMR/SNMMI Expert Consensus Recommendations for Multimodality Imaging in Cardiac Amyloidosis. J Card Fail. 2019.
- Washout en PYP/DPD/HMDP: diferenciación ATTR vs AL

---

## Notas técnicas

- **VOISphere NO tiene parámetro `label`** - solo (cz, cy, cx, radius_mm)
- **HMR raw es el valor clínico** - usar siempre hmr_raw para clasificación
- **Washout negativo = ATTR probable** (captación aumenta con el tiempo)

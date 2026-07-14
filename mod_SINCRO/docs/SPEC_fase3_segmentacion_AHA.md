# SPEC Fase 3 — Segmentación miocárdica + mapeo AHA 17 segmentos

**Autor de la especificación:** Opus 4.8 (diseño).
**Implementa:** GPT 5.3-codex.
**Fecha:** 2026-07-14.
**Contexto:** módulo SINCRO, `mod_SINCRO/`. El motor de fase (Fase 1) ya está hecho y
validado. Esta fase reemplaza la "máscara provisional por umbral" por una segmentación
real del miocardio del VI y agrega el mapeo a los 17 segmentos AHA + territorios.

> **Cómo usar esta spec (Codex):** implementá EXACTAMENTE las firmas y comportamientos
> descritos. No cambies nombres de funciones ni de campos. Cuando una decisión quede
> abierta, elegí lo más simple y dejá un `# TODO` comentado. Corré los tests al final.

---

## 0. Contexto de datos (ya resuelto en Fase 0/1)

- El cubo de entrada es `(n_gates, n_slices, H, W)` float64 (salida de `core.dicom_loader.load(...).cube`).
- Para el estudio de referencia: `(8, 19, 22, 22)`.
- **Eje corto (Short Axis):** cada `slice` es un corte perpendicular al eje largo del VI.
  El VI aparece como un **anillo (dona)** de actividad; el centro es la cavidad.
- Orden de slices: del ápex a la base (o al revés). NO asumir; ver §3.4 (detección).

Dependencias permitidas: numpy, scipy, opencv-python (cv2). Ya instaladas.

---

## 1. Archivos a crear/editar

1. `core/segmentation.py` — segmentación del miocardio (máscara 3D booleana).
2. `core/aha_segments.py` — mapeo voxel → segmento AHA (1-17) + territorios.
3. `tests/test_segmentation.py` — tests (sintético + real).

---

## 2. `core/segmentation.py`

### 2.1 Firma principal

```python
@dataclass
class SegmentationResult:
    mask: np.ndarray          # (n_slices, H, W) bool — miocardio
    center_per_slice: np.ndarray   # (n_slices, 2) float — (cy, cx) centro de cavidad por slice
    inner_radius: np.ndarray  # (n_slices,) float — radio endocárdico aprox (px)
    outer_radius: np.ndarray  # (n_slices,) float — radio epicárdico aprox (px)
    method: str
    n_voxels: int

def segment_myocardium(
    cube: np.ndarray,
    method: str = "auto",       # "auto" | "manual" | "threshold"
    threshold_frac: float = 0.35,
    smooth_sigma: float = 1.0,
    manual_rois: dict | None = None,   # solo si method="manual" (ver 2.4)
) -> SegmentationResult:
    ...
```

### 2.2 Imagen base para segmentar
Usar la **imagen sumada temporal**: `mean_img = cube.mean(axis=0)` → `(n_slices, H, W)`.
(La perfusión es estable en el tiempo; sumar mejora SNR para segmentar.)

### 2.3 method="auto" (thresholding adaptativo + morfología) — PRINCIPAL para el MVP
Por cada slice:
1. Suavizar con `scipy.ndimage.gaussian_filter(img, smooth_sigma)`.
2. Umbral relativo: `thr = threshold_frac * img.max()` (por slice). `bin = img > thr`.
3. Limpieza morfológica (cv2 o scipy): abrir (quitar puntos), cerrar (rellenar huecos del anillo).
4. Quedarse con el **componente conexo más grande** (`scipy.ndimage.label` + tamaño).
5. Guardar en `mask[s]`.
6. Centro de cavidad `center_per_slice[s]`: centroide de los píxeles del anillo
   (`scipy.ndimage.center_of_mass` sobre la máscara del slice). Si el slice está vacío
   (ápex/base sin miocardio), marcar centro = NaN y máscara vacía.
7. `inner_radius`/`outer_radius`: estimar como percentil 20 y 80 de la distancia de los
   voxels de máscara al centro (aprox; sirve para el mapeo radial de §3).

Slices sin anillo válido (área < mínimo, ej < 8 px) → máscara vacía en ese slice.

### 2.4 method="manual" (para validar contra LVSD)
`manual_rois` = dict `{slice_index: (cy, cx, r_inner, r_outer)}`. Construir un anillo
booleano por slice: voxels con `r_inner <= dist_al_centro <= r_outer`. Slices no provistos
quedan vacíos. (Sirve para dibujar la ROI a mano y comparar con la auto.)

### 2.5 method="threshold"
Igual que "auto" pero SIN quedarse con el componente más grande ni morfología (la máscara
provisional que ya usa `test_engine_real.py`). Se mantiene para retrocompatibilidad.

### 2.6 CLI
```python
if __name__ == "__main__":
    # python -m core.segmentation <archivo_SA_gated.dcm>
    # carga con dicom_loader, segmenta method="auto", imprime resumen (nº voxels, nº slices con anillo).
```

---

## 3. `core/aha_segments.py` — mapeo AHA 17 segmentos (LO DIFÍCIL, ya diseñado)

### 3.1 Concepto
El modelo AHA de 17 segmentos divide el VI en:
- **Basal (1-6):** 6 sectores angulares (60° c/u).
- **Medio (7-12):** 6 sectores angulares (60° c/u).
- **Apical (13-16):** 4 sectores angulares (90° c/u).
- **Ápex (17):** el/los slice(s) más apicales, sin división angular.

Dos coordenadas por voxel miocárdico:
- **Longitudinal (nivel):** según la posición del slice entre ápex y base → basal/medio/apical/ápex.
- **Circunferencial (ángulo):** según el ángulo del voxel respecto del centro de la cavidad de su slice.

### 3.2 División longitudinal (nivel) — ALGORITMO
1. Determinar el rango de slices con miocardio válido (máscara no vacía): `s_first..s_last`.
2. **Detectar orientación ápex→base** (§3.4). Reordenar índices para que vayan de BASE (0) a ÁPEX (1).
3. Sea `L = nº de slices válidos`. Repartir a lo largo del eje normalizado `u∈[0,1]` (0=base, 1=ápex):
   - `u < 0.35`  → **basal**
   - `0.35 ≤ u < 0.70` → **medio**
   - `0.70 ≤ u < 0.90` → **apical**
   - `u ≥ 0.90` → **ápex (segmento 17)**
   (Proporciones estándar aproximadas; ajustables. El ápex se lleva la punta.)

### 3.3 División circunferencial (ángulo) — ALGORITMO
Para cada voxel miocárdico de un slice, con centro de cavidad `(cy, cx)`:
```
angle = atan2(y - cy, x - cx)  →  grados 0-360
```
**Referencia angular (convención AHA):** 0° apunta a la unión anteroseptal y se numera en
sentido antihorario, pero para el MVP adoptamos una convención FIJA y documentada:
- 0° = dirección +x (derecha), ángulos crecientes antihorario.
- **El usuario podrá rotar/espejar después** (dejar parámetro `angle_offset_deg=0.0` y
  `clockwise=False` para calibrar contra el polar map de MyoVation).

**Asignación de sector según nivel:**
- Basal y medio (6 sectores de 60°): `sector6 = int((angle + angle_offset) % 360 // 60)` → 0..5.
- Apical (4 sectores de 90°): `sector4 = int((angle + angle_offset) % 360 // 90)` → 0..3.

**Numeración AHA (mapear sector→nº de segmento):**
```python
# nivel basal:  sectores 0..5 → segmentos 1..6
# nivel medio:  sectores 0..5 → segmentos 7..12
# nivel apical: sectores 0..3 → segmentos 13..16
# ápex:                        → segmento 17
```
El mapeo exacto sector→segmento (qué sector es "anterior", etc.) queda como
`SECTOR_TO_SEGMENT_BASAL = [1,2,3,4,5,6]` (y análogos), **calibrable** después contra
MyoVation. Para el MVP usar orden directo y dejar `# TODO calibrar orientación vs GE`.

### 3.4 Detección de orientación ápex↔base — ALGORITMO
El ápex es el extremo donde el anillo se cierra (cavidad chica / radio interno → 0) y el
área de miocardio disminuye. Heurística:
- Calcular por slice válido el `inner_radius` (de SegmentationResult) y el área de máscara.
- El extremo con **inner_radius menor y área menor** = ÁPEX. El otro = BASE.
- Devolver el orden base→ápex.

### 3.5 Firmas

```python
TERRITORY_MAP = {
    "LAD": [1, 2, 7, 8, 13, 14, 17],
    "LCx": [5, 6, 11, 12, 16],
    "RCA": [3, 4, 9, 10, 15],
}

@dataclass
class AHAResult:
    segment_map: np.ndarray      # (n_slices, H, W) int, 0=fuera, 1..17=segmento
    apex_to_base_order: list[int]
    n_per_segment: dict[int, int]

def map_to_17_segments(
    seg: "SegmentationResult",
    angle_offset_deg: float = 0.0,
    clockwise: bool = False,
) -> AHAResult:
    ...

def phase_by_segment(phase_map: np.ndarray, aha: "AHAResult") -> dict[int, float]:
    """Media CIRCULAR de fase por segmento (usar core.metrics.circular_mean_deg)."""

def territory_analysis(phase_by_seg: dict[int, float]) -> dict[str, dict]:
    """
    Para cada territorio (LAD/LCx/RCA): mean (circular), std (circular), min, max
    de las fases de sus segmentos. Usar core.metrics.circular_mean_deg / circular_std_deg.
    """
```

---

## 4. `tests/test_segmentation.py`

### 4.1 Test sintético (anillo conocido)
- Construir un cubo sintético: para cada slice, un **anillo** (dona) de actividad
  centrado, radio interno 4 px, externo 8 px, matriz 32×32, 8 gates, 10 slices.
- `segment_myocardium(cube, method="auto")` debe:
  - Detectar máscara no vacía en todos los slices con anillo.
  - `center_per_slice` ≈ centro real (±1.5 px).
- `map_to_17_segments(...)`:
  - `segment_map` debe contener valores en 1..17.
  - Cada uno de los 17 segmentos debe tener al menos 1 voxel (con 10 slices y anillo completo).
  - Suma de `n_per_segment` == nº de voxels de la máscara.

### 4.2 Test real (smoke, sobre REST_IRNCG_SA)
- Cargar con `dicom_loader`, segmentar `method="auto"`, mapear AHA.
- Imprimir: nº voxels por nivel, nº segmentos no vacíos, y `phase_by_segment` + `territory_analysis`
  usando el `phase_map` del motor (Fase 1).
- Aserción laxa: al menos 15 de 17 segmentos con voxels; los 3 territorios con datos.
- Ruta de referencia (NO versionada):
  `C:\Users\Ernesto\Desktop\INTERCAMBIO\varios stress cardiacos de xeleris\estudio uno\MYOMETRIX\myometrix results\REST_IRNCG_SA001_DS.dcm`
  (si no existe → `[SKIP]`).

---

## 5. Integración con lo existente (NO romper)

- Reutilizar `core.metrics.circular_mean_deg` y `circular_std_deg` para todo promedio angular.
- El pipeline nuevo debe encajar así:
  ```
  study = dicom_loader.load(path)
  seg   = segmentation.segment_myocardium(study.cube, method="auto")
  res   = phase_analysis(study.cube, seg.mask)        # Fase 1 (ya hecho)
  aha   = aha_segments.map_to_17_segments(seg)
  pbs   = aha_segments.phase_by_segment(res.phase_map, aha)
  terr  = aha_segments.territory_analysis(pbs)
  ```
- Actualizar `tests/test_engine_real.py` para usar `segment_myocardium(method="auto")`
  en vez de la máscara provisional por umbral (dejar la provisional como fallback).

---

## 6. Criterios de aceptación (Definition of Done)

- [ ] `py tests/test_segmentation.py` pasa (sintético + real-skip-safe).
- [ ] `py tests/test_phase_synthetic.py` sigue pasando (no romper Fase 1).
- [ ] `py -m core.segmentation <dcm>` imprime resumen sin error.
- [ ] El mapeo AHA cubre 1..17 en el test sintético (anillo completo).
- [ ] Sin dependencias nuevas (solo numpy/scipy/cv2).
- [ ] UTF-8 en todos los archivos.

## 7. Cosas deliberadamente FUERA de esta fase (no hacer)
- LVSD estilo GE (6 etapas) → futuro.
- Reorientación oblicua a SA → el input ya viene en SA.
- Polar map bullseye visual → Fase 4.
- Calibración exacta de orientación angular vs MyoVation → dejar parámetros y `# TODO`.

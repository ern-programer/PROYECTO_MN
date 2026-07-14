# SPEC Fase 4 — Visualización (polar map bullseye + histograma de fase)

**Autor de la especificación:** Opus 4.8 (diseño).
**Implementa:** GPT 5.3-codex.
**Fecha:** 2026-07-14.
**Contexto:** módulo SINCRO, `mod_SINCRO/`. Fases 1 (motor de fase) y 3 (segmentación +
mapeo AHA) ya están hechas y validadas. Esta fase produce las DOS visualizaciones
estándar del análisis de fase: el **Phase Polar Map (bullseye 17 segmentos)** y el
**Phase Histogram**. Todo con matplotlib, generando imágenes (PNG / array RGB) SIN UI
(la UI PyQt6 es Fase 6; acá solo se generan figuras guardables/embebibles).

> **Cómo usar esta spec (Codex):** implementá EXACTAMENTE las firmas y comportamientos.
> No cambies nombres. Donde algo quede abierto, elegí lo simple y dejá `# TODO`.
> Corré los tests al final con el venv propio.

---

## 0. Contexto (ya resuelto)

- `core/aha_segments.py` → `AHAResult.segment_map` (n_slices,H,W) int 1..17, `phase_by_segment(phase_map, aha)` → `dict[int,float]` (fase circular media por segmento 1..17), `TERRITORY_MAP`.
- `core/phase_analysis.py` → `PhaseResult.phases_deg` (array de fases 0-360 por voxel) y `phase_map`.
- `core/metrics.py` → `calculate_phase_metrics(phases_deg)` → dict con phase_sd, bandwidth, entropy, mean_phase, peak_phase, classification, etc. + `circular_mean_deg`.
- `core/console_utf8.py` → `enable_utf8()`.

Dependencias permitidas: numpy, matplotlib (+ scipy si hace falta). Ya instaladas. NO usar PyQt en esta fase.

Backend matplotlib: usar `matplotlib.use("Agg")` al inicio de los módulos viz (renderizado sin ventana, para generar PNG en cualquier entorno).

---

## 1. Archivos a crear/editar

1. `viz/colormaps.py` — colormap CÍCLICO para fase (0-360°).
2. `viz/polar_map.py` — bullseye de 17 segmentos.
3. `viz/histogram.py` — histograma de fase con métricas.
4. `tests/test_viz.py` — tests (generan PNG y verifican que no crashea + propiedades básicas).

---

## 2. `viz/colormaps.py` — colormap cíclico (LO IMPORTANTE #1)

**Por qué cíclico:** la fase es un ÁNGULO (0° y 360° son el mismo instante del ciclo).
Un colormap lineal (jet/viridis) pondría colores muy distintos en 1° y 359°, que en
realidad están casi juntos. Hay que usar un colormap **cíclico** donde el color en 0°
== color en 360°.

### Firmas
```python
import numpy as np
import matplotlib.cm as cm

def get_phase_cmap(name: str = "hsv"):
    """
    Devuelve un colormap matplotlib CÍCLICO para mapear fase 0-360°.
    Opciones válidas: 'hsv' (clásico de MUGA/SyncTool), 'twilight', 'twilight_shifted'.
    Default 'hsv' (el estándar en cardiología nuclear).
    """

def phase_to_rgb(phase_deg, cmap_name: str = "hsv", nan_color=(0.1, 0.1, 0.1)):
    """
    Mapea un array de fase (0-360°, puede tener NaN) a RGB (…,3) float 0-1.
    NaN → nan_color (gris oscuro). Normaliza 0-360 → 0-1 antes de aplicar el cmap.
    """
```
- Normalización: `norm = (phase_deg % 360) / 360.0`.
- El cmap se aplica sobre `norm`; devolver solo los 3 canales RGB (descartar alpha).
- Documentar en docstring: rojo/inicio = contracción temprana, otros colores = más tardía
  (la interpretación exacta depende de la referencia de fase).

---

## 3. `viz/polar_map.py` — bullseye 17 segmentos (LO IMPORTANTE #2)

### 3.1 Geometría del bullseye (diseño Opus)
El bullseye son **anillos concéntricos** divididos en sectores, vista "ojo de buey"
(el ápex en el centro, la base en el borde externo):

```
Anillo EXTERNO  (r 0.75–1.00): 6 sectores de 60° → segmentos BASALES  1-6
Anillo MEDIO    (r 0.50–0.75): 6 sectores de 60° → segmentos MEDIOS    7-12
Anillo INTERNO  (r 0.25–0.50): 4 sectores de 90° → segmentos APICALES 13-16
CENTRO          (r 0.00–0.25): 1 disco            → ÁPEX               17
```

**Ángulos de inicio de cada sector (convención, calibrable):**
- Basal/medio: sector k ocupa `[k*60, (k+1)*60)` grados, k=0..5.
- Apical: sector k ocupa `[k*90, (k+1)*90)` grados, k=0..3.
- `angle_offset_deg` (default 0) rota todo el bullseye para calibrar contra MyoVation.
- `# TODO calibrar orientación del bullseye vs MyoVation/GE` (igual que en aha_segments).

**Mapeo sector→nº de segmento:** usar los mismos LUT que `core/aha_segments.py`
(`SECTOR_TO_SEGMENT_BASAL/MEDIO/APICAL`) para consistencia. Importarlos de ahí.

### 3.2 Implementación (matplotlib con wedges)
Dibujar con `matplotlib.patches.Wedge` en un eje con `aspect="equal"`:
- Para cada segmento, un `Wedge((0,0), r_out, theta1, theta2, width=r_out-r_in)`.
- El ápex (17) es un `Circle`/`Wedge` completo de radio 0.25.
- Color del wedge = `phase_to_rgb(fase_del_segmento)` (de §2). Segmento sin dato → gris.
- Etiqueta opcional: número de segmento y/o valor de fase en el centro de cada wedge.
- Colorbar lateral: escala 0-360° con el cmap cíclico.

### 3.3 Firmas
```python
@dataclass
class PolarMapFigure:
    fig: "matplotlib.figure.Figure"
    segment_values: dict[int, float]      # fase por segmento usada
    cmap_name: str

def build_polar_map(
    phase_by_seg: dict[int, float],       # de aha_segments.phase_by_segment
    cmap_name: str = "hsv",
    angle_offset_deg: float = 0.0,
    show_values: bool = True,
    title: str | None = None,
) -> PolarMapFigure:
    ...

def save_polar_map(pmfig: "PolarMapFigure", path: str, dpi: int = 150) -> str:
    """Guarda la figura a PNG. Devuelve el path."""
```

### 3.4 Comportamiento
- Debe funcionar aunque falten segmentos (dibuja los presentes, el resto gris).
- No abrir ventanas (backend Agg). No llamar plt.show().

---

## 4. `viz/histogram.py` — histograma de fase

### 4.1 Qué mostrar (estándar SyncTool/Emory)
- Histograma de las fases (0-360°, típicamente 36–72 bins para visual; el cálculo de
  entropy usa 360, pero el gráfico puede agrupar).
- Líneas verticales: media de fase (P50 circular) y P5/P95 (bandwidth).
- Caja de texto con métricas: Phase SD, Bandwidth, Entropy, Peak Phase, Classification.
- Título con la clasificación (NORMAL/MILD/MODERATE/SEVERE).

### 4.2 Firmas
```python
def build_phase_histogram(
    phases_deg: np.ndarray,
    metrics: dict | None = None,          # de calculate_phase_metrics; si None, se calcula
    bins: int = 72,
    title: str | None = None,
):
    """Devuelve la Figure de matplotlib. Si metrics es None, llamar calculate_phase_metrics."""

def save_histogram(fig, path: str, dpi: int = 150) -> str:
    ...
```

---

## 5. `tests/test_viz.py`

### 5.1 Test colormap
- `phase_to_rgb(np.array([0., 360.]))` → los dos colores casi iguales (cíclico): 
  `np.allclose(rgb[0], rgb[1], atol=0.05)`.
- `phase_to_rgb(np.array([np.nan]))` → devuelve el nan_color.

### 5.2 Test polar map (sintético)
- Construir `phase_by_seg = {i: (i*20) % 360 for i in range(1,18)}`.
- `build_polar_map(phase_by_seg)` no crashea; `save_polar_map(..., tmp.png)` crea el archivo.
- Verificar que la figura tiene ejes y que el PNG pesa > 0 bytes.

### 5.3 Test histograma (sintético)
- `phases = np.random.default_rng(0).normal(120, 30, 2000) % 360`.
- `build_phase_histogram(phases)` no crashea; `save_histogram(..., tmp.png)` crea archivo.

### 5.4 Test integración real (skip-safe)
- Si existe `REST_IRNCG_SA001_DS.dcm`: load → segment(auto) → phase_analysis →
  map_to_17_segments → phase_by_segment → build_polar_map + build_histogram → guardar 2 PNG
  en carpeta temporal. Aserción laxa: ambos PNG creados y > 0 bytes.

Usar `tempfile` o una carpeta `output/` (ignorada por git) para los PNG. Llamar `enable_utf8()`.

---

## 6. Criterios de aceptación (Definition of Done)

- [ ] `.\.venv\Scripts\python.exe tests\test_viz.py` pasa.
- [ ] `tests/test_phase_synthetic.py` y `tests/test_segmentation.py` SIGUEN pasando (no romper F1/F3).
- [ ] Colormap es cíclico (0°≈360°).
- [ ] `build_polar_map` genera bullseye con 17 segmentos en la geometría descrita.
- [ ] PNGs se generan sin abrir ventanas (backend Agg).
- [ ] Sin dependencias nuevas (numpy/matplotlib/scipy). UTF-8. Cambios solo en `mod_SINCRO/**`.

## 7. Fuera de esta fase (no hacer)
- UI PyQt6 / interactividad / animación del ciclo → Fase 6.
- Informe PDF → Fase 5.
- Calibración fina de orientación vs MyoVation → dejar `angle_offset_deg` + `# TODO`.

## 8. AL TERMINAR (handoff)
No avanzar a otra fase. Terminar con el mensaje exacto:
> "✅ Fase 4 implementada y tests en verde. **Volvé a Opus 4.8 para la revisión** (QA del bullseye + colormap cíclico + no regresión F1/F3) antes de avanzar a la Fase 5."
Incluir resumen de archivos tocados, tests que pasan y TODO abiertos. Commits locales SIN push.

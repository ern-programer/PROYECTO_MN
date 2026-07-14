# SINCRO

Módulo de análisis de **sincronía / asincronía / disincronía cardíaca** a partir de estudios Gated SPECT (Short Axis del ventrículo izquierdo).

Genera métricas cuantitativas de fase (Phase SD, Bandwidth, Entropy, Skewness, Kurtosis), polar map (bullseye 17 segmentos AHA), análisis por territorio coronario (LAD/LCx/RCA) e informe PDF.

Proyecto de **Gammasys** (medicina nuclear, Argentina).

---

## Estado

**EN DESARROLLO** — Fase 1 (motor de fase). Inicio: 2026-07-14.

## Arquitectura

```
SINCRO/
├── core/                 # Motor puro (sin UI, testeable solo)
│   ├── dicom_loader.py   # Carga + detección gated + desempaquetado montage + auto-QC
│   ├── phase_analysis.py # FFT primer armónico → fase/amplitud por voxel
│   ├── metrics.py        # Phase SD, BW, Entropy, Skewness, Kurtosis, AI, latest site
│   ├── segmentation.py   # Segmentación miocárdica (ROI manual + auto)
│   └── aha_segments.py   # Mapeo voxel → 17 segmentos AHA + territorios
├── viz/                  # Visualización
│   ├── polar_map.py      # Bullseye 17 segmentos (colormap cíclico)
│   ├── histogram.py      # Histograma de fase
│   └── colormaps.py      # Colormaps (fase 0-360°)
├── report/               # Informe
│   └── report_generator.py
├── ui/                   # Interfaz PyQt6
│   ├── main_window.py
│   └── cine_widget.py
├── tests/                # Tests (empezar con test sintético)
│   └── test_phase_synthetic.py
├── data_test/            # DICOM de prueba (NO versionado, ver .gitignore)
└── main.py               # Entry point
```

## Fundamento científico

Basado en el algoritmo de análisis de fase de **Emory (Chen 2005, PMID 16344229)** — el mismo que usan SyncTool, QGS y 4DM.

- **FFT primer armónico** de la curva de actividad de cada voxel a lo largo del ciclo cardíaco.
- **Fase** = momento de contracción (0-360°). **Amplitud** = magnitud del cambio.
- Métricas de dispersión de fase → grado de disincronía.

| Métrica | Normal | Severo |
|---------|--------|--------|
| Phase SD | < 20° | > 60° |
| Bandwidth | < 60° | > 120° |
| Entropy | < 4.0 | > 6.0 |

## Formatos de entrada soportados

El `dicom_loader` detecta y maneja automáticamente:
- **Short Axis gated reconstruido** (ideal, uso directo). Ej: series `IRNCG_SA` de Xeleris/MyoVation.
  - Incluye desempaquetado de **montage** (cortes concatenados: `Cols = N × Rows`) y separación del frame sumado.
- **Gated crudo** (proyecciones angulares): requiere reconstrucción (futuro).

## Requisitos

Ver `requirements.txt`. Todas las dependencias son estándar (pydicom, numpy, scipy, opencv, matplotlib, reportlab, PyQt6).

## Uso rápido (loader)

```bash
python -m core.dicom_loader "ruta/al/REST_IRNCG_SA001_DS.dcm"
```

## Validación

- **Nivel 0:** test sintético (`tests/test_phase_synthetic.py`) — fase matemática conocida.
- **Nivel 1:** MyoVation/QGS (Xeleris) — EF, volúmenes, segmentación LVSD.
- **Nivel 2:** Emory Cardiac Toolbox / SyncTool — Phase SD/BW/Entropy (ground-truth de fase).

## Licencia

Propietario — Gammasys. (Definir.)

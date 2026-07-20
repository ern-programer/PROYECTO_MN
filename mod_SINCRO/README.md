# SINCRO

Módulo de análisis de **sincronía / asincronía / disincronía cardíaca** a partir de estudios Gated SPECT (Short Axis del ventrículo izquierdo).

Genera métricas cuantitativas de fase (Phase SD, Bandwidth, Entropy, Skewness, Kurtosis), polar map (bullseye 17 segmentos AHA), análisis por territorio coronario (LAD/LCx/RCA), robustez estadística (bootstrap, sensibilidad ROI) e informe PDF.

Proyecto de **Gammasys** (medicina nuclear, Argentina).

---

## Estado

**v1.8.0** — Roadmap mejoras implementado. Inicio: 2026-07-14. Última actualización: 2026-07-20.

## Arquitectura

```
SINCRO/
├── core/                 # Motor puro (sin UI, testeable solo)
│   ├── dicom_loader.py   # Carga + detección gated + desempaquetado montage + auto-QC
│   ├── phase_analysis.py # FFT primer armónico → fase/amplitud por voxel
│   ├── metrics.py        # Phase SD, BW, Entropy, Skewness, Kurtosis, AI, latest site
│   ├── segmentation.py   # Segmentación miocárdica (ROI manual + auto)
│   ├── aha_segments.py   # Mapeo voxel → 17 segmentos AHA + territorios
│   ├── robustness.py     # Robustez: segmentario AHA, bootstrap, sensibilidad ROI
│   ├── normal_db.py      # DB normal por software (QGS/ECTb/cREPO/HFV)
│   ├── export_manager.py # Exportación JSON/CSV/Excel
│   └── logging_config.py # Logging estructurado a archivo
├── viz/                  # Visualización
│   ├── polar_map.py      # Bullseye 17 segmentos (colormap cíclico)
│   ├── histogram.py      # Histograma de fase
│   └── colormaps.py      # Colormaps (fase 0-360°)
├── report/               # Informe
│   └── report_generator.py
├── ui/                   # Interfaz PyQt6
│   ├── main_window.py
│   └── cine_widget.py
├── tests/                # Tests
│   ├── test_phase_synthetic.py  # Tests sintéticos
│   └── test_integration.py      # Tests E2E
├── data_test/            # DICOM de prueba (NO versionado, ver .gitignore)
├── logs/                 # Logs estructurados (NO versionado)
└── main.py               # Entry point
```

## Fundamento científico

Basado en el algoritmo de análisis de fase de **Emory (Chen 2005, PMID 16344229)** — el mismo que usan SyncTool, QGS y 4DM.

- **FFT primer armónico** de la curva de actividad de cada voxel a lo largo del ciclo cardíaco.
- **Fase** = momento de contracción (0-360°). **Amplitud** = magnitud del cambio.
- Métricas de dispersión de fase → estimación de asincronía mecánica intraventricular del VI.

GammaSync separa dos lecturas:
- **Clasificación técnica PSD:** cortes históricos por Phase SD para orientación rápida (NORMAL/MILD/MODERATE/SEVERE).
- **Interpretación clínica vs DB:** comparación contra referencias publicadas por software/sexo/protocolo (`QGS_JSNM2023`, `ECTb_JSNM2023`, `cREPO_JSNM2023`, `HFV_JSNM2023`, etc.). PSD, BW y entropy no son intercambiables entre paquetes.
- **Robustez/QC:** además del modo voxel, calcula modo segmentario AHA, bootstrap de PSD/BW/entropy y sensibilidad a cambios de ROI ±1 px.

Entropy se informa como **Shannon (bits)** y como **normalizada (%)** cuando se compara contra literatura clínica.

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
- **Nivel 2:** Emory Cardiac Toolbox / SyncTool / 4DM — Phase SD/BW/Entropy (ground-truth de fase).
- **Nivel 3:** validación local con fantomas/estudios clínicos cuando haya acceso a comparadores externos.

## Licencia

Propietario — Gammasys. (Definir.)

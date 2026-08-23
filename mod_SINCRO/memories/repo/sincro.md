# SINCRO - Módulo de Análisis de Asincronía Cardíaca

## Estado: EN DESARROLLO

## Propósito
Módulo para análisis de asincronía cardíaca mediante SPECT Gated:
- FFT de curvas de actividad en Short Axis
- Métricas de fase y disincronía
- Polar maps de fase
- Informes clínicos

## Estructura
```
mod_SINCRO/
├── core/
│   ├── amyloid_spect.py    # Procesamiento SPECT amiloidosis + HMR-SPECT
│   └── ...
├── ui/
│   ├── amyloid_spect_panel.py  # Panel UI para AMYLO SPECT
│   └── ...
├── docs/
│   └── HMR_SPECT_PLAN.md  # Plan de implementación HMR-SPECT
└── memories/
    └── repo/
        └── sincro.md      # Este archivo
```

## HMR-SPECT (Heart-to-Mediastinum Ratio)
### Implementación completada (2026-01-07)
- **VOISphere**: Clase para regiones de interés esféricas 3D
- **HmrSpectMethod**: Enum con dos métodos:
  - `VOI_COMPLETE`: Usa toda la esfera 3D
  - `SLICE_CENTRAL`: Usa solo el slice axial central
- **compute_hmr_spect()**: Calcula ratio corazón/mediastino
- **UI**: Panel con selección de método, radios ajustables, botón de cálculo
- **PDF**: Exportación de resultados HMR al informe

### Clasificación clínica
- HMR ≥ 1.6: NEGATIVO (sin amiloidosis)
- HMR 1.5-1.6: EQUIVOCO
- HMR < 1.5: POSITIVO (amiloidosis)

### Uso
1. Cargar estudio SPECT
2. Colocar punto Anchor A en centro del corazón
3. Colocar punto B en mediastino superior
4. Seleccionar método y radios
5. Click "Calcular HMR-SPECT"
6. Exportar a PDF

## Dependencias
- PyQt6
- numpy
- scipy
- pydicom
- reportlab

## Próximos pasos
- [ ] Tests unitarios para compute_hmr_spect()
- [ ] Validación con estudios clínicos reales
- [ ] Integración con pipeline FFT fase/disincronía

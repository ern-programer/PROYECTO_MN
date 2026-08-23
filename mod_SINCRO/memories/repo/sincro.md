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
- **UI**: Sección en panel principal con selección de método, radios ajustables
- **PDF**: Exportación de resultados HMR al informe

### Clasificación clínica
- HMR ≥ 1.6: NEGATIVO (sin amiloidosis)
- HMR 1.5-1.6: EQUIVOCO
- HMR < 1.5: POSITIVO (amiloidosis)

### Flujo de uso (actualizado 2026-08-23)
1. Cargar SPECT (botón "1. Cargar SPECT")
2. Activar **"Localización"** (botón checkable)
3. **Ctrl+clic** en SPECT para posicionar cruz en centro del corazón
4. Click **"Fijar ancla A"** (guarda punto corazón)
5. **Ctrl+clic** en SPECT para posicionar cruz en mediastino superior
6. Click **"Calcular HMR-SPECT"** en sección HMR-SPECT
7. Exportar a PDF si se desea

### Dependencias
- PyQt6
- numpy
- scipy
- pydicom
- reportlab

## Próximos pasos
- [ ] Tests unitarios para compute_hmr_spect()
- [ ] Validación con estudios clínicos reales
- [ ] Integración con pipeline FFT fase/disincronía

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

### Clasificación clínica (CORREGIDO 2026-08-23)
- HMR ≥ 1.6: **POSITIVO** (alta captación cardíaca = amiloidosis)
- HMR 1.5-1.6: **EQUIVOCO**
- HMR < 1.5: **NEGATIVO** (baja captación cardíaca = sin amiloidosis)

**NOTA IMPORTANTE**: La lógica original estaba invertida. HMR ALTO significa mucha captación en corazón relativo al mediastino, lo cual indica amiloidosis.

### Flujo de uso (actualizado 2026-08-23)
1. Cargar SPECT (botón "1. Cargar SPECT")
2. Activar **"Localización"** (botón checkable)
3. **Ctrl+clic** en SPECT para posicionar cruz en centro del corazón
4. Click **"Fijar ancla A"** (guarda punto corazón)
5. **Ctrl+clic** en SPECT para posicionar cruz en mediastino superior
6. Ajustar radios de VOIs (corazón 30mm, mediastino 20mm por defecto)
7. Click **"Calcular HMR-SPECT"** en sección HMR-SPECT
8. Ver VOIs proyectados como círculos punteados en los 3 planos
9. Ver resultados: HMR(raw) y HMR(filtrado) - el raw es el valor clínico
10. Exportar a PDF si se desea

### Visualización de VOIs (2026-08-23)
- Los VOIs se muestran como círculos punteados en las 3 vistas MPR
- Corazón: círculo rojo punteado con etiqueta "Corazón XXmm"
- Mediastino: círculo azul punteado con etiqueta "Mediastino XXmm"
- Los radios se pueden ajustar antes de calcular

### Resultados duales (2026-08-23)
- **HMR (raw)**: Calculado sobre volumen base sin filtrar - **valor clínico relevante**
- **HMR (filtrado)**: Calculado sobre volumen filtrado - referencia visual
- Si no hay volumen raw disponible, solo se muestra HMR filtrado

### Referencia de escala en UI (2026-08-23)
- Se muestra escala de clasificación clínica junto al resultado:
  - **≥1.6 NEGATIVO** (verde)
  - **1.5-1.6 EQUIVOCO** (naranja)
  - **<1.5 POSITIVO** (rojo)

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

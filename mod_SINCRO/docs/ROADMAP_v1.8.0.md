# Roadmap GammaSync v1.8.0

## Objetivo
Mejorar mantenibilidad, exportación de datos, logging y testing sin cambiar el motor clínico.

## Prioridades ALTA (v1.8.0)

### 1. Refactorizar `main_window.py`
- **Problema:** ~6000 líneas, monolítico, difícil de testear.
- **Solución:** Separar en módulos especializados.
- **Módulos a crear:**
  - `ui/main_window.py` — Solo ventana principal y coordinación
  - `ui/processing_controller.py` — Lógica de procesamiento DICOM→métricas
  - `ui/preset_manager.py` — Guardar/cargar presets
  - `ui/cine_manager.py` — Gestión de cine y visualización
  - `ui/report_manager.py` — Generación de reportes PDF
  - `ui/compare_manager.py` — Comparación stress/rest
  - `ui/roi_manager.py` — Gestión de ROIs manuales/auto

### 2. Exportación JSON/CSV
- **Problema:** Solo PDF, no hay datos estructurados para análisis batch.
- **Solución:** Exportar métricas, QC, robustez en JSON/CSV.
- **Archivos:**
  - `core/export_manager.py` — Nuevo módulo
  - JSON: métricas completas + metadatos
  - CSV: métricas tabulares por paciente/estudio
  - Excel: múltiples hojas (métricas, robustez, ROI, QC)

### 3. Logging a archivo
- **Problema:** Logs van a consola/UI, no persisten.
- **Solución:** Logging estructurado a archivo con rotación.
- **Archivos:**
  - `core/logging_config.py` — Configuración logging
  - Logs en `logs/gammasync_YYYY-MM-DD.log`
  - Niveles: DEBUG, INFO, WARNING, ERROR
  - Formato: JSON con timestamp, módulo, mensaje, contexto

### 4. Tests de integración end-to-end
- **Problema:** Solo tests unitarios/sintéticos.
- **Solución:** Test completo DICOM→PDF.
- **Archivos:**
  - `tests/test_integration.py` — Test E2E
  - Usa DICOM real de `data_test/` o synthetic
  - Verifica que se genere PDF, métricas, imágenes

## Cronograma estimado

| Tarea | Días | Dependencias |
|-------|------|--------------|
| Plan roadmap | 0.5 | — |
| Refactorizar main_window | 2-3 | — |
| Exportación JSON/CSV | 1 | — |
| Logging a archivo | 0.5 | — |
| Tests integración | 1-2 | Refactorización |
| Documentación | 0.5 | Todas |

## Criterios de aceptación

### Refactorización
- [ ] `main_window.py` < 2000 líneas
- [ ] Cada manager testeable independientemente
- [ ] No rompe funcionalidad existente
- [ ] Tests pasan

### Exportación
- [ ] JSON válido con métricas + metadatos
- [ ] CSV abrible en Excel
- [ ] Incluye robustez, QC, DB eval

### Logging
- [ ] Logs persisten en archivo
- [ ] Rotación por fecha
- [ ] Niveles configurables
- [ ] No afecta performance

### Tests E2E
- [ ] Test carga DICOM real
- [ ] Test procesa completo
- [ ] Test genera PDF
- [ ] Test verifica métricas clave

## Notas
- Mantener compatibilidad con presets existentes
- No cambiar motor de fase (ya validado)
- Backup v1.7.0 ya creado en `D:\- PROGRAMACIÓN\SINCRO_backup`

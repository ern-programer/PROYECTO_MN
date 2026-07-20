# Roadmap GammaSync v1.9.0

## Objetivo
Mejorar mantenibilidad, integración clínica y precisión anatómica.

## Prioridades

### 1. Refactorizar `main_window.py` en managers
- **Problema:** ~6000 líneas, monolítico, difícil de testear y mantener.
- **Solución:** Separar en módulos especializados con responsabilidades claras.

**Managers a crear:**
- `ui/managers/processing_manager.py` — Lógica de procesamiento DICOM→métricas
- `ui/managers/preset_manager.py` — Guardar/cargar presets
- `ui/managers/cine_manager.py` — Gestión de cine y visualización
- `ui/managers/report_manager.py` — Generación de reportes PDF
- `ui/managers/compare_manager.py` — Comparación stress/rest
- `ui/managers/roi_manager.py` — Gestión de ROIs manuales/auto

**Beneficios:**
- Cada manager testeable independientemente
- `main_window.py` < 2000 líneas (solo coordinación)
- Más fácil agregar features nuevas
- Mejor separación de concerns

### 2. Integración ECG manual
- **Problema:** GammaSync no sabe si el paciente tiene BRI, QRS ancho, marcapasos.
- **Solución:** Campos manuales ECG en UI y PDF.

**Campos a agregar:**
- Ritmo (sinusal/FA/marcapasos/otro)
- FC (lpm)
- QRS (ms)
- QT/QTc (ms)
- BRI (sí/no)
- BRD (sí/no)
- Marcapasos/CRT (sí/no)
- Observaciones ECG

**Ubicación:**
- Nuevo grupo "ECG" en sidebar
- Sección en PDF "Contexto electrocardiográfico"
- Score electro-mecánico (concordancia QRS vs fase)

### 3. Atlas segmentación
- **Problema:** Segmentación automática falla en defectos, apex/base, baja resolución.
- **Solución:** Atlas probabilístico del VI + registro.

**Componentes:**
- Atlas promedio de miocardio LV (de estudios normales)
- Registro rígido/afín del estudio al atlas
- Segmentación guiada por atlas (prior probabilístico)
- Fallback a threshold si registro falla

### 4. MUGA módulo
- **Problema:** Solo SPECT MPI, no blood-pool.
- **Solución:** Módulo separado para MUGA gated.

**Features:**
- Loader MUGA planar/SPECT
- ROI VI/VD (manual/auto)
- Fase por pixel (mismo algoritmo FFT)
- FEVI más precisa (blood-pool)
- Sincronía interventricular

### 5. API REST
- **Problema:** Solo UI local, no procesamiento remoto.
- **Solución:** API REST con FastAPI.

**Endpoints:**
- `POST /process` — Procesar DICOM y devolver métricas JSON
- `GET /status/{job_id}` — Estado de procesamiento
- `GET /results/{job_id}` — Resultados completos
- `POST /batch` — Procesar múltiples estudios

## Cronograma estimado

| Tarea | Días | Prioridad |
|-------|------|-----------|
| Plan v1.9.0 | 0.5 | — |
| Refactorizar main_window | 3-4 | ALTA |
| Integración ECG | 1-2 | MEDIA |
| Atlas segmentación | 5-7 | MEDIA |
| MUGA módulo | 7-10 | BAJA |
| API REST | 5-7 | BAJA |

## Criterios de aceptación

### Refactorización
- [ ] `main_window.py` < 2000 líneas
- [ ] Cada manager testeable independientemente
- [ ] No rompe funcionalidad existente
- [ ] Tests pasan

### ECG
- [ ] Campos ECG en UI
- [ ] Sección ECG en PDF
- [ ] Score electro-mecánico calculado

### Atlas
- [ ] Atlas creado de estudios normales
- [ ] Registro funciona en estudios de prueba
- [ ] Mejora segmentación en casos difíciles

### MUGA
- [ ] Loader MUGA funciona
- [ ] Fase pixel calculada
- [ ] FEVI blood-pool más precisa

### API REST
- [ ] Endpoint /process funciona
- [ ] Procesamiento async con job_id
- [ ] Documentación OpenAPI

## Notas
- Backup v1.8.0 ya creado
- No cambiar motor de fase (ya validado)
- Mantener compatibilidad con presets existentes

# ESTADO DE FASES SINCRO (2026-07-15)

## Resumen ejecutivo

- Estado general: en desarrollo funcional, con UI operativa y pipeline clínico básico end-to-end.
- Prioridad actual: consolidar flujo clínico de uso diario (robustez, presets, visualización, reporte).
- Riesgo principal: sumar funciones avanzadas (3D, DICOM SC/SEG, calcium score) sin cerrar primero persistencia y trazabilidad por estudio.

## Roadmap propuesto y estado

### Fase 0 - Ingesta y normalización DICOM

Objetivo:
- Cargar gated SPECT Short Axis de forma robusta.
- Resolver montage, frame sumado y reshape 4D.

Estado:
- Completada funcional.

Evidencia:
- Loader con detección gated, desempaquetado y auto-QC.
- Salida normalizada (gates, slices, H, W).

Pendiente:
- Mejor metadata de paciente/estudio para persistencia clínica (PatientID/StudyUID en UI).

### Fase 1 - Motor de fase y métricas

Objetivo:
- FFT armónico 1 por voxel, mapa de fase/amplitud, métricas globales.

Estado:
- Completada funcional.

Evidencia:
- Métricas principales calculadas y usadas en UI/reportes.
- Tests sintéticos existentes.
- Cálculo de volumen miocárdico (mL) y volumen de cavidad estimado (mL) cuando el DICOM incluye spacing válido.
- Estimación preliminar de FEVI por dinámica de gates (EDV/ESV/SV/FEVI), marcada como investigación y no equivalente a Emory/4DM/CEqual/QGS.

Pendiente:
- Suite de regresión más amplia con casos clínicos reales.

### Fase 2 - Segmentación

Objetivo:
- Segmentación auto/threshold/manual, ROI por slice.

Estado:
- Completada funcional.

Evidencia:
- Flujo manual con edición en cine y replicado por volumen.
- Limpieza/borrado de ROIs y validación de ROIs inválidas.
- Auto ROI por slice y en todo el volumen, con opción de aplicar solo en slices sin ROI (preserva correcciones manuales).

Pendiente:
- Asistente guiado de ROI para reducir interacción manual.

### Fase 3 - Visualización clínica

Objetivo:
- Slices con overlay de fase, polar map, histograma, cine y ejes ortogonales.

Estado:
- Completada funcional (iterativa).

Evidencia:
- Tabs de visualización, zoom, colormaps ampliados.
- Ejes SA/HLA/VLA (HLA/VLA reconstruidos desde SA si no hay serie original).
- Histograma con referencias y métricas sin superposición.

Pendiente:
- Si llegan series originales HLA/VLA, priorizar esas sobre reconstrucción desde SA.

### Fase 4 - Reporte PDF clínico

Objetivo:
- Informe legible con tablas, figuras y resumen interpretativo.

Estado:
- Completada funcional.

Evidencia:
- PDF con imágenes de resultados y captions.
- Escalado proporcional de imágenes (sin estiramiento).

Pendiente:
- Firma digital/plantillas institucionales.

### Fase 5 - Persistencia de procesamiento por estudio/paciente

Objetivo:
- Guardar solo parámetros y configuraciones (no imágenes pesadas).
- Soportar varios presets por paciente.

Estado:
- En progreso (ya iniciado en UI).

Evidencia:
- Presets por paciente (guardar/cargar/borrar) en JSON local.

Pendiente:
- Identificación robusta por PatientID + StudyInstanceUID.
- Historial de versiones de preset y auditoría mínima.

### Fase 6 - DICOM de salida y networking

Objetivo:
- Exportar resultados para envío DICOM (SC/encapsulado y, luego, objetos estructurados).
- Integrar con recepción/envío DICOM del flujo clínico.

Estado:
- Pendiente.

Pendiente clave:
- Definir formato inicial: Secondary Capture primero, luego SEG/SR si aplica.

### Fase 7 - Visualización 3D avanzada

Objetivo:
- Render 3D interactivo del VI y overlay por territorios coronarios.

Estado:
- Pendiente.

Recomendación:
- Implementar después de cerrar Fase 5 y 6 para evitar deuda técnica.

### Fase 8 - Calcium Score (CT)

Objetivo:
- Pipeline de score de calcio (Agatston y variantes) basado en CT.

Estado:
- Pendiente estratégico (alta prioridad clínica, módulo recomendado separado).

Recomendación técnica:
- Módulo separado pero integrado al ecosistema SINCRO/FUSION.
- Requiere CT adecuado y calibración HU. No debe mezclarse con pipeline SPECT puro.

## Dónde estamos hoy

Estamos entre Fase 4 y Fase 5:
- Pipeline SPECT de sincronía usable de punta a punta.
- Reporte y visualización clínicamente útiles.
- Persistencia de presets recién iniciada (base correcta, falta robustecer identidad clínica).

## Próximos pasos sugeridos (orden)

1. Cerrar Fase 5 (persistencia robusta por paciente/estudio, auditoría mínima).
2. Iniciar Fase 6 (export y envío DICOM de resultados).
3. Definir arquitectura del módulo Calcium Score (Fase 8), con dataset CT de validación.
4. Recién después abordar Fase 7 (3D interactivo).

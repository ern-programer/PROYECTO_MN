# CCP (Cardiac Control Panel)
## Descripción general y capacidades del software

**Documento descriptivo para registro de propiedad intelectual y evaluación regulatoria**
**Fecha:** 27 de agosto de 2026
**Estado del producto:** software en desarrollo y validación técnica/clínica.

## 1. Qué es CCP

CCP (Cardiac Control Panel) es un software de escritorio para visualizar, procesar y analizar estudios de medicina nuclear cardíaca. Reúne en una misma interfaz tareas que habitualmente están separadas: lectura de imágenes DICOM, control de calidad, reconstrucción de estudios SPECT, revisión dinámica del ciclo cardíaco, segmentación del ventrículo izquierdo, análisis de sincronía mecánica, evaluación de perfusión y herramientas para estudios de amiloidosis cardíaca.

El programa está pensado como una herramienta de apoyo para profesionales capacitados en medicina nuclear e imágenes médicas. Organiza información, realiza cálculos reproducibles y presenta resultados cuantitativos y gráficos. No reemplaza el criterio médico, no emite por sí solo un diagnóstico definitivo y sus resultados deben revisarse junto con las imágenes originales, los datos clínicos y los controles de calidad.

## 2. Flujo general de trabajo

CCP recibe estudios en formato DICOM y reconoce la estructura de las series. Puede trabajar con imágenes cardíacas reconstruidas y con determinadas adquisiciones SPECT crudas. En estas últimas permite revisar las proyecciones, detectar movimiento, aplicar correcciones y reconstruir los volúmenes antes de continuar.

El flujo típico es: importar el estudio; verificar identificación, geometría, cantidad de gates y calidad de adquisición; reconstruir o normalizar los datos; orientar los ejes cardíacos; segmentar el miocardio; ejecutar los cálculos; revisar imágenes, curvas y mapas; y finalmente generar un informe o exportar los resultados.

El sistema conserva los parámetros relevantes del procesamiento y permite utilizar presets. Esto favorece la repetibilidad y deja disponibles datos técnicos para auditoría.

## 3. Capacidades principales

### Ingreso, visualización y control de calidad

- Lectura de archivos DICOM de medicina nuclear y CT utilizados por los módulos cardíacos.
- Reconocimiento de estudios gatillados, cantidad de fases, cortes y geometría de adquisición.
- Desempaquetado de series multiframe y montajes de cortes.
- Cine cardíaco por gate y navegación por cortes.
- Ajuste de ventana, nivel, color, zoom, orientación e interpolación.
- Revisión de proyecciones crudas, sinogramas, curvas de cuentas y posibles movimientos.
- Herramientas de corrección de movimiento y control del gatillado.

### Reconstrucción y preparación SPECT

- Reconstrucción tomográfica mediante métodos analíticos e iterativos configurables.
- Filtros de pre y posprocesamiento, suavizado, corrección de dispersión cuando la información está disponible y reducción de ruido.
- Generación de volúmenes gatillados y no gatillados.
- Reorientación y presentación de ejes corto, largo horizontal y largo vertical.
- Comparación visual de etapas y conservación de parámetros de procesamiento.

### Sincronía mecánica y función ventricular

CCP analiza cómo cambia la actividad del miocardio a lo largo del ciclo cardíaco. Para cada voxel incluido en la máscara miocárdica calcula la fase y amplitud mediante análisis armónico. A partir de esa distribución obtiene métricas como desviación estándar de fase, ancho de banda, entropía, fase media, fase pico e índice de asincronía.

Los resultados se presentan como histograma de fase, mapa polar de 17 segmentos, mapas sobre los cortes y resúmenes por territorios. La segmentación puede ser automática, ajustada por umbral o corregida manualmente. También se ofrecen herramientas experimentales para estimar volúmenes ventriculares y fracción de eyección, siempre identificadas como resultados en validación.

### Perfusión y comparación de estudios

- Visualización de perfusión en cortes y mapas polares.
- Resumen regional según el modelo de 17 segmentos.
- Integración visual de perfusión y fase para revisar captación y momento de contracción.
- Comparación entre dos etapas compatibles, por ejemplo reposo y esfuerzo.
- Cine polar y representación tridimensional del ventrículo izquierdo como apoyo visual.

### Amiloidosis cardíaca

CCP incluye dos espacios de trabajo. El módulo planar permite ubicar regiones de interés, calcular relaciones de captación, evaluar lavado y preparar un reporte. El módulo SPECT/CT permite cargar o reconstruir SPECT, incorporar una CT, registrar ambos volúmenes, revisar la fusión multiplanar, segmentar estructuras de interés y editar manualmente la máscara CT. Estas funciones son de apoyo cuantitativo y visual y continúan bajo validación.

## 4. Resultados y exportación

El software puede generar informes PDF con datos del estudio, parámetros empleados, imágenes, métricas, mapas, tablas y notas de control de calidad. También permite exportar resultados estructurados en JSON, CSV y planillas, además de imágenes de los paneles y determinadas series DICOM derivadas. Las salidas están orientadas a documentación, revisión profesional, investigación y auditoría del procesamiento.

## 5. Usuarios previstos, seguridad y límites

Los usuarios previstos son médicos especialistas, físicos médicos, técnicos y otros profesionales entrenados en medicina nuclear. CCP no está destinado al uso directo por pacientes ni al análisis autónomo sin supervisión.

Antes de aceptar un resultado, el usuario debe verificar identidad y estudio, calidad del gatillado, movimiento, orientación, segmentación, correspondencia anatómica y coherencia de los parámetros. El programa muestra advertencias y estados de procesamiento, conserva registros técnicos y permite revisar o corregir varias etapas.

La exactitud depende de la calidad de adquisición, del equipo, del protocolo, de la reconstrucción y de la intervención del operador. Estudios incompletos, con movimiento intenso, gatillado defectuoso, baja estadística, orientación no reconocida o artefactos pueden producir resultados no confiables. Las clasificaciones automáticas son orientativas y no constituyen un diagnóstico.

Las funciones cuantitativas avanzadas, especialmente estimaciones funcionales, reconstrucciones experimentales y análisis SPECT/CT de amiloidosis, requieren validación clínica formal antes de utilizarse como base única para decisiones asistenciales. La versión destinada a evaluación regulatoria deberá acompañarse con gestión de riesgos, verificación y validación documentadas, control de versiones, especificación de requisitos, evaluación de usabilidad, ciberseguridad y evidencia clínica según el uso previsto definitivo.

## 6. Características técnicas y aporte original

CCP es una aplicación modular desarrollada en Python con interfaz gráfica de escritorio. Separa carga DICOM, reconstrucción, segmentación, cálculos, visualización y generación de informes. Esta organización permite probar los componentes, mantener trazabilidad y ampliar funciones sin alterar todo el sistema.

El aporte original está en la integración de un recorrido cardíaco completo dentro de un único panel: desde la adquisición y su control de calidad hasta la reconstrucción, la revisión dinámica, el análisis de fase, la perfusión, la comparación de estudios, la amiloidosis y la documentación final. El programa combina automatización con controles manuales para que el profesional pueda revisar qué hizo el sistema y ajustar las etapas sensibles.

---

**Nota regulatoria:** este documento describe capacidades técnicas del software en su estado actual. No implica autorización sanitaria, certificación de desempeño ni aprobación regulatoria. La indicación de uso definitiva y las prestaciones declaradas deberán coincidir con la versión sometida a evaluación y con su evidencia de validación.
# Informe de Auditoría Científica - Módulo SINCRO

**Fecha:** 10 de Agosto de 2026
**Auditor:** Gemini
**Objetivo:** Realizar una auditoría científica a ciegas sobre el backend matemático y los informes generados por el módulo SINCRO, para asegurar la congruencia analítica, la validez física y la fiabilidad de los resultados cuantitativos.

---

## Resumen Ejecutivo

La auditoría ha revelado **fallos críticos** en la configuración del pipeline de procesamiento de imágenes que comprometen severamente la fiabilidad de los resultados. El hallazgo principal es una **configuración incorrecta y peligrosa del filtro de reconstrucción iterativa "Nítida"**, que, en lugar de mejorar las imágenes de baja cuenta, introduce sesgos masivos que invalidan tanto las métricas de sincronía como las de función ventricular.

Se identificó que el algoritmo "Nítida" (OSEM+RR) se ejecutaba sin el post-filtro de suavizado necesario para controlar el ruido, causando una amplificación de artefactos que llevaba a resultados paradójicos y clínicamente engañosos. Se ha aplicado una corrección inicial en el código de la UI para activar este filtro por defecto.

La arquitectura actual, que utiliza una reconstrucción FBP "pasajero" para los cálculos de sincronía, se considera una **medida de contención necesaria pero no ideal a largo plazo**, debido a la disonancia que crea entre los datos visualizados y los medidos.

Las recomendaciones finales se centran en:
1.  Mantener temporalmente la arquitectura del "pasajero" FBP por seguridad.
2.  Realizar una validación rigurosa de la versión corregida de "Nítida", ajustando el parámetro de suavizado.
3.  Migrar a un pipeline unificado solo cuando "Nítida" demuestre ser robusto y fiable para todos los cálculos.
4.  Aumentar la transparencia en los informes, haciendo visibles los parámetros críticos del procesamiento.

---

## Parte 1: Análisis Inicial del Código Backend

La auditoría comenzó con un análisis de los archivos Python en `core/` para entender la implementación matemática.

### 1.1. `phase_analysis.py` y `metrics.py`
- **Observación:** El cálculo de fase (basado en el primer armónico de la FFT) y las métricas estadísticas derivadas (Desviación Estándar Circular, Entropía) siguen de cerca los estándares de la literatura (método de Emory, fórmulas de Mardia). El código es robusto, bien documentado y matemáticamente sólido.
- **Hipótesis Inicial:** Se detectó una ambigüedad en el cálculo del **Bandwidth**, con dos métodos implementados (`narrowest_band_deg` vs. `bandwidth_p5_p95`). Se marcó como un punto a verificar contra los informes numéricos.

### 1.2. `ectb_lv.py`
- **Observación:** El cálculo de volúmenes y FEVI se basa en una reimplementación muy sofisticada del método del Emory Cardiac Toolbox (ECTb), que utiliza el máximo de cuentas en lugar de un umbral para definir el borde. La implementación incluye correcciones complejas como el plano valvular de dos piezas y el engrosamiento de pared basado en la curva de cuentas.
- **Hipótesis Inicial:** La alta complejidad y la dependencia de múltiples parámetros (`valve_septal_offset_mm`, `use_thickening`, etc.) fueron identificadas como una fuente potencial de error si la configuración no era la adecuada.

---

## Parte 2: Primera Verificación Numérica y Hallazgos Críticos

Se procedió al análisis de un primer informe (`fbp052-05--040-10.pdf`).

### Hallazgo 2.1 (Crítico): Filtro de Amplitud Excesivamente Agresivo
- **Observación:** El informe fue generado con el parámetro `Amp filter=0.40`.
- **Análisis:** Este valor es extraordinariamente alto (el estándar es 10-20%). Un umbral del 40% elimina del análisis a todos los vóxeles cuya contracción sea inferior al 40% del máximo.
- **Impacto:** Esto **enmascara la patología**, descartando los segmentos más enfermos (hipocinéticos/acinéticos) y produciendo una **subestimación drástica y peligrosa** de la disincronía. Las métricas de `Phase SD` y `Bandwidth` resultantes son falsamente optimistas.

### Hallazgo 2.2 (Moderado): Ambigüedad en "Bandwidth"
- **Observación:** El informe presenta un único valor de `Bandwidth` sin especificar si corresponde al método del 95% (ECTb) o al del 90% (P95-P5).
- **Impacto:** Potencial de mala interpretación al comparar con bases de datos normales o software de referencia que esperan una definición específica.

---

## Parte 3: Auditoría de Efectividad del Filtro "Nítida"

El usuario solicitó evaluar la efectividad de "Nítida" para compensar estudios de baja dosis, proporcionando un segundo informe del mismo estudio procesado con este filtro.

### Hallazgo 3.1 (Crítico): Comportamiento Paradójico de las Métricas
- **Observación:** Al comparar el informe "Nítida" con el FBP, las métricas de sincronía divergieron: `Phase SD` aumentó (peor sincronía), mientras que `Bandwidth` y `Entropía` disminuyeron (mejor sincronía).
- **Análisis:** Este comportamiento paradójico es una bandera roja que indica que el filtro no solo reduce el ruido, sino que **altera la forma de la distribución de fase de manera no física**, probablemente aplanando el pico central y creando valores atípicos en las colas (confirmado por el cambio en la Kurtosis de positiva a negativa).
- **Conclusión Parcial:** Bajo esta configuración, el filtro "Nítida" fue considerado **no confiable** para el análisis de sincronía.

### Hallazgo 3.2 (Crítico): Factor de Confusión Dominante
- **Observación:** Ambos informes (FBP y Nítida) se generaron con el `Amp filter=0.40`.
- **Análisis:** Este error de configuración común a ambos procesamientos **invalidaba científicamente la comparación**. Era imposible aislar el efecto de "Nítida" cuando ambos pipelines estaban siendo alimentados con datos ya masivamente sesgados.

---

## Parte 4: Descubrimiento de la Arquitectura y Causa Raíz

### 4.1. La Estrategia del "Pasajero FBP"
- **Aclaración del Usuario:** El usuario explicó que, consciente de los problemas de "Nítida" con la fase, había diseñado una arquitectura dual: la imagen y la FEVI se calculan con "Nítida", pero la sincronía se calcula sobre una reconstrucción FBP "pasajero".
- **Análisis de la Estrategia:**
    - **Pro:** Es una solución pragmática que aísla el cálculo de fase de los artefactos de "Nítida".
    - **Contra (Riesgo Científico):** Crea una **disonancia fundamental** entre la imagen que el médico ve (Nítida) y los números que la cuantifican (FBP). Además, introduce riesgos de inconsistencia en la segmentación. Se concluyó que es una **buena medida de contención, pero no una solución ideal a largo plazo.**

### 4.2. Análisis de `raw_reconstruction.py` y `ui/main_window.py`
- **Descubrimiento 1:** "Nítida" no es un simple filtro, sino una reconstrucción **OSEM con Recuperación de Resolución (RR)**, activada por el flag `resolution_recovery`.
- **Descubrimiento 2 (Causa Raíz):** El análisis de `ui/main_window.py` reveló la función `_on_nitida_toggled`. Esta función, al activar "Nítida", forzaba la reconstrucción OSEM pero también ejecutaba `self.cine_crudo_post_check.setChecked(False)`, **desactivando deliberadamente el post-filtro de suavizado.**

- **Conclusión de Causa Raíz:** La falla catastrófica de "Nítida" se debía a la aplicación de un algoritmo de realce de detalles (OSEM+RR) sobre datos de bajo conteo **sin la regularización (suavizado) necesaria para controlar la amplificación de ruido.**

---

## Parte 5: Prueba de Validación Definitiva

El usuario realizó la prueba de validación correcta: un estudio de 10 seg con FBP (Gold Standard) vs. uno de 5 seg con "Nítida".

### Hallazgo 5.1 (Fallo Catastrófico):
- **Observación:** Los resultados de "Nítida" sobre el estudio de 5 seg mostraron desviaciones masivas respecto al Gold Standard:
    - `Phase SD`: +24.7% (Empeoramiento drástico)
    - `FEVI`: -4.8% (Diferencia clínicamente significativa)
    - `EDV / ESV`: +39% / +110% (Sobreestimación radical de volúmenes)
- **Veredicto:** El filtro "Nítida", en su configuración sin regularización, **falla completamente** en su objetivo. Genera un resultado cuantitativamente peor y más engañoso que el del estudio de dosis completa.

---

## Parte 6: Conclusiones y Recomendaciones Finales para el Agente Programador

1.  **Acción Correctiva Realizada:** Se ha modificado `ui/main_window.py` para que al activar "Nítida" se active también por defecto el post-filtro de suavizado (`setChecked(True)`) con un FWHM de 8.0 mm. Esto corrige la causa raíz del fallo.

2.  **Estrategia de Pipeline (Recomendación):**
    - **Corto Plazo:** **Mantener la arquitectura del "pasajero" FBP.** Es la única forma garantizada de tener métricas de sincronía fiables mientras "Nítida" no esté completamente validado. La seguridad y la validez científica deben prevalecer sobre la elegancia arquitectónica.
    - **Largo Plazo:** El objetivo debe ser **eliminar el pasajero**. Para ello, se debe iniciar un proceso de validación con la nueva versión corregida de "Nítida". Se debe encontrar el "punto dulce" del parámetro de suavizado (`post_fwhm_spin`) que permita que los resultados de "Nítida" (en estudios de baja cuenta) se aproximen lo más posible a los del Gold Standard (FBP en alta cuenta) tanto para FEVI como para sincronía. Solo cuando se demuestre esta concordancia, se podrá unificar el pipeline.

3.  **Transparencia en el Informe:** Se recomienda encarecidamente que parámetros de procesamiento tan críticos como el `Amp filter`, el método de reconstrucción (`FBP` vs. `OSEM+RR`) y los parámetros de regularización (`post-filtro`) sean **explícitamente declarados en el informe final en PDF**. Esto es fundamental para la reproducibilidad, la auditoría y la confianza del usuario clínico.

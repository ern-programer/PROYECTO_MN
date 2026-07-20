# Base teórica — Integración ECG con análisis de sincronía y FEVI

**GammaSync / SINCRO · v1.9.0 · 2026-07-20**

Este documento registra la base científica de cómo los datos del ECG de 12 derivaciones inciden en la evaluación de FEVI (fracción de eyección) y de sincronía mecánica (PSD, Bandwidth, Entropy) por Gated SPECT MPI.

---

## 1. Principio fundamental

El análisis de fase SPECT mide **disincronía mecánica** (tiempos de contracción). El ECG mide **disincronía eléctrica** (tiempos de activación). Son fenómenos relacionados pero no equivalentes:

- QRS prolongado y disincronía eléctrica **no equivalen** necesariamente a disincronía mecánica.
- Los criterios clásicos de CRT (QRS ≥120-130 ms + FEVI ≤35%) dejan 20-40% de no respondedores.
- La fase SPECT aporta una medida objetiva y reproducible de la disincronía mecánica, complementaria al ECG.

Referencias: Atchley 2009 (PMID 19690935), Chen 2008 (PMID 18242490), Trimble 2007 (PMID 17556163), revisión sistemática CRT 2025 (PMID 39851177).

---

## 2. Impacto de cada dato ECG en FEVI

### 2.1 Ritmo

| Ritmo | Efecto en FEVI por gated SPECT | Mecanismo |
|---|---|---|
| **Sinusal** | Medición confiable | Gating estable |
| **FA (fibrilación auricular)** | **Subestimación** | Variabilidad RR → gating imperfecto, mezcla de latidos con distinto llenado |
| **Extrasístoles frecuentes** | Sesgo + menor estadística | Latidos ectópicos rechazados o contaminan el promedio |
| **Marcapasos** | Depende del modo | RV apical puede empeorarla; CRT puede mejorarla |

**Regla GammaSync:** si el ritmo es FA, la FEVI se marca como *preliminar/no confiable* y las métricas de fase se interpretan con máxima cautela.

### 2.2 FC (frecuencia cardíaca)

- FC >100 lpm: reduce tiempo de llenado diastólico → FEVI ligeramente subestimada.
- FC <50 lpm: puede alterar la distribución temporal de los gates.

### 2.3 QRS ancho (≥120 ms) y BRI

- La disincronía eléctrica produce contracción mecánica ineficiente → la **FEVI reducida es real**, no artefacto.
- Atchley 2009: en FEVI 35-50%, ~37% tenía disincronía mecánica significativa; controles PSD 8.8°/BW 28.7°; disfunción severa PSD 52.0°/BW 158.2°.
- La correlación QRS-disincronía mecánica es **débil**: QRS ancho no garantiza disincronía mecánica ni viceversa.

### 2.4 Marcapasos / CRT

- **Marcapasos RV apical:** patrón de activación tipo BRI → FEVI reducida por modo de estimulación, no necesariamente por miocardiopatía.
- **CRT:** en respondedores, mejora FEVI y reduce PSD/BW. Estudio 2026 en HFrEF (PMID 41912136): PSD y entropy en reposo predijeron normalización de FEVI/superrespuesta tras CRT.

---

## 3. Impacto de cada dato ECG en sincronía (PSD/BW/Entropy)

| Dato ECG | Efecto en métricas de fase | Interpretación |
|---|---|---|
| **BRI/LBBB** | PSD/BW típicamente elevados | Retraso activación lateral → contracción tardía póstero-lateral |
| **BRD/RBBB** | Efecto moderado | Retraso RV; menos impacto en VI |
| **Marcapasos RV apical** | Patrón tipo BRI | Activación desde apex |
| **CRT activo** | Reduce PSD/BW si responde | Mejora coordinación mecánica |
| **FA** | **Invalida las métricas** | Variabilidad RR → fases inestables voxel a voxel |
| **Extrasístoles** | Contaminan histograma | Latidos ectópicos tienen fase distinta |
| **QRS estrecho + PSD alto** | Discordancia | Sospechar artefacto, ROI, isquemia, o disincronía no eléctrica |

---

## 4. Matriz de concordancia electro-mecánica

GammaSync evalúa la concordancia ECG-SPECT así:

| QRS/BRI | PSD/BW | Lectura |
|---|---|---|
| Ancho/BRI | Elevado | **Concordante:** disincronía eléctrica y mecánica. Perfil CRT. |
| Ancho/BRI | Normal | **Discordante:** disincronía eléctrica sin mecánica. Posible falso positivo eléctrico. |
| Estrecho | Elevado | **Discordante:** mecánica sin eléctrica. Revisar ROI/ruido/isquemia/stunning. |
| Estrecho | Normal | Sin disincronía. |

Esta matriz se calcula automáticamente en el PDF cuando hay datos ECG cargados.

---

## 5. Cutoffs CRT heterogéneos — advertencia clave

La revisión sistemática CRT 2025 (33 estudios, 2066 pacientes) muestra cutoffs SPECT para respuesta CRT **extremadamente variables**:

- HBW: 55-152°
- PSD: 20-54°

Y dependen del software:
- **ECTb:** BW 135° / PSD 43°
- **QGS:** BW 83° / PSD 20°

**Regla GammaSync:** los cutoffs CRT son contexto, nunca indicación aislada. La decisión de CRT integra QRS, FEVI, perfusión, viabilidad, clínica y juicio médico.

---

## 6. Extracción automática de datos ECG

GammaSync soporta carga opcional de ECG de 12 derivaciones:

| Formato | Extracción | Estado |
|---|---|---|
| PDF con texto | Ritmo, FC, QRS, QT, QTc (Bazett), BRI, BRD, marcapasos | Implementado |
| PDF escaneado | Mismo set vía OCR (pytesseract + pdf2image/PyMuPDF) | Implementado (fallback) |
| SCP-ECG | Estructura básica | Stub (requiere `scp-ecg`) |
| DICOM Waveform | Metadatos | Stub (requiere estudio de campo) |

**Flujo:**
1. Usuario carga ECG (opcional).
2. `core/ecg_extractor.py` extrae datos con regex/parsers; si el PDF no tiene texto, aplica OCR.
3. Se comparan contra valores manuales: si hay diferencias, se listan una por una marcando las **significativas** (FC >20, QRS >20 ms, cambio de ritmo/BRI/BRD/marcapasos).
4. El profesional elige: aplicar valores del ECG o conservar manuales.
5. El modo manual siempre está disponible (no todos los pacientes traen ECG).

**QTc:** si no viene en el ECG, se calcula con Bazett: $QTc = QT / \sqrt{RR(s)}$.

### 6.1 ECG de adquisición (monitor de 3 derivaciones) embebido en el DICOM SPECT

El DICOM SPECT gated puede traer datos del ECG de adquisición (monitor de 3 derivaciones usado para el gating). GammaSync los extrae automáticamente al cargar el estudio (`core/dicom_loader._extract_gating_info`):

| Tag DICOM | Dato | Uso en GammaSync |
|---|---|---|
| HeartRate (0018,1088) | FC en lpm | Precarga campo FC |
| RRIntervalVector (0054,0060) | Intervalos RR en ms | RR medio, CV% (variabilidad), FC estimada (60000/RR) |
| TriggerWindow (0018,1094) | Ventana de aceptación % | Info de adquisición |
| HeartRate low/high (0018,1081/1082) | Límites FC aceptados | Info de adquisición |
| BeatRejection/PVC (0018,1080/1085) | Rechazo de latidos | QC de gating |
| TriggerSource (0018,1061) | Origen del trigger (EKG) | Info |

**Reglas:**
- Los datos de adquisición se **precargan por defecto** al cargar el estudio (FC, variabilidad RR), siempre editables manualmente.
- Si la variabilidad RR (CV%) > 15%, se marca advertencia de posible FA/extrasístoles.
- Al cargar el ECG de 12 derivaciones, se **contrastan** los datos: FC del monitor de 3 derivaciones vs FC del ECG de 12; diferencias >10 lpm se listan (>20 significativas).
- **Limitación conocida:** los DICOM exportados por Xeleris/MyoVation traen poco (solo `TriggerSource=EKG`); los crudos de la consola suelen traer más tags. La infraestructura está lista para ambos casos.

---

## 7. Qué NO hace la integración ECG

- No cambia el cálculo de fase (FFT/PSD/BW/entropy) — eso sale solo del SPECT.
- No genera un mapa de sincronía desde el ECG (el ECG no tiene resolución anatómica segmentaria).
- No reemplaza la interpretación médica del ECG.
- No invalida estudios sin ECG: son totalmente procesables en modo manual.

---

## 8. Referencias

| Tema | Referencia |
|---|---|
| Algoritmo fase Emory | Chen J et al. J Nucl Cardiol 2005 (PMID 16344229) |
| Metodología fase | Chen et al. 2008 (PMID 18242490) |
| Validación índices fase | Trimble et al. 2007 (PMID 17556163) |
| Disincronía en FEVI intermedia | Atchley et al. 2009 (PMID 19690935) |
| Normalidad por software | Kuronuma et al. J Cardiol 2023 (PMID 36858173) |
| Meta-análisis pronóstico | Lee et al. 2025 (PMID 39535673) |
| Revisión sistemática CRT | 2025 (PMID 39851177) — cutoffs HBW 55-152°, PSD 20-54° |
| CRT y G-SPECT en HFrEF | Stepien-Wroniecka et al. 2026 (PMID 41912136) |
| QPS/QGS normalidad | Hamalainen / García-Gómez Rev Colomb Cardiol 2018 |

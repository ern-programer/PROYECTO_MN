# NÍTIDA III — Fundamento técnico y diseño original

**Estado:** borrador de trabajo (rama `MATRIZ_FINA_(k3)`), 2026-08-11.
**Objetivo:** filtro/reconstrucción de calidad diagnóstica para SPECT miocárdico de **mitad de tiempo o mitad de dosis**, comparable a software licenciado (Evolution/Philips, Astonish/GE, WBR/UltraSPECT), con **reglas propias** y base física/matemática publicada.

---

## 1. El problema físico (por qué NÍTIDA/RR actual "hace lo contrario")

La recuperación de resolución (RR) modela la PSF del colimador dentro del proyector OSEM. Matemáticamente es **deconvolución**: amplifica las frecuencias que la PSF atenúa. Esas frecuencias son justamente donde el **ruido domina sobre la señal** en bajo conteo. Resultado medido en nuestros datos:

- Cada iteración OSEM+RR mejora resolución aparente y **degrada SNR** (rellena la cavidad, motea la pared).
- A **6.8 mm/voxel la PSF es sub-píxel** → la RR casi no tiene resolución real que recuperar, pero amplifica el ruido igual.

**Conclusión física:** RR sola no compensa mitad de cuentas. Lo que compensa es **regularización (prior) + información adicional que ya tenemos y no usamos**.

## 2. Qué hace la competencia (dominio público, como referencia conceptual)

| Producto | Esencia (publicado) | Clave real |
|----------|--------------------|------------|
| **WBR / UltraSPECT** (DePuey 2008-2011) | OSEM + RR + **control de ruido durante la recon** (sin post-filtro) | El freno de ruido, no la RR |
| **Evolution / Philips** (Filipczak 2017) | OSEM (2 iter) + RR + **filtro adaptativo dependiente de SNR local** | Suaviza donde hay poca señal, preserva bordes donde hay señal |
| **Astonish / GE** | OSEM + PSF + **regularización** + scatter | Penalización por iteración |

Nada es mágico: son aplicaciones de **MAP-OSEM** (maximum a posteriori) y priors edge-preserving, toda matemática publicada. No copiamos código; construimos sobre la base publicada.

## 3. Nuestra propuesta original: NÍTIDA III

Tres pilares, dos de los cuales ya construimos y validamos hoy:

### Pilar A — Feta axial restringida (YA HECHO, validado)
Reconstruir sólo la banda axial del corazón (markers Base/Ápex del usuario). En SPECT paralelo cada corte z es independiente → recorte **exacto** y 4-5× más rápido. Elimina contaminación extracardíaca y concentra todo el cómputo en la ROI. **Esto ya es una ventaja propia.**

### Pilar B — Guía ungated (alto conteo) para regularizar el gated (bajo conteo)
El mismo paciente tiene un ungated de alto conteo (suma de gates) y 8 gates de bajo conteo. El ungated tiene la **anatomía** (bordes del miocardio) con buena SNR; los gates tienen el **movimiento** pero ruido. Usar el ungated como **guía estructural** para regularizar cada gate (filtrado guiado / prior anatómico). Literatura relacionada: guided reconstruction, pero con espacio para variante propia.

### Pilar C — "Matched Recovery": RR adaptativa por SNR local
En vez de aplicar la PSF completa (amplifica ruido), aplicar una **fracción** de la recuperación controlada por la SNR local medida en cada región. Donde la SNR es alta (pared bien perfundida), recuperar más; donde es baja (cavidad, defectos), no amplificar ruido. **Formulación propia** sobre el principio MAP publicado.

## 4. Base matemática (publicada)

- **MLEM/OSEM:** Shepp & Vardi 1982; Hudson & Larkin 1994 (subsets).
- **MAP-OSEM / Green OSL:** Green 1990 — update con prior: `x ← x·BP(y/FP(x)) / (S·(1+β·∂U/∂x))`.
- **Priors edge-preserving:** Huber, Total Variation (Rudin-Osher-Fatemi 1992), mediana.
- **RR / modelado de PSF:** Tsui/Frey — PSF dependiente de profundidad del colimador.
- **Trade-off resolución-ruido:** documentado en toda la literatura de recon emisión.

## 5. Métricas objetivas (no sólo ojo)

Con el par real 5s/10s del mismo paciente:
- **CNR cavidad/pared** = (mean_pared − mean_cavidad) / std_fondo.
- **Fidelidad al 10s:** el 5s+NÍTIDA III debe acercarse al 10s FBP (referencia) en CNR y en perfil de pared.
- **Espesor de pared** (FWHM del perfil transmural): no debe inflarse (la queja del "miocardio grueso inventado").

## 6. Referencias clave (acceso legal: PubMed/PMC)

1. Ali I, et al. *Half-time SPECT myocardial perfusion imaging with attenuation correction.* J Nucl Med 2009;50:554-62. PMID 19289436 (free).
2. DePuey EG, et al. *OSEM and WBR "half-time" gated MPI SPECT: comparison to full-time FBP.* J Nucl Cardiol 2008;15:547-63. PMID 18674723.
3. DePuey EG, et al. *WBR "quarter-time" gated MPI SPECT.* J Nucl Cardiol 2009;16:736-52. PMID 19533264.
4. DePuey EG, et al. *Full-time MPI SPECT vs WBR half-time and half-dose.* J Nucl Cardiol 2011;18:273-80. PMID 21287370.
5. Armstrong IS, et al. *Reduced-count myocardial perfusion SPECT with resolution recovery.* Nucl Med Commun 2012;33:121-9. PMID 22107994.
6. Marcassa C, et al. *WBR for half-dose or half-time cardiac gated SPECT.* Eur J Nucl Med Mol Imaging 2011;38:499-508. PMID 21069317.
7. Filipczak K, et al. *Shortened gated MPI processed with "Myovation Evolution" vs full time.* Nucl Med Rev 2017;20:25-31. PMID 28198518 (free).
8. Druz RS, et al. *WBR half-time SPECT improves diagnostic certainty.* J Nucl Cardiol 2011;18:52-61. PMID 21181520.

## 7. Advertencia de honestidad intelectual

- "Patente" la define un agente de PI con búsqueda de anterioridad formal. Lo que construimos es un **algoritmo original con fundamento publicable**, que es la base para esa discusión.
- No usamos código ni secretos de Evolution/Astonish/WBR: sólo conceptos de papers públicos.

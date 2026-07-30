# Fundamento matemático de la sustracción de fondo por ROI de referencia

**Módulo:** GammaSync / SINCRO — `core/intestinal_subtraction.py`
**Fecha:** 2026-07-30
**Estado:** documento de respaldo para auditoría
**Ámbito:** justificación formal de la operación implementada, sus efectos demostrables, y delimitación explícita de lo que **no** hace.

---

## 0. Declaración de alcance (leer antes que nada)

La operación implementada en este módulo es una **sustracción de fondo aditivo estimado a partir de ROI de referencia trazadas por el operador**.

**No es, ni puede ser, una corrección de atenuación.** No es un sustituto de la corrección de Chang, ni de la corrección basada en TC, ni de una fuente lineal de transmisión. La Sección 5 contiene la demostración formal de esta imposibilidad, y la Sección 6 muestra que en lo relativo a **uniformidad regional** la sustracción de fondo actúa en **sentido opuesto** al de una corrección de atenuación.

Cualquier documento, informe o rótulo de interfaz que describa esta operación como "corrección de atenuación" es **incorrecto** y debe corregirse.

---

## 1. Nomenclatura

| Símbolo | Significado | Unidades |
|---|---|---|
| $\vec{r}=(x,y,z)$ | posición del vóxel en la imagen reconstruida | — |
| $a(\vec{r})$ | actividad real (concentración de radiotrazador) | cuentas equivalentes |
| $A(\vec{r})$ | factor de atenuación, $A\in(0,1]$ | adimensional |
| $S(\vec{r})$ | contribución de radiación dispersa (*scatter*) | cuentas |
| $B(\vec{r})$ | fondo por actividad extracardíaca difusa | cuentas |
| $F(\vec{r}) = S+B$ | término aditivo total, "fondo" | cuentas |
| $\hat{F}(\vec{r})$ | estimación del fondo hecha por este módulo | cuentas |
| $I(\vec{r})$ | intensidad medida en la imagen reconstruida | cuentas |
| $\mu$ | coeficiente de atenuación lineal | cm⁻¹ |
| $d_i(x,y)$ | distancia del píxel al borde del contorno corporal en el ángulo $i$ | cm |
| $N$ | número de ángulos de proyección | — |
| $n_g$ | número de gates (intervalos R-R) | — |
| $t$ | tiempo dentro del ciclo cardíaco | — |
| $\omega = 2\pi/T$ | frecuencia fundamental cardíaca | rad/s |

---

## 2. Modelo de formación de la imagen

Para una proyección adquirida en el ángulo $i$, las cuentas registradas en un elemento del detector son:

$$
p_i \;=\; \underbrace{\int_{L_i} a(\vec{r})\, \exp\!\left[-\int_{\vec{r}}^{\text{det}} \mu(\vec{r}\,')\,dl'\right] dl}_{\text{primarios, atenuados}} \;+\; \underbrace{S_i}_{\text{dispersos}} \;+\; \underbrace{B_i}_{\text{fondo}}
$$

Tras la reconstrucción tomográfica **sin compensación de atenuación ni de scatter**, la imagen admite el modelo aproximado, ampliamente aceptado en la literatura de SPECT cuantitativo:

$$
\boxed{\;I(\vec{r}) \;\approx\; a(\vec{r})\cdot A(\vec{r}) \;+\; F(\vec{r})\;}
\tag{2.1}
$$

La estructura de la ecuación (2.1) es el punto central de todo este documento:

- La atenuación entra como **factor multiplicativo** $A(\vec{r})$, con $0 < A \le 1$, que **varía espacialmente** según la profundidad del vóxel dentro del paciente y la composición de los tejidos interpuestos.
- El fondo entra como **término aditivo** $F(\vec{r})$.

Son dos perturbaciones de naturaleza algebraica distinta. Esta distinción no es una sutileza de notación: determina qué operación puede deshacer cada una.

---

## 3. Las dos correcciones, formalmente

### 3.1 Corrección de atenuación

$$
\hat{a}_{\text{att}}(\vec{r}) \;=\; \frac{I(\vec{r})}{A(\vec{r})}
\tag{3.1}
$$

Es una **división** por un campo escalar que depende de la posición. Requiere conocer $A(\vec{r})$, sea por medición (TC, fuente de transmisión) o por modelo geométrico con $\mu$ asumido (Chang).

### 3.2 Sustracción de fondo (lo implementado aquí)

$$
\hat{a}_{\text{sub}}(\vec{r}) \;=\; \max\!\big(\,I(\vec{r}) - \hat{F}(\vec{r})\,,\; 0\,\big)
\tag{3.2}
$$

Es una **resta** seguida de recorte a cero. Sustituyendo (2.1):

$$
\hat{a}_{\text{sub}}(\vec{r}) \;=\; a(\vec{r})\,A(\vec{r}) \;+\; \big[F(\vec{r}) - \hat{F}(\vec{r})\big]
\tag{3.3}
$$

Si la estimación es perfecta, $\hat{F} = F$, entonces:

$$
\hat{a}_{\text{sub}}(\vec{r}) \;=\; a(\vec{r})\cdot A(\vec{r})
\tag{3.4}
$$

**El factor $A(\vec{r})$ permanece intacto.** La sustracción de fondo, incluso en el caso ideal de estimación exacta, deja la imagen atenuada.

---

## 4. Estimación del fondo implementada

### 4.1 Nivel por ROI de referencia

Para cada ROI de referencia $R_k$ trazada por el operador sobre la imagen **promediada sobre gates**:

$$
\bar{I}(x,y) \;=\; \frac{1}{n_g}\sum_{g=1}^{n_g} I_g(x,y)
\tag{4.1}
$$

$$
L_k \;=\; \operatorname{mediana}\big\{\, \bar{I}(x,y) \;:\; (x,y)\in R_k \,\big\}
\tag{4.2}
$$

Se emplea la **mediana** y no la media porque el estimador debe ser robusto frente a píxeles calientes aislados (ruido Poisson, focos extracardíacos parciales incluidos por error en la ROI). El punto de ruptura de la mediana es 50 %; el de la media, 0 %.

Las ROI con menos de `MIN_REFERENCE_PIXELS` = 4 píxeles se descartan: $L_k$ no es estadísticamente utilizable.

### 4.2 Método A — interpolación por distancia inversa (`method="idw"`)

Con $D_k(x,y)$ = distancia euclídea del píxel a la ROI $R_k$ (transformada de distancia exacta):

$$
w_k(x,y) \;=\; \frac{1}{\big(D_k(x,y)+\varepsilon\big)^{\,p}}, \qquad \varepsilon = 10^{-3},\; p = 1
\tag{4.3}
$$

$$
\hat{F}(x,y) \;=\; \frac{\sum_k w_k(x,y)\,L_k}{\sum_k w_k(x,y)}
\tag{4.4}
$$

Propiedades: $\hat{F}$ es una media ponderada convexa de los $L_k$, por lo que $\min_k L_k \le \hat{F} \le \max_k L_k$ en todo el dominio (no puede extrapolar fuera del rango medido). Sobre cada ROI $R_k$ se cumple $\hat{F}\to L_k$ salvo el término de orden $\varepsilon$.

Adecuado cuando el fondo tiene **estructura espacial** conocida (por ejemplo un asa intestinal cuyo nivel cambia entre su entrada y su salida del plano).

### 4.3 Método B — media simple (`method="mean"`)

$$
\hat{F}(x,y) \;=\; \bar{L} \;=\; \frac{1}{K}\sum_{k=1}^{K} L_k \qquad \forall (x,y)
\tag{4.5}
$$

Constante en todo el plano. **Cada ROI pesa igual, con independencia de su tamaño**: es una media de los niveles $L_k$, no una media ponderada por número de píxeles. Esto evita que una ROI grande domine el estimador, lo cual es deseable cuando el operador traza varias ROI deliberadamente en regiones distintas para muestrear el fondo.

Corresponde al método clásico de ROI de fondo empleado en ventriculografía isotópica y renografía.

### 4.4 Aplicación al cubo gatillado

Sea $W(x,y)\in[0,1]$ el mapa de pesos de la región a corregir (ROI del operador con borde suavizado). Para **todos** los gates $g$:

$$
I'_g(x,y) \;=\; \max\!\big(\, I_g(x,y) \;-\; \hat{F}(x,y)\cdot W(x,y)\,,\; 0 \,\big)
\tag{4.6}
$$

**Regla inviolable:** $\hat{F}$ se estima **una sola vez** sobre la imagen promediada (4.1) y se resta **idéntica a todos los gates**. Estimar el fondo gate por gate introduciría una variación temporal artificial en $\hat{F}(t)$ que se propagaría al análisis armónico y contaminaría la fase. Como la estimación se realiza sobre el **promedio** y no sobre la **suma**, se resta $\hat{F}$ directamente y no $\hat{F}/n_g$.

---

## 5. Teorema: la sustracción de fondo no puede emular una corrección de atenuación

**Enunciado.** No existe ningún campo $\hat{F}(\vec{r})$ estimable a partir de ROI de referencia tal que la operación (3.2) recupere $a(\vec{r})$, salvo en el caso degenerado sin atenuación.

**Demostración.** Para que la sustracción recupere la actividad real se requiere:

$$
a(\vec{r})\,A(\vec{r}) + F(\vec{r}) - \hat{F}(\vec{r}) \;=\; a(\vec{r})
$$

Despejando:

$$
\boxed{\;\hat{F}(\vec{r}) \;=\; F(\vec{r}) \;-\; a(\vec{r})\,\big[\,1 - A(\vec{r})\,\big]\;}
\tag{5.1}
$$

El término corrector exigido, $a(\vec{r})\,[1-A(\vec{r})]$, **depende del producto de la actividad desconocida $a(\vec{r})$ por el déficit de atenuación $1-A(\vec{r})$**. Se sigue que:

1. **Circularidad.** Determinar $\hat{F}$ requiere conocer $a$, que es precisamente la incógnita. Si $a$ fuera conocida, no habría nada que corregir.
2. **No estimabilidad desde ROI de fondo.** Las ROI de referencia se trazan sobre regiones **exteriores** al miocardio. El término $a(\vec{r})[1-A(\vec{r})]$ es una propiedad **interna** del miocardio, específica de cada vóxel. Ninguna medición fuera del objeto puede recuperarlo.
3. **Caso degenerado.** La igualdad se reduce a $\hat{F}=F$ únicamente si $A(\vec{r})\equiv 1$, es decir, en ausencia total de atenuación. $\blacksquare$

**Corolario (interpretación geométrica).** En el espacio de intensidades, la sustracción de un fondo es una **traslación**; la corrección de atenuación es un **reescalado dependiente de la posición**. Una traslación no pertenece al grupo de las homotecias no uniformes. Ninguna composición de traslaciones puede generar un reescalado espacialmente variable.

---

## 6. Efecto sobre la uniformidad regional: la sustracción actúa en sentido contrario

Este resultado es contraintuitivo y constituye el **principal riesgo clínico** del uso propuesto. Se documenta explícitamente.

### 6.1 Planteo

Considérense dos regiones del miocardio con **idéntica actividad real** $a$ pero distinta atenuación: la pared anterior, más superficial ($A_{\text{ant}}$), y la pared inferior, más profunda y afectada por diafragma y contenido abdominal ($A_{\text{inf}}$), con

$$A_{\text{inf}} < A_{\text{ant}}$$

Las intensidades medidas, con fondo $F>0$ homogéneo:

$$
I_{\text{ant}} = a\,A_{\text{ant}} + F, \qquad I_{\text{inf}} = a\,A_{\text{inf}} + F
$$

Se define el cociente de uniformidad inferior/anterior, que es la magnitud que el clínico lee como "defecto":

$$
\rho \;=\; \frac{I_{\text{inf}}}{I_{\text{ant}}} \;=\; \frac{a\,A_{\text{inf}} + F}{a\,A_{\text{ant}} + F}
\tag{6.1}
$$

Tras la sustracción perfecta del fondo:

$$
\rho' \;=\; \frac{a\,A_{\text{inf}}}{a\,A_{\text{ant}}} \;=\; \frac{A_{\text{inf}}}{A_{\text{ant}}}
\tag{6.2}
$$

### 6.2 Proposición

$$\rho' < \rho \qquad \text{para todo } F>0 \text{ con } A_{\text{inf}}<A_{\text{ant}}$$

**Demostración.** Sean $x = a\,A_{\text{inf}}$ e $y = a\,A_{\text{ant}}$, con $0<x<y$. Hay que probar $\dfrac{x}{y} < \dfrac{x+F}{y+F}$. Multiplicando en cruz (denominadores positivos):

$$x\,(y+F) < y\,(x+F) \iff xy + xF < xy + yF \iff xF < yF \iff x < y$$

que se cumple por hipótesis. $\blacksquare$

**Además:** $\lim_{F\to\infty}\rho = 1$. Es decir, **el fondo enmascara el gradiente de atenuación**, aproximando artificialmente el cociente a la unidad. Restarlo lo **desenmascara**.

### 6.3 Ejemplo numérico

Actividad real uniforme $a = 100$; $A_{\text{ant}} = 0{,}70$; $A_{\text{inf}} = 0{,}50$; fondo $F = 20$.

| Magnitud | Antes de restar | Después de restar |
|---|---|---|
| $I_{\text{ant}}$ | 90 | 70 |
| $I_{\text{inf}}$ | 70 | 50 |
| Cociente $\rho$ | 0,778 | 0,714 |
| "Defecto" aparente | −22,2 % | −28,6 % |

El defecto inferior aparente **se agrava en 6,4 puntos porcentuales**. Una corrección de atenuación correcta habría llevado el cociente a 1,000 (defecto 0 %, que es la verdad: la actividad era uniforme).

### 6.4 Conclusión de la sección

La sustracción de fondo **acentúa** los artefactos de atenuación en lugar de compensarlos. Empleada con la intención de "corregir atenuación" puede **fabricar o agravar un defecto inferior inexistente**, con el consiguiente riesgo de falso positivo.

---

## 7. Lo que la sustracción de fondo sí mejora, demostrado

### 7.1 Cociente pared/cavidad

$$
R = \frac{I_{\text{pared}}}{I_{\text{cavidad}}} = \frac{a_p A_p + F}{a_c A_c + F} \;\xrightarrow{\;-F\;}\; R' = \frac{a_p A_p}{a_c A_c}
\tag{7.1}
$$

Por la misma desigualdad de la Sección 6.2 (con los papeles invertidos, pues ahora el numerador es mayor), $R' > R$: **el contraste pared/cavidad aumenta**. Y como pared y cavidad distan pocos centímetros, $A_p \approx A_c$, de modo que

$$R' \approx \frac{a_p}{a_c}$$

es decir, el cociente **sí queda limpio de fondo** y se aproxima al cociente de actividades reales. Este es un beneficio genuino y cuantificable.

### 7.2 Amplitud relativa del primer armónico (relevante para el análisis de fase)

Modelo de la señal temporal en un vóxel miocárdico, con fondo estático $F$:

$$
s(t) \;=\; m_0 \;+\; a_1\cos(\omega t + \phi) \;+\; F
\tag{7.2}
$$

La transformada discreta de Fourier separa:

- **Bin 0 (continua):** $m_0 + F$
- **Bin 1 (primer armónico):** módulo $\propto a_1$, argumento $\phi$

De aquí se derivan tres resultados:

**(a) La fase es invariante ante el fondo.** $F$ es constante en $t$, luego contribuye exclusivamente al bin 0. El argumento del bin 1 no se altera:

$$\phi' = \phi$$

Restar fondo **no desplaza la fase**. Esto es lo que legitima la operación en un módulo de sincronía: no introduce sesgo en la magnitud primaria que se reporta.

**(b) La amplitud relativa sí mejora.** La métrica que emplea el filtro de amplitud es

$$
\alpha \;=\; \frac{a_1}{m_0 + F} \;\xrightarrow{\;-F\;}\; \alpha' = \frac{a_1}{m_0} \;>\; \alpha
\tag{7.3}
$$

**(c) La atenuación multiplicativa no altera la amplitud relativa.** Si en cambio se multiplica la señal por un factor $k$ (que es lo que hace la herramienta de atenuación porcentual del visor):

$$
s_k(t) = k\big[m_0 + a_1\cos(\omega t+\phi) + F\big]
\;\Longrightarrow\;
\alpha_k = \frac{k\,a_1}{k\,(m_0+F)} = \frac{a_1}{m_0+F} = \alpha
\tag{7.4}
$$

El factor $k$ se cancela. **Escalar no mejora la amplitud relativa; solo restar lo hace.** Esta es la razón fundacional por la que se implementó la sustracción además de la atenuación porcentual preexistente.

### 7.3 Control de calidad derivado

De (7.3) se sigue un criterio de validación objetivo: **tras restar, $\alpha$ debe aumentar**. Si $\alpha' \le \alpha$, la región corregida no contenía miocardio recuperable sino fondo, o bien las ROI de referencia estaban mal situadas. El módulo calcula y registra $\alpha$ antes y después por cada corte.

---

## 8. Corrección de Chang: especificación completa

Se documenta el método correcto, a efectos de dejar constancia de qué sería necesario implementar y por qué no es asimilable a lo actual.

**Referencia:** Chang LT. *A Method for Attenuation Correction in Radionuclide Computed Tomography.* IEEE Transactions on Nuclear Science, 1978; NS-25(1):638–643.

### 8.1 Formulación (orden cero)

Para cada píxel $(x,y)$ de un corte **transaxial** reconstruido:

$$
\boxed{\;C(x,y) \;=\; \frac{N}{\displaystyle\sum_{i=1}^{N} \exp\big[-\mu\, d_i(x,y)\big]}\;}
\tag{8.1}
$$

$$
I_{\text{Chang}}(x,y) \;=\; I(x,y)\cdot C(x,y)
\tag{8.2}
$$

donde $d_i(x,y)$ es la longitud del trayecto desde el píxel hasta el borde del **contorno corporal** en la dirección de detección correspondiente al ángulo $i$, y $N$ es el número de ángulos considerados, distribuidos uniformemente en el arco de adquisición.

Obsérvese que $C(x,y) \ge 1$ y que es el recíproco de la media aritmética de los factores de atenuación sobre todos los ángulos: es, en efecto, una **división** por un $A(\vec{r})$ estimado geométricamente, en total coherencia con (3.1) y en total contraste con (3.2).

### 8.2 Chang de orden superior (iterativo)

1. Aplicar (8.2) para obtener $I^{(1)}$.
2. **Reproyectar** $I^{(1)}$ incorporando atenuación, obteniendo proyecciones simuladas $\tilde{p}_i$.
3. Reconstruir el error $p_i - \tilde{p}_i$ y sumarlo a $I^{(1)}$.
4. Repetir. En la práctica raramente se supera una o dos iteraciones por amplificación de ruido.

### 8.3 Coeficientes de atenuación lineal

> **Nota de auditoría:** los valores siguientes son de literatura general y deben verificarse contra fuente primaria (tablas NIST XCOM o el protocolo del servicio) antes de incorporarse a un documento regulatorio.

| Medio | Tc-99m (140 keV) | Tl-201 (~70–80 keV) |
|---|---|---|
| Agua / tejido blando, haz estrecho | ≈ 0,15 cm⁻¹ | ≈ 0,19 cm⁻¹ |
| Valor "efectivo" haz ancho (scatter incluido) | ≈ 0,12 cm⁻¹ | ≈ 0,15 cm⁻¹ |
| Pulmón | ≈ 0,04–0,05 cm⁻¹ | — |
| Hueso cortical | ≈ 0,25–0,30 cm⁻¹ | — |

El valor "efectivo" reducido se emplea cuando **no** se ha aplicado corrección de scatter, para no sobrecorregir.

### 8.4 Requisitos que el módulo actual no satisface

| Requisito de Chang | Estado en SINCRO |
|---|---|
| Contorno **corporal** (no del miocardio) | No disponible |
| Tamaño de píxel en **centímetros** | Disponible vía DICOM, no utilizado con este fin |
| Ángulos de proyección y centro de rotación | No propagados hasta esta etapa |
| Trazado de rayos para $d_i(x,y)$ | No implementado |
| Trabajar sobre cortes **transaxiales** | **No**: el módulo opera en **eje corto ya reorientado** |

El último punto es dirimente. La corrección de Chang debe aplicarse sobre los cortes transaxiales, cuyo plano es perpendicular al eje de rotación del gantry, porque la geometría $d_i(x,y)$ está definida respecto de ese eje. Una vez reorientado el volumen a eje corto, los planos ya no guardan relación con la geometría de adquisición y (8.1) deja de tener sentido.

### 8.5 Limitación intrínseca de Chang en cardiología

Chang asume $\mu$ **uniforme** dentro del contorno. El tórax es marcadamente heterogéneo (pulmón, hueso, tejido blando, mama). Por ello la aplicación de Chang uniforme en SPECT cardíaco es discutida y puede **sobrecorregir** la pared inferior. Las guías de práctica favorecen la corrección basada en TC o en fuentes de transmisión. Un Chang uniforme no debe presentarse como equivalente a estas.

---

## 9. Alternativas válidas para mitigar el artefacto de atenuación sin TC

Se listan por constituir la respuesta correcta a la necesidad clínica que motivó la consulta:

1. **Adquisición complementaria en decúbito prono.** Desplaza el diafragma y el contenido abdominal, modificando el patrón de atenuación inferior. Un defecto que se normaliza en prono es, con alta probabilidad, atenuación.
2. **Análisis de motilidad y engrosamiento parietal a partir del gated.** Un segmento con perfusión disminuida por atenuación conserva motilidad y engrosamiento normales; un infarto no. **Este es el discriminador estándar y SINCRO ya dispone de la información gatillada necesaria.**
3. **Bases de datos de normalidad estratificadas por sexo y hábito corporal.** Compensan estadísticamente el patrón esperado de atenuación mamaria y diafragmática.
4. **Corrección basada en TC o fuente lineal de transmisión.** Única solución física completa.

Ninguna de estas es sustituible por la sustracción de fondo.

---

## 10. Riesgos, limitaciones y condiciones de uso

### 10.1 Riesgos identificados

| Riesgo | Mecanismo | Mitigación implementada |
|---|---|---|
| Agravamiento del defecto inferior | Sección 6 | Advertencia en informe; QC de amplitud relativa |
| Sobresustracción (píxeles a cero) | $\hat{F} > I$ localmente | Recorte a 0, indicador `clipped_fraction`, alerta `oversubtracted` |
| ROI de referencia sobre **aire** | Mediana ≈ 0, subestima el fondo | Requiere criterio del operador; **no** detectado automáticamente |
| ROI de referencia sobre otro órgano captante | Sobreestima el fondo | Requiere criterio del operador |
| Fondo intracardíaco ≠ fondo extracardíaco | El scatter dentro del corazón procede en parte del propio miocardio y de la sangre cavitaria | Limitación **no corregible** por este método |
| Interpretación errónea como corrección de atenuación | Nomenclatura | Este documento; declaración en el informe |

### 10.2 Condición de validez del estimador

El método presupone que el fondo medido **fuera** de la región a corregir es representativo del fondo **dentro** de ella:

$$\hat{F}\big|_{\text{ROI referencia}} \;\approx\; F\big|_{\text{región corregida}}$$

Esta hipótesis es razonable para una estructura extracardíaca contigua y de nivel homogéneo (asa intestinal, borde hepático). Es **más débil** cuando las ROI de referencia se distribuyen por toda la imagen para estimar un fondo global, porque el scatter dentro del volumen cardíaco tiene una componente originada en el propio corazón que por construcción no está presente en las regiones de referencia. En ese régimen el método **subestima** sistemáticamente el fondo intracardíaco.

### 10.3 Declaración obligatoria en el informe

Todo estudio procesado con esta corrección debe declararlo, con indicación de los cortes afectados, las cuentas sustraídas y el porcentaje correspondiente, junto con la advertencia de que los hallazgos en los cortes corregidos deben interpretarse con cautela y correlacionarse siempre con la clínica del paciente, la perfusión regional y el resto del estudio. Debe constar asimismo que la operación **no constituye corrección de atenuación**.

### 10.4 Modos de uso admitidos

La operación admite dos configuraciones geométricas, con distinto grado de respaldo. Ambas ejecutan la misma matemática (Sección 4); lo que cambia es dónde se sitúan las ROI y, en consecuencia, qué conclusiones son legítimas.

#### Modo A — Fondo local de estructura contigua

**Configuración.** ROI a corregir sobre la zona de solapamiento con una estructura extracardíaca captante (asa intestinal, borde hepático). ROI de referencia sobre esa misma estructura donde se ve aislada.

**Hipótesis.** El nivel medido en la estructura fuera del solapamiento es representativo de su contribución dentro del solapamiento. Es una hipótesis **fuerte**, porque se muestrea la misma estructura física a pocos centímetros.

**Conclusiones legítimas.** Recuperación de la señal miocárdica en los cortes afectados; mejora de amplitud relativa; segmentación no capturada por el foco extracardíaco.

#### Modo B — Fondo general del campo aplicado al miocardio

**Configuración.** ROI a corregir abarcando el miocardio. ROI de referencia distribuidas por el resto del campo de visión.

**Hipótesis.** El fondo medido fuera del corazón es representativo del fondo dentro del volumen cardíaco. Es una hipótesis **más débil** que la del Modo A, por tres razones documentadas:

1. **Componente de autoscatter.** Una fracción del scatter presente dentro del volumen cardíaco se origina en el propio miocardio y en la sangre de la cavidad. Esa componente no está presente en ninguna ROI externa, de modo que el estimador $\hat F$ **subestima sistemáticamente** el fondo intracardíaco.
2. **Riesgo de contaminación del estimador.** ROI trazadas sobre aire producen mediana ≈ 0 y arrastran $\bar L$ a la baja; ROI que rocen hígado, intestino o pared torácica la arrastran al alza. En Modo B el operador muestrea un campo heterogéneo, por lo que la varianza entre $L_k$ es alta y el promedio (4.5) puede no representar a ninguna región real.
3. **Homogeneidad no verificada.** La ecuación (4.5) impone un fondo constante. En el Modo B esta suposición no se contrasta con ninguna medida; conviene inspeccionar la dispersión de los $L_k$ antes de aceptarla.

**Conclusiones legítimas en Modo B:**

- ✔ **Análisis de fase y sincronía.** Es el uso respaldado. Por (7.2)–(7.3), restar un término constante en el tiempo **no desplaza la fase** y **aumenta la amplitud relativa** $\alpha$, que es el criterio del filtro de amplitud. Los vóxeles miocárdicos de baja modulación que el fondo empujaba por debajo del umbral pasan a ser admitidos, con su fase correcta. Este beneficio es demostrable y se verifica automáticamente mediante el QC de la Sección 7.3.
- ✔ **Contraste pared/cavidad**, por (7.1).

**Conclusiones ilegítimas en Modo B:**

- ✘ **Juicio de perfusión regional.** Por la Sección 6, la sustracción **acentúa** los gradientes de origen atenuativo. Un cociente inferior/anterior calculado tras la sustracción **no** es interpretable como perfusión, y su comparación contra bases de datos de normalidad (construidas sin esta corrección) carece de validez.
- ✘ **Cuantificación absoluta** de captación miocárdica.
- ✘ Cualquier afirmación que presente el resultado como compensación de atenuación (Sección 5).

**Requisitos de trazabilidad para Modo B.** Debe registrarse el número de ROI de referencia, sus niveles $L_k$ individuales, su dispersión, el método de combinación empleado (4.4 o 4.5) y el valor final $\hat F$ aplicado. El módulo registra estos datos en el informe y en el registro de procesamiento.

---

## 11. Trazabilidad de la implementación

| Elemento del documento | Implementación |
|---|---|
| Ec. (4.1), (4.2) | `reference_levels()` |
| Ec. (4.3), (4.4) | `estimate_background_map(method="idw")` |
| Ec. (4.5) | `estimate_background_map(method="mean")` |
| Ec. (4.6) | `subtract_background_from_slice()` |
| Regla de estimación única sobre el promedio | `apply_intestinal_subtraction()` |
| Métrica $\alpha$ de la Sec. 7.2 | `relative_first_harmonic_amplitude()` |
| Verificación de (7.3) y (7.4) | `tests/test_intestinal_subtraction.py::test_restar_dc_intestinal_mejora_amplitud_sin_mover_la_fase` |
| Declaración en informe (Sec. 10.3) | `report/report_generator.py` |

---

## 12. Referencias

1. Chang LT. A Method for Attenuation Correction in Radionuclide Computed Tomography. *IEEE Trans Nucl Sci.* 1978;NS-25(1):638–643.
2. Jaszczak RJ, Greer KL, Floyd CE, Harris CC, Coleman RE. Improved SPECT quantification using compensation for scattered photons. *J Nucl Med.* 1984;25(8):893–900.
3. King MA, Glick SJ, Pretorius PH, et al. Attenuation, scatter, and spatial resolution compensation in SPECT. En: *Emission Tomography: The Fundamentals of PET and SPECT.* Academic Press.

> **Nota de auditoría:** las referencias 2 y 3 se citan por su pertinencia temática y deben verificarse en su edición y paginación exactas antes de la presentación formal del expediente.

---

## 13. Control de versiones del documento

| Versión | Fecha | Cambio |
|---|---|---|
| 1.0 | 2026-07-30 | Redacción inicial. Modelo de formación, teorema de imposibilidad (Sec. 5), efecto sobre uniformidad regional (Sec. 6), beneficios demostrados (Sec. 7), especificación de Chang (Sec. 8). |

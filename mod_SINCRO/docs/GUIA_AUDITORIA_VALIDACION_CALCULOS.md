# GUIA DE AUDITORIA Y VALIDACION DE CALCULOS (SINCRO)

Fecha: 2026-07-15
Estado: operativo (uso interno de validacion)

## 1. Objetivo clinico
Esta guia describe:
- Como se construyen los resultados (segmentacion, fase, volumenes, FE preliminar).
- Como usar ROI en slices apicales y basales cuando no se visualiza cavidad.
- Que impacto tiene cada decision sobre los indicadores reportados.

## 2. Flujo de calculo (resumen)
1. Se carga estudio gated SPECT (cubo: gates x slices x H x W).
2. Se segmenta miocardio (auto/threshold/manual).
3. Se calcula fase sobre la mascara de miocardio final.
4. Se derivan metricas globales de fase y analisis AHA/territorios.
5. Se estiman volumenes geometricos y FE preliminar (si hay condiciones minimas).

## 2.1 Que es AHA y que es el mapa AHA
- AHA significa American Heart Association.
- El mapa AHA (modelo de 17 segmentos) divide el VI en regiones estandarizadas:
	- Basal: 6 segmentos
	- Medio: 6 segmentos
	- Apical: 4 segmentos
	- Apex: 1 segmento
- Finalidad clinica:
	- Ubicar regionalmente la alteracion de fase/perfusion.
	- Estandarizar reportes entre software y entre estudios seriados.
	- Resumir informacion por territorios coronarios (LAD/LCx/RCA) para decision clinica.

Donde se muestra en SINCRO:
- UI: pestaña/panel "polar_map" (mapa polar AHA).
- PDF: figura del mapa polar AHA (Figura 2, bullseye 17 segmentos).

Como se usa en los calculos:
- La fase se calcula voxel a voxel sobre la mascara segmentada.
- Luego se proyecta esa fase a los 17 segmentos AHA.
- De ahi salen fase por segmento y por territorio, ademas de la clasificacion global.

## 3. Segmentacion manual por ROI
Formato por slice:
- slice,cy,cx,r_inner,r_outer

Ejemplos:
- Con cavidad visible: 9,12,11,4,7
- Apex/base sin cavidad visible: 9,12,11,-,7

Reglas:
- Centro y radio externo son obligatorios.
- r_inner puede ser '-' (sin interno) solo cuando no hay cavidad distinguible.
- Si r_inner existe, debe cumplir: 0 <= r_inner < r_outer.

Interpretacion geometrica:
- ROI con interno: anillo miocardico.
- ROI sin interno: disco hasta r_outer (sin exclusion central).

## 4. Recomendacion clinica de uso (apex y base)
Usar ROI sin interno (r_inner='-') cuando:
- El frame/slice no muestra cavidad ventricular clara.
- Se trata de apex extremo o base extrema con colapso/indistincion de la luz.

No usar ROI sin interno cuando:
- La cavidad es visible y delimitable.
- El objetivo es cuantificacion volumetrica mas estable de cavidad.

Criterio practico:
- Slices medios: preferir anillo completo (interno + externo).
- Apex/base extremos: permitido sin interno, con justificacion visual.

## 5. Impacto en resultados
### 5.1 Metricas de fase
- La fase se calcula sobre la mascara segmentada final.
- Si un slice se dibuja sin interno, entran mas voxeles centrales en la mascara de ese slice.
- Puede modificar histogramas y dispersion de fase si esos voxeles agregados tienen señal distinta.

### 5.2 Volumenes
- Volumen miocardico: puede aumentar en slices sin interno.
- Volumen de cavidad: en slices sin interno no hay contribucion directa de cavidad por radio interno.
- Resultado esperado: tendencia a mayor miocardio y menor cavidad en esos niveles.

### 5.3 FE preliminar
- La FE preliminar excluye slices sin radio interno finito.
- Esto reduce sesgo por cavidad no visible, pero puede bajar cantidad de slices validos.
- Si hay muy pocos slices validos, FE puede quedar no disponible.

## 6. Ecuaciones y definiciones usadas
### 6.1 Volumen de voxel
- voxel_ml = (dx_mm * dy_mm * dz_mm) / 1000

### 6.2 Volumen miocardico
- myocardial_ml = numero_de_voxeles_en_mascara * voxel_ml

### 6.3 Volumen de cavidad (geometrico)
- Se estima por conteo de pixeles dentro de r_inner por slice valido.
- cavity_ml = cavity_voxels * voxel_ml

### 6.4 FE preliminar
- EDV = max(volumen_cavidad_por_gate)
- ESV = min(volumen_cavidad_por_gate)
- EF(%) = ((EDV - ESV) / EDV) * 100

Nota:
- FE en este modulo se considera preliminar (investigacion/validacion), no sustituto de paquete comercial validado.

## 7. Requisitos de auditoria recomendados
Registrar por estudio:
- Metodo de segmentacion usado (auto/manual/threshold).
- Slices con ROI sin interno y justificacion.
- Parametros activos (threshold, sigma, harmonics, filtro amplitud).
- Version del software y fecha.

Checklist minimo:
- Revision visual de ROI por slice.
- Confirmar consistencia apex->medio->base.
- Verificar que FE preliminar tenga slices suficientes.

## 8. Limitaciones actuales
- FE marcada como preliminar.
- Segmentacion manual depende del operador.
- Slices sin interno pueden afectar comparabilidad entre estudios si se usan de forma inconsistente.

## 9. Uso para validacion
Este documento sirve para:
- Justificar tecnicamente el pipeline durante auditorias internas.
- Estandarizar criterios de carga manual ROI.
- Facilitar trazabilidad de decisiones y reproducibilidad.

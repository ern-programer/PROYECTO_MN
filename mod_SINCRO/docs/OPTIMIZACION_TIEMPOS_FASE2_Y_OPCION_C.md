# Optimizacion de Tiempos - Fase 2 (pendiente) y Opcion C (plan)

## Estado actual
- Rama de trabajo: OPTIMIZACION_TIEMPOS
- Implementado (Fase 1 / Opcion B):
  - Modo "Tiempo real (rapido) + HQ diferido".
  - En carga dual (stress/rest): fast-pass inmediato + pipeline HQ diferido.
  - En flujo principal con avanzado activo: fast-pass (basico) y render HQ diferido.
  - Objetivo: mejorar respuesta percibida sin bajar calidad final exportada.

## Fase 2 (guardar backlog)
Objetivo: reducir tiempo total real, no solo tiempo percibido.

1. Cola de renders cancelable por token
- Si el usuario mueve sliders/cambia ROI, cancelar job HQ en curso y quedarse con el ultimo.
- Evita trabajo desperdiciado y reduce bloqueos UI.

2. Render bajo demanda por pestaña (lazy)
- No renderizar todas las pestañas al procesar.
- Renderizar solo:
  - pestaña activa,
  - comparacion_stress_rest,
  - mini set clinico minimo.
- El resto se renderiza al abrir pestaña.
- Estado 2026-07-17:
  - Implementado lazy render al abrir pestañas pesadas.
  - Implementado render selectivo para `ventriculograma` y `bullseye_directo`.
  - Implementada carga selectiva de previews para evitar refrescar todo el set en cada procesamiento.
  - Fase 2.3: todas las pestañas pasan a render diferido; se conserva en disco/memoria de preview hasta que cambien parámetros o se reprocesen.

3. Composicion dual diferida y selectiva
- _compose_dual_tab_images solo cuando el usuario activa modo comparativo visual o exporta.
- Evitar componer todas las pestañas en cada carga dual.

4. GIF/MP4 bajo demanda
- polar_cine.gif/mp4 y comparacion_ejes cine solo al solicitar reproducir/exportar.
- En procesado normal, guardar frame estatico representativo.

5. Cache de imagenes por pestaña con invalidacion fina
- Cache por firma: (estudio, seg, fase, params visuales, estado ROI intestinal).
- Invalidar solo las pestañas impactadas por cada cambio.

6. Paralelizacion controlada
- Separar render secundario y composicion en tareas de worker (QThreadPool/QRunnable).
- Mantener UI libre con señales de progreso.

## Opcion C (explicacion)
Opcion C es una re-arquitectura de pipeline por grafo de dependencias.
No solo acelera la UI: reduce computo total al minimo necesario por cambio.

Idea central:
- Cada salida (histograma, polar, panel funcional, comparacion_ejes, etc.) depende de nodos previos.
- Nodos base:
  - carga DICOM,
  - preprocesado/atenuacion intestinal,
  - segmentacion,
  - fase,
  - metricas,
  - render por pestaña.
- Al cambiar un parametro, se recalculan solo nodos descendientes afectados.

Beneficios:
- Escala mejor en dual mode.
- Facilita paralelizar nodos independientes.
- Reduce rerender global innecesario.

Costo:
- Mayor complejidad de implementacion.
- Requiere definir contrato de datos por nodo + scheduler + invalidacion.

## Plan Opcion C (propuesto)
### C1 - Preparacion (bajo riesgo)
1. Definir estructuras de firma por nodo (hash inputs/outputs).
2. Mapear dependencias actuales por pestaña.
3. Extraer funciones puras por etapa (sin side effects de UI).

### C2 - Motor de dependencias
1. Implementar scheduler simple DAG.
2. Cache de nodos con almacenamiento en memoria.
3. Invalidacion por cambio de parametros/ROI/estudio.

### C3 - Integracion gradual
1. Migrar primero comparacion_stress_rest e histograma.
2. Luego polar y panel funcional.
3. Finalmente comparacion_ejes y cine polar.

### C4 - Rendimiento avanzado
1. Worker pool para nodos pesados.
2. Priorizacion por pestaña activa.
3. Prewarm de nodos probables en idle.

## Criterios de exito
- Interaccion < 300 ms en ajustes visuales frecuentes.
- Carga dual percibida rapida (preview en < 1.5 s en hardware objetivo).
- Render HQ completo sustancialmente menor en promedio.
- Calidad final de exportes igual a baseline clinico actual.

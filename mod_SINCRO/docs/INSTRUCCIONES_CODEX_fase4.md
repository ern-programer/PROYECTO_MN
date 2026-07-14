# Instrucciones para Codex — Implementar Fase 4 (SINCRO)

> Copiá y pegá el bloque de abajo en tu sesión con GPT 5.3-codex.

---

```txt
Proyecto: D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\

Tarea:
Implementar Fase 4 (visualización) siguiendo EXACTAMENTE:
docs/SPEC_fase4_visualizacion.md

Archivos a implementar:
- viz/colormaps.py   (colormap CÍCLICO para fase 0-360°)
- viz/polar_map.py   (bullseye 17 segmentos)
- viz/histogram.py   (histograma de fase con métricas)
- tests/test_viz.py

Contexto obligatorio (leer antes, NO modificar):
- core/aha_segments.py  (AHAResult.segment_map, phase_by_segment, TERRITORY_MAP,
                         SECTOR_TO_SEGMENT_BASAL/MEDIO/APICAL — importar los LUT)
- core/phase_analysis.py (PhaseResult)
- core/metrics.py        (calculate_phase_metrics, circular_mean_deg)
- core/console_utf8.py   (enable_utf8() en scripts con print)

Reglas:
- Respetar firmas y nombres de la spec.
- matplotlib con backend Agg (matplotlib.use("Agg")). NO usar PyQt en esta fase. NO plt.show().
- Sin dependencias nuevas (numpy, matplotlib, scipy).
- UTF-8 en todo. Cambios solo en mod_SINCRO/**.
- Donde quede algo abierto, elegir lo simple y dejar # TODO.
- No implementar lo marcado "Fuera de esta fase".

Entorno (venv propio, NO global):
cd "d:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO"
.\.venv\Scripts\python.exe tests\test_viz.py
.\.venv\Scripts\python.exe tests\test_phase_synthetic.py
.\.venv\Scripts\python.exe tests\test_segmentation.py

Definition of Done:
- test_viz pasa (colormap cíclico 0°≈360°, polar map y histograma generan PNG > 0 bytes)
- test_phase_synthetic y test_segmentation SIGUEN pasando (no romper F1/F3)
- PNGs se generan sin abrir ventanas

Git:
- Commits locales descriptivos. NO hacer git push.

Al terminar:
No avanzar a Fase 5. Terminar con este mensaje exacto:

✅ Fase 4 implementada y tests en verde. Volvé a Opus 4.8 para la revisión (QA del bullseye + colormap cíclico + no regresión F1/F3) antes de avanzar a la Fase 5.

Incluir resumen de: archivos tocados, tests que pasan, TODO abiertos.
```

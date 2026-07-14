# Instrucciones para Codex — Implementar Fase 3 (SINCRO)

> Copiá y pegá el bloque de abajo en tu sesión con GPT 5.3-codex.

---

## PROMPT PARA CODEX

Estás trabajando en el proyecto **mod_SINCRO** (análisis de asincronía cardíaca por fase, Gated SPECT). Ubicación: `D:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO\`.

### Tu tarea
Implementá la **Fase 3** siguiendo AL PIE DE LA LETRA la especificación en:
`mod_SINCRO/docs/SPEC_fase3_segmentacion_AHA.md`

Tenés que crear/completar:
1. `core/segmentation.py` — segmentación del miocardio (métodos auto/manual/threshold).
2. `core/aha_segments.py` — mapeo voxel → 17 segmentos AHA + territorios coronarios.
3. `tests/test_segmentation.py` — tests sintético + real (skip-safe).

### Contexto que DEBÉS leer antes de escribir (ya existe, NO lo modifiques salvo lo indicado)
- `core/dicom_loader.py` — cómo se carga el estudio y qué devuelve (`GatedStudy.cube`, shape `(n_gates, n_slices, H, W)`).
- `core/phase_analysis.py` — el motor de fase (Fase 1, ya validado). Devuelve `PhaseResult` con `phase_map`.
- `core/metrics.py` — **reutilizá** `circular_mean_deg` y `circular_std_deg` para TODO promedio angular. NO reimplementes medias circulares.
- `core/console_utf8.py` — llamá `enable_utf8()` al inicio de cualquier script con `print` (evita crash en consola Windows).

### Reglas obligatorias
- Respetá EXACTAMENTE las firmas, nombres de funciones, dataclasses y campos de la spec. No los cambies.
- Sin dependencias nuevas: solo `numpy`, `scipy`, `opencv-python` (cv2). Ya están instaladas.
- Todos los archivos en UTF-8. Todo entry point con `print` debe llamar `enable_utf8()`.
- Donde la spec deje una decisión abierta, elegí lo más simple y dejá un comentario `# TODO`.
- NO implementes cosas marcadas como "FUERA de esta fase" (LVSD 6 etapas, reorientación oblicua, polar map visual).

### Entorno para correr (IMPORTANTE)
Usá el venv propio del módulo, NO el global ni el de FUSION:
```powershell
cd "d:\- PROGRAMACIÓN\PROYECTO_MN\mod_SINCRO"
.\.venv\Scripts\python.exe tests\test_segmentation.py
.\.venv\Scripts\python.exe tests\test_phase_synthetic.py   # verificá que NO se rompió
```

### Criterios de aceptación (Definition of Done) — verificalos vos antes de terminar
- [ ] `tests/test_segmentation.py` pasa (sintético con anillo cubre segmentos 1..17; real es skip-safe si no está el DICOM).
- [ ] `tests/test_phase_synthetic.py` SIGUE pasando (no rompiste la Fase 1).
- [ ] `.\.venv\Scripts\python.exe -m core.segmentation <un_dcm_SA>` imprime resumen sin error.
- [ ] Sin dependencias nuevas. UTF-8 en todo.
- [ ] El pipeline de integración de la §5 de la spec funciona end-to-end.

### Commits
- Hacé commits locales descriptivos (`git commit`), pero **NO hagas `git push`** (regla del repo: el push lo autoriza el usuario).

### ⛔ AL TERMINAR — MUY IMPORTANTE
Cuando completes la implementación y los tests pasen, **NO sigas con la Fase 4 ni con nada más**.
Terminá tu respuesta con este mensaje EXACTO al usuario:

> "✅ Fase 3 implementada y tests en verde. **Volvé a Opus 4.8 para la revisión** (control de calidad del mapeo AHA + validación de que no se rompió la Fase 1) antes de avanzar a la Fase 4."

Dejá un resumen de: qué archivos tocaste, qué tests pasan, y cualquier decisión/`TODO` que hayas dejado abierto, para que Opus lo revise.

---

## Por qué este reparto
- **Opus** diseñó el algoritmo difícil (mapeo AHA, detección ápex/base) en la spec.
- **Codex** (vos) implementás el código mecánico siguiendo la spec.
- **Opus** revisa al final (control de calidad clínico + arquitectura).
Esto optimiza costos: la cabeza cara (Opus) solo diseña y revisa; el grueso del tipeo lo hace el modelo intermedio.

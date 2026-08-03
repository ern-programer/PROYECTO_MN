# Estado de restauración tras cierre inesperado de VS Code (2026-08-01)

## Lo que sobrevivió al cierre y commit `f7d9f6a`:
- [x] **Botones del sidebar**: Process verde (#16a34a), Restart rojo (#dc2626), enroque de posiciones (Process-izquierda, PDF-centro, Restart-derecha, proporción 3:3:1).
- [x] **Botón Auto eliminado** del sidebar.
- [x] **PDF fusionado** en un botón "PDF ▾" con menú (Abrir / Guardar como).
- [x] **Grid de Acciones reorganizado**: Cargar + Comparar en misma fila (Cargar col0, Comparar col1-2). Advanced, Export, Config, ECTb, GQC, Asincronía en filas siguientes.
- [x] **Ajuste manual DENTRO de Corrección de movimiento**: botón "Ajuste manual ▾" al final de toolbar3 (dentro del menú "Corrección de movimiento"). El botón separado "Ajuste manual y exportación ▾" eliminado de la fila de grupos.
- [x] **closeEvent con warning** si hay estudio procesado sin PDF guardado (`_check_unsaved_study`).
- [x] **Símbolos de play** ▶/⏸ en todos los botones (polar_cine, cine_crudo, compare_axes).
- [x] **Leading zeros** en label de frame (`Img 01/32` en vez de `Img 1/32`).

## Lo que se perdió y NO se restauró:

Todos los items fueron restaurados en el commit `v1.19.0` (2026-08-03). Detalle de lo que se re-aplicó:

- [x] Botón "Carpeta" movido a Configuración
- [x] Anchos fijos en combos de cine_crudo: `source_combo` (80px), `mode_combo` (100px), `frame_label` (180px)
- [x] Drag label "drag:" eliminado
- [x] `_update_cine_crudo_toggle_text` con símbolos ▶/⏸
- [x] `_update_compare_axes_toggle_text` con símbolos ▶/⏸
- [x] `_refresh_readonly_results_panel()` al final de `_handle_raw_projections_loaded`
- [x] `load_compare_study` con `_check_unsaved_study`
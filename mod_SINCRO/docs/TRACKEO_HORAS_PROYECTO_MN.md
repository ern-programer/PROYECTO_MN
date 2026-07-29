# Trackeo de horas trabajadas - PROYECTO_MN (era GammaSync)

Generado: 2026-07-28 23:18 por `_track_horas_proyecto.py`

## Resumen ejecutivo

| Metrica | Valor |
|---|---|
| Primer rastro (creacion repo) | 2026-07-14 12:12 |
| Ultimo rastro | 2026-07-28 23:00 |
| Dias calendario del periodo | 15 |
| **Dias con actividad** | **14** |
| Sesiones detectadas | 37 |
| **Horas crudas** (spans activos) | **55h 23m** |
| **Horas estimadas** (+20 min cola/sesion) | **67h 43m** |
| Promedio por dia activo (est.) | 4h 50m |

> **Lectura honesta:** las horas crudas son el piso medible (actividad
> digital registrada). Las estimadas agregan cola por arranque/cierre.
> El tiempo real (pensar, leer papers, chats sin tocar archivos) es MAYOR.

## Metodologia

- **Sesion:** secuencia de eventos con gaps <= 60 min. Un gap mayor
  cierra la sesion (pausa, corte, interrupcion) y abre otra.
- **Duracion cruda:** ultimo evento - primer evento de cada sesion.
- **Duracion estimada:** cruda + 20 min por sesion.
- **Fuentes de eventos:**
  - commits git (todas las ramas): 88
  - lineas de log de la app (arranques, procesamientos, exports): 1229
  - exports output_demo (json/csv/xlsx/npz): 656
  - presets guardados: 2
  - mtimes de .md de memoria/docs (trabajo en otros chats): 15

## Jornadas por dia

| Dia | Sesiones | Horarios (inicio-fin) | Crudas | Estimadas | Commits | Exports | Versiones |
|---|---|---|---|---|---|---|---|
| 2026-07-14 | 3 | 12:12-12:14, 17:10-18:51, 20:30-20:49 | 2h 03m | 3h 03m | 8 | 0 |  |
| 2026-07-15 | 3 | 13:57-13:57, 15:54-15:54, 20:05-22:50 | 2h 44m | 3h 44m | 14 | 0 | v1.0.0 |
| 2026-07-16 | 1 | 12:24-13:34 | 1h 10m | 1h 30m | 5 | 0 |  |
| 2026-07-17 | 3 | 01:11-01:54, 11:49-11:49, 14:29-14:29 | 0h 43m | 1h 43m | 3 | 0 | v1.4.0, v1.5.0 |
| 2026-07-18 | 3 | 12:11-12:11, 13:34-13:34, 15:14-15:14 | 0h 00m | 1h 00m | 2 | 0 | v1.5.1, v1.6.0 |
| 2026-07-19 | 1 | 12:07-12:07 | 0h 00m | 0h 20m | 1 | 0 | v1.6.1 |
| 2026-07-20 | 2 | 15:27-17:35, 20:12-00:26 | 6h 22m | 7h 02m | 13 | 41 | v1.7.0, v1.8.0, v1.9.0 |
| 2026-07-21 | 3 | 08:57-12:00, 13:01-15:26, 17:21-23:28 | 11h 34m | 12h 34m | 35 | 3 | v1.10.0 |
| 2026-07-22 | 5 | 10:59-12:04, 13:53-14:46, 16:01-17:46, 18:49-20:05, 21:10-01:01 | 8h 51m | 10h 31m | 2 | 39 | v1.10.1 |
| 2026-07-23 | 1 | 09:13-18:46 | 9h 34m | 9h 54m | 1 | 93 | v1.11.0 |
| 2026-07-24 | 3 | 09:01-11:31, 12:37-12:50, 20:12-20:15 | 2h 46m | 3h 46m | 0 | 42 |  |
| 2026-07-25 | 2 | 01:50-01:58, 11:13-12:31 | 1h 27m | 2h 07m | 0 | 102 |  |
| 2026-07-27 | 3 | 09:47-09:59, 13:30-17:23, 22:32-22:42 | 4h 16m | 5h 16m | 1 | 180 | v1.12.0 |
| 2026-07-28 | 4 | 15:12-15:58, 17:05-17:17, 18:35-19:58, 21:27-23:00 | 3h 54m | 5h 14m | 3 | 156 | v1.13.0 |

| **TOTAL** | **37** | | **55h 23m** | **67h 43m** | **88** | **656** | |

## Limitaciones

- Solo mide actividad digital: commits, corridas de la app, exports, guardados.
- Tiempo pensando, leyendo bibliografia o conversando en chats sin tocar
  archivos NO queda registrado (subestimacion).
- Una sesion con un solo evento cuenta span 0 (solo aporta la cola estimada).
- El 2026-07-13 (reanudacion del proyecto, creacion de memoria) queda fuera
  porque el monorepo se creo el 2026-07-14.

## Como regenerar

```powershell
& "C:\Users\Ernesto\AppData\Local\Programs\Python\Python313\python.exe" "D:\- PROGRAMACIÓN\PROYECTO_MN\FUSION\_track_horas_proyecto.py"
```

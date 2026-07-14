# PROYECTO_MN

Monorepo de **Gammasys** (medicina nuclear, Argentina). Aloja el motor principal y los módulos de procesamiento de estudios de medicina nuclear.

## Estructura

```
PROYECTO_MN/
├── mod_SINCRO/          # Módulo de análisis de asincronía cardíaca (Gated SPECT → fase)
├── motor_principal/     # (futuro) Software motor principal — sin nombre definitivo aún
└── mod_XXX/             # (futuro) Otros módulos
```

Cada módulo es autocontenido (su propio README, requirements y tests) pero comparte este repositorio.

> **Nota:** los proyectos legacy (FUSION, OPEN, visor_dicom) y todos los datos DICOM
> quedan FUERA del control de versiones (ver `.gitignore`). Este monorepo versiona
> únicamente los módulos nuevos y el motor principal.

## Módulos

| Módulo | Descripción | Estado |
|--------|-------------|--------|
| [`mod_SINCRO`](mod_SINCRO/README.md) | Análisis de sincronía/asincronía/disincronía cardíaca desde Gated SPECT Short Axis (fase FFT, polar map 17 seg AHA, informe). | En desarrollo (Fase 1) |
| `motor_principal` | Software motor principal (visualización, gestión de estudios, orquestación de módulos). | Pendiente |

## Convenciones

- Cada módulo nuevo se agrega como carpeta `mod_<NOMBRE>/` y se habilita en `.gitignore` raíz.
- Datos médicos (`*.dcm`, `data_test/`) NUNCA se versionan.
- UTF-8 en todos los archivos de texto.

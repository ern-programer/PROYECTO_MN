# Plan HMR-SPECT: Ratio Corazón/Mediastino en SPECT 3D

## Contexto

El módulo actual de amiloidosis calcula HMR (Heart-to-Mediastinum Ratio) en imágenes planar usando ROIs circulares 2D. Este plan extiende el cálculo a SPECT 3D usando ROIs esféricas posicionadas con el sistema de localización ya implementado.

## Objetivo

Calcular HMR-SPECT = Cuentas ROI esférica corazón / Cuentas ROI esférica mediastino

## Fundamento

- **HMR planar**: ROIs circulares 2D sobre proyección AP
- **HMR-SPECT**: ROIs esféricas 3D sobre volumen reconstruido
- **Ventaja**: Mayor precisión anatómica, menos contaminación de estructuras adyacentes

## Diseño Técnico

### 1. Estructura de Datos

```python
@dataclass
class ROISphere:
    """ROI esférica 3D en coordenadas ZYX (índices de volumen)."""
    cz: float      # centro Z (axial)
    cy: float      # centro Y (coronal)
    cx: float      # centro X (sagittal)
    radius_mm: float
    
    def mask(self, shape: tuple[int, int, int], spacing_zyx: tuple[float, float, float]) -> np.ndarray:
        """Genera máscara 3D con radio en mm."""
        ...
    
    def radius_voxels(self, spacing_zyx: tuple[float, float, float]) -> tuple[float, float, float]:
        """Convierte radio mm a voxeles según spacing."""
        ...
```

### 2. Cálculo HMR-SPECT

```python
@dataclass
class HmrSpectResult:
    """Resultado del cálculo HMR en SPECT 3D."""
    hmr: float
    heart_counts: float
    mediastinum_counts: float
    heart_volume_ml: float
    mediastinum_volume_ml: float
    roi_heart: ROISphere
    roi_mediastinum: ROISphere
    classification: str  # POSITIVO/EQUIVOCO/NEGATIVO
```

### 3. Posicionamiento de VOIs

**Opción A - Manual con puntos de localización:**
- Usuario posiciona anchor A (centro corazón) y point B (referencia)
- VOI corazón: esfera centrada en anchor A
- VOI mediastino: esfera centrada en point B (o desplazamiento relativo)

**Opción B - Semi-automático:**
- Detección automática del centro cardíaco (ya existe en `core/lv_center.py`)
- Usuario ajusta posición si es necesario
- VOI mediastino: posición predefinida relativa al corazón

### 4. Métodos de Cálculo (seleccionables por usuario)

| Método | Descripción | Ventaja |
|--------|-------------|---------|
| **VOI esférica completa** | Integra cuentas de todos los voxeles dentro de la esfera 3D | Mayor precisión, incluye todo el volumen |
| **ROI slice central** | Usa solo el slice axial donde mejor se ve el corazón | Más simple, comparable con HMR planar |

**UI:**
```
Método HMR-SPECT: [VOI esférica completa ▼]
                  └─ VOI esférica completa (integra cuentas 3D)
                  └─ ROI slice central (slice único comparable a planar)
```

### 4. Integración con UI

- Panel AMYLO SPECT ya tiene sistema de localización (anchor A + point B)
- Agregar modo "HMR-SPECT" que use esos puntos para ROIs esféricas
- Mostrar resultado en vivo como HMR planar

### 5. Parámetros por Defecto

| Parámetro | Valor | Nota |
|-----------|-------|------|
| Radio ROI corazón | 30 mm | Ajustable |
| Radio ROI mediastino | 20 mm | Ajustable |
| Offset mediastino Z | +40 mm | Superior al corazón |
| Offset mediastino Y | 0 mm | Misma coronal |
| Offset mediastino X | 0 mm | Mismo sagittal |

## Cutoffs Diagnósticos (mismos que planar)

- HMR ≥ 1.5: POSITIVO (sugiere ATTR)
- HMR 1.0–1.5: EQUIVOCO
- HMR < 1.0: NEGATIVO

## Fases de Implementación

### Fase 1: Core (este sprint)
- [x] Definir ROISphere en amyloid_spect.py
- [ ] Implementar compute_hmr_spect()
- [ ] Tests unitarios

### Fase 2: UI
- [ ] Agregar modo HMR-SPECT en panel
- [ ] Controles de radio ROI
- [ ] Visualización de ROIs en MPR

### Fase 3: Informe
- [ ] Exportar HMR-SPECT a PDF
- [ ] Comparación con HMR planar si disponible

## Dependencias

- Sistema de localización existente (`_localization_anchor_zyx`, `_localization_point_zyx`)
- Spacing del volumen (`_spect_spacing_zyx`)
- MPR rendering para visualización

## Referencias

- Bokhari S, et al. ASNC practice points for 99mTc-PYP imaging. J Nucl Cardiol. 2021.
- EMORY Cardiac Toolbox documentation

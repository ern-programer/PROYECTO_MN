# Modelos anatómicos 3D del corazón

Coloque aquí los archivos de malla 3D (`.obj`, `.stl`, `.ply`, `.vtp`, `.vtk`)
para usarlos en el visor anatómico de SINCRO.

Si esta carpeta está vacía, SINCRO genera automáticamente un **corazón
procedural** simplificado (no anatómicamente exacto) para no bloquear la
visualización.

## Fuentes gratuitas / open source recomendadas

| Fuente | Licencia | URL | Atribución |
|--------|----------|-----|------------|
| **BodyParts3D** | CC BY-SA 2.1 JP | https://lifesciencedb.jp/bp3d/ | Obligatoria |
| **NIH 3D Print Exchange** | Dominio público | https://3dprint.nih.gov/ | Recomendada |
| **Open Anatomy Project** | MIT | https://www.openanatomy.org/ | Recomendada |

## Convención de nombres para atribución automática

SINCRO detecta la licencia por el nombre del archivo. Incluya una de estas
palabras clave en el nombre para que se muestre el crédito correcto:

- `bodyparts3d` → BodyParts3D (CC BY-SA)
- `nih` → NIH 3D Print Exchange (dominio público)
- `openanatomy` → Open Anatomy Project (MIT)

Ejemplo: `heart_bodyparts3d.obj`

## Notas

- Los modelos deben estar centrados y en escala razonable (cm).
- Las mallas se triangulan automáticamente al cargarse.
- Formatos con textura embebida (OBJ+MTL) cargan solo la geometría.

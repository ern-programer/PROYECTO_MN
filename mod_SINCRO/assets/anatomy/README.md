# Modelos anatómicos 3D del corazón

Coloque aquí los archivos de malla 3D (`.obj`, `.stl`, `.ply`, `.vtp`, `.vtk`)
para usarlos en el visor anatómico de SINCRO.

Si esta carpeta está vacía, SINCRO genera automáticamente un **corazón
procedural** simplificado (no anatómicamente exacto) para no bloquear la
visualización.

## Fuentes gratuitas / open source recomendadas

| Fuente | Licencia | URL | Atribución |
| ------ | -------- | --- | ---------- |
| **BodyParts3D** | CC BY-SA 2.1 JP | <https://lifesciencedb.jp/bp3d/> | Obligatoria |
| **NIH 3D Print Exchange** | Dominio público | <https://3dprint.nih.gov/> | Recomendada |
| **Open Anatomy Project** | MIT | <https://www.openanatomy.org/> | Recomendada |
| **TotalSegmentator (subject s1397)** | CC BY 4.0 | <https://zenodo.org/records/10047292> | Obligatoria |

## Convención de nombres para atribución automática

SINCRO detecta la licencia por el nombre del archivo. Incluya una de estas
palabras clave en el nombre para que se muestre el crédito correcto:

- `bodyparts3d` → BodyParts3D (CC BY-SA)
- `nih` → NIH 3D Print Exchange (dominio público)
- `openanatomy` → Open Anatomy Project (MIT)
- `totalsegmentator` o `ccby4` → TotalSegmentator (CC BY 4.0)

Ejemplo: `heart_bodyparts3d.obj`

## Malla incluida en este repo

- `heart_totalsegmentator_male_ccby4.vtp`
  - Origen: TotalSegmentator v2.0.1 (subject `s1397`) vía dataset anatómico usado por PyVista.
  - Licencia: **CC BY 4.0**.
  - Referencia: Wasserthal J. et al., *TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images*, doi:10.1148/ryai.230024.

## Mallas adicionales incluidas

- `heart_totalsegmentator_female_ccby4.vtp`
- `aorta_totalsegmentator_male_ccby4.vtp`
- `atrial_appendage_left_totalsegmentator_male_ccby4.vtp`
- `pulmonary_vein_totalsegmentator_male_ccby4.vtp`
- `superior_vena_cava_totalsegmentator_male_ccby4.vtp`
- `inferior_vena_cava_totalsegmentator_male_ccby4.vtp`

Todas estas mallas derivan de TotalSegmentator y están bajo **CC BY 4.0**.

## Notas

- Los modelos deben estar centrados y en escala razonable (cm).
- Las mallas se triangulan automáticamente al cargarse.
- Formatos con textura embebida (OBJ+MTL) cargan solo la geometría.

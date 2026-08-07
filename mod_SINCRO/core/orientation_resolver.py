"""Resolvedor de orientacion anatomica multi-camara para vistas de referencia.

Objetivo: dado un estudio SPECT crudo (de cualquier camara), decidir de forma
automatica como presentar las dos vistas de referencia que se usan para fijar
los limites base/apex del cilindro y para reorientar:
  - **anterior (AP)**  a la izquierda,
  - **lateral izquierda (LL)** a la derecha,
ambas con la cabeza arriba y la lateralidad L/R correcta, sea la camara que sea.

Fuentes de verdad (por prioridad):
  1. **DetectorInformationSequence[0].ImageOrientationPatient** (0054,0022 -> 0020,0037):
     6 cosenos director (row, col) en coords paciente LPS (+X=izq, +Y=post, +Z=cabeza).
     El vector de columna 'col' (direccion de indice de fila creciente en la
     imagen = eje vertical del detector) mapea al eje Z del volumen reconstruido
     por SINCRO (volumen (H,W,W): z = filas del detector). De ahi sale, sin
     ambiguedad, si el eje Z del volumen crece hacia la cabeza o hacia los pies
     -> flip vertical para dejar la cabeza arriba.
  2. **PatientPosition** (0018,5100): HFS/FFS/HFP/FFP -> confirmacion cabeza/pies
     y supino/prono. Secundaria (a veces ausente).
  3. **Perfil por (fabricante, modelo, tipo de orbita)**: offsets angulares
     anterior/lateral y espejos L/R. Necesario cuando la camara NO trae IOP ni
     PatientPosition (p.ej. GVI OnePass, sentada). Se calibra visualmente.

NOTA de ingenieria: derivar los angulos anterior/lateral por trigonometria pura
desde el IOP es AMBIGUO en el signo (la convencion del angulo de gantry varia por
fabricante y no siempre coincide con el sentido espacial). Por eso los angulos se
derivan de un PERFIL calibrable (offsets desde StartAngle segun el sentido de giro),
igual que el sistema de presets de reorientacion ya existente, y el IOP se usa solo
para lo que es inequivoco (cabeza/pies). Los espejos L/R quedan como default por
perfil y siguen siendo corregibles a mano (rotacion manual / presets).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Ejes del paciente en DICOM LPS.
_LEFT = np.array([1.0, 0.0, 0.0])       # +X
_POSTERIOR = np.array([0.0, 1.0, 0.0])  # +Y
_HEAD = np.array([0.0, 0.0, 1.0])       # +Z

# Offsets estandar de la orbita cardiaca dedicada de 180 (RAO45 -> LPO45):
#   anterior = start + sign*45 ; lateral izq = start + sign*135.
_CARDIAC_ANTERIOR_OFFSET = 45.0
_CARDIAC_LATERAL_OFFSET = 135.0


@dataclass(frozen=True)
class CameraProfile:
    """Perfil de orientacion por familia de camara (calibrable a ojo)."""

    key: str
    manufacturer_startswith: str = ""       # match por prefijo (normalizado)
    model_contains: str = ""                # match adicional opcional
    anterior_offset_deg: float = _CARDIAC_ANTERIOR_OFFSET
    lateral_offset_deg: float = _CARDIAC_LATERAL_OFFSET
    # Espejos de display (default; corregibles a mano):
    mirror_ap_lr: bool = False              # espejar horizontal la vista anterior
    mirror_ll_lr: bool = False              # espejar horizontal la lateral izq
    # flip_z: None => decidir por IOP/PatientPosition; True/False => forzar.
    force_flip_z: bool | None = None
    calibrated: bool = False                # True cuando se valido a ojo
    note: str = ""


# Tabla de perfiles. El primero que matchea gana; el ultimo es el fallback.
# Los offsets estandar (45/135) sirven para la orbita cardiaca dedicada de 180
# de GE/Marconi/Millennium/Picker (confirmado por la auditoria de start/dir/arco).
# GVI OnePass (sentada, orbita ~254, sin IOP ni PatientPosition) queda PENDIENTE
# de calibracion visual.
_PROFILES: tuple[CameraProfile, ...] = (
    CameraProfile(
        key="gvi_onepass", manufacturer_startswith="gvi",
        anterior_offset_deg=_CARDIAC_ANTERIOR_OFFSET, lateral_offset_deg=_CARDIAC_LATERAL_OFFSET,
        calibrated=False,
        note="GVI OnePass (sentada, sin IOP): offsets provisionales, PENDIENTE calibracion visual.",
    ),
    CameraProfile(
        key="ge_cardiac", manufacturer_startswith="ge",
        mirror_ap_lr=True,
        note="GE (Infinia/Millennium/Starcam/Ventri/Xeleris): orbita cardiaca 180 estandar; AP espejado L/R (validado a ojo con estudios Xeleris).",
        calibrated=True,
    ),
    CameraProfile(
        key="marconi_picker", manufacturer_startswith="marconi",
        mirror_ap_lr=True,
        note="Marconi/Picker axis: orbita cardiaca 180 estandar; AP espejado L/R (validado a ojo).",
        calibrated=True,
    ),
    CameraProfile(
        key="generic", manufacturer_startswith="",
        note="Fallback generico: orbita cardiaca 180 estandar.",
        calibrated=False,
    ),
)


@dataclass
class OrientationResult:
    """Decision de orientacion para las vistas de referencia."""

    anterior_angle_deg: float | None
    left_lateral_angle_deg: float | None
    flip_z: bool                 # voltear eje vertical para cabeza arriba
    mirror_ap_lr: bool           # espejar horizontal vista anterior
    mirror_ll_lr: bool           # espejar horizontal vista lateral izq
    profile_key: str
    source: str                  # 'iop', 'patient_position', 'profile'
    calibrated: bool
    notes: list[str] = field(default_factory=list)


def _norm_txt(text: str) -> str:
    return "".join(ch for ch in str(text or "").strip().lower() if ch.isalnum())


def parse_iop(iop) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Devuelve (row, col, normal) unitarios desde un ImageOrientationPatient (6 floats)."""
    if iop is None:
        return None
    try:
        vals = [float(v) for v in iop]
    except (TypeError, ValueError):
        return None
    if len(vals) < 6:
        return None
    row = np.array(vals[0:3], dtype=np.float64)
    col = np.array(vals[3:6], dtype=np.float64)
    rn = np.linalg.norm(row)
    cn = np.linalg.norm(col)
    if rn <= 1e-6 or cn <= 1e-6:
        return None
    row = row / rn
    col = col / cn
    normal = np.cross(row, col)
    nn = np.linalg.norm(normal)
    if nn > 1e-6:
        normal = normal / nn
    return row, col, normal


def _flip_z_from_iop(col: np.ndarray) -> bool | None:
    """Decide el flip vertical (cabeza arriba) desde el vector columna del IOP.

    El volumen reconstruido tiene z = filas del detector = direccion 'col'. Si
    'col' apunta hacia la cabeza (+Z), el indice z crece hacia la cabeza, por lo
    que la fila 0 (arriba en imshow) queda hacia los pies -> hay que voltear.
    Si 'col' apunta hacia los pies (-Z), z=0 ya es la cabeza -> no voltear.
    """
    cz = float(np.dot(col, _HEAD))
    if abs(cz) < 0.2:
        return None  # detector casi horizontal: indefinido por IOP
    return cz > 0.0


def _flip_z_from_patient_position(patient_position: str) -> bool | None:
    """Fallback de cabeza-arriba por PatientPosition (HF*/FF*)."""
    pp = str(patient_position or "").strip().upper()
    if pp.startswith("HF"):
        return False   # head-first: el corte 0 suele quedar hacia la cabeza
    if pp.startswith("FF"):
        return True    # feet-first: voltear para dejar la cabeza arriba
    return None


def _rotation_sign(rotation_direction: str) -> float:
    """+1 si el angulo crece con el indice (CC/CCW), -1 si decrece (CW)."""
    rd = str(rotation_direction or "").strip().upper()
    return -1.0 if rd.startswith("CW") else 1.0


def select_profile(manufacturer: str = "", model: str = "") -> CameraProfile:
    man = _norm_txt(manufacturer)
    mod = _norm_txt(model)
    for prof in _PROFILES:
        ms = _norm_txt(prof.manufacturer_startswith)
        if ms and not man.startswith(ms):
            continue
        mc = _norm_txt(prof.model_contains)
        if mc and mc not in mod:
            continue
        return prof
    return _PROFILES[-1]


def resolve_orientation(
    *,
    manufacturer: str = "",
    model: str = "",
    patient_position: str = "",
    start_angle: float | None = None,
    rotation_direction: str = "",
    scan_arc: float | None = None,
    detector_iop=None,
) -> OrientationResult:
    """Resuelve la orientacion de las vistas de referencia AP / lateral izq.

    Combina el perfil de camara (offsets angulares + espejos default) con el IOP
    del detector (cabeza/pies inequivoco) y PatientPosition (confirmacion).
    """
    notes: list[str] = []
    prof = select_profile(manufacturer, model)

    # Angulos anterior / lateral izquierda desde StartAngle segun sentido de giro.
    ant_angle = lat_angle = None
    if start_angle is not None:
        sign = _rotation_sign(rotation_direction)
        ant_angle = (float(start_angle) + sign * prof.anterior_offset_deg) % 360.0
        lat_angle = (float(start_angle) + sign * prof.lateral_offset_deg) % 360.0
        notes.append(
            f"Angulos por perfil '{prof.key}': anterior={ant_angle:.1f}, lateral={lat_angle:.1f} "
            f"(start={float(start_angle):.1f}, dir={rotation_direction or '?'}, "
            f"offsets={prof.anterior_offset_deg:.0f}/{prof.lateral_offset_deg:.0f})."
        )
    else:
        notes.append(f"Sin StartAngle: perfil '{prof.key}' sin angulos; se usara fallback ortogonal.")

    # Flip vertical (cabeza arriba): perfil forzado > IOP > PatientPosition.
    flip_z = None
    source = "profile"
    if prof.force_flip_z is not None:
        flip_z = bool(prof.force_flip_z)
        notes.append(f"flip_z forzado por perfil = {flip_z}.")
    else:
        parsed = parse_iop(detector_iop)
        if parsed is not None:
            _row, col, _normal = parsed
            fz = _flip_z_from_iop(col)
            if fz is not None:
                flip_z = fz
                source = "iop"
                notes.append(f"flip_z por IOP (col={np.round(col, 2).tolist()}) = {flip_z}.")
        if flip_z is None:
            fz = _flip_z_from_patient_position(patient_position)
            if fz is not None:
                flip_z = fz
                source = "patient_position"
                notes.append(f"flip_z por PatientPosition '{patient_position}' = {flip_z}.")
    if flip_z is None:
        flip_z = False
        notes.append("flip_z indefinido por metadatos: se asume False (corregible a mano).")

    if not prof.calibrated:
        notes.append(f"Perfil '{prof.key}' NO calibrado a ojo: verificar AP/lateral y L/R visualmente.")

    return OrientationResult(
        anterior_angle_deg=ant_angle,
        left_lateral_angle_deg=lat_angle,
        flip_z=bool(flip_z),
        mirror_ap_lr=bool(prof.mirror_ap_lr),
        mirror_ll_lr=bool(prof.mirror_ll_lr),
        profile_key=prof.key,
        source=source,
        calibrated=bool(prof.calibrated),
        notes=notes,
    )


def read_detector_iop(ds):
    """Lee DetectorInformationSequence[0].ImageOrientationPatient (0054,0022 -> 0020,0037).

    Devuelve la lista de 6 cosenos o None. NO usa el IOP de nivel superior
    (0020,0037), que en NM SPECT suele estar ausente o pertenece al CT.
    """
    try:
        dis = ds[(0x0054, 0x0022)].value if (0x0054, 0x0022) in ds else None
    except Exception:
        dis = None
    if not dis:
        return None
    try:
        item = dis[0]
        if (0x0020, 0x0037) in item:
            return list(item[(0x0020, 0x0037)].value)
    except Exception:
        return None
    return None

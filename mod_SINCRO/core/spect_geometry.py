"""Geometría de adquisición SPECT y vistas anatómicas de referencia.

Este módulo traduce la metadata de adquisición DICOM (posición del paciente,
ángulo inicial, sentido de giro, arco de barrido) en las **dos vistas de
referencia** que Xeleris/Odyssey usan para reorientar el corazón antes de
generar los cortes: la vista **anterior (AP)** y la **lateral izquierda**.

Fundamento (manuales LX/Xeleris + geometría estándar del corazón izquierdo):
- La órbita cardíaca estándar de 180° recorre RAO 45° → LPO 45°, pasando por
  ANTERIOR (0°) → LAO 45° → LATERAL IZQUIERDA (LAO 90°) → LPO 45°.
- Por lo tanto, medido desde el StartAngle y avanzando en el sentido de giro:
    * ANTERIOR      = start + 45°
    * LATERAL IZQ.  = start + 135°
  El signo del avance lo fija RotationDirection: CW resta, CC/CCW suma.
- Como el volumen ya está reconstruido, cada vista se obtiene **reproyectando**
  el volumen al ángulo de detector correspondiente (Radon a un solo ángulo),
  garantizando que la vista coincide con la proyección física a ese ángulo.

Nota clínica: para un arco < 135° la lateral izquierda no fue adquirida
directamente; la reproyección la **sintetiza** desde el volumen reconstruido
(menor calidad, pero anatómicamente consistente para posicionar markers).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.raw_reconstruction import _forward_project_slice

# Offsets angulares (grados) desde el StartAngle, medidos AVANZANDO en el
# sentido de giro, para la órbita estándar del corazón izquierdo.
_ANTERIOR_OFFSET_DEG = 45.0
_LEFT_LATERAL_OFFSET_DEG = 135.0


@dataclass(frozen=True)
class SpectGeometry:
    """Geometría de adquisición extraída del bruto DICOM."""

    patient_position: str = ""       # HFS/FFS/HFP/FFP
    start_angle: float | None = None       # grados
    angular_step: float | None = None       # grados
    rotation_direction: str = ""     # CW / CC
    scan_arc: float | None = None          # grados
    n_angles: int = 0

    # --- Derivados de la posición del paciente ---
    @property
    def head_first(self) -> bool:
        return self.patient_position.upper().startswith("HF")

    @property
    def supine(self) -> bool:
        # HFS/FFS = supino; HFP/FFP = prono.
        return self.patient_position.upper().endswith("S")

    @property
    def rotation_sign(self) -> float:
        """+1 si el ángulo crece con el índice de proyección, -1 si decrece."""
        return -1.0 if self.rotation_direction.upper().startswith("CW") else 1.0

    def anterior_angle(self) -> float | None:
        """Ángulo de detector (grados) de la vista anterior (AP)."""
        if self.start_angle is None:
            return None
        return (self.start_angle + self.rotation_sign * _ANTERIOR_OFFSET_DEG) % 360.0

    def left_lateral_angle(self) -> float | None:
        """Ángulo de detector (grados) de la vista lateral izquierda."""
        if self.start_angle is None:
            return None
        return (self.start_angle + self.rotation_sign * _LEFT_LATERAL_OFFSET_DEG) % 360.0

    @classmethod
    def from_raw_projections(cls, raw) -> "SpectGeometry":
        """Construye la geometría desde un RawGatedProjections."""
        return cls(
            patient_position=str(getattr(raw, "patient_position", "") or ""),
            start_angle=getattr(raw, "start_angle", None),
            angular_step=getattr(raw, "angular_step", None),
            rotation_direction=str(getattr(raw, "rotation_direction", "") or ""),
            scan_arc=getattr(raw, "scan_arc", None),
            n_angles=int(getattr(raw, "n_angles", 0) or 0),
        )


def reproject_view(volume: np.ndarray, angle_deg: float, *, detector_size: int | None = None) -> np.ndarray:
    """Reproyecta un volumen transaxial (z,y,x) a un ángulo de detector.

    Devuelve una vista planar (filas = eje z craneocaudal, columnas = detector),
    equivalente a la proyección física adquirida a ese ángulo de gantry.
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"volume debe ser 3D (z,y,x); recibió {vol.shape}")
    nz, _, w = vol.shape
    det = int(detector_size or w)
    theta = np.asarray([float(angle_deg)], dtype=np.float64)
    view = np.zeros((nz, det), dtype=np.float64)
    for z in range(nz):
        view[z] = _forward_project_slice(vol[z], theta, detector_size=det)[:, 0]
    return view


def _orient_for_display(view: np.ndarray, geometry: SpectGeometry, *, mirror_lr: bool) -> np.ndarray:
    """Orienta una vista planar para display: superior arriba y L/R anatómico.

    - Eje vertical (z): se coloca la cabeza arriba. La convención de índice de
      corte depende de head-first vs feet-first, por eso se voltea en feet-first.
    - Eje horizontal (detector): `mirror_lr` corrige la lateralidad según el
      sentido de giro/posición para que coincida con la convención radiológica.
    """
    out = np.asarray(view, dtype=np.float64)
    # Superior arriba: en feet-first el corte 0 suele quedar hacia la cabeza.
    if not geometry.head_first:
        out = out[::-1, :]
    if mirror_lr:
        out = out[:, ::-1]
    return out


def reference_views(volume: np.ndarray, geometry: SpectGeometry) -> dict:
    """Genera las vistas de referencia anterior (AP) y lateral izquierda.

    Devuelve un dict con:
      - 'anterior': ndarray 2D orientada para display
      - 'left_lateral': ndarray 2D orientada para display
      - 'anterior_angle', 'left_lateral_angle': ángulos de detector usados
      - 'synthesized_lateral': bool, True si la lateral cae fuera del arco
      - 'notes': list[str]
    Si no hay geometría angular, cae a proyecciones ortogonales del volumen
    (coronal para anterior, sagital para lateral) como aproximación.
    """
    vol = np.asarray(volume, dtype=np.float64)
    notes: list[str] = []
    ant_angle = geometry.anterior_angle()
    lat_angle = geometry.left_lateral_angle()

    if ant_angle is None or lat_angle is None:
        # Fallback sin metadata angular: proyecciones ortogonales (MIP-sum).
        anterior = vol.sum(axis=1)          # colapsa eje y -> (z, x) ~ coronal
        left_lateral = vol.sum(axis=2)      # colapsa eje x -> (z, y) ~ sagital
        notes.append("Sin geometría angular: vistas anterior/lateral aproximadas por proyección ortogonal.")
        anterior = _orient_for_display(anterior, geometry, mirror_lr=False)
        left_lateral = _orient_for_display(left_lateral, geometry, mirror_lr=False)
        return {
            "anterior": anterior,
            "left_lateral": left_lateral,
            "anterior_angle": None,
            "left_lateral_angle": None,
            "synthesized_lateral": True,
            "notes": notes,
        }

    anterior = reproject_view(vol, ant_angle)
    left_lateral = reproject_view(vol, lat_angle)

    # La lateral izquierda requiere avanzar 135° desde el start; si el arco es
    # menor, la vista se sintetiza desde el volumen (calidad reducida).
    synthesized = bool(geometry.scan_arc is not None and geometry.scan_arc < _LEFT_LATERAL_OFFSET_DEG)
    if synthesized:
        notes.append(
            f"Lateral izquierda sintetizada: arco {geometry.scan_arc:.0f}° < 135° "
            "(la proyección lateral no fue adquirida directamente)."
        )

    # En supino la vista anterior mira al paciente de frente; se refleja L/R
    # para presentar el lado izquierdo del paciente a la derecha de la imagen.
    anterior = _orient_for_display(anterior, geometry, mirror_lr=geometry.supine)
    left_lateral = _orient_for_display(left_lateral, geometry, mirror_lr=False)

    notes.append(
        f"Vistas por reproyección: anterior @ {ant_angle:.1f}°, lateral izq @ {lat_angle:.1f}° "
        f"(start {geometry.start_angle:.1f}°, dir {geometry.rotation_direction or '?'})."
    )
    return {
        "anterior": anterior,
        "left_lateral": left_lateral,
        "anterior_angle": ant_angle,
        "left_lateral_angle": lat_angle,
        "synthesized_lateral": synthesized,
        "notes": notes,
    }

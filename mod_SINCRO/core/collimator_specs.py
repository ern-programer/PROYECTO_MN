"""Base de colimadores y lectura de geometría de adquisición SPECT.

Este módulo es el habilitador *multi-fabricante* del motor de recuperación de
resolución (RR). El DICOM estándar entrega el **nombre** del colimador
(0018,1180), su **tipo** (0018,1181), el **pixel spacing** (0028,0030) y el
**radio de órbita** (0018,1142) para cualquier fabricante — pero NO entrega la
geometría física del colimador (diámetro de hueco, largo, septa). Esa física se
resuelve acá con una **tabla editable** indexada por (fabricante, nombre).

Modelo físico de resolución de un colimador de huecos paralelos (Anger):
    R_geom(b) = d · (L_eff + b) / L_eff        (resolución geométrica)
    R_sys(b)  = sqrt( R_geom(b)² + R_int² )     (resolución del sistema)
donde:
    d      = diámetro efectivo del hueco [mm]
    L_eff  = largo efectivo del hueco [mm]  (≈ L - 2/µ; acá se usa L)
    b      = distancia fuente→cara del colimador [mm]
    R_int  = resolución intrínseca del detector [mm FWHM]

La distancia b se obtiene del DICOM: b ≈ radio_de_órbita ± offset del vóxel.

IMPORTANTE: los valores de la tabla son representativos de cada familia de
colimador (datasheets públicos). Para uso clínico cuantitativo, reemplazar por
las specs exactas del fabricante/modelo instalado.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# FWHM = 2·sqrt(2·ln2)·sigma
FWHM_TO_SIGMA = 1.0 / 2.354820045


@dataclass(frozen=True)
class CollimatorSpec:
    """Especificación física de un colimador SPECT."""

    name: str
    manufacturer: str
    geometry: str = "parallel"          # 'parallel' | 'fanbeam'
    hole_diameter_mm: float = 1.5
    hole_length_mm: float = 35.0
    septal_mm: float = 0.2
    intrinsic_fwhm_mm: float = 3.8
    focal_length_mm: float | None = None       # solo fan-beam
    axial_magnification: float | None = None    # fan-beam: estiramiento en Y (None = desconocido)
    aliases: tuple[str, ...] = ()
    notes: str = ""

    def geometric_fwhm_mm(self, source_to_collimator_mm: float) -> float:
        """Resolución geométrica R_geom(b) en mm FWHM."""
        b = max(0.0, float(source_to_collimator_mm))
        return self.hole_diameter_mm * (self.hole_length_mm + b) / self.hole_length_mm

    def system_fwhm_mm(self, source_to_collimator_mm: float) -> float:
        """Resolución del sistema R_sys(b) = sqrt(R_geom² + R_int²) en mm FWHM."""
        rg = self.geometric_fwhm_mm(source_to_collimator_mm)
        return float((rg * rg + self.intrinsic_fwhm_mm ** 2) ** 0.5)

    def sigma_px(self, source_to_collimator_mm: float, pixel_mm: float) -> float:
        """Sigma de la PSF en píxeles a una distancia dada."""
        if pixel_mm <= 0:
            return 0.0
        return self.system_fwhm_mm(source_to_collimator_mm) * FWHM_TO_SIGMA / float(pixel_mm)


# =========================================================================
# Base de colimadores (representativa; editable)
# =========================================================================
_GENERIC_LEHR = CollimatorSpec(
    name="LEHR (genérico)", manufacturer="",
    hole_diameter_mm=1.5, hole_length_mm=35.0, septal_mm=0.2, intrinsic_fwhm_mm=4.0,
    notes="Fallback cuando el colimador no está en la tabla.",
)

_TABLE: tuple[CollimatorSpec, ...] = (
    # --- GE ---
    # hole_length = largo EFECTIVO (31 mm), no el geométrico (35 mm): calibrado a la
    # resolución de sistema publicada del GE LEHR, 7.4 mm FWHM @ 100 mm. Usar L=35
    # daba 6.9 mm (7% optimista) porque el modelo ignora la penetración septal.
    CollimatorSpec("LEHR", "GE", hole_diameter_mm=1.5, hole_length_mm=31.0, septal_mm=0.2,
                   intrinsic_fwhm_mm=3.8, aliases=("le_hr", "lehr_par", "gelehr"),
                   notes="GE LEHR baja energía alta resolución (Tc-99m). L_eff=31mm cal. a 7.4mm@10cm."),
    CollimatorSpec("LEGP", "GE", hole_diameter_mm=1.9, hole_length_mm=35.0, septal_mm=0.2,
                   intrinsic_fwhm_mm=3.8, aliases=("leap", "le_gp", "gap"),
                   notes="GE LEGP/LEAP baja energía propósito general."),
    CollimatorSpec("STARCAM-GP", "GE", hole_diameter_mm=1.9, hole_length_mm=32.0, septal_mm=0.2,
                   intrinsic_fwhm_mm=4.5, aliases=("99", "starcam", "star"),
                   notes="GE STARCAM (equipo antiguo); nombre de colimador '99' en DICOM."),
    # --- Siemens ---
    CollimatorSpec("LEHR", "SIEMENS", hole_diameter_mm=1.11, hole_length_mm=24.05, septal_mm=0.16,
                   intrinsic_fwhm_mm=3.8, aliases=("siemens_lehr", "ecam_lehr"),
                   notes="Siemens Symbia/E.CAM LEHR."),
    # --- Philips ---
    CollimatorSpec("VXGP", "PHILIPS", hole_diameter_mm=1.4, hole_length_mm=27.0, septal_mm=0.2,
                   intrinsic_fwhm_mm=3.6, aliases=("philips_lehr", "brightview_lehr", "vertex_lehr"),
                   notes="Philips BrightView/Vertex baja energía."),
    # --- GVI (cámara cardíaca dedicada, fan-beam multi-pinhole vertical) ---
    CollimatorSpec("NGSPECT", "GVI", geometry="fanbeam", hole_diameter_mm=1.5, hole_length_mm=35.0,
                   intrinsic_fwhm_mm=3.5, focal_length_mm=152.0, axial_magnification=None,
                   aliases=("ngspect", "onepass", "gvi_fanbeam"),
                   notes="GVI OnePass: colimador fan-beam de pinholes verticales; estira el "
                         "eje Y. La corrección axial requiere la magnificación real del datasheet."),
)


def _norm(text: str) -> str:
    return "".join(ch for ch in str(text or "").strip().lower() if ch.isalnum())


def lookup_collimator(manufacturer: str = "", name: str = "",
                      collimator_type: str = "") -> CollimatorSpec:
    """Devuelve la CollimatorSpec que mejor coincide con la metadata DICOM.

    Prioridad: (fabricante + nombre) > nombre/alias > fallback LEHR genérico.
    Si el tipo DICOM es fan-beam pero no hay match, marca la geometría fan-beam
    en el fallback para que el pipeline aplique la corrección axial.
    """
    man = _norm(manufacturer)
    nm = _norm(name)
    ctype = _norm(collimator_type)

    def matches_name(spec: CollimatorSpec) -> bool:
        keys = {_norm(spec.name), *(_norm(a) for a in spec.aliases)}
        return nm in keys if nm else False

    # 1) fabricante + nombre
    if nm:
        for spec in _TABLE:
            if matches_name(spec) and man and man.startswith(_norm(spec.manufacturer)):
                return spec
        # 2) solo nombre/alias
        for spec in _TABLE:
            if matches_name(spec):
                return spec
    # 3) solo fabricante fan-beam conocido
    if man:
        for spec in _TABLE:
            if _norm(spec.manufacturer) and man.startswith(_norm(spec.manufacturer)) and spec.geometry == "fanbeam":
                return spec

    # 4) fallback
    if ctype.startswith("fan"):
        return CollimatorSpec("FANBEAM (genérico)", manufacturer, geometry="fanbeam",
                              intrinsic_fwhm_mm=_GENERIC_LEHR.intrinsic_fwhm_mm,
                              notes="Fan-beam desconocido: corrección axial pendiente de datasheet.")
    return _GENERIC_LEHR


# =========================================================================
# Lectura de geometría de adquisición desde DICOM
# =========================================================================
@dataclass(frozen=True)
class AcquisitionGeometry:
    """Geometría física leída del DICOM para alimentar el RR."""

    manufacturer: str = ""
    model: str = ""
    collimator_name: str = ""
    collimator_type: str = ""       # PARA / FANB / CONE
    pixel_mm: float | None = None
    radius_mm: float | None = None       # RadialPosition (0018,1142): detector→centro
    focal_length_mm: float | None = None       # FocalDistance (0018,1182)
    notes: list[str] = field(default_factory=list)

    @property
    def is_fanbeam(self) -> bool:
        return str(self.collimator_type or "").upper().startswith("FAN")


def _first(seq, default=None):
    try:
        return seq[0]
    except Exception:  # noqa: BLE001
        return default


def read_acquisition_geometry(ds) -> AcquisitionGeometry:
    """Extrae la geometría física relevante de un dataset pydicom."""

    def g(tag, default=None):
        return ds[tag].value if tag in ds else default

    manufacturer = str(g(0x00080070, "") or "")
    model = str(g(0x00081090, "") or "")
    collimator_name = str(g(0x00181180, "") or "")
    collimator_type = str(g(0x00181181, "") or "")
    notes: list[str] = []

    pixel_mm = None
    ps = g(0x00280030, None)
    if ps is not None:
        try:
            pixel_mm = float(ps[0]) if hasattr(ps, "__len__") and not isinstance(ps, str) else float(ps)
        except Exception:  # noqa: BLE001
            pixel_mm = None

    radius_mm = None
    focal_length_mm = None
    # Detector Information Sequence: colimador + distancia focal por detector.
    det = g(0x00540022, None)
    d0 = _first(det) if det is not None else None
    if d0 is not None:
        if not collimator_name:
            collimator_name = str(d0[0x00181180].value if 0x00181180 in d0 else "" or "")
        if not collimator_type:
            collimator_type = str(d0[0x00181181].value if 0x00181181 in d0 else "" or "")
        if 0x00181182 in d0:
            try:
                focal_length_mm = float(d0[0x00181182].value) or None
            except Exception:  # noqa: BLE001
                focal_length_mm = None

    # Rotation Information Sequence: radio de órbita (constante o por vista).
    rot = g(0x00540052, None)
    r0 = _first(rot) if rot is not None else None
    if r0 is not None and 0x00181142 in r0:
        rp = r0[0x00181142].value
        try:
            vals = [float(v) for v in (rp if hasattr(rp, "__len__") and not isinstance(rp, str) else [rp])]
            vals = [v for v in vals if v > 0]
            if vals:
                radius_mm = float(sum(vals) / len(vals))
                if len(vals) > 1:
                    notes.append(f"Radio de órbita por vista (contorno): {min(vals):.0f}–{max(vals):.0f} mm, medio {radius_mm:.0f}.")
        except Exception:  # noqa: BLE001
            radius_mm = None

    return AcquisitionGeometry(
        manufacturer=manufacturer.strip(),
        model=model.strip(),
        collimator_name=collimator_name.strip(),
        collimator_type=collimator_type.strip(),
        pixel_mm=pixel_mm,
        radius_mm=radius_mm,
        focal_length_mm=focal_length_mm,
        notes=notes,
    )

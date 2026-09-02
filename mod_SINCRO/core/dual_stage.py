# -*- coding: utf-8 -*-
"""Modelo de datos para el flujo dual de perfusión (Esfuerzo / Reposo).

Plan C (rama PERFU_RyE): cada etapa es un objeto de primera clase en vez de un
conjunto de atributos sueltos tageados con un string de etapa. Este módulo es
PURO (sin PyQt, sin I/O): solo transporta el estado del pipeline por etapa y
expone el progreso del flujo. La orquestación real y la UI viven en main_window;
acá está el contrato de datos, testeable sin GUI.

Etapas canónicas: "stress" (esfuerzo) y "rest" (reposo).
El pipeline por etapa es: raw → motion → recon → reorient → cuts → analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


# Etapas soportadas. Diseñado para admitir más adelante una tercera (p.ej.
# "redistribution") sin cambiar el contrato: DualSession itera sobre las claves.
STAGE_STRESS = "stress"
STAGE_REST = "rest"
STAGES = (STAGE_STRESS, STAGE_REST)

# Etiquetas de presentación (la UI puede sobreescribir con lo detectado por DICOM).
STAGE_LABELS = {
    STAGE_STRESS: "Esfuerzo",
    STAGE_REST: "Reposo",
}

# Pasos del pipeline en orden. Cada uno tiene un atributo "testigo" en StageState
# cuya presencia marca el paso como cumplido (ver StageState.progress).
PIPELINE_STEPS = ("raw", "motion", "recon", "reorient", "cuts", "analysis")


def normalize_stage(stage: str | None) -> str:
    """Normaliza alias comunes a la etapa canónica. Default: stress."""
    s = str(stage or "").strip().lower()
    if s in ("rest", "reposo", "reposo basal", "basal", "resting"):
        return STAGE_REST
    return STAGE_STRESS


@dataclass
class StageState:
    """Estado del pipeline de UNA etapa (esfuerzo o reposo).

    Los campos reflejan los atributos que hoy viven sueltos en MainWindow como
    ``cine_crudo_*`` / ``compare_*``. Ninguno se toca acá: este objeto es el
    contenedor canónico al que esos atributos delegarán vía properties (Fase 1).
    """

    stage: str = STAGE_STRESS

    # --- crudo / preparación ---
    raw_study: Any = None                 # GatedStudy crudo (no reconstruido)
    source_path: str = ""
    motion_result: dict | None = None
    corrected_projections: Any = None     # proyecciones tras motion correction
    ref_index: int | None = None

    # --- reconstrucción ---
    recon_result: Any = None              # RawReconResult (nítido)
    recon_result_phase: Any = None        # pasajero FBP para fase (si NÍTIDA on)
    recon_study: Any = None
    raw_study_for_recon: Any = None       # estudio efectivamente reconstruido

    # --- reorientación (volúmenes reorientados a eje corto) ---
    reoriented_ungated: Any = None
    reoriented_gated: Any = None
    reoriented_phase: Any = None
    reoriented_mf: Any = None
    reoriented_ct: Any = None

    # --- cortes SA/HLA/VLA ---
    cut_study: Any = None                 # GatedStudy con cube SA (para process_current)
    cut_source_label: str = ""
    axes: dict = field(default_factory=dict)          # {"SA","HLA","VLA"} nítido/gated
    axes_ungated: dict = field(default_factory=dict)  # ídem ungated (perfusión)
    axes_mf: dict = field(default_factory=dict)       # ídem motion-frozen (si MF on)
    axes_ct: dict = field(default_factory=dict)       # ídem CT registrado (fusión)
    cut_thickness_mm: float = 0.0

    # --- análisis (fase / asincronía / FEVI) ---
    seg: Any = None
    phase: Any = None
    metrics: dict | None = None
    metrics_raw: dict | None = None
    phase_by_seg: Any = None
    territory: Any = None
    ef: dict | None = None

    # --- CT / atenuación (cada etapa con SU PROPIO CT; no se comparte) ---
    ct_path: str = ""
    mu_map_native: Any = None             # μ-map (cm^-1) en grilla nativa de la fuente
    mu_map_spacing_zyx: Any = None        # spacing (z,y,x) mm del μ-map nativo
    mu_map_source: str = ""               # "att_export" | "ct_bilineal"
    mu_map_description: str = ""
    mu_map_recon_grid: Any = None         # cache: μ-map remuestreado/registrado a grilla recon
    mu_map_shift_zyx: Any = None          # Δ del refinamiento NCC (voxeles z,y,x)
    mu_map_manual_shift_zyx: Any = None   # Δ manual del visor de fusión (prioridad sobre NCC)
    mu_map_flip_zyx: Any = None           # espejos (z,y,x) del CT fijados en el visor de fusión
    ct_volume_native: Any = None          # volumen de display (HU si vino CT; μ si vino ATTMAP)
    ct_affine_ijk_to_lps: Any = None      # affine DICOM del CT/ATT (clave para orientación consistente)
    ct_spacing_zyx: Any = None            # spacing propio del CT display (no confundir con el del μ-map)

    # --- metadatos de presentación ---
    label: str = ""       # etiqueta clínica ("Esfuerzo"/"Reposo"/personalizada)
    patient_id: str = ""

    def __post_init__(self):
        self.stage = normalize_stage(self.stage)
        if not self.label:
            self.label = STAGE_LABELS.get(self.stage, self.stage)

    # ---- estado del pipeline (barato, sin efectos) ----
    def has_raw(self) -> bool:
        return self.raw_study is not None

    def has_recon(self) -> bool:
        return self.recon_result is not None

    def has_reorient(self) -> bool:
        return self.reoriented_ungated is not None or self.reoriented_gated is not None

    def has_cuts(self) -> bool:
        return self.cut_study is not None or bool(self.axes)

    def has_analysis(self) -> bool:
        return self.metrics is not None

    def step_done(self, step: str) -> bool:
        s = str(step).lower()
        if s == "raw":
            return self.has_raw()
        if s == "motion":
            return self.motion_result is not None
        if s == "recon":
            return self.has_recon()
        if s == "reorient":
            return self.has_reorient()
        if s == "cuts":
            return self.has_cuts()
        if s == "analysis":
            return self.has_analysis()
        raise ValueError(f"Paso de pipeline desconocido: {step!r}")

    def progress(self) -> dict[str, bool]:
        """Mapa paso→cumplido, en orden de PIPELINE_STEPS."""
        return {step: self.step_done(step) for step in PIPELINE_STEPS}

    def last_completed_step(self) -> str | None:
        """Último paso contiguo cumplido desde el inicio (para el cockpit dual)."""
        last = None
        for step in PIPELINE_STEPS:
            if self.step_done(step):
                last = step
            else:
                break
        return last

    def is_complete(self) -> bool:
        """True si el pipeline llegó hasta análisis."""
        return self.has_analysis()

    def reset_from(self, step: str) -> None:
        """Invalida el paso indicado y todos los posteriores (cambió un input aguas arriba).

        No borra ``raw_study``/``source_path`` salvo que se pida "raw" (recarga
        del estudio). Mantener los datos base evita recargar DICOM sin necesidad.
        """
        s = str(step).lower()
        if s not in PIPELINE_STEPS:
            raise ValueError(f"Paso de pipeline desconocido: {step!r}")
        idx = PIPELINE_STEPS.index(s)
        order = PIPELINE_STEPS[idx:]
        if "motion" in order:
            self.motion_result = None
            self.corrected_projections = None
            self.ref_index = None
        if "recon" in order:
            self.recon_result = None
            self.recon_result_phase = None
            self.recon_study = None
            self.raw_study_for_recon = None
        if "reorient" in order:
            self.reoriented_ungated = None
            self.reoriented_gated = None
            self.reoriented_phase = None
            self.reoriented_mf = None
        if "cuts" in order:
            self.cut_study = None
            self.cut_source_label = ""
            self.axes = {}
            self.axes_ungated = {}
            self.axes_mf = {}
            self.cut_thickness_mm = 0.0
        if "analysis" in order:
            self.seg = None
            self.phase = None
            self.metrics = None
            self.metrics_raw = None
            self.phase_by_seg = None
            self.territory = None
            self.ef = None
        if "raw" in order:
            self.raw_study = None
            self.source_path = ""


@dataclass
class DualSession:
    """Sesión de perfusión con dos (o más) etapas y parámetros compartidos.

    ``active`` es la etapa que ven las herramientas de UI que operan sobre "la
    etapa actual" (compatibilidad con el selector Etapa existente). Los parámetros
    que deben ser idénticos entre etapas para no falsear la comparación (zoom de
    adquisición, VOI de reorientación) viven acá, no en cada StageState.
    """

    stages: dict[str, StageState] = field(default_factory=dict)
    active: str = STAGE_STRESS

    # Parámetros compartidos entre etapas (consistencia clínica).
    locked_voi: dict | None = None        # semiejes de la elipse VOI (zoom fijo entre etapas)
    locked_voi_stage: str | None = None   # etapa que fijó el VOI
    recon_config: Any = None              # config de recon aplicada a ambas
    cut_thickness_mm: float = 0.0         # espesor de corte compartido (0 = por etapa)

    # Resultado de la comparación stress-rest (delta de fase, etc.).
    comparison: dict | None = None

    def __post_init__(self):
        self.active = normalize_stage(self.active)
        # Garantizar que existan las etapas canónicas.
        for st in STAGES:
            if st not in self.stages:
                self.stages[st] = StageState(stage=st)

    def stage(self, which: str | None = None) -> StageState:
        """Devuelve el StageState de la etapa dada (o la activa)."""
        key = normalize_stage(which) if which is not None else self.active
        if key not in self.stages:
            self.stages[key] = StageState(stage=key)
        return self.stages[key]

    def active_stage(self) -> StageState:
        return self.stage(self.active)

    def other_stage(self, which: str | None = None) -> StageState:
        """La etapa complementaria a la dada (o a la activa)."""
        key = normalize_stage(which) if which is not None else self.active
        other = STAGE_REST if key == STAGE_STRESS else STAGE_STRESS
        return self.stage(other)

    def set_active(self, which: str) -> StageState:
        self.active = normalize_stage(which)
        return self.active_stage()

    def has_both(self) -> bool:
        """True si ambas etapas canónicas tienen al menos el crudo cargado."""
        return all(self.stage(st).has_raw() for st in STAGES)

    def both_complete(self) -> bool:
        return all(self.stage(st).is_complete() for st in STAGES)

    def loaded_stages(self) -> list[str]:
        return [st for st in STAGES if self.stage(st).has_raw()]

    def lock_voi(self, voi: dict, stage: str) -> None:
        """Fija el VOI de reorientación con la PRIMERA etapa; las demás lo heredan."""
        if self.locked_voi is None:
            self.locked_voi = dict(voi) if voi else None
            self.locked_voi_stage = normalize_stage(stage)

    def clear(self) -> None:
        """Reinicia la sesión a vacío (nuevo par de estudios)."""
        self.stages = {st: StageState(stage=st) for st in STAGES}
        self.active = STAGE_STRESS
        self.locked_voi = None
        self.locked_voi_stage = None
        self.recon_config = None
        self.cut_thickness_mm = 0.0
        self.comparison = None

    def snapshot_progress(self) -> dict[str, dict[str, bool]]:
        """Progreso de ambas etapas, para pintar el cockpit dual."""
        return {st: self.stage(st).progress() for st in STAGES}

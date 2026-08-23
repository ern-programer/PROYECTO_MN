# -*- coding: utf-8 -*-
"""Washout SPECT para amiloidosis cardíaca (PYP/DPD/HMDP).

Permite procesar dos estudios SPECT a diferentes tiempos (ej: 1h y 3h)
y calcular el washout cardíaco, un parámetro diagnóstico importante.

Referencia:
- Dorbala et al. ASNC/AHA/ASE/EANM/HFSA/ISA/SCMR/SNMMI Expert Consensus
  Recommendations for Multimodality Imaging in Cardiac Amyloidosis.
  J Card Fail. 2019.
"""

from __future__ import annotations

import enum
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.amyloid_spect import HmrSpectResult, VOISphere


@dataclass
class WashoutSpectResult:
    """Resultado del análisis de washout SPECT.
    
    El washout cardíaco en PYP/DPD/HMDP es un parámetro diagnóstico
    que ayuda a diferenciar ATTR-CM de otras causas de captación.
    
    Fórmula:
        Washout % = (1 - HMR_t2 / HMR_t1) × 100
        
    Interpretación (según literatura):
        - Washout < 0% (negativo): Captación aumenta con el tiempo
          → Sugiere ATTR-CM (amiloidosis por transtiretina)
        - Washout > 20%: Clearance rápido
          → Sugiere captación no específica o AL (cadena ligera)
        - Washout 0-20%: Zona gris
    """
    
    # HMRs individuales
    hmr_t1: HmrSpectResult | None = None  # Tiempo 1 (ej: 1h)
    hmr_t2: HmrSpectResult | None = None  # Tiempo 2 (ej: 3h)
    
    # Tiempos de adquisición (en horas desde inyección)
    time_t1_h: float = 1.0
    time_t2_h: float = 3.0
    
    # Washout calculado
    washout_pct: float | None = None  # Porcentaje de washout
    
    # Clasificación
    interpretation: str = ""  # "ATTR probable", "AL posible", "Indeterminado"
    
    # Metadatos
    patient_id: str = ""
    study_date: str = ""
    radiotracer: str = "Tc-99m PYP"  # PYP, DPD, HMDP
    
    @property
    def washout_text(self) -> str:
        """Texto formateado del washout."""
        if self.washout_pct is None:
            return "Washout: N/D"
        sign = "+" if self.washout_pct >= 0 else ""
        return f"Washout: {sign}{self.washout_pct:.1f}%"
    
    def calculate(self) -> None:
        """Calcula el washout a partir de los HMRs."""
        if self.hmr_t1 is None or self.hmr_t2 is None:
            self.washout_pct = None
            self.interpretation = "Faltan datos"
            return
        
        # Usar HMR raw (clínicamente relevante)
        hmr1 = self.hmr_t1.hmr_raw if self.hmr_t1.hmr_raw is not None else self.hmr_t1.hmr
        hmr2 = self.hmr_t2.hmr_raw if self.hmr_t2.hmr_raw is not None else self.hmr_t2.hmr
        
        # Fórmula de washout
        if hmr1 > 0:
            self.washout_pct = (1.0 - hmr2 / hmr1) * 100.0
        else:
            self.washout_pct = None
            self.interpretation = "HMR inválido"
            return
        
        # Interpretación según literatura
        # Nota: Esta es una guía general, el diagnóstico final
        # debe considerar el contexto clínico completo
        if self.washout_pct < 0:
            # Captación aumenta con el tiempo (negativo washout)
            # Típico de ATTR-CM donde el PYP se acumula en amiloide
            self.interpretation = "Washout negativo → ATTR-CM probable"
        elif self.washout_pct > 20:
            # Clearance rápido, típico de AL o captación no específica
            self.interpretation = "Washout elevado → AL posible o captación inespecífica"
        else:
            # Zona intermedia
            self.interpretation = "Washout intermedio → Indeterminado"
    
    @property
    def is_attr_pattern(self) -> bool:
        """True si el patrón sugiere ATTR-CM."""
        return self.washout_pct is not None and self.washout_pct < 0
    
    @property
    def is_al_pattern(self) -> bool:
        """True si el patrón sugiere AL (cadena ligera)."""
        return self.washout_pct is not None and self.washout_pct > 20


@dataclass
class DualSpectSession:
    """Sesión de análisis dual-SPECT para washout.
    
    Mantiene el estado de dos estudios SPECT cargados
    y permite comparación lado a lado.
    """
    
    # Volúmenes
    volume_t1: np.ndarray | None = None
    volume_t2: np.ndarray | None = None
    
    # Spacing
    spacing_t1: tuple[float, float, float] | None = None
    spacing_t2: tuple[float, float, float] | None = None
    
    # Paths DICOM
    path_t1: str = ""
    path_t2: str = ""
    
    # Metadatos temporales
    time_t1_h: float = 1.0  # Horas desde inyección
    time_t2_h: float = 3.0
    
    # Labels para UI
    label_t1: str = "1h"
    label_t2: str = "3h"
    
    # Resultados HMR
    hmr_t1: HmrSpectResult | None = None
    hmr_t2: HmrSpectResult | None = None
    
    # Resultado washout
    washout: WashoutSpectResult = field(default_factory=WashoutSpectResult)
    
    # Estado
    is_loaded_t1: bool = False
    is_loaded_t2: bool = False
    is_reconstructed_t1: bool = False
    is_reconstructed_t2: bool = False
    
    def calculate_washout(self) -> None:
        """Calcula el washout si ambos HMRs están disponibles."""
        self.washout.hmr_t1 = self.hmr_t1
        self.washout.hmr_t2 = self.hmr_t2
        self.washout.time_t1_h = self.time_t1_h
        self.washout.time_t2_h = self.time_t2_h
        self.washout.calculate()
    
    @property
    def is_complete(self) -> bool:
        """True si ambos estudios están cargados y procesados."""
        return (
            self.is_loaded_t1 and 
            self.is_loaded_t2 and 
            self.hmr_t1 is not None and 
            self.hmr_t2 is not None
        )
    
    @property
    def status_text(self) -> str:
        """Texto de estado para UI."""
        parts = []
        if self.is_loaded_t1:
            parts.append(f"T1 ({self.label_t1}): ✓ cargado")
            if self.hmr_t1:
                parts[-1] += f", HMR={self.hmr_t1.hmr:.2f}"
        else:
            parts.append(f"T1 ({self.label_t1}): ⏳ pendiente")
        
        if self.is_loaded_t2:
            parts.append(f"T2 ({self.label_t2}): ✓ cargado")
            if self.hmr_t2:
                parts[-1] += f", HMR={self.hmr_t2.hmr:.2f}"
        else:
            parts.append(f"T2 ({self.label_t2}): ⏳ pendiente")
        
        if self.is_complete:
            parts.append(f"Washout: {self.washout.washout_text}")
        
        return " | ".join(parts)


class WashoutInterpretation(enum.Enum):
    """Interpretación del washout SPECT."""
    ATTR_PROBABLE = "ATTR-CM probable"
    AL_POSIBLE = "AL posible"
    INDETERMINADO = "Indeterminado"
    PENDIENTE = "Pendiente de análisis"


def interpret_washout(washout_pct: float | None, hmr_t1: float | None) -> WashoutInterpretation:
    """Interpreta el washout en contexto clínico.
    
    Args:
        washout_pct: Porcentaje de washout (puede ser negativo)
        hmr_t1: HMR del primer estudio (para contexto)
        
    Returns:
        Interpretación clínica
    """
    if washout_pct is None:
        return WashoutInterpretation.PENDIENTE
    
    # Si HMR inicial es bajo, el washout es menos relevante
    if hmr_t1 is not None and hmr_t1 < 1.5:
        return WashoutInterpretation.INDETERMINADO
    
    # Patrón ATTR: washout negativo (captación aumenta)
    if washout_pct < 0:
        return WashoutInterpretation.ATTR_PROBABLE
    
    # Patrón AL: washout elevado (clearance rápido)
    if washout_pct > 20:
        return WashoutInterpretation.AL_POSIBLE
    
    return WashoutInterpretation.INDETERMINADO


def calculate_washout_from_hmr(hmr_t1: float, hmr_t2: float) -> float:
    """Calcula el washout a partir de dos HMRs.
    
    Args:
        hmr_t1: HMR del estudio temprano
        hmr_t2: HMR del estudio tardío
        
    Returns:
        Porcentaje de washout (puede ser negativo)
    """
    if hmr_t1 <= 0:
        raise ValueError("HMR t1 debe ser positivo")
    
    return (1.0 - hmr_t2 / hmr_t1) * 100.0


def format_washout_report(result: WashoutSpectResult) -> str:
    """Genera un texto de reporte formateado.
    
    Args:
        result: Resultado del washout
        
    Returns:
        Texto formateado para informe
    """
    lines = [
        "ANÁLISIS DE WASHOUT SPECT",
        "=" * 40,
        "",
        f"Radiotrazador: {result.radiotracer}",
        f"Tiempo 1: {result.time_t1_h:.1f}h → HMR = {result.hmr_t1.hmr:.2f}" if result.hmr_t1 else "Tiempo 1: N/D",
        f"Tiempo 2: {result.time_t2_h:.1f}h → HMR = {result.hmr_t2.hmr:.2f}" if result.hmr_t2 else "Tiempo 2: N/D",
        "",
        f"Washout: {result.washout_text}",
        "",
        f"Interpretación: {result.interpretation}",
        "",
        "Nota: El washout negativo (captación creciente) sugiere ATTR-CM.",
        "El washout elevado (>20%) sugiere AL o captación inespecífica.",
        "El diagnóstico final debe integrar contexto clínico completo.",
    ]
    
    return "\n".join(lines)

"""CompareManager - Gestión de comparación stress/rest."""
from __future__ import annotations

from typing import Any

import numpy as np


class CompareManager:
    """Gestiona comparación entre estudios (stress vs rest)."""

    def __init__(self):
        self.compare_bundle: dict | None = None
        self.compare_label: str | None = None
        self.compare_metrics: dict | None = None
        self.compare_ef: dict | None = None
        self.dual_mode_active = False

    def set_compare_bundle(self, bundle: dict, label: str | None = None):
        """Establece el bundle de comparación."""
        self.compare_bundle = bundle
        self.compare_label = label or bundle.get("label", "Comparación")
        self.compare_metrics = bundle.get("metrics")
        self.compare_ef = bundle.get("ef")
        self.dual_mode_active = True

    def clear_compare(self):
        """Limpia la comparación."""
        self.compare_bundle = None
        self.compare_label = None
        self.compare_metrics = None
        self.compare_ef = None
        self.dual_mode_active = False

    def get_compare_summary(self) -> dict[str, Any]:
        """Retorna resumen de la comparación."""
        if self.compare_bundle is None:
            return {"active": False}
        return {
            "active": True,
            "label": self.compare_label,
            "has_metrics": self.compare_metrics is not None,
            "has_ef": self.compare_ef is not None,
            "study_path": self.compare_bundle.get("path", ""),
        }

    def compute_delta_metrics(self, primary_metrics: dict) -> dict[str, Any]:
        """Calcula delta entre métricas primarias y comparación."""
        if self.compare_metrics is None:
            return {}
        delta = {}
        for key in ["phase_sd", "bandwidth", "entropy_normalized_pct"]:
            primary_val = float(primary_metrics.get(key, np.nan))
            compare_val = float(self.compare_metrics.get(key, np.nan))
            if np.isfinite(primary_val) and np.isfinite(compare_val):
                delta[f"delta_{key}"] = round(primary_val - compare_val, 2)
        return delta

    def is_stunning_suggestive(self, primary_metrics: dict) -> bool:
        """Detecta si hay sugestión de stunning isquémico."""
        delta = self.compute_delta_metrics(primary_metrics)
        d_psd = delta.get("delta_phase_sd", 0.0)
        d_bw = delta.get("delta_bandwidth", 0.0)
        return bool(d_psd > 3.0 and d_bw > 8.0)

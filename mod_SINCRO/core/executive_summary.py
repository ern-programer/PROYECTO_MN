"""Resumen ejecutivo determinista para el informe SINCRO.

Genera una síntesis en lenguaje natural del hallazgo del estudio (fase/sincronía,
territorios, función ventricular) a partir de los diccionarios de métricas que ya
calcula la app. No emite un veredicto de confiabilidad ni semáforo: solo redacta
el hallazgo.

El resultado es reutilizable tanto por el PDF (reportlab) como por el panel de la
app (texto plano). La estructura ``structured`` expone los valores crudos para una
eventual redacción asistida por IA en el futuro, sin depender hoy de ningún modelo
externo (generación 100% por reglas, reproducible y auditable).
"""

from __future__ import annotations

import math
from typing import Any


def _num(value: Any) -> float | None:
    """Convierte a float finito o devuelve None."""
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


def _fmt(value: Any, decimals: int = 1, unit: str = "") -> str:
    fv = _num(value)
    if fv is None:
        return "N/D"
    return f"{fv:.{decimals}f}{unit}"


def _most_delayed_territory(territory: dict | None) -> tuple[str | None, float | None]:
    """Devuelve (nombre, fase_media) del territorio con mayor fase media."""
    if not territory:
        return None, None
    best_name: str | None = None
    best_mean: float | None = None
    for name in ("LAD", "LCx", "RCA"):
        data = territory.get(name) or {}
        mean_v = _num(data.get("mean"))
        if mean_v is None:
            continue
        if best_mean is None or mean_v > best_mean:
            best_mean = mean_v
            best_name = name
    return best_name, best_mean


def build_executive_summary(
    *,
    metrics: dict,
    ef: dict | None = None,
    territory: dict | None = None,
    volumes: dict | None = None,
    phase_label: str = "Estudio",
    db_eval: dict | None = None,
) -> dict:
    """Construye el resumen ejecutivo (hallazgo en lenguaje natural).

    Parameters
    ----------
    metrics : dict
        Diccionario de métricas robustas (phase_sd, bandwidth, asynchrony_index,
        technical_classification, latest_activation_phase, ...). Puede incluir
        ``normal_db_eval`` como fallback si ``db_eval`` no se pasa.
    ef : dict, optional
        Resultado de FEVI (ef_pct, edv_ml, esv_ml, pfr_text, tvmax_text,
        thickening_pct, wall_thickness_ed_mm, wall_thickness_es_mm).
    territory : dict, optional
        Fase por territorio coronario (LAD/LCx/RCA con mean/std/n).
    volumes : dict, optional
        Volúmenes miocárdicos/cavidad en mL.
    phase_label : str
        Etiqueta de la fase procesada (Esfuerzo/Reposo/Estudio).
    db_eval : dict, optional
        Evaluación vs DB normal ({"dyssynchrony": bool, ...}). Si no se pasa se
        intenta ``metrics["normal_db_eval"]``.

    Returns
    -------
    dict
        {
          "available": bool,
          "phase_label": str,
          "sections": [ {"title": str, "text": str}, ... ],
          "plain_text": str,
          "structured": { ... valores crudos para IA futura ... },
        }
    """
    metrics = metrics or {}
    ef = ef or {}
    territory = territory or {}
    volumes = volumes or {}
    if db_eval is None:
        db_eval = metrics.get("normal_db_eval") or {}

    psd = _num(metrics.get("phase_sd"))
    bw = _num(metrics.get("bandwidth"))
    ai = _num(metrics.get("asynchrony_index"))
    entropy_norm = _num(metrics.get("entropy_normalized_pct"))
    latest = _num(metrics.get("latest_activation_phase"))
    tech_class = str(metrics.get("technical_classification", metrics.get("classification", "N/D")))

    sections: list[dict] = []

    # --- Fase y sincronía ------------------------------------------------
    frase_fase = (
        f"El análisis de fase del ventrículo izquierdo ({phase_label.lower()}) muestra una "
        f"dispersión estándar (PSD) de {_fmt(psd, 1, '°')} y un ancho de banda de {_fmt(bw, 1, '°')} "
        f"(clase técnica orientativa: {tech_class})."
    )
    if ai is not None:
        frase_fase += f" El índice de asincronía es {_fmt(ai, 1, '%')}."
    if entropy_norm is not None:
        frase_fase += f" La entropía normalizada de fase es {_fmt(entropy_norm, 1, '%')}."
    if latest is not None:
        frase_fase += f" La activación más tardía se registra hacia los {_fmt(latest, 0, '°')} del ciclo."
    if db_eval:
        lectura = "fuera de referencia" if db_eval.get("dyssynchrony") else "dentro de referencia"
        frase_fase += f" Frente a la base de datos normal, los parámetros se encuentran {lectura}."
    sections.append({"title": "Fase y sincronía", "text": frase_fase})

    # --- Territorios coronarios -----------------------------------------
    terr_name, terr_mean = _most_delayed_territory(territory)
    if terr_name is not None:
        sections.append({
            "title": "Distribución territorial",
            "text": (
                f"El territorio con mayor retraso relativo de fase es {terr_name} "
                f"(fase media {_fmt(terr_mean, 1, '°')}). Correlacionar con el mapa polar de fase."
            ),
        })

    # --- Función ventricular --------------------------------------------
    if ef.get("available"):
        ef_pct = _num(ef.get("ef_pct"))
        edv = _num(ef.get("edv_ml"))
        esv = _num(ef.get("esv_ml"))
        frase_func = f"FEVI preliminar {_fmt(ef_pct, 1, '%')}"
        if edv is not None and esv is not None:
            frase_func += f" (EDV {_fmt(edv, 0, ' mL')} / ESV {_fmt(esv, 0, ' mL')})"
        frase_func += "."
        if ef.get("pfr_text"):
            frase_func += f" PFR: {ef.get('pfr_text')}."
        if ef.get("tvmax_text"):
            frase_func += f" TVmáx: {ef.get('tvmax_text')}."
        thk = _num(ef.get("thickening_pct"))
        if thk is not None:
            frase_func += f" Engrosamiento sistólico global {_fmt(thk, 1, '%')}."
        sections.append({"title": "Función ventricular", "text": frase_func})

    # --- Cierre / recomendación de lectura ------------------------------
    sections.append({
        "title": "Lectura recomendada",
        "text": (
            "Síntesis orientativa, no diagnóstica. Integrar fase, perfusión, viabilidad, "
            "FEVI, QRS y contexto clínico antes de concluir. La FEVI es una estimación "
            "preliminar de investigación."
        ),
    })

    plain_lines: list[str] = []
    for sec in sections:
        plain_lines.append(sec["title"])
        plain_lines.append(f"  {sec['text']}")
        plain_lines.append("")
    plain_text = "\n".join(plain_lines).rstrip()

    structured = {
        "phase_label": phase_label,
        "phase_sd": psd,
        "bandwidth": bw,
        "asynchrony_index": ai,
        "entropy_normalized_pct": entropy_norm,
        "latest_activation_phase": latest,
        "technical_classification": tech_class,
        "db_dyssynchrony": bool(db_eval.get("dyssynchrony")) if db_eval else None,
        "most_delayed_territory": terr_name,
        "most_delayed_territory_phase": terr_mean,
        "ef_pct": _num(ef.get("ef_pct")) if ef else None,
        "edv_ml": _num(ef.get("edv_ml")) if ef else None,
        "esv_ml": _num(ef.get("esv_ml")) if ef else None,
        "thickening_pct": _num(ef.get("thickening_pct")) if ef else None,
        "myocardial_ml": _num(volumes.get("myocardial_ml")) if volumes else None,
    }

    return {
        "available": True,
        "phase_label": phase_label,
        "sections": sections,
        "plain_text": plain_text,
        "structured": structured,
    }

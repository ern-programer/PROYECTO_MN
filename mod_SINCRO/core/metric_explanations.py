"""
SINCRO - core.metric_explanations
==================================

Registro consultable de explicaciones de métricas: qué es, cómo se calcula, en
qué unidades, cutoffs de referencia y bibliografía. La idea es que TODA métrica
expuesta por el software tenga acá su explicación, y que la UI (tooltips, panel
de ayuda) y el informe puedan leer de una sola fuente en vez de duplicar texto.

Piloto: PFR y TVmáx (función diastólica). Se irán agregando el resto.
"""
from __future__ import annotations

from typing import Any

# Cada entrada: clave estable → dict con campos consultables.
#   title       : nombre humano de la métrica
#   short       : una línea (apto tooltip)
#   what        : qué mide, en 1-3 frases
#   how         : cómo se calcula en SINCRO
#   units       : convención de unidades
#   cutoffs     : rangos/umbrales de referencia (informativo, no diagnóstico)
#   reference   : bibliografía / origen del método
_EXPLANATIONS: dict[str, dict[str, str]] = {
    "pfr": {
        "title": "PFR — Peak Filling Rate (tasa pico de llenado)",
        "short": "Máxima velocidad de llenado ventricular en diástole; marcador de función diastólica.",
        "what": (
            "Es la máxima tasa a la que el ventrículo izquierdo se llena durante "
            "la diástole. Un PFR bajo sugiere disfunción diastólica (relajación "
            "alterada), incluso con FEVI conservada."
        ),
        "how": (
            "Se toma la curva de volumen por gate (un ciclo RR), se calcula su "
            "derivada dV/dgate y se busca el máximo positivo en la fase de llenado "
            "(desde fin de sístole hacia el siguiente fin de diástole). Ese máximo "
            "se normaliza por el VTD."
        ),
        "units": (
            "Doble convención ECTb: 'VTD/s' (fracción del VTD por segundo, requiere "
            "intervalo RR) y 'VTD/RR' (fracción del VTD por intervalo RR, geométrica). "
            "Se muestra como 'X.XX VTD/s [Y.YY VTD/RR]'."
        ),
        "cutoffs": (
            "Orientativo: PFR ≥ ~2.5 VTD/s se considera normal; valores menores "
            "apuntan a disfunción diastólica. Depende de FC, edad y protocolo; "
            "interpretar con la clínica."
        ),
        "reference": "Emory Cardiac Toolbox — curva tiempo-volumen gated SPECT (a_07/b_06).",
    },
    "tvmax": {
        "title": "TVmáx — Time to Peak Filling (tiempo al pico de llenado)",
        "short": "Tiempo desde fin de sístole hasta el pico de llenado; se alarga en disfunción diastólica.",
        "what": (
            "Es el tiempo que tarda el ventrículo, desde el fin de sístole, en "
            "alcanzar su máxima velocidad de llenado. Se prolonga cuando la "
            "relajación diastólica está alterada."
        ),
        "how": (
            "Sobre la misma curva de volumen por gate, es el tiempo entre el gate "
            "de fin de sístole y el gate donde ocurre el PFR."
        ),
        "units": (
            "'ms' (milisegundos, requiere intervalo RR) y '%RR' (porcentaje del "
            "intervalo RR, geométrico). Se muestra como 'X ms [Y %RR]'."
        ),
        "cutoffs": (
            "Orientativo: TVmáx ≤ ~180 ms se considera normal; tiempos mayores "
            "sugieren llenado retardado. Depende de FC; interpretar con la clínica."
        ),
        "reference": "Emory Cardiac Toolbox — curva tiempo-volumen gated SPECT (a_07/b_06).",
    },
    "phase_seg": {
        "title": "Fase por segmento AHA",
        "short": "Momento de contracción de cada segmento del VI (fase del 1er armónico).",
        "what": (
            "Es la fase media de contracción de cada uno de los 17 segmentos AHA. "
            "Segmentos con fase muy tardía respecto al resto indican activación "
            "retrasada (dato clave para localizar el sitio de estimulación en TRC)."
        ),
        "how": (
            "Media circular de la fase (1er armónico FFT del ciclo gated) de los "
            "voxeles de cada segmento AHA (17 segmentos)."
        ),
        "units": "Grados (0–360°). Se resalta el segmento de activación más tardía.",
        "cutoffs": (
            "No hay un cutoff único por segmento: se lee el patrón (dispersión y "
            "el segmento más tardío) junto a PSD/Bandwidth globales."
        ),
        "reference": "AHA 17-segment model; análisis de fase gated SPECT (Emory).",
    },
    "perfusion_pct": {
        "title": "Perfusión segmentaria (% del máximo)",
        "short": "Captación relativa de cada segmento respecto al más captante (=100%).",
        "what": (
            "Cuánta actividad capta cada segmento comparado con el segmento de "
            "mayor captación del estudio. Segmentos con % bajo son hipoperfundidos."
        ),
        "how": (
            "Captación media por segmento AHA normalizada al máximo segmentario "
            "(el segmento más captante = 100%)."
        ),
        "units": "% del máximo segmentario.",
        "cutoffs": (
            "Orientativo (parametrizable): ≥70% conservada, 50–70% hipoperfusión, "
            "<50% defecto severo. Interpretar con el protocolo y la clínica."
        ),
        "reference": "Mapas polares de perfusión SPECT (convención % del máximo).",
    },
    "viability": {
        "title": "Viabilidad por perfusión",
        "short": "Clasificación viable / dudosa / no viable derivada de la perfusión segmentaria.",
        "what": (
            "Estima si el miocardio del segmento es viable a partir de su nivel de "
            "perfusión relativa. Es una aproximación por captación en reposo, no "
            "reemplaza a viabilidad por PET FDG o realce tardío por RMN."
        ),
        "how": (
            "Se clasifica el % de perfusión segmentaria: ≥ umbral viable → 'viable'; "
            "entre los dos umbrales → 'dudosa'; por debajo → 'no viable'."
        ),
        "units": "Categórica: viable / dudosa / no viable.",
        "cutoffs": (
            "Defaults 70% (viable) y 50% (dudosa/no viable), ajustables. La "
            "perfusión de reposo ≥50% se asocia a viabilidad en MPI."
        ),
        "reference": "Umbrales de viabilidad por MPI de reposo (aproximación clínica).",
    },
    "tid": {
        "title": "TID — Dilatación Isquémica Transitoria (gatillada)",
        "short": "Cociente del tamaño del VI esfuerzo/reposo; elevado sugiere isquemia extensa.",
        "what": (
            "Compara el tamaño de la cavidad del ventrículo izquierdo entre "
            "esfuerzo y reposo. Cuando la cavidad se ve mayor en esfuerzo (TID "
            "elevado) puede reflejar isquemia extensa (enfermedad multivaso o de "
            "tronco) o dilatación subendocárdica difusa; es un marcador pronóstico "
            "clásico del SPECT de perfusión."
        ),
        "how": (
            "Versión GATILLADA: cociente del volumen de fin de diástole (EDV) del "
            "mismo método (ECTb) entre esfuerzo y reposo: TID = EDV_esfuerzo / "
            "EDV_reposo. Al ser un cociente del mismo método, cancela el sesgo del "
            "volumen absoluto. (El TID clásico se mide sobre perfusión ungated "
            "sumada; esa variante queda como desarrollo futuro.)"
        ),
        "units": "Cociente adimensional (esfuerzo/reposo).",
        "cutoffs": (
            "Orientativo (NO diagnóstico): gatillado ~≥1.20; el TID clásico ungated "
            "suele citarse ~1.22. Depende de protocolo, cámara y población: "
            "interpretar con la perfusión y la clínica."
        ),
        "reference": "TID en SPECT de perfusión (marcador pronóstico); variante gatillada por EDV ratio.",
    },
}


def get_explanation(key: str) -> dict[str, str] | None:
    """Devuelve la explicación de una métrica por su clave, o None si no existe."""
    return _EXPLANATIONS.get(str(key).strip().lower())


def explanation_short(key: str) -> str:
    """Línea corta apta para tooltip. Cadena vacía si no hay explicación."""
    exp = get_explanation(key)
    return exp.get("short", "") if exp else ""


def explanation_tooltip(key: str) -> str:
    """Texto multilínea para tooltip enriquecido (título + qué + cómo + unidades)."""
    exp = get_explanation(key)
    if not exp:
        return ""
    return (
        f"{exp.get('title', key)}\n\n"
        f"{exp.get('what', '')}\n\n"
        f"Cálculo: {exp.get('how', '')}\n"
        f"Unidades: {exp.get('units', '')}\n"
        f"Referencia: {exp.get('reference', '')}"
    )


def all_explanations() -> dict[str, dict[str, str]]:
    """Copia del registro completo (para paneles de ayuda o export al informe)."""
    return {k: dict(v) for k, v in _EXPLANATIONS.items()}

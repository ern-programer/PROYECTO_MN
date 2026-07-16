"""
SINCRO - core.normal_db
========================

Base de datos de valores normales de sincronía mecánica del VI para comparar las
métricas de fase de un paciente (PSD y Bandwidth/PHB) contra rangos publicados,
estratificados por SEXO y PROTOCOLO (stress/rest).

Fundamento (por qué estratificar):
- Mukherjee 2016 (Indian J Nucl Med 31(4):255-9): demostró que PSD y PHB difieren
  SIGNIFICATIVAMENTE por sexo (H>M) y protocolo (stress>rest). No existe un único
  valor normal universal.
- Chen 2005 (J Nucl Cardiol): normal database original Emory (población Western).
- Cutoff de disincronía = media + 2·SD del grupo normal correspondiente.

Los valores están guardados como (mean, sd). El z-score de un paciente es:
    z = (valor_paciente - mean) / sd
y se considera disincronía si z > 2 (equivale a > media + 2·SD).

Extensible: se puede cargar una base propia desde un JSON con la misma estructura
usando `load_custom_db(path)`. Así, cuando el usuario tenga sus propios controles
normales (por población/cámara), reemplaza o agrega datasets sin tocar el código.
"""
from __future__ import annotations

import json
import os
from typing import Any

# Estructura: dataset -> sexo -> protocolo -> métrica -> {"mean", "sd", "range"}
# métricas: "phase_sd" (PSD, grados), "bandwidth" (PHB, grados).
_PUBLISHED_DB: dict[str, Any] = {
    "Mukherjee2016_India": {
        "descripcion": "Población India, 120 pac (60H/60F), Tc-99m MIBI, SyncTool/Emory, GE Infinia, 8 frames.",
        "referencia": "Mukherjee A et al. Indian J Nucl Med 2016;31(4):255-9.",
        "male": {
            "stress": {
                "phase_sd": {"mean": 14.3, "sd": 4.7, "range": [9.2, 25.2]},
                "bandwidth": {"mean": 40.1, "sd": 11.9, "range": [23.0, 72.0]},
            },
            "rest": {
                "phase_sd": {"mean": 8.9, "sd": 2.9, "range": [3.6, 18.2]},
                "bandwidth": {"mean": 30.6, "sd": 7.6, "range": [20.0, 54.0]},
            },
        },
        "female": {
            "stress": {
                "phase_sd": {"mean": 11.0, "sd": 4.0, "range": [4.7, 20.8]},
                "bandwidth": {"mean": 34.7, "sd": 12.6, "range": None},
            },
            "rest": {
                "phase_sd": {"mean": 7.7, "sd": 2.7, "range": [3.3, 16.5]},
                "bandwidth": {"mean": 25.3, "sd": 8.6, "range": None},
            },
        },
    },
    "Chen2005_Western": {
        "descripcion": "Normal database original Emory (población Western), dual isótopo rest/stress.",
        "referencia": "Chen J et al. J Nucl Cardiol 2005;12(6):687-95. PMID 16344229.",
        # Chen 2005 no separa por sexo en la publicación original; se usa el mismo
        # valor para ambos (PSD ~14.2 hombres / distintos por estudio). Valores
        # ampliamente citados como referencia general.
        "male": {
            "stress": {
                "phase_sd": {"mean": 14.2, "sd": 5.1, "range": None},
                "bandwidth": {"mean": 38.7, "sd": 11.8, "range": None},
            },
            "rest": {
                "phase_sd": {"mean": 14.2, "sd": 5.1, "range": None},
                "bandwidth": {"mean": 38.7, "sd": 11.8, "range": None},
            },
        },
        "female": {
            "stress": {
                "phase_sd": {"mean": 11.8, "sd": 5.2, "range": None},
                "bandwidth": {"mean": 30.6, "sd": 10.3, "range": None},
            },
            "rest": {
                "phase_sd": {"mean": 11.8, "sd": 5.2, "range": None},
                "bandwidth": {"mean": 30.6, "sd": 10.3, "range": None},
            },
        },
    },
}

Z_CUTOFF = 2.0  # z > 2 = disincronía (media + 2 SD)


def available_datasets() -> list[str]:
    return list(_PUBLISHED_DB.keys())


def dataset_info(dataset: str) -> dict:
    d = _PUBLISHED_DB.get(dataset, {})
    return {"descripcion": d.get("descripcion", ""), "referencia": d.get("referencia", "")}


def load_custom_db(path: str) -> bool:
    """Carga/mergea una base propia desde JSON (misma estructura). Devuelve True si OK."""
    if not os.path.exists(path):
        return False
    with open(path, "rb") as fh:
        data = json.loads(fh.read().decode("utf-8"))
    if not isinstance(data, dict):
        return False
    _PUBLISHED_DB.update(data)
    return True


def _get_ref(dataset: str, sex: str, protocol: str, metric: str) -> dict | None:
    sex = (sex or "").strip().lower()
    protocol = (protocol or "").strip().lower()
    if sex in ("m", "male", "hombre", "masculino", "h"):
        sex = "male"
    elif sex in ("f", "female", "mujer", "femenino"):
        sex = "female"
    try:
        return _PUBLISHED_DB[dataset][sex][protocol][metric]
    except (KeyError, TypeError):
        return None


def z_score(value: float, dataset: str, sex: str, protocol: str, metric: str) -> float | None:
    ref = _get_ref(dataset, sex, protocol, metric)
    if not ref or not ref.get("sd"):
        return None
    return (float(value) - float(ref["mean"])) / float(ref["sd"])


def evaluate(
    phase_sd: float,
    bandwidth: float,
    dataset: str = "Mukherjee2016_India",
    sex: str = "male",
    protocol: str = "stress",
) -> dict:
    """Evalúa PSD y BW del paciente contra la DB normal. Devuelve por métrica el
    valor, referencia (mean±sd), cutoff (mean+2sd), z-score y flag de disincronía.
    """
    out: dict[str, Any] = {
        "dataset": dataset,
        "sex": sex,
        "protocol": protocol,
        "info": dataset_info(dataset),
        "metrics": {},
        "dyssynchrony": False,
    }
    for metric, value in (("phase_sd", phase_sd), ("bandwidth", bandwidth)):
        ref = _get_ref(dataset, sex, protocol, metric)
        if not ref:
            out["metrics"][metric] = {"value": float(value), "available": False}
            continue
        mean = float(ref["mean"])
        sd = float(ref["sd"])
        cutoff = mean + Z_CUTOFF * sd
        z = (float(value) - mean) / sd if sd else None
        is_abn = float(value) > cutoff
        out["metrics"][metric] = {
            "value": float(value),
            "available": True,
            "mean": mean,
            "sd": sd,
            "cutoff": cutoff,
            "z": z,
            "abnormal": is_abn,
        }
        if is_abn:
            out["dyssynchrony"] = True
    return out

"""
SINCRO - core.normal_db
========================

Base de datos de valores normales de sincronía mecánica del VI para comparar las
métricas de fase de un paciente (PSD, Bandwidth/PHB y entropy %) contra rangos publicados,
estratificados por SEXO y PROTOCOLO (stress/rest).

Fundamento (por qué estratificar):
- Mukherjee 2016 (Indian J Nucl Med 31(4):255-9): demostró que PSD y PHB difieren
  SIGNIFICATIVAMENTE por sexo (H>M) y protocolo (stress>rest). No existe un único
  valor normal universal.
- Chen 2005 (J Nucl Cardiol): normal database original Emory (población Western).
- Cutoff de disincronía = media + 2·SD del grupo normal correspondiente.

Los valores pueden estar guardados como (mean, sd) o como upper_limit directo.
El z-score de un paciente, cuando hay media/SD, es:
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
# o {"upper_limit"}. Métricas: phase_sd (PSD, grados), bandwidth (PHB, grados),
# entropy_normalized_pct (0-100%).
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
    "QGS_JSNM2023": {
        "descripcion": "JSNM working group normal database, QGS/Cedars-Sinai. Upper normal global y por sexo; stress/rest no separado.",
        "referencia": "Kuronuma K et al. J Cardiol 2023;82:87-92; Nakajima K et al. J Nucl Cardiol 2017;24:611-621.",
        "male": {
            "stress": {
                "phase_sd": {"mean": 6.2, "sd": 3.0, "upper_limit": 12.0, "range": None},
                "bandwidth": {"mean": 25.0, "sd": 8.9, "upper_limit": 43.0, "range": None},
                "entropy_normalized_pct": {"mean": 27.8, "sd": 7.8, "upper_limit": 43.0, "range": None},
            },
            "rest": {
                "phase_sd": {"mean": 6.2, "sd": 3.0, "upper_limit": 12.0, "range": None},
                "bandwidth": {"mean": 25.0, "sd": 8.9, "upper_limit": 43.0, "range": None},
                "entropy_normalized_pct": {"mean": 27.8, "sd": 7.8, "upper_limit": 43.0, "range": None},
            },
        },
        "female": {
            "stress": {
                "phase_sd": {"mean": 4.3, "sd": 2.7, "upper_limit": 10.0, "range": None},
                "bandwidth": {"mean": 18.5, "sd": 6.9, "upper_limit": 32.0, "range": None},
                "entropy_normalized_pct": {"mean": 19.8, "sd": 6.7, "upper_limit": 33.0, "range": None},
            },
            "rest": {
                "phase_sd": {"mean": 4.3, "sd": 2.7, "upper_limit": 10.0, "range": None},
                "bandwidth": {"mean": 18.5, "sd": 6.9, "upper_limit": 32.0, "range": None},
                "entropy_normalized_pct": {"mean": 19.8, "sd": 6.7, "upper_limit": 33.0, "range": None},
            },
        },
    },
    "ECTb_JSNM2023": {
        "descripcion": "JSNM working group normal database, Emory Cardiac Toolbox. Upper normal global y por sexo; stress/rest no separado.",
        "referencia": "Kuronuma K et al. J Cardiol 2023;82:87-92; Nakajima K et al. J Nucl Cardiol 2017;24:611-621.",
        "male": {
            "stress": {
                "phase_sd": {"mean": 12.8, "sd": 6.2, "upper_limit": 25.0, "range": None},
                "bandwidth": {"mean": 31.3, "sd": 9.4, "upper_limit": 50.0, "range": None},
            },
            "rest": {
                "phase_sd": {"mean": 12.8, "sd": 6.2, "upper_limit": 25.0, "range": None},
                "bandwidth": {"mean": 31.3, "sd": 9.4, "upper_limit": 50.0, "range": None},
            },
        },
        "female": {
            "stress": {
                "phase_sd": {"mean": 10.1, "sd": 4.3, "upper_limit": 19.0, "range": None},
                "bandwidth": {"mean": 27.3, "sd": 8.9, "upper_limit": 45.0, "range": None},
            },
            "rest": {
                "phase_sd": {"mean": 10.1, "sd": 4.3, "upper_limit": 19.0, "range": None},
                "bandwidth": {"mean": 27.3, "sd": 8.9, "upper_limit": 45.0, "range": None},
            },
        },
    },
    "cREPO_JSNM2023": {
        "descripcion": "JSNM working group normal database, cardioREPO/cREPO. Upper normal global y por sexo; stress/rest no separado.",
        "referencia": "Kuronuma K et al. J Cardiol 2023;82:87-92; Nakajima K et al. J Nucl Cardiol 2017;24:611-621.",
        "male": {
            "stress": {
                "phase_sd": {"mean": 11.4, "sd": 3.7, "upper_limit": 19.0, "range": None},
                "bandwidth": {"mean": 43.7, "sd": 12.8, "upper_limit": 69.0, "range": None},
                "entropy_normalized_pct": {"mean": 45.9, "sd": 5.6, "upper_limit": 57.0, "range": None},
            },
            "rest": {
                "phase_sd": {"mean": 11.4, "sd": 3.7, "upper_limit": 19.0, "range": None},
                "bandwidth": {"mean": 43.7, "sd": 12.8, "upper_limit": 69.0, "range": None},
                "entropy_normalized_pct": {"mean": 45.9, "sd": 5.6, "upper_limit": 57.0, "range": None},
            },
        },
        "female": {
            "stress": {
                "phase_sd": {"mean": 9.1, "sd": 2.0, "upper_limit": 13.0, "range": None},
                "bandwidth": {"mean": 36.6, "sd": 9.0, "upper_limit": 54.0, "range": None},
                "entropy_normalized_pct": {"mean": 40.0, "sd": 5.8, "upper_limit": 52.0, "range": None},
            },
            "rest": {
                "phase_sd": {"mean": 9.1, "sd": 2.0, "upper_limit": 13.0, "range": None},
                "bandwidth": {"mean": 36.6, "sd": 9.0, "upper_limit": 54.0, "range": None},
                "entropy_normalized_pct": {"mean": 40.0, "sd": 5.8, "upper_limit": 52.0, "range": None},
            },
        },
    },
    "HFV_JSNM2023": {
        "descripcion": "JSNM working group normal database, Heart Function View. Upper normal global y por sexo; stress/rest no separado.",
        "referencia": "Kuronuma K et al. J Cardiol 2023;82:87-92; Nakajima K et al. J Nucl Cardiol 2017;24:611-621.",
        "male": {
            "stress": {
                "phase_sd": {"mean": 6.2, "sd": 2.7, "upper_limit": 12.0, "range": None},
                "bandwidth": {"mean": 23.1, "sd": 9.5, "upper_limit": 42.0, "range": None},
            },
            "rest": {
                "phase_sd": {"mean": 6.2, "sd": 2.7, "upper_limit": 12.0, "range": None},
                "bandwidth": {"mean": 23.1, "sd": 9.5, "upper_limit": 42.0, "range": None},
            },
        },
        "female": {
            "stress": {
                "phase_sd": {"mean": 4.4, "sd": 1.8, "upper_limit": 8.0, "range": None},
                "bandwidth": {"mean": 16.5, "sd": 7.2, "upper_limit": 31.0, "range": None},
            },
            "rest": {
                "phase_sd": {"mean": 4.4, "sd": 1.8, "upper_limit": 8.0, "range": None},
                "bandwidth": {"mean": 16.5, "sd": 7.2, "upper_limit": 31.0, "range": None},
            },
        },
    },
    "QGS_Hamalainen2018": {
        "descripcion": "QPS/QGS usado en Rev Colomb Cardiol 2018; normalidad Hamalainen et al.; stress/rest no separado.",
        "referencia": "Garcia-Gomez FJ et al. Rev Colomb Cardiol 2018;25(3):192-199; Hamalainen et al.",
        "male": {
            "stress": {
                "phase_sd": {"mean": 10.2, "sd": 6.1, "range": None},
                "bandwidth": {"mean": 30.9, "sd": 12.7, "range": None},
                "peak_phase": {"mean": 142.3, "sd": 13.6, "range": None},
            },
            "rest": {
                "phase_sd": {"mean": 10.2, "sd": 6.1, "range": None},
                "bandwidth": {"mean": 30.9, "sd": 12.7, "range": None},
                "peak_phase": {"mean": 142.3, "sd": 13.6, "range": None},
            },
        },
        "female": {
            "stress": {
                "phase_sd": {"mean": 10.2, "sd": 6.1, "range": None},
                "bandwidth": {"mean": 30.9, "sd": 12.7, "range": None},
                "peak_phase": {"mean": 142.3, "sd": 13.6, "range": None},
            },
            "rest": {
                "phase_sd": {"mean": 10.2, "sd": 6.1, "range": None},
                "bandwidth": {"mean": 30.9, "sd": 12.7, "range": None},
                "peak_phase": {"mean": 142.3, "sd": 13.6, "range": None},
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


def _metric_eval(value: float, ref: dict | None) -> dict[str, Any]:
    out: dict[str, Any] = {"value": float(value), "available": bool(ref)}
    if not ref:
        return out
    mean = ref.get("mean")
    sd = ref.get("sd")
    upper_limit = ref.get("upper_limit")
    if upper_limit is None and mean is not None and sd:
        upper_limit = float(mean) + Z_CUTOFF * float(sd)
    out.update({
        "mean": float(mean) if mean is not None else None,
        "sd": float(sd) if sd is not None else None,
        "cutoff": float(upper_limit) if upper_limit is not None else None,
        "upper_limit": float(upper_limit) if upper_limit is not None else None,
        "z": ((float(value) - float(mean)) / float(sd)) if mean is not None and sd else None,
        "abnormal": bool(upper_limit is not None and float(value) > float(upper_limit)),
        "method": "upper_limit" if ref.get("upper_limit") is not None else "mean_plus_2sd",
    })
    return out


def evaluate(
    phase_sd: float,
    bandwidth: float,
    entropy_normalized_pct: float | None = None,
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
    metric_values: list[tuple[str, float]] = [("phase_sd", phase_sd), ("bandwidth", bandwidth)]
    if entropy_normalized_pct is not None:
        metric_values.append(("entropy_normalized_pct", entropy_normalized_pct))
    for metric, value in metric_values:
        ref = _get_ref(dataset, sex, protocol, metric)
        out["metrics"][metric] = _metric_eval(float(value), ref)
        if out["metrics"][metric].get("abnormal"):
            out["dyssynchrony"] = True
    out["clinical_classification"] = "ANORMAL" if out["dyssynchrony"] else "normal"
    out["interpretation"] = "disincronía mecánica intraventricular VI vs DB" if out["dyssynchrony"] else "sin disincronía significativa vs DB"
    return out

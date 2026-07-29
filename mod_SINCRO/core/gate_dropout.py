"""SINCRO - core.gate_dropout

Corrección del "dropout" de cuentas del último gate en Gated SPECT.

EL PROBLEMA
-----------
En un estudio gatillado el ciclo R-R se divide en N gates (típicamente 8 o 16).
Como el intervalo R-R no es constante latido a latido, los latidos más cortos
que la ventana de aceptación terminan antes de llenar el último gate. Resultado:
**el último gate acumula sistemáticamente menos cuentas que los demás**, aunque
el corazón esté en la misma posición anatómica que al inicio del ciclo.

Ese déficit es un artefacto de adquisición, no fisiología. Contamina dos cosas:

1. **La curva de volumen/FEVI**: el último gate parece tener "menos miocardio",
   se subestima el volumen telediastólico final y la curva no cierra el ciclo.
2. **El análisis de fase (FFT del 1er armónico)**: la caída artificial del
   último punto de la curva de actividad mete un escalón que desplaza la fase
   estimada y ensancha la distribución (PSD/BW mayores de lo real).

LA CORRECCIÓN (Emory Cardiac Toolbox 4.0, Technical Overview secc. 22.8)
------------------------------------------------------------------------
ECTb escala **todas las muestras del último gate** por un único factor, de modo
que la suma total de cuentas del último gate iguale la suma del primero:

    factor = sum(gate[0]) / sum(gate[-1])
    gate[-1] <- gate[-1] * factor

Es una corrección global de ganancia: no cambia la forma espacial del último
gate (no mueve bordes ni deforma el miocardio), solo restituye la estadística
que se perdió por el R-R corto.

EL SUPUESTO (y su límite)
-------------------------
El método asume que el primer y el último gate son adyacentes en el ciclo
(ambos en telediastole) y por lo tanto **deberían** tener cuentas comparables.
Es un supuesto razonable en un gated cardiaco normal, pero no exacto: si la
contracción real hace que el último gate tenga legítimamente algo menos de
cuentas que el primero, la corrección mete un pequeño error en sentido
contrario. En la práctica ese residuo es mucho menor que el dropout que
corrige (típicamente 10-20% de déficit por R-R vs pocos % de diferencia
fisiológica), por eso ECTb lo aplica por defecto.

CUÁNDO NO APLICARLA
-------------------
- Si el estudio ya viene corregido por la consola (Xeleris/Odyssey a veces lo
  hacen internamente): se detecta porque el dropout medido es ~0%.
- Si el déficit es enorme (>50%), el último gate tiene tan pocas cuentas que
  amplificarlo solo amplifica ruido. Por eso `correct_last_gate_dropout` acepta
  un `max_scale` de seguridad y avisa cuando lo alcanza.

USO TÍPICO
----------
    from core.gate_dropout import analyze_gate_dropout, correct_last_gate_dropout

    info = analyze_gate_dropout(cube)          # diagnóstico, no modifica nada
    if info["significant"]:
        cube, applied = correct_last_gate_dropout(cube)
        print(applied["message"])
"""
from __future__ import annotations

import numpy as np

#: Por debajo de este déficit (en %) la corrección se considera cosmética y
#: `analyze_gate_dropout` marca `significant=False`. 2% está por debajo del
#: ruido estadístico típico de un gate de SPECT cardíaco.
SIGNIFICANT_DROPOUT_PCT = 2.0

#: Tope de amplificación por seguridad. Un factor 2.0 ya implica que el último
#: gate tenía la mitad de las cuentas: más que eso es ruido, no señal.
DEFAULT_MAX_SCALE = 2.0


def gate_total_counts(cube: np.ndarray) -> np.ndarray:
    """Cuentas totales por gate.

    Parameters
    ----------
    cube : ndarray (n_gates, n_slices, H, W)

    Returns
    -------
    ndarray (n_gates,) con la suma de cuentas de cada gate.
    """
    arr = np.asarray(cube, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"cube debe ser 4D (n_gates, n_slices, H, W); recibió {arr.shape}")
    return arr.sum(axis=(1, 2, 3))


def analyze_gate_dropout(cube: np.ndarray, max_scale: float = DEFAULT_MAX_SCALE) -> dict:
    """Mide el déficit de cuentas del último gate sin modificar el cubo.

    Es la función barata: se puede llamar en cada refresco de UI (es un
    ``sum`` sobre el cubo, del orden de milisegundos para un gated típico).

    Returns
    -------
    dict con:
        n_gates          : int
        gate_counts      : list[float] — cuentas por gate
        first_counts     : float
        last_counts      : float
        dropout_pct      : float — cuánto le falta al último gate vs el primero (%)
        scale            : float — factor que igualaría last con first (1.0 = nada que hacer)
        scale_clipped    : float — `scale` limitado por `max_scale`
        clipped          : bool  — True si hubo que recortar el factor
        significant      : bool  — True si vale la pena corregir
        applicable       : bool  — False si no se puede (cubo raro, gate vacío)
        message          : str   — resumen legible para log/UI
    """
    counts = gate_total_counts(cube)
    n_gates = int(counts.size)
    out: dict = {
        "n_gates": n_gates,
        "gate_counts": [float(c) for c in counts],
        "first_counts": float(counts[0]) if n_gates else 0.0,
        "last_counts": float(counts[-1]) if n_gates else 0.0,
        "dropout_pct": 0.0,
        "scale": 1.0,
        "scale_clipped": 1.0,
        "clipped": False,
        "significant": False,
        "applicable": False,
        "message": "",
    }

    if n_gates < 3:
        out["message"] = f"Solo {n_gates} gate(s): la corrección de dropout no aplica."
        return out

    first = float(counts[0])
    last = float(counts[-1])
    if first <= 0.0 or last <= 0.0:
        out["message"] = "Primer o último gate sin cuentas: no se puede estimar el dropout."
        return out

    out["applicable"] = True
    scale = first / last
    dropout_pct = (1.0 - last / first) * 100.0
    scale_clipped = float(min(scale, float(max_scale))) if scale > 1.0 else float(scale)

    out["scale"] = float(scale)
    out["scale_clipped"] = scale_clipped
    out["clipped"] = bool(scale > float(max_scale))
    out["dropout_pct"] = float(dropout_pct)
    out["significant"] = bool(dropout_pct >= SIGNIFICANT_DROPOUT_PCT)

    if dropout_pct < 0.0:
        out["message"] = (
            f"El último gate tiene {abs(dropout_pct):.1f}% MÁS cuentas que el primero "
            f"(factor {scale:.3f}); no hay dropout que corregir."
        )
    elif not out["significant"]:
        out["message"] = (
            f"Dropout del último gate {dropout_pct:.1f}% (< {SIGNIFICANT_DROPOUT_PCT:.0f}%): despreciable."
        )
    elif out["clipped"]:
        out["message"] = (
            f"Dropout del último gate {dropout_pct:.1f}% — factor {scale:.2f} recortado a "
            f"{scale_clipped:.2f} por seguridad. Revisá la ventana de aceptación del R-R."
        )
    else:
        out["message"] = (
            f"Dropout del último gate {dropout_pct:.1f}% — se corrige con factor {scale:.3f} (ECTb 22.8)."
        )
    return out


def correct_last_gate_dropout(
    cube: np.ndarray,
    max_scale: float = DEFAULT_MAX_SCALE,
    force: bool = False,
) -> tuple[np.ndarray, dict]:
    """Escala el último gate para que iguale las cuentas del primero (ECTb 22.8).

    Parameters
    ----------
    cube : ndarray (n_gates, n_slices, H, W)
    max_scale : float
        Tope de amplificación. Evita inflar ruido cuando el último gate quedó
        casi vacío.
    force : bool
        Si True aplica la corrección aunque el dropout sea despreciable
        (< `SIGNIFICANT_DROPOUT_PCT`). Útil para comparar A/B.

    Returns
    -------
    (cube_corregido, info)
        `cube_corregido` es un array float64 nuevo; el original no se toca.
        Si no se aplicó nada, se devuelve una copia igual al original.
        `info` es el dict de `analyze_gate_dropout` más la clave `applied`.
    """
    info = analyze_gate_dropout(cube, max_scale=max_scale)
    arr = np.asarray(cube, dtype=np.float64)

    should_apply = info["applicable"] and info["scale_clipped"] > 1.0 and (info["significant"] or force)
    info["applied"] = bool(should_apply)
    if not should_apply:
        return arr.copy(), info

    out = arr.copy()
    out[-1] = out[-1] * float(info["scale_clipped"])
    return out, info

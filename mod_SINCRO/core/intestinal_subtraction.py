"""SINCRO - core.intestinal_subtraction

Sustracción de fondo intestinal por ROI del operador.

EL PROBLEMA
-----------
En SPECT cardíaco con trazadores de tecnecio, un asa intestinal cargada puede
quedar "pegada" a la cara inferior del ventrículo izquierdo. Cuando la
separación anatómica es imposible, hoy el estudio termina informado con la cara
inferior **no evaluable**, o directamente se pierde.

El atenuador intestinal preexistente (``CineWidget._attenuate_image_with_intestinal_roi``)
es **multiplicativo**: ``img * (1 - atten * mask)``. Ese modelo es físicamente
incorrecto para este caso, porque las cuentas de intestino y de miocardio no se
multiplican: **se suman** en el vóxel, son emisiones superpuestas. Con 100
cuentas de miocardio + 200 de intestino = 300, atenuar al 50% deja 150 cuando lo
correcto sería 100; y además castiga proporcionalmente al miocardio sano que
caiga dentro del ROI.

POR QUÉ IMPORTA ESPECÍFICAMENTE EN ANÁLISIS DE FASE
---------------------------------------------------
La señal de un vóxel a lo largo de los gates es::

    s(t) = m0 + a * cos(w*t + phi) + I

donde ``I`` es la contribución intestinal. **El intestino no late con el R-R**,
así que ``I`` es esencialmente una constante (componente DC).

- Atenuación multiplicativa ``k * s(t)``:
  amplitud relativa = ``k*a / (k*m0 + k*I)`` = ``a / (m0 + I)``. **Idéntica**:
  el factor ``k`` se cancela. Para el filtro de amplitud del análisis de fase,
  atenuar no aporta absolutamente nada.
- Sustracción ``s(t) - I``:
  amplitud relativa = ``a / m0``. **Mejora real**, y recupera vóxeles que el
  ``amplitude_threshold_frac`` estaba descartando.

Y como ``I`` es DC puro, restarlo **no desplaza la fase del primer armónico**.
Es una corrección de bajo riesgo para el valor clínico medido y de beneficio
concreto para la retención de vóxeles.

EL MÉTODO (análogo a la sustracción de paratiroides)
-----------------------------------------------------
En un estudio de paratiroides se resta ``sestamibi - k * pertecnetato``: hay una
**imagen medida** de la fuente que se quiere eliminar. Acá no existe una imagen
de "solo intestino" en el mismo sitio, así que hay que **estimarla**:

1. El operador marca una o dos ROI de **referencia** sobre el asa donde se ve
   limpia, sin miocardio encima (la "entrada" y la "salida" del asa).
2. De cada referencia se toma la **mediana** de cuentas (no el promedio: la
   mediana aguanta píxeles calientes puntuales).
3. En la zona de solapamiento, el nivel intestinal se interpola por **distancia
   inversa** (IDW) entre las referencias::

       B(p) = sum_i(w_i * level_i) / sum_i(w_i),   w_i = 1 / (d_i + eps)^power

   Esto es el "promedio entre entrada y salida" propuesto, pero sin necesidad de
   definir el eje del asa, que en 3D con un asa curva es ambiguo. Con una sola
   referencia degenera a un nivel constante, que también es válido.
4. Se resta con recorte a cero.

LA REGLA QUE NO SE PUEDE VIOLAR
--------------------------------
**El fondo se estima UNA sola vez sobre el promedio de gates y se resta el mismo
mapa a todos los gates.** Si se estimara gate por gate se introduciría variación
temporal artificial en la región, contaminando justamente la fase que se quiere
medir. El intestino es una constante: la corrección tiene que ser una constante.

Como la estimación se hace sobre ``cube.mean(axis=0)``, el nivel ``B`` ya está en
unidades de "cuentas de un gate", así que a cada gate se le resta ``B`` directo
(no ``B / n_gates``, que sería el caso si se hubiera estimado sobre la suma).

CONTROL DE CALIDAD
------------------
Después de restar, el tejido que queda en la zona corregida **tiene que latir**.
Si la amplitud relativa del primer armónico no mejora, lo que había ahí era
intestino y no miocardio recuperable. Ese chequeo lo da
``relative_first_harmonic_amplitude`` y se reporta como ``rel_amp_before`` /
``rel_amp_after``. Es un verificador objetivo que la sustracción de paratiroides
no puede hacer, porque allá no existe la dimensión temporal.

LÍMITES Y RIESGOS (leer antes de confiar en el resultado)
----------------------------------------------------------
- **Dirección del error**: restar de menos deja el punto caliente y puede tapar
  un defecto real; restar de más **fabrica un defecto inferior** donde no lo hay,
  justo en la región que ya tiene más falsos positivos por atenuación
  diafragmática. ``clipped_fraction`` alerta sobre esto.
- **Ruido**: restar suma varianzas (Poisson: ``var(a-b) = var(a) + var(b)``).
  Donde queden pocas cuentas tras la resta, la fase se vuelve inestable.
- **No deshace la reconstrucción**: un asa muy caliente además distorsiona la
  reconstrucción iterativa (roba cuentas, genera streaks). Corregir en el espacio
  imagen no revierte eso. Lo más correcto sería enmascarar en las proyecciones
  crudas y re-reconstruir (ver ``core.raw_projections``).
- **Es operador-dependiente**: como el factor ``k`` de la sustracción de
  paratiroides, lo elige una persona. Por eso toda corrección aplicada debe
  quedar declarada en el informe, con las cuentas restadas, y correlacionarse
  siempre con la clínica del paciente.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

#: Exponente de la interpolación por distancia inversa. 1.0 = interpolación
#: aproximadamente lineal entre las referencias, que es lo que uno espera de un
#: asa que entra y sale de la zona de solapamiento.
DEFAULT_IDW_POWER = 1.0

#: Métodos de estimación del nivel de fondo.
#:
#: - ``"idw"``: interpolación por distancia inversa entre las referencias. El
#:   fondo varía dentro de la zona a corregir, acercándose al nivel de la
#:   referencia más próxima. Modela un asa cuyo nivel cambia entre la entrada y
#:   la salida.
#: - ``"mean"``: **media aritmética de los niveles** de todas las referencias,
#:   aplicada como una única constante a toda la zona. Cada ROI pesa igual, sin
#:   importar su tamaño, para que una referencia grande no domine el resultado.
#:   Es el equivalente al ROI de fondo clásico de MUGA o del renograma: un solo
#:   número, predecible y fácil de auditar.
BACKGROUND_METHODS = ("idw", "mean")
DEFAULT_BACKGROUND_METHOD = "idw"

#: Una referencia con menos píxeles que esto es ruido, no una medición de nivel.
MIN_REFERENCE_PIXELS = 4

#: Si más de esta fracción de la zona corregida quedó recortada en cero, el
#: fondo se sobreestimó y lo más probable es que se esté fabricando un defecto.
OVERSUBTRACTION_WARN_FRAC = 0.35

#: Evita división por cero en el peso IDW de los píxeles que caen dentro de la
#: propia ROI de referencia (distancia 0).
_EPS_DISTANCE = 1e-3


def reference_levels(mean_img: np.ndarray, reference_masks: list[np.ndarray]) -> list[float]:
    """Nivel intestinal (mediana de cuentas) de cada ROI de referencia.

    Se usa mediana y no media porque un solo píxel muy caliente dentro de la
    referencia desplazaría el nivel hacia arriba y produciría sobre-sustracción.

    Las referencias con menos de `MIN_REFERENCE_PIXELS` píxeles se descartan
    (devuelven ``nan``), porque un nivel medido sobre 2 o 3 píxeles no es un
    nivel: es ruido.
    """
    img = np.asarray(mean_img, dtype=np.float64)
    levels: list[float] = []
    for mask in reference_masks:
        m = np.asarray(mask, dtype=bool)
        if m.shape != img.shape or int(m.sum()) < MIN_REFERENCE_PIXELS:
            levels.append(float("nan"))
            continue
        levels.append(float(np.median(img[m])))
    return levels


def estimate_background_map(
    mean_img: np.ndarray,
    reference_masks: list[np.ndarray],
    *,
    power: float = DEFAULT_IDW_POWER,
    method: str = DEFAULT_BACKGROUND_METHOD,
) -> tuple[np.ndarray, dict]:
    """Mapa de fondo intestinal estimado a partir de las ROI de referencia.

    Parameters
    ----------
    mean_img : ndarray (H, W)
        Imagen **promediada sobre gates** del corte. Promediar y no sumar deja el
        nivel en unidades de un gate, que es lo que después se resta a cada gate.
    reference_masks : list of ndarray (H, W) bool
        ROIs dibujadas por el operador sobre el asa donde se ve limpia.
    power : float
        Exponente IDW. Mayor valor = transición más abrupta entre referencias.
        Solo aplica con ``method="idw"``.
    method : str
        ``"idw"`` (interpolado entre referencias) o ``"mean"`` (media aritmética
        de los niveles, constante en toda la zona). Ver `BACKGROUND_METHODS`.

    Returns
    -------
    (background, info)
        `background` es (H, W) float64 con el nivel estimado en todo el plano.
        Si no hay referencias válidas devuelve ceros (no restar nada es la
        opción segura).
    """
    img = np.asarray(mean_img, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"mean_img debe ser 2D (H, W); recibió {img.shape}")
    mode = str(method or DEFAULT_BACKGROUND_METHOD).strip().lower()
    if mode not in BACKGROUND_METHODS:
        mode = DEFAULT_BACKGROUND_METHOD

    levels = reference_levels(img, reference_masks or [])
    valid = [
        (np.asarray(m, dtype=bool), float(lv))
        for m, lv in zip(reference_masks or [], levels)
        if np.isfinite(lv)
    ]

    info: dict = {
        "n_references": len(valid),
        "n_references_discarded": len(levels) - len(valid),
        "levels": [float(lv) for _, lv in valid],
        "method": mode,
        "applicable": bool(valid),
    }

    if not valid:
        info["message"] = (
            "Sin ROI de referencia válida: no se resta nada. "
            f"Cada referencia necesita al menos {MIN_REFERENCE_PIXELS} píxeles."
        )
        return np.zeros_like(img), info

    if mode == "mean" or len(valid) == 1:
        level = float(np.mean([lv for _, lv in valid]))
        info["background_level"] = level
        if len(valid) == 1:
            info["message"] = (
                f"Una sola referencia: se resta un nivel constante de {level:.1f} cuentas/gate."
            )
        else:
            lv_txt = ", ".join(f"{lv:.1f}" for lv in info["levels"])
            info["message"] = (
                f"{len(valid)} referencias (niveles: {lv_txt} cuentas/gate); "
                f"media simple = {level:.1f} cuentas/gate aplicada como constante."
            )
        return np.full_like(img, level), info

    weights_sum = np.zeros_like(img)
    weighted = np.zeros_like(img)
    for mask, level in valid:
        # distancia de cada píxel a la ROI de referencia
        dist = distance_transform_edt(~mask).astype(np.float64)
        w = 1.0 / np.power(dist + _EPS_DISTANCE, float(power))
        weights_sum += w
        weighted += w * level

    background = np.divide(
        weighted,
        weights_sum,
        out=np.zeros_like(img),
        where=weights_sum > 0,
    )
    lv_txt = ", ".join(f"{lv:.1f}" for lv in info["levels"])
    info["message"] = (
        f"{len(valid)} referencias (niveles: {lv_txt} cuentas/gate); "
        "fondo interpolado por distancia inversa."
    )
    return background, info


def relative_first_harmonic_amplitude(gate_stack: np.ndarray, mask: np.ndarray) -> float:
    """Amplitud relativa media del 1er armónico en los vóxeles de `mask`.

    Es el mismo criterio que usa el filtro de amplitud del análisis de fase:
    ``a1 / media_temporal`` por vóxel. Sirve para verificar que lo que quedó tras
    la sustracción efectivamente late.

    Devuelve ``nan`` si no hay vóxeles útiles.
    """
    arr = np.asarray(gate_stack, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if arr.ndim != 3 or m.shape != arr.shape[1:]:
        return float("nan")
    n_gates = int(arr.shape[0])
    if n_gates < 3 or not m.any():
        return float("nan")

    series = arr[:, m]  # (gates, n_vox)
    mean_level = series.mean(axis=0)
    useful = mean_level > 0
    if not np.any(useful):
        return float("nan")

    spectrum = np.fft.rfft(series[:, useful], axis=0)
    a1 = 2.0 * np.abs(spectrum[1]) / float(n_gates)
    return float(np.mean(a1 / mean_level[useful]))


def subtract_background_from_slice(
    gate_stack: np.ndarray,
    background: np.ndarray,
    weight_map: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Resta el mismo mapa de fondo a todos los gates de un corte.

    Parameters
    ----------
    gate_stack : ndarray (n_gates, H, W)
        Todos los gates de un mismo corte.
    background : ndarray (H, W)
        Nivel intestinal estimado, en cuentas de un gate.
    weight_map : ndarray (H, W)
        Peso de aplicación en [0, 1] (la máscara del ROI con su borde blando).
        Fuera del ROI vale 0 y no se toca nada.

    Returns
    -------
    (corrected, info)
    """
    arr = np.asarray(gate_stack, dtype=np.float64)
    bg = np.asarray(background, dtype=np.float64)
    w = np.clip(np.asarray(weight_map, dtype=np.float64), 0.0, 1.0)

    if arr.ndim != 3:
        raise ValueError(f"gate_stack debe ser 3D (n_gates, H, W); recibió {arr.shape}")
    if bg.shape != arr.shape[1:] or w.shape != arr.shape[1:]:
        raise ValueError("background y weight_map deben tener la forma (H, W) del corte")

    region = w > 0.5
    to_subtract = bg * w
    raw = arr - to_subtract[None, :, :]
    corrected = np.clip(raw, 0.0, None)

    subtracted = float(np.sum(arr - corrected))
    original_total = float(np.sum(arr))
    if region.any():
        clipped_fraction = float(np.mean(raw[:, region] < 0.0))
    else:
        clipped_fraction = 0.0

    info = {
        "counts_subtracted": subtracted,
        "counts_original": original_total,
        "subtracted_pct": float(100.0 * subtracted / original_total) if original_total > 0 else 0.0,
        "clipped_fraction": clipped_fraction,
        "oversubtracted": bool(clipped_fraction > OVERSUBTRACTION_WARN_FRAC),
        "region_pixels": int(region.sum()),
        "rel_amp_before": relative_first_harmonic_amplitude(arr, region),
        "rel_amp_after": relative_first_harmonic_amplitude(corrected, region),
    }
    return corrected, info


def apply_intestinal_subtraction(
    cube: np.ndarray,
    target_weights: dict[int, np.ndarray],
    reference_masks: dict[int, list[np.ndarray]],
    *,
    power: float = DEFAULT_IDW_POWER,
    method: str = DEFAULT_BACKGROUND_METHOD,
) -> tuple[np.ndarray, dict]:
    """Aplica la sustracción de fondo intestinal a un cubo gatillado completo.

    Parameters
    ----------
    cube : ndarray (n_gates, n_slices, H, W)
    target_weights : dict[int, ndarray (H, W)]
        Por índice de corte, el peso de aplicación en [0, 1] (ROI a corregir con
        su borde blando). Los cortes ausentes no se tocan.
    reference_masks : dict[int, list[ndarray (H, W) bool]]
        Por índice de corte, las ROI de referencia sobre el asa limpia.
    power : float
        Exponente de la interpolación por distancia inversa.
    method : str
        Método de estimación del fondo (ver `BACKGROUND_METHODS`).

    Returns
    -------
    (cube_corregido, info)
        El cubo original no se modifica. `info["per_slice"]` trae el detalle por
        corte y el resto son agregados para log e informe.
    """
    arr = np.asarray(cube, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"cube debe ser 4D (n_gates, n_slices, H, W); recibió {arr.shape}")

    out = arr.copy()
    per_slice: dict[int, dict] = {}
    total_subtracted = 0.0
    slices_touched: list[int] = []
    slices_oversubtracted: list[int] = []
    slices_without_reference: list[int] = []

    for slice_index, weight in sorted((target_weights or {}).items()):
        s = int(slice_index)
        if s < 0 or s >= int(arr.shape[1]):
            continue
        w = np.clip(np.asarray(weight, dtype=np.float64), 0.0, 1.0)
        if w.shape != arr.shape[2:] or not np.any(w > 0):
            continue

        refs = list((reference_masks or {}).get(s, []) or [])
        # El fondo se estima sobre el PROMEDIO de gates: una sola vez, nunca por
        # gate, para no inyectar variación temporal artificial en la región.
        mean_img = arr[:, s, :, :].mean(axis=0)
        background, bg_info = estimate_background_map(mean_img, refs, power=power, method=method)
        if not bg_info["applicable"]:
            slices_without_reference.append(s)
            per_slice[s] = {"applied": False, **bg_info}
            continue

        corrected, sub_info = subtract_background_from_slice(arr[:, s, :, :], background, w)
        out[:, s, :, :] = corrected
        total_subtracted += float(sub_info["counts_subtracted"])
        slices_touched.append(s)
        if sub_info["oversubtracted"]:
            slices_oversubtracted.append(s)
        per_slice[s] = {"applied": True, **bg_info, **sub_info}

    original_total = float(np.sum(arr))
    info = {
        "applied": bool(slices_touched),
        "method": str(method),
        "slices_corrected": slices_touched,
        "slices_oversubtracted": slices_oversubtracted,
        "slices_without_reference": slices_without_reference,
        "counts_subtracted": total_subtracted,
        "counts_original": original_total,
        "subtracted_pct": float(100.0 * total_subtracted / original_total) if original_total > 0 else 0.0,
        "per_slice": per_slice,
        "message": "",
    }

    if not slices_touched:
        if slices_without_reference:
            info["message"] = (
                f"Sustracción de fondo sin aplicar: {len(slices_without_reference)} corte(s) "
                "con ROI a corregir pero sin ROI de referencia."
            )
        else:
            info["message"] = "Sustracción de fondo sin aplicar: no hay ROI a corregir."
        return out, info

    msg = (
        f"Sustracción de fondo aplicada en {len(slices_touched)} corte(s): "
        f"{total_subtracted:,.0f} cuentas ({info['subtracted_pct']:.2f}% del total)."
    )
    if slices_oversubtracted:
        msg += (
            f" ATENCIÓN: posible sobre-sustracción en el/los corte(s) {slices_oversubtracted} "
            "(mucha región recortada en cero): puede estar fabricando un defecto inferior."
        )
    if slices_without_reference:
        msg += f" Sin referencia y por lo tanto sin corregir: corte(s) {slices_without_reference}."
    info["message"] = msg
    return out, info

"""SINCRO - core.ectb_lv

Cuantificación de función ventricular izquierda con el método del
**Emory Cardiac Toolbox 4.0** (Technical Overview, secc. 22.8).

POR QUÉ OTRO MÉTODO
-------------------
El método que veníamos usando (`_estimate_lv_ef_preliminary`) busca el borde
endocárdico como "el primer radio donde la actividad supera un umbral relativo
al pico" (`cavity_frac`), y después corrige el volumen con un factor empírico
(`basal_pad`) para que los mililitros den fisiológicos. Eso tiene dos problemas:

1. **El umbral depende del contraste.** Un estudio con más ruido de fondo, o
   con hígado pegado, mueve el borde y con él el ESV, que es justo el que manda
   en la FEVI.
2. **`basal_pad` es un fudge factor.** Escala el radio para que los volúmenes
   absolutos den bien, sin base física.

EL MÉTODO ECTb
--------------
ECTb no busca un borde por umbral: busca el **máximo de cuentas**, que es una
referencia mucho más estable porque no depende del nivel de fondo ni del
contraste, solo de dónde está el centro de la pared.

El pipeline es:

1. **Línea de centro miocárdico.** Se trazan perfiles radiales desde el eje
   largo del VI. En cada perfil, el máximo de cuentas marca el centro de la
   pared. Se refina con interpolación subpíxel (parábola por los 3 puntos
   alrededor del máximo).

2. **Espesor de pared en ED = 10 mm.** ECTb asume un espesor telediastólico
   fijo y coloca el endocardio 5 mm hacia adentro del centro y el epicardio
   5 mm hacia afuera. No es una medición: es una convención que ancla la escala
   absoluta de los volúmenes (reemplaza a nuestro `basal_pad`).

3. **Engrosamiento por el 1er armónico.** En SPECT el espesor de pared no se
   puede medir directamente (la resolución es peor que la pared), pero por
   efecto de volumen parcial las **cuentas máximas son proporcionales al
   espesor** mientras la pared sea más fina que ~2×FWHM. Entonces se ajusta el
   primer armónico a la curva de cuentas máximas de cada punto y se deriva el
   espesor de cada gate:

       espesor(t) = 10 mm × cuentas_ajustadas(t) / cuentas_ajustadas(ED)

   El endocardio de cada gate es centro(t) − espesor(t)/2.

4. **Suavizado de radios: mediana 7×7 y después 3×3** sobre la matriz
   (corte × ángulo). La mediana quita outliers puntuales (un píxel caliente de
   hígado) sin correr los bordes reales, cosa que un gaussiano sí haría.

5. **Volumen** por integración de la superficie endocárdica y **masa
   miocárdica** = volumen de pared × 1.05 g/mL (densidad del músculo cardíaco).

6. **Plano valvular de dos piezas.** El anillo mitral NO es perpendicular al
   eje largo: del lado septal la cavidad termina más cerca del ápex (ahí están
   el tracto de salida aórtico y el septum membranoso, que no son cavidad del
   VI), mientras que del lado lateral llega más arriba. Si se corta con un plano
   perpendicular único hay que elegir entre comerse pared lateral o contar como
   cavidad lo que no lo es; ECTb resuelve el dilema con un plano de dos piezas:
   perpendicular en toda la mitad lateral y angulado en la mitad septal.

   Acá se implementa con un corte dependiente del ángulo:

       u_corte(θ) = u_base − (offset_mm / dz) · max(0, cos(θ − θ_septal))

   El coseno rectificado vale 1 en el medio del septum, cae a 0 en las uniones
   antero-septal e infero-septal y queda en 0 en TODA la mitad lateral, que es
   exactamente la definición de "dos piezas". Cada corte aporta al volumen con
   un peso 0-1 según qué fracción suya queda por debajo del plano, así el
   volumen varía de forma continua al mover el parámetro (nada de saltos de un
   corte entero, que en vivo se verían como escalones).

   Sin esto el volumen basal se sobreestima y la FEVI queda inflada.

7. **Índice de esfericidad.** Cociente entre el diámetro de eje corto (el del
   corte más ancho) y la longitud del eje largo de la cavidad. El VI normal es
   un elipsoide alargado, así que el índice da bastante menor que 1; a medida
   que el ventrículo se remodela se vuelve más esférico y el índice sube hacia
   1. Se reporta en telediástole y en telesístole.

   Es un descriptor de forma, no de función: puede estar alterado con FEVI
   todavía conservada, y por eso aporta información que la FEVI sola no da. El
   punto de corte concreto depende del software y de la población, así que acá
   se informa el valor y la comparación queda a criterio del lector.

LO QUE ESTE MÓDULO TODAVÍA NO HACE
----------------------------------
El ángulo del septum se asume fijo (convención de despliegue SA del módulo:
septal a la izquierda de la imagen). No se detecta automáticamente a partir del
ventrículo derecho, que sería lo robusto si alguna vez entran cortes con otra
orientación. Por eso el ángulo quedó como parámetro editable.

USO
---
    from core.ectb_lv import ECTbLVConfig, analyze_lv_ectb

    res = analyze_lv_ectb(cube, seg, pixel_mm=(6.4, 6.4), slice_mm=6.4)
    if res.available:
        print(res.ef_pct, res.edv_ml, res.esv_ml, res.myocardial_mass_g)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import map_coordinates, median_filter

#: Espesor telediastólico de pared asumido por ECTb (mm).
ED_WALL_THICKNESS_MM = 10.0

#: Densidad del músculo cardíaco (g/mL) usada para la masa.
MYOCARDIAL_DENSITY_G_ML = 1.05

#: Límites de seguridad del espesor derivado del 1er armónico, como múltiplos
#: del espesor en ED. Evita que un punto con cuentas casi nulas genere una pared
#: absurda (o negativa) y rompa el volumen.
MIN_THICKNESS_FRAC = 0.55
MAX_THICKNESS_FRAC = 2.20

#: Cuánto más apical termina la cavidad del lado septal que del lado lateral
#: (mm). Es el "angulado" del plano valvular de dos piezas del ECTb.
VALVE_SEPTAL_OFFSET_MM = 10.0

#: Ángulo (grados) donde cae el medio del septum en el corte de eje corto. Con
#: la convención de despliegue del módulo (SEPTAL izquierda, LATERAL derecha,
#: ANTERIOR arriba) y ángulos medidos con x=cosθ / y=sinθ sobre la imagen, el
#: medio del septum cae en 180°.
SEPTAL_ANGLE_DEG = 180.0


@dataclass
class ECTbLVConfig:
    """Parámetros del método. Los defaults son los del ECTb 4.0."""

    #: Espesor de pared asumido en telediástole (mm). Ancla la escala absoluta
    #: de EDV/ESV. Subirlo achica la cavidad; bajarlo la agranda.
    ed_wall_thickness_mm: float = ED_WALL_THICKNESS_MM
    #: Perfiles radiales por corte. Más ángulos = contorno más fino y más lento.
    n_angles: int = 64
    #: Muestras por píxel a lo largo de cada rayo (resolución radial).
    radial_oversample: float = 4.0
    #: Hasta dónde se extiende el rayo, como múltiplo del radio externo del ROI.
    radial_extent_frac: float = 1.35
    #: Kernel de la primera mediana (corte × ángulo). 0 lo desactiva.
    median_kernel_large: int = 7
    #: Kernel de la segunda mediana. 0 lo desactiva.
    median_kernel_small: int = 3
    #: Un corte entra al cálculo si su área miocárdica supera esta fracción del
    #: corte con más miocardio. Descarta base abierta y ápex sin cavidad.
    min_myo_area_frac: float = 0.30
    #: Gate que se toma como telediástole para anclar el espesor. En un gated
    #: el gating arranca en la onda R, así que el gate 1 (índice 0) es ED.
    ed_gate_index: int = 0
    #: Densidad del miocardio para la masa (g/mL).
    myocardial_density_g_ml: float = MYOCARDIAL_DENSITY_G_ML
    #: Aplicar el engrosamiento por 1er armónico. Si es False, el espesor queda
    #: fijo en todos los gates (útil para aislar el efecto del engrosamiento).
    use_thickening: bool = True
    #: Recortar la base con el plano valvular de dos piezas. Si es False se
    #: integran los cortes válidos enteros (plano perpendicular único).
    use_valve_plane: bool = True
    #: Cuánto retrocede el plano hacia el ápex en el medio del septum (mm).
    valve_septal_offset_mm: float = VALVE_SEPTAL_OFFSET_MM
    #: Ángulo del medio del septum en el corte de eje corto (grados).
    septal_angle_deg: float = SEPTAL_ANGLE_DEG


@dataclass
class ECTbLVResult:
    """Resultado del análisis. `available=False` si no se pudo calcular."""

    available: bool = False
    reason: str = ""
    method: str = "ectb_max_counts"
    edv_ml: float = 0.0
    esv_ml: float = 0.0
    sv_ml: float = 0.0
    ef_pct: float = 0.0
    ed_gate: int = 0
    es_gate: int = 0
    gate_volumes_ml: np.ndarray = field(default_factory=lambda: np.zeros(0))
    myocardial_volume_ml: float = 0.0
    myocardial_mass_g: float = 0.0
    mean_wall_thickness_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    thickening_pct: float = 0.0
    #: Espesor medio de pared en el gate ED y en el gate ES (mm), para mostrar el
    #: valor absoluto del engrosamiento y no solo el porcentaje.
    wall_thickness_ed_mm: float = 0.0
    wall_thickness_es_mm: float = 0.0
    #: Índice de esfericidad = diámetro de eje corto / longitud de eje largo.
    #: 0-1: el VI normal es un elipsoide alargado (valores bajos); cuanto más se
    #: acerca a 1, más esférico y por lo tanto más remodelado.
    shape_index_ed: float = 0.0
    shape_index_es: float = 0.0
    #: Dimensiones que dan origen al índice, en mm (para poder auditarlo).
    long_axis_mm: float = 0.0
    short_axis_ed_mm: float = 0.0
    short_axis_es_mm: float = 0.0
    valid_slices: tuple[int, ...] = ()
    #: Cortes válidos ordenados del ápex a la base. El plano valvular se define
    #: en esta escala, no en el índice absoluto de corte.
    apex_to_base_slices: tuple[int, ...] = ()
    #: (n_angles,) posición del plano valvular en la pila ápex→base, en índices
    #: fraccionarios de corte. Sirve para dibujarlo sobre los cortes de eje largo.
    valve_cut_u: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: mL de cavidad telediastólica que el plano valvular sacó de la base.
    valve_removed_ml: float = 0.0
    #: Cortes totales del estudio, para poner `valid_slices` en contexto.
    n_slices_total: int = 0
    #: (n_gates, n_slices_validos, n_angles) en mm — para dibujar contornos.
    center_radii_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    endo_radii_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    epi_radii_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    config: ECTbLVConfig = field(default_factory=ECTbLVConfig)
    notes: list[str] = field(default_factory=list)


def _valid_slices(mask: np.ndarray, outer: np.ndarray, centers: np.ndarray, min_frac: float) -> list[int]:
    """Cortes con miocardio suficiente y geometría utilizable."""
    n_slices = int(mask.shape[0])
    area = mask.reshape(n_slices, -1).sum(axis=1).astype(np.float64)
    max_area = float(area.max()) if area.size else 0.0
    if max_area <= 0.0:
        return []
    return [
        s
        for s in range(n_slices)
        if area[s] >= float(min_frac) * max_area
        and np.isfinite(outer[s])
        and outer[s] > 2.0
        and np.isfinite(centers[s, 0])
        and np.isfinite(centers[s, 1])
    ]


def _sample_radial_profiles(
    volume: np.ndarray,
    centers: np.ndarray,
    r_line: np.ndarray,
    sin_a: np.ndarray,
    cos_a: np.ndarray,
) -> np.ndarray:
    """Muestrea perfiles radiales de un volumen 3D en un solo paso.

    Parameters
    ----------
    volume : (n_slices, H, W) — un gate.
    centers : (n_slices, 2) — (cy, cx) por corte.
    r_line : (n_r,) — radios de muestreo en píxeles.

    Returns
    -------
    (n_slices, n_angles, n_r) con las cuentas interpoladas bilinealmente.
    """
    n_slices = int(centers.shape[0])
    cy = centers[:, 0][:, None, None]
    cx = centers[:, 1][:, None, None]
    yy = cy + r_line[None, None, :] * sin_a[None, :, None]
    xx = cx + r_line[None, None, :] * cos_a[None, :, None]
    ss = np.broadcast_to(np.arange(n_slices, dtype=np.float64)[:, None, None], yy.shape)
    coords = np.stack([ss, yy, xx], axis=0)
    return map_coordinates(volume, coords, order=1, mode="nearest")


def _subpixel_peak(profiles: np.ndarray, r_line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ubica el máximo de cada perfil con precisión subpíxel.

    Ajusta una parábola por los tres puntos alrededor del máximo discreto. Es
    el refinamiento estándar para picos muestreados: con un pico razonablemente
    simétrico da el vértice sin necesidad de ajustar una gaussiana completa.

    Returns
    -------
    (radios_pico, cuentas_pico) con la forma de `profiles` sin el último eje.
    """
    n_r = int(profiles.shape[-1])
    dr = float(r_line[1] - r_line[0]) if n_r > 1 else 1.0
    idx = np.argmax(profiles, axis=-1)
    idx_c = np.clip(idx, 1, n_r - 2)

    take = lambda off: np.take_along_axis(profiles, (idx_c + off)[..., None], axis=-1)[..., 0]  # noqa: E731
    y0 = take(-1)
    y1 = take(0)
    y2 = take(1)

    denom = y0 - 2.0 * y1 + y2
    delta = np.zeros_like(y1)
    # Solo refinar donde hay curvatura real (pico, no meseta ni borde).
    ok = np.abs(denom) > 1e-12
    delta[ok] = 0.5 * (y0[ok] - y2[ok]) / denom[ok]
    delta = np.clip(delta, -1.0, 1.0)

    r_peak = r_line[idx_c] + delta * dr
    peak_counts = y1 - 0.25 * (y0 - y2) * delta
    # Donde el máximo cayó en el borde del rayo no hay pico interpretable.
    edge = (idx == 0) | (idx == n_r - 1)
    peak_counts = np.where(edge, 0.0, peak_counts)
    return r_peak, np.maximum(peak_counts, 0.0)


def _first_harmonic_fit(curves: np.ndarray) -> np.ndarray:
    """Reconstruye cada curva temporal con su componente DC + 1er armónico.

    Parameters
    ----------
    curves : (n_gates, ...) — curva de cuentas por gate en cada punto.

    Returns
    -------
    Array de la misma forma con la curva suavizada por el primer armónico.
    """
    n_gates = int(curves.shape[0])
    dc = curves.mean(axis=0)
    comp = np.fft.fft(curves - dc, axis=0)[1]
    t = np.arange(n_gates, dtype=np.float64)
    phase = np.exp(2j * np.pi * t / n_gates)
    shape = (n_gates,) + (1,) * (curves.ndim - 1)
    return dc[None, ...] + (2.0 / n_gates) * np.real(comp[None, ...] * phase.reshape(shape))


def _median_smooth_radii(radii: np.ndarray, kernel: int) -> np.ndarray:
    """Mediana 2D sobre (corte × ángulo) con envoltura circular en el ángulo.

    El eje angular es periódico (0° y 359° son vecinos), así que se padea de
    forma circular antes de filtrar; el eje de cortes se extiende por réplica.
    `median_filter` no admite un modo distinto por eje, de ahí el padding manual.
    """
    k = int(kernel)
    if k < 2:
        return radii
    if k % 2 == 0:
        k += 1
    pad = k // 2
    out = np.empty_like(radii)
    for g in range(radii.shape[0]):
        plane = radii[g]
        padded = np.pad(plane, ((pad, pad), (pad, pad)), mode="wrap")
        padded[:pad, :] = padded[pad : pad + 1, :]
        padded[-pad:, :] = padded[-pad - 1 : -pad, :]
        filtered = median_filter(padded, size=k, mode="nearest")
        out[g] = filtered[pad:-pad, pad:-pad]
    return out


def _wedge_areas(radii_mm: np.ndarray, n_angles: int) -> np.ndarray:
    """Área de cada cuña angular del contorno polar (mm²).

    El área encerrada por un contorno r(θ) es 0.5 ∫ r² dθ; acá se devuelve el
    integrando por ángulo SIN sumar, porque el plano valvular recorta una
    cantidad distinta en cada ángulo y hay que ponderar cuña por cuña antes de
    integrar. Sumando el último eje se recupera el área del corte completo.
    """
    dtheta = 2.0 * np.pi / float(n_angles)
    return 0.5 * np.asarray(radii_mm) ** 2 * dtheta


def _base_is_last(seg, valid: list[int]) -> bool:
    """¿El último corte válido es la base (True) o el ápex (False)?

    La pila de eje corto puede venir ordenada ápex→base o base→ápex según cómo
    se haya reorientado el estudio, y el plano valvular necesita saber para qué
    lado está la base. Se usa el mismo criterio que `core.aha_segments`: hacia la
    base el miocardio abarca más área y la cavidad es más ancha; hacia el ápex
    se cierra. Con eso alcanza y no hace falta metadata de orientación.
    """
    if len(valid) < 2:
        return True
    mask = np.asarray(getattr(seg, "mask"), dtype=bool)
    inner = np.asarray(getattr(seg, "inner_radius", np.full(mask.shape[0], np.nan)), dtype=np.float64)

    def score(s: int) -> float:
        area = float(mask[s].sum())
        r_in = float(inner[s]) if inner.size > s and np.isfinite(inner[s]) else 0.0
        return area + 10.0 * r_in

    return score(valid[-1]) >= score(valid[0])


def _valve_plane_weights(
    n_valid: int,
    angles: np.ndarray,
    dz_mm: float,
    septal_angle_deg: float,
    offset_mm: float,
    base_is_last: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Peso 0-1 con que cada corte entra al volumen, según ángulo.

    El plano valvular de dos piezas se define en la escala ápex→base:

        u_corte(θ) = (u_base + 0.5) − (offset_mm / dz) · max(0, cos(θ − θ_septal))

    El `+0.5` pone la referencia en la CARA externa del último corte, de modo
    que con `offset_mm = 0` todos los pesos dan exactamente 1 y el resultado
    coincide con el plano perpendicular de siempre.

    El peso de cada corte es la fracción de su espesor que queda por debajo del
    plano, recortada a [0, 1]. Al ser lineal, mover el offset cambia el volumen
    de forma continua: en vivo se ve como un deslizamiento, no como escalones de
    un corte entero.

    Returns
    -------
    (pesos, u_corte) con formas (n_valid, n_angles) y (n_angles,).
    """
    u = np.arange(n_valid, dtype=np.float64)
    if not base_is_last:
        # `u` tiene que crecer SIEMPRE hacia la base, sin importar cómo venga
        # ordenada la pila.
        u = u[::-1].copy()
    u_base = float(n_valid - 1)

    taper = np.maximum(0.0, np.cos(angles - np.deg2rad(float(septal_angle_deg))))
    u_cut = (u_base + 0.5) - (float(offset_mm) / float(dz_mm)) * taper
    weights = np.clip(u_cut[None, :] - u[:, None] + 0.5, 0.0, 1.0)
    return weights, u_cut


def analyze_lv_ectb(
    cube: np.ndarray,
    seg,
    pixel_mm: tuple[float, float],
    slice_mm: float,
    config: ECTbLVConfig | None = None,
) -> ECTbLVResult:
    """Cuantifica volúmenes, FEVI y masa del VI con el método ECTb.

    Parameters
    ----------
    cube : ndarray (n_gates, n_slices, H, W)
        Gated SPECT eje corto. Conviene pasarlo ya corregido por dropout del
        último gate (ver `core.gate_dropout`).
    seg : objeto de segmentación
        Necesita `mask` (n_slices,H,W), `center_per_slice` (n_slices,2) y
        `outer_radius` (n_slices,).
    pixel_mm : (dy, dx) tamaño de píxel en el plano, en mm.
    slice_mm : separación entre cortes, en mm.
    config : ECTbLVConfig | None

    Returns
    -------
    ECTbLVResult
    """
    cfg = config or ECTbLVConfig()
    res = ECTbLVResult(config=cfg)

    arr = np.asarray(cube, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[0] < 3:
        res.reason = f"Se necesita un gated 4D con >=3 gates; se recibió {arr.shape}."
        return res

    n_gates, n_slices, height, width = arr.shape
    centers_all = np.asarray(getattr(seg, "center_per_slice", np.empty((0, 2))), dtype=np.float64)
    outer_all = np.asarray(getattr(seg, "outer_radius", np.empty((0,))), dtype=np.float64)
    mask_all = np.asarray(getattr(seg, "mask", np.empty((0,))), dtype=bool)
    if centers_all.shape[0] != n_slices or outer_all.shape[0] != n_slices or mask_all.shape[0] != n_slices:
        res.reason = "La segmentación no coincide con las dimensiones del estudio."
        return res

    valid = _valid_slices(mask_all, outer_all, centers_all, cfg.min_myo_area_frac)
    if len(valid) < max(3, n_slices // 4):
        res.reason = f"Solo {len(valid)} cortes válidos: insuficiente para integrar el volumen."
        return res

    px_mm = float(np.mean([abs(float(pixel_mm[0])), abs(float(pixel_mm[1]))]))
    dz_mm = abs(float(slice_mm))
    if px_mm <= 0.0 or dz_mm <= 0.0:
        res.reason = "Spacing inválido: no se puede convertir píxeles a mm."
        return res

    centers = centers_all[valid]
    outer = outer_all[valid]
    n_ang = max(8, int(cfg.n_angles))
    angles = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
    sin_a = np.sin(angles)
    cos_a = np.cos(angles)

    r_max_px = float(np.max(outer) * float(cfg.radial_extent_frac))
    step = 1.0 / max(1.0, float(cfg.radial_oversample))
    r_line = np.arange(0.0, r_max_px + step, step)
    if r_line.size < 5:
        res.reason = "Radio externo demasiado chico para trazar perfiles."
        return res

    # --- Paso 1: línea de centro miocárdico (máximo de cuentas por rayo) ------
    center_r_px = np.empty((n_gates, len(valid), n_ang), dtype=np.float64)
    peak_counts = np.empty_like(center_r_px)
    for g in range(n_gates):
        profiles = _sample_radial_profiles(arr[g][valid], centers, r_line, sin_a, cos_a)
        center_r_px[g], peak_counts[g] = _subpixel_peak(profiles, r_line)

    center_r_mm = center_r_px * px_mm

    # --- Paso 2-3: espesor de pared por gate ---------------------------------
    ed_idx_cfg = int(np.clip(cfg.ed_gate_index, 0, n_gates - 1))
    if cfg.use_thickening:
        # Las cuentas máximas son proporcionales al espesor mientras la pared sea
        # más fina que ~2xFWHM (volumen parcial). Se ajusta el 1er armónico para
        # quitar ruido antes de derivar el espesor.
        fitted = _first_harmonic_fit(peak_counts)
        ref = fitted[ed_idx_cfg]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(ref > 0.0, fitted / ref[None, ...], 1.0)
        ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
        ratio = np.clip(ratio, MIN_THICKNESS_FRAC, MAX_THICKNESS_FRAC)
    else:
        ratio = np.ones_like(center_r_mm)
        res.notes.append("Engrosamiento desactivado: espesor constante en todos los gates.")

    thickness_mm = float(cfg.ed_wall_thickness_mm) * ratio

    # --- Paso 4: endocardio/epicardio y suavizado por mediana ----------------
    endo_mm = center_r_mm - 0.5 * thickness_mm
    epi_mm = center_r_mm + 0.5 * thickness_mm
    endo_mm = np.clip(endo_mm, 0.0, None)

    for kernel in (cfg.median_kernel_large, cfg.median_kernel_small):
        endo_mm = _median_smooth_radii(endo_mm, kernel)
        epi_mm = _median_smooth_radii(epi_mm, kernel)
    epi_mm = np.maximum(epi_mm, endo_mm)

    # --- Paso 5: plano valvular y volúmenes ----------------------------------
    endo_wedges = _wedge_areas(endo_mm, n_ang)              # (n_gates, n_valid, n_ang)
    epi_wedges = _wedge_areas(epi_mm, n_ang)

    base_last = _base_is_last(seg, valid)
    if cfg.use_valve_plane and float(cfg.valve_septal_offset_mm) > 0.0:
        weights, u_cut = _valve_plane_weights(
            len(valid), angles, dz_mm, cfg.septal_angle_deg, cfg.valve_septal_offset_mm, base_last
        )
    else:
        weights = np.ones((len(valid), n_ang), dtype=np.float64)
        u_cut = np.full(n_ang, float(len(valid)) - 0.5)

    endo_area_mm2 = (endo_wedges * weights[None, ...]).sum(axis=-1)   # (n_gates, n_valid)
    epi_area_mm2 = (epi_wedges * weights[None, ...]).sum(axis=-1)
    gate_volumes_ml = endo_area_mm2.sum(axis=1) * dz_mm / 1000.0
    myo_volume_ml = float(np.mean((epi_area_mm2 - endo_area_mm2).sum(axis=1) * dz_mm / 1000.0))

    # Cuánto sacó el plano valvular, para poder auditarlo desde la UI.
    gate_volumes_flat_ml = endo_wedges.sum(axis=-1).sum(axis=1) * dz_mm / 1000.0

    if not np.isfinite(gate_volumes_ml).all() or gate_volumes_ml.max() <= 0.0:
        res.reason = "Los volúmenes calculados no son válidos."
        return res

    ed_idx = int(np.argmax(gate_volumes_ml))
    es_idx = int(np.argmin(gate_volumes_ml))
    edv = float(gate_volumes_ml[ed_idx])
    esv = float(gate_volumes_ml[es_idx])

    mean_thickness = thickness_mm.reshape(n_gates, -1).mean(axis=1)
    thk_ed = float(mean_thickness[ed_idx])
    thk_es = float(mean_thickness[es_idx])
    thickening = float((thk_es - thk_ed) / thk_ed * 100.0) if thk_ed > 0.0 else 0.0
    # --- Índice de esfericidad -----------------------------------------------
    # Diámetro de eje corto: se promedia el radio endocárdico sobre los ángulos
    # (así una espiga aislada no define el diámetro) y se toma el corte más ancho,
    # que es el ecuador del ventrículo.
    short_axis_mm = 2.0 * endo_mm.mean(axis=2).max(axis=1)      # (n_gates,)
    # Eje largo: longitud efectiva de la cavidad. Se usa la suma de los pesos del
    # plano valvular para no contar como largo lo que el plano ya recortó.
    long_axis_mm = float(weights.mean(axis=1).sum()) * dz_mm

    res.available = True
    res.edv_ml = edv
    res.esv_ml = esv
    res.sv_ml = float(edv - esv)
    res.ef_pct = float((edv - esv) / edv * 100.0)
    res.ed_gate = ed_idx + 1
    res.es_gate = es_idx + 1
    res.gate_volumes_ml = gate_volumes_ml
    res.myocardial_volume_ml = myo_volume_ml
    res.myocardial_mass_g = float(myo_volume_ml * float(cfg.myocardial_density_g_ml))
    res.mean_wall_thickness_mm = mean_thickness
    res.thickening_pct = thickening
    res.wall_thickness_ed_mm = thk_ed
    res.wall_thickness_es_mm = thk_es
    res.long_axis_mm = long_axis_mm
    res.short_axis_ed_mm = float(short_axis_mm[ed_idx])
    res.short_axis_es_mm = float(short_axis_mm[es_idx])
    if long_axis_mm > 0.0:
        res.shape_index_ed = float(short_axis_mm[ed_idx] / long_axis_mm)
        res.shape_index_es = float(short_axis_mm[es_idx] / long_axis_mm)
    res.valid_slices = tuple(int(s) for s in valid)
    res.apex_to_base_slices = tuple(int(s) for s in (valid if base_last else valid[::-1]))
    res.valve_cut_u = u_cut
    res.valve_removed_ml = float(gate_volumes_flat_ml[ed_idx] - edv)
    res.n_slices_total = int(n_slices)
    res.center_radii_mm = center_r_mm
    res.endo_radii_mm = endo_mm
    res.epi_radii_mm = epi_mm

    if ed_idx != ed_idx_cfg:
        res.notes.append(
            f"El volumen máximo cayó en el gate {ed_idx + 1}, no en el gate {ed_idx_cfg + 1} "
            "que se usó como referencia de espesor telediastólico."
        )
    if res.valve_removed_ml > 0.0 and gate_volumes_flat_ml[ed_idx] > 0.0:
        pct = res.valve_removed_ml / float(gate_volumes_flat_ml[ed_idx]) * 100.0
        res.notes.append(
            f"Plano valvular de dos piezas: descontó {res.valve_removed_ml:.1f} mL del EDV "
            f"({pct:.1f}%) en el lado septal."
        )
    elif not cfg.use_valve_plane:
        res.notes.append("Plano valvular desactivado: la base se corta con un plano perpendicular.")
    if thickening < 0.0:
        res.notes.append(
            "El engrosamiento sistólico dio negativo: revisá la segmentación o el gating "
            "(en un ventrículo normal la pared engrosa en sístole)."
        )
    return res


def apply_regression(ef_fraction: float, slope: float, intercept: float) -> float:
    """Aplica una regresión lineal de conversión entre softwares.

    ECTb publica regresiones para expresar su FEVI en la escala de otros
    paquetes (QGS, MUGA). Se trabaja con la FEVI como **fracción 0-1**, que es
    la convención de las ecuaciones publicadas.
    """
    return float(slope) * float(ef_fraction) + float(intercept)


# ----------------------------------------------------------------------------
# Conversión de la FEVI a la escala de otros softwares
# ----------------------------------------------------------------------------
#
# POR QUÉ HACE FALTA
# ------------------
# Cada paquete de cuantificación pone el borde endocárdico en un lugar
# ligeramente distinto, así que la misma adquisición da FEVI distintas según el
# software. No es un error de ninguno: son escalas diferentes. El ECTb, por
# ejemplo, corre alrededor de 7-8 puntos por encima del QGS de 8 gates.
#
# Esto importa cuando el paciente trae un estudio previo informado con otro
# equipo: comparar 58% de ECTb contra 51% de QGS y concluir que "empeoró" es un
# error de lectura, no un hallazgo. Las regresiones publicadas permiten expresar
# nuestro resultado en la escala del informe anterior para compararlos de igual
# a igual.
#
# OJO CON LAS UNIDADES: las ecuaciones no están todas en la misma escala. Las de
# QGS-8 y MUGA se publican con la FEVI como fracción 0-1; la de QGS-16, en
# porcentaje. Por eso cada regresión declara su unidad en vez de asumirla.


@dataclass(frozen=True)
class EFRegression:
    """Regresión lineal publicada para convertir la FEVI a otra escala."""

    key: str
    label: str
    slope: float
    intercept: float
    #: "fraction" si la ecuación trabaja con 0-1, "percent" si trabaja con 0-100.
    units: str
    note: str = ""


#: Regresiones publicadas por el Emory Cardiac Toolbox.
EF_REGRESSIONS: dict[str, EFRegression] = {
    "qgs8": EFRegression(
        key="qgs8",
        label="QGS — 8 gates",
        slope=0.96,
        intercept=-0.053,
        units="fraction",
        note="ECTb corre alrededor de 7-8 puntos por encima del QGS de 8 gates.",
    ),
    "muga": EFRegression(
        key="muga",
        label="MUGA (ventriculografía radioisotópica)",
        slope=1.22,
        intercept=-0.072,
        units="fraction",
        note="Pendiente > 1: la diferencia entre ambos crece a FEVI altas.",
    ),
    "qgs16": EFRegression(
        key="qgs16",
        label="QGS — 16 gates",
        slope=0.855,
        intercept=1.73,
        units="percent",
        note="Ecuación publicada en porcentaje, no en fracción.",
    ),
}


def convert_ef_pct(ef_pct: float, regression: EFRegression | str) -> float:
    """Convierte una FEVI en porcentaje a la escala de otro software.

    Se encarga de la conversión de unidades: recibe y devuelve **porcentaje**,
    sin importar en qué escala esté publicada la ecuación.

    Raises
    ------
    KeyError
        Si se pasa una clave de regresión que no existe.
    """
    reg = EF_REGRESSIONS[regression] if isinstance(regression, str) else regression
    value = float(ef_pct)
    if reg.units == "fraction":
        return (reg.slope * (value / 100.0) + reg.intercept) * 100.0
    return reg.slope * value + reg.intercept


def regression_equation_text(regression: EFRegression | str) -> str:
    """Ecuación legible, con la unidad explícita para que no se malinterprete."""
    reg = EF_REGRESSIONS[regression] if isinstance(regression, str) else regression
    unit = "fracción 0-1" if reg.units == "fraction" else "porcentaje 0-100"
    sign = "+" if reg.intercept >= 0 else "−"
    return f"y = {reg.slope:g}·x {sign} {abs(reg.intercept):g}   (x en {unit})"


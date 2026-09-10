"""SINCRO - core.lv_center

Centro de la cavidad del ventrículo izquierdo.

EL PROBLEMA
-----------
Casi todo el módulo venía tomando como "centro del VI" el **centroide de la
máscara de miocardio** (``center_of_mass(mask)``). Eso es el centro de masa del
*músculo*, no el centro del *hueco*, y los dos solo coinciden si la captación
fuese perfectamente uniforme en los 360°. En la práctica nunca lo es: si la
pared lateral capta más que el septum, o hay un defecto, o hay un foco
hepato-intestinal pegado al anillo, el centroide se corre hacia el lado
caliente.

POR QUÉ IMPORTA
---------------
El centro no es un detalle de dibujo, es el origen del sistema de coordenadas
polares de todo el análisis:

- Los radios del ROI circular se miden desde ahí (centro corrido → radio
  inflado de un lado y corto del otro).
- El ECTb tira sus perfiles radiales desde ahí (centro corrido → los rayos
  cruzan la pared en diagonal de un lado y el máximo de cuentas queda mal
  ubicado, con lo que endo/epi salen deformados).
- El ángulo de cada voxel se mide desde ahí, y con el ángulo se asignan los 17
  segmentos AHA y los territorios coronarios (centro corrido → la fase se
  reparte a segmentos que no le corresponden).

CÓMO SE RESUELVE ACÁ
--------------------
Dos caminos, en orden de confiabilidad:

1. **Geométrico.** Si la máscara del anillo encierra un hueco (lo que
   ``binary_fill_holes`` rellena y el anillo no ocupa), el centroide de ese
   hueco *es* el centro de la cavidad. Es el camino preferido porque no depende
   de intensidades.

2. **Por hipocaptación.** En matrices chicas (22×22 es lo habitual) la cavidad
   muchas veces no llega a cerrar un hueco de píxeles enteros. Entonces se la
   busca por contraste: se pondera cada píxel por cuánto está *por debajo* del
   nivel de la pared, dentro de un disco alrededor del centro previo y con un
   prior gaussiano para no escaparse al fondo negro de la imagen.

Si ninguno de los dos aplica, se devuelve el centro de entrada sin tocar: la
función nunca empeora el punto de partida.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes, center_of_mass

#: Fracción del radio externo dentro de la cual se busca la cavidad. Acota la
#: búsqueda al interior del anillo para no engancharse con el fondo.
SEARCH_DISK_FRAC = 0.74

#: Fracción del radio externo que define la región usada para estimar el nivel
#: de pared. Tiene que abarcar el anillo completo: si se midiera sobre toda la
#: imagen, el fondo negro (que es la mayor parte del campo en SPECT) se llevaría
#: el percentil y el nivel de pared saldría 0.
WALL_REGION_FRAC = 1.25

#: Percentil, dentro de la región de pared, que se toma como "nivel de pared".
#: Todo lo que esté por debajo cuenta como candidato a cavidad.
WALL_LEVEL_PCT_LOW_RES = 78.0
WALL_LEVEL_PCT = 72.0

#: Mínimo de píxeles para que un hueco geométrico se considere cavidad real.
MIN_CAVITY_PIXELS = 3

#: Iteraciones del refinamiento por hipocaptación. Cada pasada recentra el disco
#: de búsqueda, así que el centro converge en vez de quedarse a medio camino.
MAX_REFINE_ITERS = 4

#: Desplazamiento por debajo del cual se considera convergido, en píxeles.
CONVERGENCE_PX = 0.05

#: Nº de direcciones angulares que se muestrean alrededor del candidato para
#: verificar que esté rodeado de pared (guard anti-fondo).
ANGULAR_COVERAGE_BINS = 16

#: Fracción mínima de direcciones angulares que tienen que encontrar pared para
#: aceptar el candidato como cavidad real. Una cavidad cerrada da ~1.0; media
#: pared (herradura de base/ápex, caso legítimo) da ~0.5; un candidato que se
#: descolgó al fondo negro ve la banda miocárdica bajo un ángulo chico y cae por
#: debajo de este umbral. Se deja en 1/3 para aceptar con holgura la media pared
#: y rechazar solo los descuelgues francos hacia el fondo.
ANGULAR_COVERAGE_MIN = 0.34


def cavity_center_from_mask(mask: np.ndarray) -> tuple[float, float] | None:
    """Centro del hueco encerrado por el anillo, o None si no hay hueco.

    Camino preferido: no depende de intensidades, solo de la topología de la
    máscara. Falla legítimamente cuando el anillo está abierto (base) o cuando
    la cavidad es más chica que un píxel (ápex, matrices chicas).
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return None
    cavity = binary_fill_holes(mask) & (~mask)
    if int(np.count_nonzero(cavity)) < MIN_CAVITY_PIXELS:
        return None
    cy, cx = center_of_mass(cavity)
    if not (np.isfinite(cy) and np.isfinite(cx)):
        return None
    return float(cy), float(cx)


def _wall_angular_coverage(
    img: np.ndarray,
    cy: float,
    cx: float,
    r_outer: float,
    wall_level: float,
    *,
    n_bins: int = ANGULAR_COVERAGE_BINS,
) -> float:
    """Fracción de direcciones angulares alrededor de (cy, cx) que ven pared.

    Dispara ``n_bins`` rayos desde el candidato y, en cada uno, recorre el anillo
    ``[0.30·r_outer, WALL_REGION_FRAC·r_outer]`` buscando algún píxel a nivel de
    pared. Una cavidad real está rodeada de pared en (casi) todas las
    direcciones; el fondo negro pegado a una banda miocárdica (base/ápex sin
    cavidad cerrada) solo tiene pared en un arco chico. Sirve para distinguir un
    centro de cavidad legítimo de un candidato descolgado al fondo.
    """
    height, width = img.shape
    r_min = max(1.5, 0.30 * float(r_outer))
    r_max = max(r_min + 2.0, WALL_REGION_FRAC * float(r_outer))
    n_steps = max(4, int(np.ceil(r_max - r_min)) + 1)
    radii = np.linspace(r_min, r_max, n_steps)
    covered = 0
    for b in range(int(n_bins)):
        theta = 2.0 * np.pi * b / float(n_bins)
        dy = np.sin(theta)
        dx = np.cos(theta)
        iy = np.round(cy + radii * dy).astype(np.intp)
        ix = np.round(cx + radii * dx).astype(np.intp)
        valid = (iy >= 0) & (iy < height) & (ix >= 0) & (ix < width)
        if not np.any(valid):
            continue
        if np.any(img[iy[valid], ix[valid]] >= wall_level):
            covered += 1
    return covered / float(n_bins)


def cavity_center_from_image(
    img: np.ndarray,
    cy: float,
    cx: float,
    r_outer: float,
    *,
    low_res: bool = False,
) -> tuple[float, float] | None:
    """Centro de la cavidad por hipocaptación, o None si no se pudo estimar.

    Pondera cada píxel por cuánto está por debajo del nivel de pared, restringe
    la búsqueda a un disco interior y aplica un prior gaussiano para no derivar
    hacia el fondo. Se itera unas pocas veces: cada pasada recentra el disco de
    búsqueda sobre el candidato anterior, de modo que el centro converge en vez
    de quedarse a mitad de camino cuando el punto de partida venía muy corrido.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2 or img.size == 0:
        return None
    if not (np.isfinite(cy) and np.isfinite(cx) and np.isfinite(r_outer)) or r_outer <= 0.0:
        return None
    if not np.isfinite(img).any():
        return None

    height, width = img.shape
    ys, xs = np.ogrid[:height, :width]
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    wall_pct = WALL_LEVEL_PCT_LOW_RES if low_res else WALL_LEVEL_PCT
    search_radius = max(2.0, SEARCH_DISK_FRAC * float(r_outer))
    wall_radius = max(3.0, WALL_REGION_FRAC * float(r_outer))
    sigma = max(2.0, 0.60 * float(r_outer))

    cur_cy = float(cy)
    cur_cx = float(cx)
    moved = False

    for _ in range(MAX_REFINE_ITERS):
        dist = np.sqrt((ys - cur_cy) ** 2 + (xs - cur_cx) ** 2)

        wall_region = dist <= wall_radius
        if int(np.count_nonzero(wall_region)) < 12:
            break
        wall_level = float(np.percentile(img[wall_region], wall_pct))
        if not np.isfinite(wall_level) or wall_level <= 0.0:
            # Región sin señal: no hay contraste del que deducir una cavidad.
            break

        search = dist <= search_radius
        if int(np.count_nonzero(search)) < 8:
            break

        below_wall = np.clip(wall_level - img, 0.0, None)
        prior = np.exp(-0.5 * (dist / sigma) ** 2)
        weights = np.where(search, below_wall * prior, 0.0)
        total = float(np.sum(weights))
        if total <= 1e-8:
            break

        new_cy = float(np.sum(weights * ys) / total)
        new_cx = float(np.sum(weights * xs) / total)
        if not (np.isfinite(new_cy) and np.isfinite(new_cx)):
            break

        shift = float(np.hypot(new_cy - cur_cy, new_cx - cur_cx))
        cur_cy, cur_cx = new_cy, new_cx
        moved = True
        if shift < CONVERGENCE_PX:
            break

    if not moved:
        return None

    # Guard anti-fondo: en base/ápex/cortes apaisados el miocardio es una banda
    # abierta (herradura) sin cavidad cerrada, y la mayor masa "por debajo del
    # nivel de pared" es el fondo negro pegado a la banda. El centro se descuelga
    # ahí y deforma el ROI. Se acepta el candidato solo si está rodeado de pared
    # en una fracción mínima de direcciones angulares; si no, se descarta y el
    # llamador conserva el centro de entrada (o el hueco geométrico).
    final_dist = np.sqrt((ys - cur_cy) ** 2 + (xs - cur_cx) ** 2)
    wall_region = final_dist <= wall_radius
    if int(np.count_nonzero(wall_region)) >= 12:
        wall_level = float(np.percentile(img[wall_region], wall_pct))
        if np.isfinite(wall_level) and wall_level > 0.0:
            coverage = _wall_angular_coverage(img, cur_cy, cur_cx, float(r_outer), wall_level)
            if coverage < ANGULAR_COVERAGE_MIN:
                return None

    return cur_cy, cur_cx


def refine_center_to_cavity(
    cy: float,
    cx: float,
    r_outer: float,
    *,
    mask: np.ndarray | None = None,
    img: np.ndarray | None = None,
    low_res: bool = False,
    max_shift_px: float | None = None,
) -> tuple[float, float]:
    """Mueve (cy, cx) del centroide del músculo al centro de la cavidad.

    El hueco geométrico sirve como punto de partida cuando existe, pero no
    alcanza por sí solo: ese hueco sale de una máscara umbralizada, y el umbral
    se come más píxeles del lado frío que del caliente, así que el hueco queda
    corrido hacia el lado frío (justo al revés que el centroide del músculo,
    que se corre hacia el caliente). Por eso el refinamiento final se hace
    siempre sobre la imagen, que no depende del umbral.

    Nunca devuelve algo peor que la entrada: si ningún camino aplica, o si el
    desplazamiento propuesto es implausible, se conserva el centro original.

    Parameters
    ----------
    cy, cx : centro de partida (típicamente el centroide de la máscara).
    r_outer : radio externo del ROI, en píxeles. Acota búsqueda y salto.
    mask : máscara del anillo miocárdico, para inicializar sobre el hueco.
    img : imagen del corte, para el refinamiento por hipocaptación.
    low_res : True en matrices chicas (22×22 y similares).
    max_shift_px : salto máximo tolerado respecto del centro de entrada. Por
        defecto, el radio externo completo. No conviene apretarlo más: el
        desplazamiento real ya está acotado por el disco de búsqueda
        (``SEARCH_DISK_FRAC``), y un tope chico rechaza justamente los casos
        graves, que son los que más hay que corregir — con media pared visible
        el centroide del músculo se corre más que medio radio.

    Returns
    -------
    (cy, cx) refinado, o el de entrada si no se pudo mejorar.
    """
    if not (np.isfinite(cy) and np.isfinite(cx)):
        return float(cy), float(cx)

    start = cavity_center_from_mask(mask) if mask is not None else None
    if start is None:
        start = (float(cy), float(cx))

    candidate = None
    if img is not None:
        candidate = cavity_center_from_image(
            img, start[0], start[1], float(r_outer), low_res=low_res
        )
    if candidate is None:
        # Sin imagen utilizable queda el hueco geométrico, si lo hubo.
        candidate = start
    if candidate == (float(cy), float(cx)):
        return float(cy), float(cx)

    new_cy, new_cx = candidate
    limit = float(max_shift_px) if max_shift_px is not None else float(r_outer)
    if np.isfinite(limit) and limit > 0.0:
        shift = float(np.hypot(new_cy - float(cy), new_cx - float(cx)))
        if shift > limit:
            # Salto implausible (foco extracardíaco, anillo roto): no se mueve.
            return float(cy), float(cx)
    return float(new_cy), float(new_cx)


# ---------------------------------------------------------------------------
# Banda axial del corazón desde el SPECT (sin CT)
# ---------------------------------------------------------------------------

#: Fracción del pico del perfil axial que separa "banda cardíaca" de fondo.
#: El corazón es la estructura más caliente del FOV torácico en un MIBI/tetro;
#: por debajo de esta fracción del máximo se considera fuera de la banda.
HEART_AXIAL_LEVEL_FRAC = 0.35

#: Fracción del máximo por debajo de la cual un valle entre dos jorobas cuenta
#: como separación real (corazón vs hígado). Un valle poco profundo NO parte la
#: banda (es la propia caída base→ápex del miocardio).
HEART_AXIAL_VALLEY_FRAC = 0.55

#: Umbral para el perfil por LATIDO (amplitud temporal). Solo el miocardio late,
#: así que el fondo es casi nulo y se puede exigir un umbral más alto que en el
#: perfil de intensidad (donde el hígado inflaba la base).
HEART_AXIAL_BEAT_LEVEL_FRAC = 0.30

#: Nº mínimo de gates para intentar el perfil por latido (bajo esto la FFT/rango
#: temporal no es confiable; se cae al perfil de intensidad).
HEART_MIN_GATES_FOR_BEAT = 3

#: Fracción de modulación mínima (|F1|/DC) para contar un voxel como pulsátil.
#: El miocardio ronda 0.15–0.40; el hígado/arrastre global queda por debajo.
BEAT_FRACTION_MIN = 0.12

#: Un voxel se considera "con señal" si su DC supera esta fracción del DC de
#: referencia (percentil 60 de los voxels con señal). Descarta el fondo, donde
#: |F1|/DC explota por ruido.
BEAT_DC_MIN_FRAC = 0.35


def _bounds_from_axial_profile(
    profile: np.ndarray,
    *,
    level_frac: float,
    valley_frac: float,
    margin: int,
) -> tuple[int, int] | None:
    """Banda contigua [lo, hi] (0-based inclusivo) desde un perfil axial 1D.

    Umbral a ``level_frac``·pico; toma la banda que contiene el pico global y
    fusiona vecinas separadas por un valle poco profundo (>= ``valley_frac``·pico
    = caída base→ápex del propio miocardio, no otra víscera). Añade ``margin`` y
    clampea. Devuelve None si el perfil no tiene señal.
    """
    profile = np.asarray(profile, dtype=np.float64).ravel()
    n = int(profile.shape[0])
    if n < 3:
        return None
    peak = float(profile.max())
    if peak <= 0.0:
        return None

    above = profile >= float(level_frac) * peak
    if not above.any():
        return None

    bands: list[tuple[int, int]] = []
    start = None
    for i in range(n):
        if above[i] and start is None:
            start = i
        elif not above[i] and start is not None:
            bands.append((start, i - 1))
            start = None
    if start is not None:
        bands.append((start, n - 1))
    if not bands:
        return None

    peak_idx = int(np.argmax(profile))
    heart_band = next((b for b in bands if b[0] <= peak_idx <= b[1]), bands[0])
    lo, hi = heart_band

    valley_level = float(valley_frac) * peak
    for b in bands:
        if b == heart_band:
            continue
        gap_lo, gap_hi = (hi + 1, b[0] - 1) if b[0] > hi else (b[1] + 1, lo - 1)
        if gap_lo > gap_hi:
            lo, hi = min(lo, b[0]), max(hi, b[1])
            continue
        if float(profile[gap_lo:gap_hi + 1].min()) >= valley_level:
            lo, hi = min(lo, b[0]), max(hi, b[1])

    lo = int(np.clip(lo - int(margin), 0, n - 1))
    hi = int(np.clip(hi + int(margin), 0, n - 1))
    if hi <= lo:
        return None
    return lo, hi


def _beat_amplitude_axial_profile(gated: np.ndarray) -> np.ndarray | None:
    """Perfil axial de LATIDO desde un volumen gated (g, K, H, W).

    La amplitud absoluta ``|F1|`` no basta: con movimiento del paciente TODO el
    volumen se traslada entre gates, y el borde de una estructura brillante que
    se desplaza (hígado, FOV) da ``|F1|`` alto sin ser miocardio. La fracción de
    modulación ``|F1|/DC`` ayuda pero tampoco alcanza: un voxel de borde que pasa
    de brillante a oscuro por la traslación también modula fuerte.

    El discriminador que sí separa corazón de hígado es la **coherencia de
    fase**. La contracción cardíaca es un cambio de volumen COHERENTE: todos los
    voxels del miocardio laten (engrosan/adelgazan) en fase, así que sus primeros
    armónicos ``F1`` (complejos) apuntan en la misma dirección y **suman
    constructivamente**. La traslación de una estructura rígida, en cambio, hace
    que los bordes opuestos laten en ANTIFASE (uno se aclara mientras el otro se
    oscurece): sus ``F1`` apuntan en direcciones opuestas y **se cancelan** al
    sumarlos.

    Por eso el perfil suma el ``F1`` COMPLEJO por corte y toma el módulo:
      - Corazón (en fase)  → ``|Σ F1|`` ≈ ``Σ |F1|``  (no se cancela).
      - Hígado (antifase) → ``|Σ F1|`` ≪ ``Σ |F1|``  (se cancela).
    Se enmascara antes a voxels con señal (DC no despreciable) y con fracción de
    modulación mínima, para no arrastrar ruido de fondo.

    Devuelve la suma coherente por corte, o ``None`` si no hay latido detectable.
    """
    arr = np.asarray(gated, dtype=np.float64)
    if arr.ndim != 4 or arr.shape[0] < HEART_MIN_GATES_FOR_BEAT:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    g, K, H, W = arr.shape

    spectrum = np.fft.fft(arr, axis=0)
    dc = np.abs(spectrum[0]) / float(g)          # media temporal por voxel
    f1 = spectrum[1] * (2.0 / float(g))          # 1er armónico COMPLEJO por voxel
    amp = np.abs(f1)                             # |F1|

    # Voxels con señal real (evita arrastrar ruido de fondo con DC≈0).
    dc_ref = float(np.percentile(dc[dc > 0], 60)) if np.any(dc > 0) else 0.0
    if dc_ref <= 0.0:
        return None
    signal = dc >= (BEAT_DC_MIN_FRAC * dc_ref)

    frac = np.zeros_like(dc)
    np.divide(amp, dc, out=frac, where=signal & (dc > 0))
    mask = signal & (frac >= BEAT_FRACTION_MIN)
    f1_masked = f1 * mask                        # anula fondo/baja modulación

    # Suma COHERENTE por corte: la contracción en fase sobrevive, la traslación
    # en antifase se cancela.
    profile = np.abs(f1_masked.reshape(K, -1).sum(axis=1))
    if float(profile.max()) <= 0.0:
        return None
    return profile


def heart_axial_bounds_from_spect(
    volume: np.ndarray,
    *,
    level_frac: float = HEART_AXIAL_LEVEL_FRAC,
    margin: int = 1,
) -> tuple[int, int] | None:
    """Estima la banda axial [z_base, z_apex] del corazón desde el SPECT.

    Sin SPECT-CT no hay máscara anatómica para ubicar los límites Base/Ápex de
    la feta; hay que deducirlos del propio volumen reconstruido.

    Dos caminos, en orden de confiabilidad:

    1. **Por latido** (si ``volume`` es gated 4D con ≥3 gates). El corazón es la
       única estructura que se contrae; el perfil de amplitud temporal (1er
       armónico FFT) enciende el miocardio y apaga el hígado/intestino/fondo, así
       que la banda sale ajustada incluso con contaminación caudal fuerte.
    2. **Por intensidad** (fallback: volumen estático o 1 gate). El corazón es la
       estructura compacta más caliente → joroba en el perfil de cuentas por
       corte. Puede estirarse hacia el hígado caudal, por eso es el fallback.

    En ambos casos: umbral relativo al pico → banda(s) contigua(s) → se toma la
    del pico global fusionando vecinas separadas por valles poco profundos →
    ``margin`` a cada lado → clamp.

    ``volume`` puede ser 3D ``(K, H, W)`` o 4D ``(g, K, H, W)``. Devuelve índices
    **0-based** ``(z_lo, z_hi)`` inclusivos, o ``None`` si no se pudo estimar (el
    llamador conserva su punto de partida, nunca lo empeora).
    """
    arr = np.asarray(volume, dtype=np.float64)

    # Camino 1: perfil por latido (gated). Umbral propio, más alto (fondo ~0).
    beat_profile = _beat_amplitude_axial_profile(arr) if arr.ndim == 4 else None
    if beat_profile is not None:
        bounds = _bounds_from_axial_profile(
            beat_profile,
            level_frac=HEART_AXIAL_BEAT_LEVEL_FRAC,
            valley_frac=HEART_AXIAL_VALLEY_FRAC,
            margin=margin,
        )
        if bounds is not None:
            return bounds

    # Camino 2 (fallback): perfil de intensidad (suma de gates si es 4D).
    if arr.ndim == 4:
        arr = arr.sum(axis=0)
    if arr.ndim != 3 or arr.shape[0] < 3:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    profile = arr.reshape(arr.shape[0], -1).sum(axis=1)
    return _bounds_from_axial_profile(
        profile,
        level_frac=float(level_frac),
        valley_frac=HEART_AXIAL_VALLEY_FRAC,
        margin=margin,
    )

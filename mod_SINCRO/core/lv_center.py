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

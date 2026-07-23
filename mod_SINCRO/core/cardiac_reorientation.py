"""Reorientacion oblicua cardiaca (estilo Rec/Ref Xeleris / Odyssey).

Toma un volumen reconstruido transaxial (no reorientado) y lo reslicea a lo
largo del eje largo del VI definido por el usuario (azimut + elevacion + centro),
produciendo un volumen alineado a eje corto (SA) del que derivan HLA y VLA.

Convenciones de ejes del volumen de entrada: ``vol[z, y, x]``.
El volumen reorientado de salida usa ``out[k, j, i]`` donde:
- ``k`` (axis 0) recorre el eje largo del VI (base -> apex),
- ``j`` (axis 1) y ``i`` (axis 2) forman el plano de eje corto (SA).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - scipy siempre presente en el entorno del proyecto
    from scipy.ndimage import affine_transform
except Exception:  # pragma: no cover
    affine_transform = None


@dataclass
class ReorientationParams:
    """Parametros geometricos de la reorientacion oblicua.

    - ``center`` centro del VI en coordenadas de volumen ``(z, y, x)``.
    - ``theta`` azimut (rad): angulo del eje largo proyectado en el plano axial (y, x).
    - ``phi`` elevacion (rad): angulo del eje largo respecto del plano axial.
    - ``out_size`` lado del cubo reorientado de salida (cubo isotropico).
    """

    center: tuple[float, float, float]
    theta: float
    phi: float
    out_size: int


def long_axis_vector(theta: float, phi: float) -> np.ndarray:
    """Vector unitario del eje largo en coordenadas ``(z, y, x)``.

    ``theta`` gira en el plano axial (y, x); ``phi`` inclina fuera del plano axial.
    """
    cz = np.sin(phi)
    cy = np.cos(phi) * np.sin(theta)
    cx = np.cos(phi) * np.cos(theta)
    u = np.array([cz, cy, cx], dtype=np.float64)
    n = np.linalg.norm(u)
    return u / n if n > 0 else np.array([1.0, 0.0, 0.0])


def _basis_from_long_axis(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Base ortonormal (u, e_j, e_i) a partir del eje largo ``u`` en ``(z, y, x)``.

    El plano de eje corto (SA) se fija de forma **anatómicamente reproducible**
    anclándolo al eje axial (cráneo-caudal) del paciente, que en el volumen
    SPECT ``(z, y, x)`` corresponde al eje ``z`` de las slices transaxiales
    (``a = [1, 0, 0]``). Esto replica el convenio Cedars/Emory (Xeleris/Odyssey):
    la rotación del anillo SA NO depende del ángulo del gantry, por lo que
    HLA/VLA quedan siempre con la misma orientación anatómica y no se confunden.

    Devuelve ``(u, e_j, e_i)`` donde:
    - ``u`` = eje largo del VI (base→ápex),
    - ``e_j`` = dirección in-plane derivada del eje axial del paciente
      (eje "vertical" anatómico del corte: anterior/inferior),
    - ``e_i`` = tercera dirección ortogonal (eje "horizontal": septal/lateral).
    """
    u = np.asarray(u, dtype=np.float64)
    u = u / (np.linalg.norm(u) or 1.0)
    # Eje axial del paciente (cráneo-caudal) en coordenadas de volumen (z, y, x).
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    # Si el eje largo es casi paralelo al axial, usar el eje AP como respaldo
    # para evitar degeneración del producto cruz.
    if abs(float(np.dot(u, a))) > 0.98:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    # e_j = componente de 'a' perpendicular a u (Gram-Schmidt) → vertical anatómico
    # (anterior/inferior). Es PAR en u: invertir base↔ápex NO lo cambia.
    e_j = a - float(np.dot(a, u)) * u
    e_j = e_j / (np.linalg.norm(e_j) or 1.0)
    # e_i = tercera dirección ortogonal → horizontal anatómico (septal/lateral).
    # Se ANCLA su signo al eje x del paciente (izquierda/derecha) para que la
    # orientación septal↔lateral del anillo SA/HLA NO se invierta al cambiar el
    # signo del eje largo (base↔ápex). Sin este anclaje, e_i = u×e_j cambia de
    # signo con u y desalinea SA/HLA/VLA entre sí.
    e_i = np.cross(u, e_j)
    n_ei = np.linalg.norm(e_i)
    e_i = e_i / (n_ei or 1.0)
    x_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # eje x del paciente
    if float(np.dot(e_i, x_ref)) < 0:
        e_i = -e_i
        e_j = -e_j  # mantener terna derecha (u, e_j, e_i)
    return u, e_j, e_i


def reslice_oblique(volume: np.ndarray, params: ReorientationParams, order: int = 1) -> np.ndarray:
    """Reslicea ``volume`` (z, y, x) al marco de eje corto -> ``out[k, j, i]``.

    ``out[k]`` es un corte SA (perpendicular al eje largo). ``k`` recorre base->apex.
    """
    if affine_transform is None:
        raise RuntimeError("scipy.ndimage.affine_transform no disponible")
    vol = np.asarray(volume, dtype=np.float64)
    u = long_axis_vector(params.theta, params.phi)
    u, e_j, e_i = _basis_from_long_axis(u)
    # Columnas de M en orden de salida (k, j, i) expresadas en (z, y, x).
    M = np.stack([u, e_j, e_i], axis=1)
    n = int(params.out_size)
    out_shape = (n, n, n)
    oc = (np.array(out_shape, dtype=np.float64) - 1.0) / 2.0
    center = np.asarray(params.center, dtype=np.float64)
    offset = center - M @ oc
    return affine_transform(
        vol, M, offset=offset, output_shape=out_shape,
        order=order, mode="constant", cval=0.0,
    )


def reslice_oblique_gated(cube: np.ndarray, params: ReorientationParams, order: int = 1) -> np.ndarray:
    """Aplica :func:`reslice_oblique` por gate. ``cube`` es ``(g, z, y, x)``."""
    cube = np.asarray(cube, dtype=np.float64)
    gates = [reslice_oblique(cube[g], params, order=order) for g in range(cube.shape[0])]
    return np.stack(gates, axis=0)


def reslice_from_vector(
    volume: np.ndarray,
    center: tuple[float, float, float],
    long_axis: np.ndarray,
    out_size: int,
    order: int = 1,
) -> np.ndarray:
    """Reslicea ``volume`` (z, y, x) usando directamente un vector de eje largo.

    Pensado para el flujo Rec/Ref de dos vistas ortogonales (anterior + lateral):
    el eje largo del VI se define trazando su proyección en cada vista y de ahí se
    arma el vector 3D ``long_axis`` en coordenadas ``(z, y, x)``.
    """
    if affine_transform is None:
        raise RuntimeError("scipy.ndimage.affine_transform no disponible")
    vol = np.asarray(volume, dtype=np.float64)
    u = np.asarray(long_axis, dtype=np.float64)
    if np.linalg.norm(u) <= 0:
        u = np.array([1.0, 0.0, 0.0])
    u, e_j, e_i = _basis_from_long_axis(u)
    M = np.stack([u, e_j, e_i], axis=1)
    n = int(out_size)
    out_shape = (n, n, n)
    oc = (np.array(out_shape, dtype=np.float64) - 1.0) / 2.0
    c = np.asarray(center, dtype=np.float64)
    offset = c - M @ oc
    return affine_transform(
        vol, M, offset=offset, output_shape=out_shape,
        order=order, mode="constant", cval=0.0,
    )


def reslice_from_vector_gated(
    cube: np.ndarray,
    center: tuple[float, float, float],
    long_axis: np.ndarray,
    out_size: int,
    order: int = 1,
) -> np.ndarray:
    """Aplica :func:`reslice_from_vector` por gate. ``cube`` es ``(g, z, y, x)``."""
    cube = np.asarray(cube, dtype=np.float64)
    gates = [reslice_from_vector(cube[g], center, long_axis, out_size, order=order) for g in range(cube.shape[0])]
    return np.stack(gates, axis=0)



def default_center(volume: np.ndarray) -> tuple[float, float, float]:
    """Centro de masa (z, y, x) del volumen sobre un umbral robusto."""
    vol = np.asarray(volume, dtype=np.float64)
    mx = float(vol.max()) if vol.size else 0.0
    if mx <= 0.0:
        return tuple((np.array(vol.shape, dtype=np.float64) - 1.0) / 2.0)  # type: ignore[return-value]
    mask = vol > 0.35 * mx
    if not mask.any():
        return tuple((np.array(vol.shape, dtype=np.float64) - 1.0) / 2.0)  # type: ignore[return-value]
    idx = np.array(np.where(mask), dtype=np.float64)
    w = vol[mask]
    cz = float(np.average(idx[0], weights=w))
    cy = float(np.average(idx[1], weights=w))
    cx = float(np.average(idx[2], weights=w))
    return (cz, cy, cx)


def auto_orient_lv(gated_cube, ungated_volume=None):
    """Detecta el VI automáticamente usando el movimiento del gated SPECT.

    El miocardio se contrae/relaja a lo largo del ciclo cardíaco (alta
    variabilidad temporal entre gates), mientras que el hígado y otras
    estructuras permanecen estáticas (baja variabilidad). El mapa de
    desviación estándar temporal aísla así el VI del hígado adyacente.

    A partir de la máscara de movimiento se estima:
    - ``center`` (z, y, x): centro de masa ponderado por movimiento,
    - ``long_axis`` (z, y, x): autovector principal (PCA ponderado) = eje largo,
    - ``semiaxes`` (z, y, x): semiejes de la VOI elíptica que envuelve el VI,
    - ``half_length``: media longitud sugerida para la línea del eje largo.

    El signo base→ápex se resuelve heurísticamente (el ápex apunta hacia afuera
    del centro del cuerpo). Devuelve ``None`` si no hay gated utilizable.
    """
    if gated_cube is None:
        return None
    cube = np.asarray(gated_cube, dtype=np.float64)
    if cube.ndim != 4 or cube.shape[0] < 3:
        return None

    motion = cube.std(axis=0)  # (z, y, x): mapa de movimiento
    try:  # pragma: no cover - scipy presente en el entorno
        from scipy.ndimage import gaussian_filter, label as _label
    except Exception:  # pragma: no cover
        gaussian_filter = None
        _label = None
    if gaussian_filter is not None:
        motion = gaussian_filter(motion, sigma=1.0)

    mmax = float(motion.max())
    if mmax <= 0.0:
        return None
    mask = motion >= 0.35 * mmax
    if not mask.any():
        return None
    if _label is not None:
        lbl, nlab = _label(mask)
        if nlab > 1:
            sizes = np.bincount(lbl.ravel())
            sizes[0] = 0
            mask = lbl == int(sizes.argmax())

    zz, yy, xx = np.nonzero(mask)
    wv = motion[zz, yy, xx]
    sw = float(wv.sum())
    if sw <= 0.0:
        return None
    cz = float((zz * wv).sum() / sw)
    cy = float((yy * wv).sum() / sw)
    cx = float((xx * wv).sum() / sw)

    pts = np.stack([zz - cz, yy - cy, xx - cx], axis=1)
    cov = (pts.T * wv) @ pts / sw
    evals, evecs = np.linalg.eigh(cov)
    u = np.asarray(evecs[:, -1], dtype=np.float64)  # mayor autovalor = eje largo
    nrm = float(np.linalg.norm(u))
    u = u / (nrm if nrm > 0 else 1.0)

    # Signo base→ápex: el ápex del VI apunta hacia afuera del centro del cuerpo.
    if ungated_volume is not None:
        vol = np.asarray(ungated_volume, dtype=np.float64)
        tot = float(vol.sum())
        if tot > 0:
            zc = float((vol.sum(axis=(1, 2)) * np.arange(vol.shape[0])).sum() / tot)
            yc = float((vol.sum(axis=(0, 2)) * np.arange(vol.shape[1])).sum() / tot)
            xc = float((vol.sum(axis=(0, 1)) * np.arange(vol.shape[2])).sum() / tot)
            to_apex = np.array([cz - zc, cy - yc, cx - xc], dtype=np.float64)
            if float(np.dot(u, to_apex)) < 0:
                u = -u

    proj = pts @ u
    half = float(1.3 * proj.std())
    half = float(np.clip(half if half > 0 else 6.0, 5.0, 0.5 * max(cube.shape[1:])))
    rz = float(np.clip(1.6 * zz.std() + 3.0, 6.0, cube.shape[1]))
    ry = float(np.clip(1.6 * yy.std() + 3.0, 6.0, cube.shape[2]))
    rx = float(np.clip(1.6 * xx.std() + 3.0, 6.0, cube.shape[3]))

    return {
        "center": (cz, cy, cx),
        "long_axis": u,
        "semiaxes": (rz, ry, rx),
        "half_length": half,
    }


# ---------------------------------------------------------------------------
# Extracción de cortes anatómicos con convención Xeleris/Odyssey (Cedars/Emory)
# ---------------------------------------------------------------------------
#
# El volumen reorientado tiene ejes ``reo[k, j, i]``:
#   - ``k`` (axis 0): eje largo del VI, base -> ápex,
#   - ``j`` (axis 1): eje in-plane "vertical" anatómico (``e_j``, ancla axial),
#   - ``i`` (axis 2): eje in-plane "horizontal" anatómico (``e_i``).
#
# Convención de despliegue objetivo (paciente supino), IDÉNTICA a Xeleris 2 y
# Odyssey/ECToolbox, para que HLA y VLA NUNCA se confundan y la cara inferior
# sea inequívoca:
#   - SA  (short axis):     ANTERIOR arriba, INFERIOR abajo, SEPTAL izq, LATERAL der.
#                           La pila recorre ápex -> base.
#   - HLA (horizontal LA):  APEX arriba, BASE abajo, SEPTAL izq, LATERAL der.
#                           La pila recorre inferior -> anterior.
#   - VLA (vertical LA):    ANTERIOR arriba, INFERIOR abajo, BASE izq, APEX der.
#                           La pila recorre septal -> lateral.
#
# NOTA: los signos de flip asumen la base anatómica de ``_basis_from_long_axis``
# (e_j ~ anterior/inferior, e_i ~ septal/lateral, u = base->ápex). Si al validar
# contra los screenshots del usuario algún eje quedara espejado, se ajusta el
# flip del eje correspondiente aquí (un único lugar, consistente en toda la app).

# Flips/transposición por eje calibrados contra la convención Xeleris/Odyssey.
# ``transpose`` intercambia filas/columnas del corte 2D ANTES de los flips;
# ``row``/``col`` invierten el eje correspondiente después de transponer.
#
# SA  : plano (j, i).  Objetivo: ANT↑ INF↓ SEP← LAT→.
# HLA : plano (k, i).  Objetivo: APEX↑ BASE↓ SEP← LAT→ (eje largo VERTICAL).
# VLA : plano (k, j).  Objetivo: ANT↑ INF↓ BASE← APEX→ (eje largo HORIZONTAL),
#       forma de "C invertida" ⊂. Requiere TRANSPONER para pasar k (base→ápex)
#       de filas a columnas.
_SA_FLIP = {"transpose": False, "row": False, "col": False}
_HLA_FLIP = {"transpose": False, "row": True, "col": False}
_VLA_FLIP = {"transpose": True, "row": False, "col": False}


def _apply_2d_flips(img2d: np.ndarray, flips: dict) -> np.ndarray:
    out = np.asarray(img2d)
    if flips.get("transpose"):
        out = out.T
    if flips.get("row"):
        out = out[::-1, :]
    if flips.get("col"):
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def sa_slice(reo: np.ndarray, k: int) -> np.ndarray:
    """Corte SA (short axis) en el índice ``k`` del volumen reorientado ``reo[k,j,i]``."""
    reo = np.asarray(reo, dtype=np.float64)
    k = int(np.clip(k, 0, reo.shape[0] - 1))
    return _apply_2d_flips(reo[k], _SA_FLIP)


def hla_slice(reo: np.ndarray, j: int) -> np.ndarray:
    """Corte HLA (horizontal long axis) en el índice ``j`` de ``reo[k,j,i]``.

    Plano (k, i). Se orienta con APEX arriba / BASE abajo (fila = k invertido).
    """
    reo = np.asarray(reo, dtype=np.float64)
    j = int(np.clip(j, 0, reo.shape[1] - 1))
    return _apply_2d_flips(reo[:, j, :], _HLA_FLIP)


def vla_slice(reo: np.ndarray, i: int) -> np.ndarray:
    """Corte VLA (vertical long axis) en el índice ``i`` de ``reo[k,j,i]``.

    Plano (k, j). Se orienta con BASE izq / APEX der, ANTERIOR arriba.
    """
    reo = np.asarray(reo, dtype=np.float64)
    i = int(np.clip(i, 0, reo.shape[2] - 1))
    return _apply_2d_flips(reo[:, :, i], _VLA_FLIP)


def sa_stack(reo: np.ndarray) -> np.ndarray:
    """Pila SA completa ``(k, filas, cols)`` con orientación anatómica fija."""
    reo = np.asarray(reo, dtype=np.float64)
    return np.stack([sa_slice(reo, k) for k in range(reo.shape[0])], axis=0)


def anatomical_cuts_gated(reo_gated: np.ndarray) -> dict:
    """Devuelve pilas SA/HLA/VLA gated con la convención Xeleris/Odyssey.

    ``reo_gated`` es ``(g, k, j, i)`` (volumen reorientado por gate). Salida:
    dict con ``sa`` ``(g, k, fj, fi)``, ``hla`` ``(g, j, fk, fi)`` y
    ``vla`` ``(g, i, fk, fj)`` ya orientados anatómicamente.
    """
    cube = np.asarray(reo_gated, dtype=np.float64)
    g, K, J, I = cube.shape
    sa = np.stack([[sa_slice(cube[gg], k) for k in range(K)] for gg in range(g)], axis=0)
    hla = np.stack([[hla_slice(cube[gg], j) for j in range(J)] for gg in range(g)], axis=0)
    vla = np.stack([[vla_slice(cube[gg], i) for i in range(I)] for gg in range(g)], axis=0)
    return {"sa": sa, "hla": hla, "vla": vla}

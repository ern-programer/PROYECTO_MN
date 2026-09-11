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


def _reslice_matrix(u: np.ndarray, e_j: np.ndarray, e_i: np.ndarray, sample_scale: float = 1.0) -> np.ndarray:
    """Matriz de reslice con control de muestreo fino/grueso.

    ``sample_scale < 1`` hace que el mismo volumen físico ocupe más píxeles del
    cubo de salida, reduciendo la apariencia de pared engrosada en la SA. Es el
    ajuste que reproduce el look más limpio de Xeleris/Odyssey sin cambiar el
    tamaño del cubo final.
    """
    scale = float(sample_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return np.stack([u, e_j, e_i], axis=1) * scale


def reslice_oblique(
    volume: np.ndarray,
    params: ReorientationParams,
    order: int = 1,
    sample_scale: float = 1.0,
) -> np.ndarray:
    """Reslicea ``volume`` (z, y, x) al marco de eje corto -> ``out[k, j, i]``.

    ``out[k]`` es un corte SA (perpendicular al eje largo). ``k`` recorre base->apex.
    """
    if affine_transform is None:
        raise RuntimeError("scipy.ndimage.affine_transform no disponible")
    vol = np.asarray(volume, dtype=np.float64)
    u = long_axis_vector(params.theta, params.phi)
    u, e_j, e_i = _basis_from_long_axis(u)
    # Columnas de M en orden de salida (k, j, i) expresadas en (z, y, x).
    M = _reslice_matrix(u, e_j, e_i, sample_scale=sample_scale)
    n = int(params.out_size)
    out_shape = (n, n, n)
    oc = (np.array(out_shape, dtype=np.float64) - 1.0) / 2.0
    center = np.asarray(params.center, dtype=np.float64)
    offset = center - M @ oc
    return affine_transform(
        vol, M, offset=offset, output_shape=out_shape,
        order=order, mode="constant", cval=0.0,
    )


def reslice_oblique_gated(
    cube: np.ndarray,
    params: ReorientationParams,
    order: int = 1,
    sample_scale: float = 1.0,
) -> np.ndarray:
    """Aplica :func:`reslice_oblique` por gate. ``cube`` es ``(g, z, y, x)``."""
    cube = np.asarray(cube, dtype=np.float64)
    gates = [reslice_oblique(cube[g], params, order=order, sample_scale=sample_scale) for g in range(cube.shape[0])]
    return np.stack(gates, axis=0)


def reslice_from_vector(
    volume: np.ndarray,
    center: tuple[float, float, float],
    long_axis: np.ndarray,
    out_size: int,
    order: int = 1,
    sample_scale: float = 1.0,
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
    M = _reslice_matrix(u, e_j, e_i, sample_scale=sample_scale)
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
    sample_scale: float = 1.0,
) -> np.ndarray:
    """Aplica :func:`reslice_from_vector` por gate. ``cube`` es ``(g, z, y, x)``."""
    cube = np.asarray(cube, dtype=np.float64)
    gates = [reslice_from_vector(cube[g], center, long_axis, out_size, order=order, sample_scale=sample_scale) for g in range(cube.shape[0])]
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


def _perp_eccentricity(pts: np.ndarray, w: np.ndarray, sw: float, u: np.ndarray) -> float:
    """Excentricidad (λmax/λmin) de la nube proyectada en el plano ⟂ a ``u``.

    Si ``u`` coincide con el eje largo del VI, los cortes SA son anillos
    circulares y la proyección perpendicular es isótropa → excentricidad ≈ 1.
    Al desalinear ``u``, la sección se vuelve elíptica → excentricidad > 1.
    """
    _, e_j, e_i = _basis_from_long_axis(u)
    pj = pts @ e_j
    pi = pts @ e_i
    a = float((w * pj * pj).sum() / sw)
    b = float((w * pj * pi).sum() / sw)
    d = float((w * pi * pi).sum() / sw)
    tr = a + d
    disc = max(tr * tr - 4.0 * (a * d - b * b), 0.0) ** 0.5
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)
    return l1 / (l2 + 1e-9)


def _refine_long_axis_circularity(
    pts: np.ndarray,
    w: np.ndarray,
    u0: np.ndarray,
    max_deg: float = 20.0,
) -> np.ndarray:
    """Refina el eje largo maximizando la circularidad de la sección ⟂.

    El PCA de máxima varianza es inestable cuando la nube del VI está poco
    elongada (autovalores casi degenerados) y tiende a pegarse al eje ``z`` del
    scanner, dando un eje subinclinado. Este refinamiento parte del eje PCA y
    busca localmente la dirección que hace más circulares los anillos SA, que
    SÍ distingue el eje largo oblicuo real del eje axial (un tubo oblicuo
    cortado en axial da elipses, cortado ⟂ da círculos).

    ``max_deg`` acota la desviación respecto del eje PCA. Se mantiene DELIBERADAMENTE
    estrecho (20°) por dos razones validadas contra ground-truth manual:
    (1) un cono amplio deja que el óptimo de excentricidad salte al hemisferio
    equivocado (eje 40-60° errado, inestable entre reconstrucciones); y
    (2) empujar la excentricidad hasta su mínimo global SOBREPASA el eje largo
    real (el eje correcto no es el MÁS circular, sino uno ligeramente elíptico).
    Un cono de 20° ancla el resultado al PCA estable y solo corrige la
    subinclinación, quedando más cerca del eje manual que un cono amplio.
    """
    u = np.asarray(u0, dtype=np.float64)
    u = u / (np.linalg.norm(u) or 1.0)
    u_ref = u.copy()
    sw = float(w.sum())
    if sw <= 0.0:
        return u
    cos_lim = float(np.cos(np.deg2rad(max_deg)))
    best_u = u.copy()
    best_cost = _perp_eccentricity(pts, w, sw, u)
    step = np.deg2rad(12.0)
    for _ in range(80):
        improved = False
        _, e_j, e_i = _basis_from_long_axis(best_u)
        for direction in (e_j, e_i):
            for s in (step, -step):
                cand = best_u * np.cos(s) + direction * np.sin(s)
                cn = float(np.linalg.norm(cand))
                if cn <= 0.0:
                    continue
                cand = cand / cn
                if abs(float(np.dot(cand, u_ref))) < cos_lim:
                    continue  # fuera del cono permitido respecto del PCA
                cost = _perp_eccentricity(pts, w, sw, cand)
                if cost < best_cost - 1e-6:
                    best_cost = cost
                    best_u = cand
                    improved = True
        if not improved:
            step *= 0.5
            if step < np.deg2rad(0.5):
                break
    if float(np.dot(best_u, u_ref)) < 0:
        best_u = -best_u
    return best_u


def _resolve_apex_sign(pts: np.ndarray, w: np.ndarray, u: np.ndarray) -> int:
    """Signo ``s`` tal que ``s*u`` apunta base→ápex (hacia el ápex).

    El VI es un elipsoide truncado hueco: ancho y pesado en la BASE (plano
    valvular) y se AFINA hasta cerrarse en el ÁPEX. Por eso el hemisferio del
    ápex tiene MENOR radio perpendicular (más angosto) y MENOR masa que el de
    la base. El estrechamiento (taper) es la discriminante primaria; la masa
    confirma. Es mucho más robusto que "el ápex apunta lejos del centro del
    cuerpo", que se sesga por el hígado/intestino captantes del ungated.

    Devuelve ``+1`` si ``u`` ya apunta al ápex, ``-1`` si hay que invertirlo, y
    ``0`` si ambos cues son ambiguos (empate) y conviene un desempate externo.
    """
    u = np.asarray(u, dtype=np.float64)
    u = u / (np.linalg.norm(u) or 1.0)
    proj = pts @ u
    perp_r = np.sqrt(np.maximum((pts * pts).sum(axis=1) - proj * proj, 0.0))
    pos = proj > 0.0
    neg = ~pos
    wp = w[pos]
    wn = w[neg]
    mp = float(wp.sum())
    mn = float(wn.sum())
    if mp <= 0.0 or mn <= 0.0:
        return 0
    r_pos = float((perp_r[pos] * wp).sum() / mp)
    r_neg = float((perp_r[neg] * wn).sum() / mn)
    rmean = 0.5 * (r_pos + r_neg) + 1e-9
    taper = (r_neg - r_pos) / rmean   # >0 si +u es el lado angosto (ápex)
    mass = (mn - mp) / (mp + mn)      # >0 si +u es el lado liviano (ápex)
    # Voto: taper pesa doble (señal geométrica más directa del cono del VI).
    score = 2.0 * taper + mass
    if abs(score) < 0.02:
        return 0  # empate: dejar desempate al llamador
    return 1 if score > 0.0 else -1


def _ungated_shell_axis(ungated_volume, center, radius, u_ref):
    """Eje largo por PCA de la cáscara miocárdica ungated (perfusión).

    Sirve para desambiguar el SIGNO del tilt en x del eje de movimiento: la
    nube del movimiento gated es escasa y casi simétrica respecto al plano y-z,
    así que el signo de su componente x queda indeterminado y se refleja según
    la reconstrucción (visto en app vs probe: uz,uy idénticos pero ux con signo
    opuesto). La cáscara de perfusión (muchos más vóxeles) es estable en ese
    signo. Devuelve un vector unitario (z,y,x) con signo alineado a ``u_ref``,
    o ``None`` si no hay suficiente señal.
    """
    if ungated_volume is None:
        return None
    v = np.asarray(ungated_volume, dtype=np.float64)
    if v.ndim != 3:
        return None
    Z, Y, X = v.shape
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
    cz, cy, cx = center
    d2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    loc = v * (d2 <= float(radius) ** 2)
    lmax = float(loc.max())
    if lmax <= 0.0:
        return None
    m = loc >= 0.45 * lmax
    if int(m.sum()) < 10:
        return None
    pz, py, px = np.nonzero(m)
    w = v[pz, py, px]
    sw = float(w.sum())
    if sw <= 0.0:
        return None
    mz = float((pz * w).sum() / sw)
    my = float((py * w).sum() / sw)
    mx = float((px * w).sum() / sw)
    P = np.stack([pz - mz, py - my, px - mx], axis=1)
    cov = (P.T * w) @ P / sw
    _, evec = np.linalg.eigh(cov)
    us = np.asarray(evec[:, -1], dtype=np.float64)
    n = float(np.linalg.norm(us))
    if n <= 0.0:
        return None
    us = us / n
    if float(np.dot(us, np.asarray(u_ref, dtype=np.float64))) < 0:
        us = -us
    return us


# Prior anatómico del eje largo del VI (base→ápex) en (z, y, x), paciente
# supino, convención de la reconstrucción propia de SINCRO. Es la media de 9
# ground-truths manuales (stress+rest, 2026-09) cuya dispersión inter-paciente
# fue ≤ ~10°: el eje real del VI varía POCO entre pacientes, mientras que el
# PCA del movimiento gated variaba 20-90°. El refinamiento por circularidad se
# ancla a este prior (no al PCA) y personaliza dentro de un cono estrecho.
LV_AXIS_PRIOR = np.array([0.3627, 0.7804, 0.5094]) / np.linalg.norm([0.3627, 0.7804, 0.5094])

# Cono de personalización alrededor del prior. Validado contra ground-truth:
# el mínimo de excentricidad NO coincide con el eje real (converge a un
# atractor propio y se clava en el borde del cono), así que el cono acota el
# DAÑO máximo del refinamiento, no solo su alcance. Con 6° el peor caso queda
# dentro del ruido de trazado manual (~±8° entre sesiones del mismo operador).
LV_PRIOR_CONE_DEG = 6.0


def auto_orient_lv(gated_cube, ungated_volume=None):
    """Detecta el VI automáticamente: prior anatómico + movimiento gated.

    El miocardio se contrae/relaja una vez por ciclo cardíaco, así que su
    señal temporal entre gates tiene un primer armónico (1x/ciclo) fuerte.
    El hígado permanece estático y el intestino muy captante solo aporta
    ruido de Poisson (crece con sqrt(cuentas), sin oscilación coherente).
    La amplitud del primer armónico |FFT[1]| por vóxel aísla así el VI
    incluso cuando hay actividad intestinal que "roba" cuentas.

    Del movimiento se estima el CENTRO y el tamaño de la VOI. Para el EJE, en
    cambio, el PCA de la nube de movimiento demostró ser inestable (errores de
    20-90° vs. eje manual según la reconstrucción), mientras que el eje manual
    real varía ≤ ~10° entre pacientes. Por eso el eje parte del prior anatómico
    ``LV_AXIS_PRIOR`` y se personaliza maximizando la circularidad de los
    anillos SA dentro de un cono de ±``LV_PRIOR_CONE_DEG``. El signo base→ápex
    queda definido por el prior (elimina las heurísticas de signo, que eran la
    principal fuente de ejes invertidos).

    Devuelve dict con ``center``, ``long_axis``, ``semiaxes``, ``half_length``
    o ``None`` si no hay gated utilizable. ``ungated_volume`` se acepta por
    compatibilidad (ya no se usa para el eje).
    """
    if gated_cube is None:
        return None
    cube = np.asarray(gated_cube, dtype=np.float64)
    if cube.ndim != 4 or cube.shape[0] < 3:
        return None

    # Amplitud del primer armónico (1x/ciclo) por vóxel: aísla la contracción
    # miocárdica del ruido de Poisson del intestino captante. Misma convención
    # que core/phase_analysis (DC removido, |FFT[1]|).
    c = cube - cube.mean(axis=0, keepdims=True)
    motion = np.abs(np.fft.fft(c, axis=0)[1])  # (z, y, x): mapa de movimiento
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
    _diag_u_pca = np.asarray(evecs[:, -1], dtype=np.float64)  # solo diagnóstico
    _diag_u_pca = _diag_u_pca / (float(np.linalg.norm(_diag_u_pca)) or 1.0)

    # Eje: refinamiento por circularidad ANCLADO AL PRIOR ANATÓMICO (no al PCA).
    # El prior fija hemisferio y signo base→ápex; la circularidad personaliza
    # dentro del cono estrecho según la anatomía real del paciente. Si el sitio
    # calibró su propia BD de normales (core.axis_normals), se usa ese prior.
    prior = LV_AXIS_PRIOR
    try:
        from core.axis_normals import get_prior as _get_prior
        prior = _get_prior(LV_AXIS_PRIOR)
    except Exception:
        pass
    u = _refine_long_axis_circularity(pts, wv, prior, max_deg=LV_PRIOR_CONE_DEG)
    if float(np.dot(u, prior)) < 0:
        u = -u
    _diag_u_circ = u.copy()

    proj = pts @ u
    half = float(1.3 * proj.std())
    half = float(np.clip(half if half > 0 else 6.0, 5.0, 0.5 * max(cube.shape[1:])))
    rz = float(np.clip(1.6 * zz.std() + 3.0, 6.0, cube.shape[1]))
    ry = float(np.clip(1.6 * yy.std() + 3.0, 6.0, cube.shape[2]))
    rx = float(np.clip(1.6 * xx.std() + 3.0, 6.0, cube.shape[3]))

    try:  # DIAGNÓSTICO TEMPORAL (quitar): trazar cada paso del eje auto.
        import os as _os
        import datetime as _dt

        def _r(v):
            return None if v is None else np.round(np.asarray(v, float), 4).tolist()

        def _ang(a, b):
            if a is None or b is None:
                return None
            d = abs(float(np.dot(np.asarray(a, float), np.asarray(b, float))))
            return round(float(np.degrees(np.arccos(min(1.0, d)))), 1)

        ecc_pca = _perp_eccentricity(pts, wv, sw, _diag_u_pca)
        ecc_prior = _perp_eccentricity(pts, wv, sw, prior)
        ecc_circ = _perp_eccentricity(pts, wv, sw, _diag_u_circ)
        ev = np.sort(np.asarray(evals, float))[::-1]
        elong = float(ev[0] / (ev[1] + 1e-9))
        _lines = [
            f"\n=== auto_orient_lv(prior) {_dt.datetime.now():%H:%M:%S} ===\n",
            f"  cube={tuple(int(x) for x in cube.shape)} nvox_mask={len(zz)} mmax={mmax:.3g}\n",
            f"  center=({cz:.1f},{cy:.1f},{cx:.1f}) semiaxes=({rz:.1f},{ry:.1f},{rx:.1f})\n",
            f"  evals={np.round(ev,2).tolist()} elong(l1/l2)={elong:.2f}\n",
            f"  u_pca ={_r(_diag_u_pca)} ecc={ecc_pca:.3f} (solo referencia)\n",
            f"  prior ={_r(prior)} ecc={ecc_prior:.3f}\n",
            f"  u_final={_r(u)} ecc={ecc_circ:.3f} ang_prior->final={_ang(prior, u)}\n",
        ]
        _log = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "_auto_axis_diag.log")
        with open(_log, "a", encoding="utf-8") as _fh:
            _fh.writelines(_lines)
    except Exception:
        pass

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

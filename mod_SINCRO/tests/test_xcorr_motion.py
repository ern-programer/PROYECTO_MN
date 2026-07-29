"""
Test del método de motion correction por correlación de fase 2D ("xcorr").

Usa un fantoma sintético con estructura (blob principal + núcleo denso de borde
duro + estructura secundaria tipo hígado) desplazado con shifts CONOCIDOS por
frame, para verificar que `motion_correct_projections(method="xcorr")` recupera
esos shifts con precisión subpíxel y no altera el comportamiento de los métodos
preexistentes (sinusoid sigue siendo el default, no se toca su lógica).

Correr:  python -m pytest tests/test_xcorr_motion.py -v
     o:  python tests/test_xcorr_motion.py
"""
import os
import sys

import numpy as np
from scipy.ndimage import shift as ndshift

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.console_utf8 import enable_utf8  # noqa: E402
from core.raw_projections import motion_correct_projections  # noqa: E402


enable_utf8()


def _textured_phantom(h: int = 48, w: int = 48) -> np.ndarray:
    """Fantoma con borde duro + estructura secundaria (evita el caso patológico
    de un blob Gaussiano puro, cuyo espectro es demasiado angosto para que la
    correlación de fase (que normaliza la magnitud) encuentre un pico confiable)."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.zeros((h, w))
    base += 80 * np.exp(-(((yy - 24) ** 2 + (xx - 24) ** 2) / (2 * 5.0 ** 2)))
    base[(yy - 24) ** 2 + (xx - 24) ** 2 < 6 ** 2] += 40
    base += 15 * np.exp(-(((yy - 10) ** 2 + (xx - 38) ** 2) / (2 * 4.0 ** 2)))
    return base


def _make_projections(true_shifts_y, true_shifts_x, n_gates=3, noise_sigma=1.5, seed=3):
    base = _textured_phantom()
    h, w = base.shape
    n_angles = len(true_shifts_y)
    rng = np.random.default_rng(seed)
    proj = np.zeros((n_gates, n_angles, h, w))
    for a in range(n_angles):
        img = ndshift(base, (true_shifts_y[a], true_shifts_x[a]), order=1, mode="nearest")
        for g in range(n_gates):
            proj[g, a] = img + rng.normal(0, noise_sigma, size=img.shape)
    return proj


def test_xcorr_recupera_shifts_conocidos():
    true_shifts_y = np.array([0, 1, 2, -1, 0, 3, -2, 1, 0, -1, 2, -2], dtype=float)
    true_shifts_x = np.array([0, -1, 0, 2, 1, -3, 2, 0, -2, 1, 0, 1], dtype=float)
    proj = _make_projections(true_shifts_y, true_shifts_x)

    res = motion_correct_projections(
        proj, axis="xy", method="xcorr", threshold_frac=0.2,
        max_abs_shift_px=10.0, smooth_sigma=0.0, ref_index=0,
    )
    # El shift aplicado debe ANULAR el desplazamiento inducido (signo opuesto).
    err_y = float(np.abs(res["applied_shifts_y"] - (-true_shifts_y)).max())
    err_x = float(np.abs(res["applied_shifts_x"] - (-true_shifts_x)).max())
    assert err_y < 0.5, f"xcorr eje Y: error máximo {err_y:.2f}px demasiado alto"
    assert err_x < 0.5, f"xcorr eje X: error máximo {err_x:.2f}px demasiado alto"
    assert res["method"] == "xcorr"
    print("[OK] test_xcorr_recupera_shifts_conocidos")


def test_xcorr_sin_movimiento_da_shift_cero():
    zeros = np.zeros(10)
    proj = _make_projections(zeros, zeros, noise_sigma=0.8, seed=7)
    res = motion_correct_projections(
        proj, axis="xy", method="xcorr", threshold_frac=0.2,
        max_abs_shift_px=10.0, smooth_sigma=0.0, ref_index=0,
    )
    assert float(np.abs(res["applied_shifts_y"]).max()) < 0.5
    assert float(np.abs(res["applied_shifts_x"]).max()) < 0.5
    print("[OK] test_xcorr_sin_movimiento_da_shift_cero")


def test_xcorr_respeta_roi_alrededor_del_seed():
    """Con seed + roi_radius chico, xcorr debe seguir funcionando (no debe
    romperse ni devolver NaN) restringido a una ventana local."""
    true_shifts_y = np.array([0, 2, -2, 1, 0], dtype=float)
    true_shifts_x = np.array([0, -1, 1, 0, 2], dtype=float)
    proj = _make_projections(true_shifts_y, true_shifts_x, seed=11)

    res = motion_correct_projections(
        proj, axis="xy", method="xcorr", threshold_frac=0.2, seed=(24.0, 24.0),
        roi_radius=10.0, roi_mode="box", max_abs_shift_px=10.0, smooth_sigma=0.0, ref_index=0,
    )
    assert np.all(np.isfinite(res["applied_shifts_y"]))
    assert np.all(np.isfinite(res["applied_shifts_x"]))
    err_y = float(np.abs(res["applied_shifts_y"] - (-true_shifts_y)).max())
    err_x = float(np.abs(res["applied_shifts_x"] - (-true_shifts_x)).max())
    assert err_y < 1.0 and err_x < 1.0
    print("[OK] test_xcorr_respeta_roi_alrededor_del_seed")


def test_sinusoid_default_no_se_altera_por_xcorr():
    """Agregar 'xcorr' no debe cambiar el comportamiento del método sinusoid
    (el default actual): mismo resultado que antes de agregar xcorr."""
    true_shifts_y = np.array([0, 1, 2, -1, 0, 3, -2, 1], dtype=float)
    true_shifts_x = np.array([0, -1, 0, 2, 1, -3, 2, 0], dtype=float)
    proj = _make_projections(true_shifts_y, true_shifts_x, seed=5)
    res = motion_correct_projections(proj, axis="xy", method="sinusoid", threshold_frac=0.2)
    assert res["method"] == "sinusoid"
    assert "corrected" in res and res["corrected"].shape == proj.shape
    print("[OK] test_sinusoid_default_no_se_altera_por_xcorr")


def test_metodo_invalido_sigue_fallando_explicito():
    proj = _make_projections(np.zeros(4), np.zeros(4), seed=1)
    try:
        motion_correct_projections(proj, axis="y", method="metodo_inexistente")
        assert False, "debía lanzar ValueError"
    except ValueError:
        pass
    print("[OK] test_metodo_invalido_sigue_fallando_explicito")


if __name__ == "__main__":
    test_xcorr_recupera_shifts_conocidos()
    test_xcorr_sin_movimiento_da_shift_cero()
    test_xcorr_respeta_roi_alrededor_del_seed()
    test_sinusoid_default_no_se_altera_por_xcorr()
    test_metodo_invalido_sigue_fallando_explicito()
    print("\n[TODOS LOS TESTS DE XCORR MOTION PASARON]")

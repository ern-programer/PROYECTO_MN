"""
Test sintético del motor de fase (Nivel 0 de validación).

Genera miocardios sintéticos con fase CONOCIDA y verifica que phase_analysis + metrics
recuperan lo esperado. Valida el algoritmo contra verdad matemática ANTES de comparar
con datos reales o software externo (SyncTool).

Correr:  python -m pytest tests/test_phase_synthetic.py -v
     o:  python tests/test_phase_synthetic.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.phase_analysis import phase_analysis  # noqa: E402
from core.metrics import calculate_phase_metrics, circular_mean_deg  # noqa: E402
from core.console_utf8 import enable_utf8  # noqa: E402

enable_utf8()


def _make_gated(phase_map_deg, n_gates=16, amplitude=100.0, dc=200.0):
    """
    Construye un cubo gated sintético (n_gates, n_slices, H, W) donde cada voxel
    oscila cosenoidalmente con la fase indicada en phase_map_deg (n_slices,H,W).
    A(t) = dc + amplitude*cos(2π t/N - φ). La fase del 1er armónico debe recuperar φ.
    """
    n_slices, H, W = phase_map_deg.shape
    t = np.arange(n_gates)
    cube = np.zeros((n_gates, n_slices, H, W), dtype=np.float64)
    phi = np.radians(phase_map_deg)
    for k in range(n_gates):
        cube[k] = dc + amplitude * np.cos(2 * np.pi * k / n_gates - phi)
    return cube


def test_uniform_phase_is_synchronous():
    """Fase uniforme (todos a 90°) → Phase SD ≈ 0, clasificación NORMAL."""
    mask = np.ones((4, 8, 8), dtype=bool)
    phase_map = np.full((4, 8, 8), 90.0)
    cube = _make_gated(phase_map, n_gates=16)
    res = phase_analysis(cube, mask, amplitude_threshold_frac=0.05)
    m = calculate_phase_metrics(res.phases_deg)
    assert abs(m["mean_phase"] - 90.0) < 2.0, f"mean={m['mean_phase']}"
    assert m["phase_sd"] < 1.0, f"SD={m['phase_sd']} (esperaba ~0)"
    assert m["classification"] == "NORMAL"
    print(f"[OK] uniforme: mean={m['mean_phase']}° SD={m['phase_sd']}° → {m['classification']}")


def test_recovers_known_phase_gradient():
    """Gradiente de fase conocido → la media circular debe coincidir."""
    n_slices, H, W = 1, 10, 10
    # Gradiente lineal 60°→120° a lo largo de X
    xs = np.linspace(60, 120, W)
    phase_map = np.tile(xs, (n_slices, H, 1))
    mask = np.ones((n_slices, H, W), dtype=bool)
    cube = _make_gated(phase_map, n_gates=16)
    res = phase_analysis(cube, mask, amplitude_threshold_frac=0.05)
    # La media de un gradiente 60-120 es ~90
    assert abs(res.phases_deg.mean() - 90.0) < 3.0, f"mean={res.phases_deg.mean()}"
    # El rango recuperado debe cubrir ~60-120
    assert res.phases_deg.min() < 70 and res.phases_deg.max() > 110
    print(f"[OK] gradiente: rango {res.phases_deg.min():.1f}-{res.phases_deg.max():.1f}°")


def test_dyssynchrony_increases_sd():
    """Más dispersión de fase → mayor Phase SD y peor clasificación."""
    mask = np.ones((2, 10, 10), dtype=bool)
    rng = np.random.default_rng(0)

    narrow = np.clip(rng.normal(90, 8, (2, 10, 10)), 0, 359)   # sincrónico
    wide = np.clip(rng.normal(90, 45, (2, 10, 10)), 0, 359)    # disincrónico

    m_narrow = calculate_phase_metrics(
        phase_analysis(_make_gated(narrow), mask, amplitude_threshold_frac=0.05).phases_deg
    )
    m_wide = calculate_phase_metrics(
        phase_analysis(_make_gated(wide), mask, amplitude_threshold_frac=0.05).phases_deg
    )
    assert m_wide["phase_sd"] > m_narrow["phase_sd"], "SD debería crecer con la dispersión"
    assert m_wide["bandwidth"] > m_narrow["bandwidth"]
    assert m_wide["entropy"] > m_narrow["entropy"]
    print(f"[OK] disincronía: SD {m_narrow['phase_sd']}° ({m_narrow['classification']}) "
          f"→ {m_wide['phase_sd']}° ({m_wide['classification']})")


def test_amplitude_filter_drops_noise():
    """Voxels de baja amplitud (ruido) se descartan por el umbral de amplitud."""
    mask = np.ones((1, 10, 10), dtype=bool)
    phase_map = np.full((1, 10, 10), 90.0)
    cube = _make_gated(phase_map, n_gates=16, amplitude=100.0)
    # Anular la amplitud de la mitad de los voxels (constantes = ruido sin señal)
    cube[:, 0, :5, :] = 200.0  # DC constante → amplitud ~0
    res = phase_analysis(cube, mask, amplitude_threshold_frac=0.10)
    assert res.n_voxels_kept < res.n_voxels_total, "debería descartar voxels de baja amplitud"
    print(f"[OK] filtro amplitud: {res.n_voxels_kept}/{res.n_voxels_total} voxels conservados")


def _run_all():
    for fn in [test_uniform_phase_is_synchronous, test_recovers_known_phase_gradient,
               test_dyssynchrony_increases_sd, test_amplitude_filter_drops_noise]:
        fn()
    print("\n[TODOS LOS TESTS SINTÉTICOS PASARON]")


if __name__ == "__main__":
    _run_all()

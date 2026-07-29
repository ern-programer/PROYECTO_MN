"""
Test de la corrección de dropout de cuentas del último gate (ECTb 4.0, secc. 22.8).

Verifica que:
- el déficit del último gate se mide correctamente,
- la corrección iguala las cuentas del último gate con las del primero,
- no se toca ningún otro gate ni la forma espacial,
- no se corrige cuando el déficit es despreciable o negativo,
- el factor se recorta cuando el último gate quedó casi vacío.

Correr:  python -m pytest tests/test_gate_dropout.py -v
     o:  python tests/test_gate_dropout.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.console_utf8 import enable_utf8  # noqa: E402
from core.gate_dropout import (  # noqa: E402
    DEFAULT_MAX_SCALE,
    analyze_gate_dropout,
    correct_last_gate_dropout,
    gate_total_counts,
)

enable_utf8()


def _make_cube(n_gates=8, n_slices=10, size=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n_gates, n_slices, size, size)) * 100.0


def test_corrige_dropout_real():
    """Un déficit del 18% se detecta y se corrige igualando al primer gate."""
    cube = _make_cube()
    cube[-1] *= 0.82

    info = analyze_gate_dropout(cube)
    assert info["applicable"]
    assert info["significant"]
    assert not info["clipped"]
    assert info["dropout_pct"] > 10.0

    out, applied = correct_last_gate_dropout(cube)
    assert applied["applied"]

    counts = gate_total_counts(out)
    assert np.isclose(counts[0], counts[-1]), "el último gate debe igualar al primero"
    assert np.allclose(out[:-1], cube[:-1]), "los demás gates no se tocan"
    # Corrección de ganancia: la forma espacial del último gate no cambia.
    assert np.allclose(out[-1] / out[-1].sum(), cube[-1] / cube[-1].sum())
    print(f"[OK] dropout real: {info['dropout_pct']:.1f}% corregido con factor {info['scale']:.3f}")


def test_no_corrige_dropout_despreciable():
    """Un déficit por debajo del umbral no dispara ninguna modificación."""
    cube = _make_cube(seed=1)
    counts = gate_total_counts(cube)
    # Forzar exactamente 0.5% de déficit respecto del primer gate.
    cube[-1] *= (counts[0] * 0.995) / counts[-1]

    info = analyze_gate_dropout(cube)
    assert info["applicable"]
    assert not info["significant"]

    out, applied = correct_last_gate_dropout(cube)
    assert not applied["applied"]
    assert np.allclose(out, cube), "sin dropout significativo el cubo no se modifica"
    print(f"[OK] dropout despreciable ({info['dropout_pct']:.1f}%): no se modifica el cubo")


def test_no_corrige_si_el_ultimo_gate_tiene_mas_cuentas():
    """Si el último gate tiene MÁS cuentas que el primero no hay nada que corregir."""
    cube = _make_cube(seed=2)
    counts = gate_total_counts(cube)
    cube[-1] *= (counts[0] * 1.10) / counts[-1]

    info = analyze_gate_dropout(cube)
    assert info["dropout_pct"] < 0.0
    out, applied = correct_last_gate_dropout(cube)
    assert not applied["applied"]
    assert np.allclose(out, cube)
    print(f"[OK] exceso de cuentas ({-info['dropout_pct']:.1f}%): no se aplica corrección")


def test_recorta_factor_en_dropout_extremo():
    """Con el último gate casi vacío el factor se recorta para no amplificar ruido."""
    cube = _make_cube(seed=3)
    counts = gate_total_counts(cube)
    cube[-1] *= (counts[0] * 0.30) / counts[-1]

    info = analyze_gate_dropout(cube)
    assert info["clipped"]
    assert info["scale"] > DEFAULT_MAX_SCALE
    assert np.isclose(info["scale_clipped"], DEFAULT_MAX_SCALE)

    out, applied = correct_last_gate_dropout(cube)
    assert applied["applied"]
    out_counts = gate_total_counts(out)
    assert out_counts[-1] < out_counts[0], "con el factor recortado no llega a igualar al primero"
    print(f"[OK] dropout extremo ({info['dropout_pct']:.1f}%): factor recortado a {DEFAULT_MAX_SCALE}")


def test_force_aplica_aunque_sea_despreciable():
    """`force=True` permite comparar A/B aunque el déficit sea mínimo."""
    cube = _make_cube(seed=4)
    counts = gate_total_counts(cube)
    cube[-1] *= (counts[0] * 0.995) / counts[-1]

    out, applied = correct_last_gate_dropout(cube, force=True)
    assert applied["applied"]
    assert np.isclose(gate_total_counts(out)[0], gate_total_counts(out)[-1])
    print("[OK] force=True aplica la corrección aunque el dropout sea despreciable")


def test_rechaza_cubos_no_4d():
    """La entrada debe ser un gated 4D."""
    for bad in (np.zeros((8, 16, 16)), np.zeros((8, 16))):
        try:
            analyze_gate_dropout(bad)
        except ValueError:
            continue
        raise AssertionError(f"debería rechazar shape {bad.shape}")
    print("[OK] cubos no 4D rechazados con ValueError")


if __name__ == "__main__":
    test_corrige_dropout_real()
    test_no_corrige_dropout_despreciable()
    test_no_corrige_si_el_ultimo_gate_tiene_mas_cuentas()
    test_recorta_factor_en_dropout_extremo()
    test_force_aplica_aunque_sea_despreciable()
    test_rechaza_cubos_no_4d()
    print("\n[TODOS LOS TESTS DE GATE DROPOUT PASARON]")

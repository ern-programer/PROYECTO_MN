"""Tests del bandwidth con la definición del ECTb (banda del 95%).

Se ejecuta directo:
    & ".venv\\Scripts\\python.exe" tests/test_bandwidth_ectb.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.metrics import BANDWIDTH_COVERAGE, narrowest_band_deg  # noqa: E402


def test_cobertura_total_da_el_rango_completo():
    """Con coverage=1 la banda tiene que ser exactamente el rango de los datos."""
    datos = np.array([10.0, 20.0, 35.0, 40.0, 90.0])
    width, low, high = narrowest_band_deg(datos, coverage=1.0)
    assert abs(low - 10.0) < 1e-9, low
    assert abs(high - 90.0) < 1e-9, high
    assert abs(width - 80.0) < 1e-9, width
    print("[OK] coverage=1 devuelve el rango completo")


def test_elige_la_banda_mas_angosta_no_la_centrada():
    """La clave de la definición ECTb: es la banda MÁS ANGOSTA, no la simétrica.

    Con una distribución asimétrica (un grupo compacto más una cola larga), la
    banda del 95% tiene que quedarse pegada al grupo compacto y dejar la cola
    afuera. Un criterio de percentiles simétricos partiría la cola por el medio
    y daría una banda más ancha.
    """
    compacto = np.linspace(100.0, 110.0, 95)
    cola = np.linspace(200.0, 400.0, 5)
    datos = np.concatenate([compacto, cola])

    width, low, high = narrowest_band_deg(datos, coverage=0.95)
    assert low >= 100.0 - 1e-9, low
    assert high <= 111.0, high
    assert width < 12.0, width

    simetrico = float(np.percentile(datos, 97.5) - np.percentile(datos, 2.5))
    assert width < simetrico, (width, simetrico)
    print(f"[OK] banda mas angosta {width:.1f} vs percentiles simetricos {simetrico:.1f}")


def test_banda_95_es_mayor_o_igual_que_p5_p95():
    """El 95% siempre abarca al menos tanto como el 90% del criterio viejo.

    Es el punto que motivó el cambio: el bandwidth histórico (P95-P5 = 90% de
    los elementos) es sistemáticamente MENOR y por eso subestimaba la
    disincronía frente a las bases de datos normales publicadas, que están en
    la escala del 95%.
    """
    rng = np.random.default_rng(7)
    datos = rng.normal(120.0, 15.0, 4000)

    width_95, _, _ = narrowest_band_deg(datos, coverage=BANDWIDTH_COVERAGE)
    width_90 = float(np.percentile(datos, 95) - np.percentile(datos, 5))
    assert width_95 > width_90, (width_95, width_90)
    print(f"[OK] banda 95% {width_95:.1f} > P95-P5 {width_90:.1f}")


def test_distribucion_ancha_da_banda_mas_ancha():
    """Chequeo de monotonía: más dispersión, más bandwidth."""
    rng = np.random.default_rng(11)
    angosta, _, _ = narrowest_band_deg(rng.normal(100.0, 5.0, 2000))
    ancha, _, _ = narrowest_band_deg(rng.normal(100.0, 30.0, 2000))
    assert ancha > angosta * 3.0, (angosta, ancha)
    print(f"[OK] dispersion mayor -> banda mayor ({angosta:.1f} vs {ancha:.1f})")


def test_pocos_elementos_no_rompe():
    """Con muy pocas muestras tiene que devolver algo finito y coherente."""
    for n in (1, 2, 3):
        datos = np.linspace(0.0, 10.0, n)
        width, low, high = narrowest_band_deg(datos)
        assert np.isfinite(width) and width >= 0.0, (n, width)
        assert high >= low, (n, low, high)
    print("[OK] casos degenerados (n<=3) no rompen")


if __name__ == "__main__":
    test_cobertura_total_da_el_rango_completo()
    test_elige_la_banda_mas_angosta_no_la_centrada()
    test_banda_95_es_mayor_o_igual_que_p5_p95()
    test_distribucion_ancha_da_banda_mas_ancha()
    test_pocos_elementos_no_rompe()
    print("\n[TODOS LOS TESTS DE BANDWIDTH ECTb PASARON]")

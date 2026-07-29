"""
Test del método ECTb de cuantificación de VI (core/ectb_lv.py).

Usa un fantoma sintético de geometría CONOCIDA: un cilindro miocárdico que se
contrae y engrosa entre telediástole y telesístole. Como el volumen verdadero
es calculable analíticamente, se puede verificar que el método recupera la FEVI.

Correr:  python -m pytest tests/test_ectb_lv.py -v
     o:  python tests/test_ectb_lv.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.console_utf8 import enable_utf8  # noqa: E402
from core.ectb_lv import (  # noqa: E402
    SEPTAL_ANGLE_DEG,
    ECTbLVConfig,
    _base_is_last,
    _valve_plane_weights,
    analyze_lv_ectb,
    apply_regression,
    convert_ef_pct,
    regression_equation_text,
)

enable_utf8()

PIXEL_MM = 6.4
SLICE_MM = 6.4


class _FakeSeg:
    """Segmentación mínima con lo que consume `analyze_lv_ectb`."""

    def __init__(self, mask, centers, outer, inner=None):
        self.mask = mask
        self.center_per_slice = centers
        self.outer_radius = outer
        self.inner_radius = (
            inner if inner is not None else np.full((mask.shape[0],), np.nan)
        )


def _make_phantom(n_gates=8, n_slices=12, size=32, ef_target=0.55, wall_ed_mm=10.0, seed=0):
    """Cilindro miocárdico gatillado con FEVI conocida.

    El radio endocárdico varía cosenoidalmente entre ED y ES de modo que la
    relación de áreas dé la FEVI pedida. La pared engrosa en sístole
    conservando el área miocárdica (músculo incompresible), y las cuentas se
    hacen proporcionales al espesor para reproducir el volumen parcial.
    """
    rng = np.random.default_rng(seed)
    cy = cx = size / 2.0
    wall_ed_px = wall_ed_mm / PIXEL_MM
    r_endo_ed_px = 7.0

    # Area_ES = Area_ED * (1 - EF)  ->  r_ES = r_ED * sqrt(1 - EF)
    r_endo_es_px = r_endo_ed_px * np.sqrt(1.0 - ef_target)
    area_myo = np.pi * ((r_endo_ed_px + wall_ed_px) ** 2 - r_endo_ed_px ** 2)

    cube = np.zeros((n_gates, n_slices, size, size), dtype=np.float64)
    ys, xs = np.ogrid[:size, :size]
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

    for g in range(n_gates):
        # gate 0 = ED, mitad del ciclo = ES
        u = 0.5 * (1.0 + np.cos(2.0 * np.pi * g / n_gates))  # 1 en ED, 0 en ES
        r_endo = r_endo_es_px + (r_endo_ed_px - r_endo_es_px) * u
        r_epi = np.sqrt(r_endo ** 2 + area_myo / np.pi)
        thickness = r_epi - r_endo
        # Volumen parcial: cuentas proporcionales al espesor.
        counts = 1000.0 * (thickness / wall_ed_px)
        ring = (dist >= r_endo) & (dist <= r_epi)
        for s in range(n_slices):
            cube[g, s][ring] = counts
        cube[g] += rng.normal(0.0, 2.0, size=(n_slices, size, size))

    cube = np.clip(cube, 0.0, None)
    mask = cube[0] > (0.35 * cube[0].max())
    centers = np.tile([cy, cx], (n_slices, 1))
    outer = np.full((n_slices,), r_endo_ed_px + wall_ed_px + 2.0)
    return cube, _FakeSeg(mask, centers, outer)


def test_recupera_fevi_conocida():
    """Sobre un fantoma con FEVI 55% el método debe caer cerca de ese valor."""
    cube, seg = _make_phantom(ef_target=0.55)
    res = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM)

    assert res.available, res.reason
    assert 40.0 <= res.ef_pct <= 70.0, f"FEVI fuera de rango: {res.ef_pct:.1f}%"
    assert res.edv_ml > res.esv_ml > 0.0
    assert np.isclose(res.sv_ml, res.edv_ml - res.esv_ml)
    print(f"[OK] FEVI recuperada: {res.ef_pct:.1f}% (objetivo 55%) | "
          f"EDV={res.edv_ml:.1f} ESV={res.esv_ml:.1f} mL")


def test_detecta_ventriculo_deprimido():
    """Un fantoma con FEVI baja tiene que dar claramente menos que uno normal."""
    cube_norm, seg_norm = _make_phantom(ef_target=0.60, seed=1)
    cube_low, seg_low = _make_phantom(ef_target=0.25, seed=1)

    ef_norm = analyze_lv_ectb(cube_norm, seg_norm, (PIXEL_MM, PIXEL_MM), SLICE_MM).ef_pct
    ef_low = analyze_lv_ectb(cube_low, seg_low, (PIXEL_MM, PIXEL_MM), SLICE_MM).ef_pct

    assert ef_norm - ef_low > 15.0, f"no discrimina: normal={ef_norm:.1f}% bajo={ef_low:.1f}%"
    print(f"[OK] discrimina función: normal={ef_norm:.1f}% vs deprimido={ef_low:.1f}%")


def test_ed_gate_y_es_gate_coherentes():
    """El gate de volumen máximo debe ser ED y el de mínimo, ES."""
    cube, seg = _make_phantom(ef_target=0.55, seed=2)
    res = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM)

    vols = res.gate_volumes_ml
    assert res.ed_gate == int(np.argmax(vols)) + 1
    assert res.es_gate == int(np.argmin(vols)) + 1
    assert res.ed_gate == 1, f"el fantoma tiene ED en el gate 1, dio {res.ed_gate}"
    print(f"[OK] ED gate={res.ed_gate}, ES gate={res.es_gate}")


def test_espesor_ed_escala_los_volumenes():
    """Asumir una pared más gruesa desplaza el endocardio y achica la cavidad."""
    cube, seg = _make_phantom(seed=3)
    fino = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM,
                           ECTbLVConfig(ed_wall_thickness_mm=8.0))
    grueso = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM,
                             ECTbLVConfig(ed_wall_thickness_mm=14.0))

    assert grueso.edv_ml < fino.edv_ml, "más espesor debe achicar la cavidad"
    assert grueso.myocardial_mass_g > fino.myocardial_mass_g
    print(f"[OK] espesor ED escala volúmenes: 8mm->EDV {fino.edv_ml:.1f} mL | "
          f"14mm->EDV {grueso.edv_ml:.1f} mL")


def test_masa_miocardica_usa_densidad():
    """La masa es el volumen de pared por 1.05 g/mL."""
    cube, seg = _make_phantom(seed=4)
    res = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM)

    assert res.myocardial_volume_ml > 0.0
    assert np.isclose(res.myocardial_mass_g, res.myocardial_volume_ml * 1.05)
    print(f"[OK] masa miocárdica: {res.myocardial_mass_g:.1f} g "
          f"(volumen {res.myocardial_volume_ml:.1f} mL x 1.05)")


def test_engrosamiento_positivo():
    """En un ventrículo que se contrae la pared engrosa en sístole."""
    cube, seg = _make_phantom(ef_target=0.55, seed=5)
    res = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM)

    assert res.thickening_pct > 0.0, f"engrosamiento no positivo: {res.thickening_pct:.1f}%"
    print(f"[OK] engrosamiento sistólico: {res.thickening_pct:.1f}%")


def test_rechaza_entradas_invalidas():
    """Cubos no gatillados o segmentación incoherente devuelven available=False."""
    cube, seg = _make_phantom(seed=6)

    res = analyze_lv_ectb(cube[:2], seg, (PIXEL_MM, PIXEL_MM), SLICE_MM)
    assert not res.available and res.reason

    res = analyze_lv_ectb(cube, seg, (0.0, 0.0), SLICE_MM)
    assert not res.available and "spacing" in res.reason.lower()
    print("[OK] entradas inválidas rechazadas con motivo explícito")


def test_regresion_qgs():
    """La regresión R1 publicada convierte FEVI de ECTb a escala QGS."""
    # y = 0.96x - 0.053 con x como fracción 0-1
    assert np.isclose(apply_regression(0.60, 0.96, -0.053), 0.5230)
    # ECTb corre ~7-8% por encima de QGS-8: la conversión debe bajar el valor.
    assert apply_regression(0.60, 0.96, -0.053) < 0.60
    print("[OK] regresión ECTb->QGS aplicada correctamente")


def test_convert_ef_pct_maneja_unidades():
    """convert_ef_pct recibe y devuelve %, sin importar la unidad de la ecuación."""
    # QGS-8 está publicada en fracción: 60% -> 0.96*0.60 - 0.053 = 0.523 -> 52.3%
    assert np.isclose(convert_ef_pct(60.0, "qgs8"), 52.3)
    # QGS-16 está publicada en porcentaje: 0.855*60 + 1.73 = 53.03%
    assert np.isclose(convert_ef_pct(60.0, "qgs16"), 53.03)
    # MUGA, en fracción: 1.22*0.60 - 0.072 = 0.660 -> 66.0%
    assert np.isclose(convert_ef_pct(60.0, "muga"), 66.0)
    print("[OK] conversión de unidades correcta en las tres regresiones")


def test_conversion_baja_respecto_de_qgs():
    """El sesgo conocido: ECTb informa más alto que QGS en el rango clínico."""
    for ef in (35.0, 45.0, 55.0, 65.0):
        assert convert_ef_pct(ef, "qgs8") < ef, f"a {ef}% QGS-8 no dio menor"
        assert convert_ef_pct(ef, "qgs16") < ef, f"a {ef}% QGS-16 no dio menor"
    print("[OK] la conversión a escala QGS baja el valor en todo el rango clínico")


def test_regresion_desconocida_falla_explicito():
    """Pedir una regresión inexistente tiene que romper con KeyError, no en silencio."""
    try:
        convert_ef_pct(55.0, "inexistente")
    except KeyError:
        print("[OK] regresión inexistente rechazada con KeyError")
        return
    raise AssertionError("debería haber fallado con KeyError")


def test_texto_de_ecuacion_declara_unidad():
    """El texto de la ecuación tiene que dejar clara la unidad de x."""
    assert "fracción" in regression_equation_text("qgs8")
    assert "porcentaje" in regression_equation_text("qgs16")
    # El signo del término independiente debe leerse correcto.
    assert "−" in regression_equation_text("qgs8")
    assert "+" in regression_equation_text("qgs16")
    print("[OK] las ecuaciones declaran su unidad y su signo")


def test_radios_permiten_reconstruir_la_cota():
    """La cota manual se apoya en endo/epi: su separación es el espesor usado."""
    cube, seg = _make_phantom(seed=7)
    cfg = ECTbLVConfig(ed_wall_thickness_mm=12.0, median_kernel_large=0, median_kernel_small=0)
    res = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM, cfg)

    assert res.available, res.reason
    gate = res.ed_gate - 1
    separacion = res.epi_radii_mm[gate] - res.endo_radii_mm[gate]
    # En el gate de referencia el espesor debe ser el configurado.
    assert np.allclose(separacion, 12.0, atol=0.6), (
        f"separación endo-epi {separacion.mean():.2f} mm, esperada 12.0 mm"
    )
    print(f"[OK] cota reconstruible: separación endo-epi = {separacion.mean():.2f} mm")


# ---------------------------------------------------------------------------
# Plano valvular de dos piezas
# ---------------------------------------------------------------------------

def test_plano_valvular_no_toca_la_mitad_lateral():
    """"Dos piezas" significa: perpendicular en TODO el lado lateral."""
    n_ang = 64
    angles = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
    w, u_cut = _valve_plane_weights(
        n_valid=10, angles=angles, dz_mm=SLICE_MM,
        septal_angle_deg=SEPTAL_ANGLE_DEG, offset_mm=10.0, base_is_last=True,
    )
    # Lateral = donde cos(theta - 180) <= 0, o sea la mitad opuesta al septum.
    lateral = np.cos(angles - np.deg2rad(SEPTAL_ANGLE_DEG)) <= 0.0
    assert np.allclose(w[:, lateral], 1.0), "el plano recortó del lado lateral"
    assert np.isclose(u_cut[lateral].min(), u_cut[lateral].max()), "el lado lateral no quedó plano"
    # En el medio del septum tiene que recortar el offset pedido.
    i_sept = int(np.argmin(np.abs(angles - np.deg2rad(SEPTAL_ANGLE_DEG))))
    recorte_cortes = u_cut[lateral][0] - u_cut[i_sept]
    assert np.isclose(recorte_cortes, 10.0 / SLICE_MM, atol=1e-6)
    print(f"[OK] plano de 2 piezas: lateral intacto, septum recortado {recorte_cortes:.2f} cortes")


def test_plano_valvular_con_offset_cero_no_cambia_nada():
    """Con offset 0 los pesos deben dar 1: idem plano perpendicular de siempre."""
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    w, _ = _valve_plane_weights(8, angles, SLICE_MM, SEPTAL_ANGLE_DEG, 0.0, True)
    assert np.allclose(w, 1.0)

    cube, seg = _make_phantom(seed=8)
    sin_plano = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM,
                                ECTbLVConfig(use_valve_plane=False))
    offset_cero = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM,
                                  ECTbLVConfig(valve_septal_offset_mm=0.0))
    assert np.isclose(sin_plano.edv_ml, offset_cero.edv_ml)
    assert np.isclose(sin_plano.ef_pct, offset_cero.ef_pct)
    print("[OK] desactivar el plano equivale a offset 0 mm")


def test_plano_valvular_descuenta_volumen_basal():
    """El plano tiene que sacar cavidad de la base, no agregarla."""
    cube, seg = _make_phantom(seed=9)
    plano = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM,
                            ECTbLVConfig(use_valve_plane=False))
    ectb = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM)

    assert ectb.edv_ml < plano.edv_ml, "el plano valvular no redujo el EDV"
    assert ectb.esv_ml < plano.esv_ml
    assert ectb.valve_removed_ml > 0.0
    assert np.isclose(ectb.valve_removed_ml, plano.edv_ml - ectb.edv_ml, atol=0.5)
    assert any("valvular" in n.lower() for n in ectb.notes), "falta la nota de auditoría"
    print(f"[OK] plano valvular descontó {ectb.valve_removed_ml:.1f} mL del EDV "
          f"({plano.edv_ml:.1f} -> {ectb.edv_ml:.1f} mL)")


def test_plano_valvular_es_continuo():
    """Mover el offset no debe producir saltos de un corte entero.

    Es el requisito para que el control se pueda mover en vivo sin que el
    número pegue escalones.
    """
    cube, seg = _make_phantom(seed=10)
    offsets = np.arange(0.0, 13.0, 0.5)
    edvs = [
        analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM,
                        ECTbLVConfig(valve_septal_offset_mm=float(o))).edv_ml
        for o in offsets
    ]
    edvs = np.asarray(edvs)
    saltos = np.abs(np.diff(edvs))
    assert np.all(np.diff(edvs) <= 1e-9), "el EDV no baja de forma monótona"
    assert saltos.max() < 0.03 * edvs[0], (
        f"salto de {saltos.max():.1f} mL entre pasos de 0.5 mm: no es continuo"
    )
    print(f"[OK] respuesta continua: salto máximo {saltos.max():.2f} mL por 0.5 mm")


def test_detecta_de_que_lado_esta_la_base():
    """El plano necesita saber hacia dónde queda la base, venga como venga la pila."""
    n_slices, size = 8, 24
    mask = np.zeros((n_slices, size, size), dtype=bool)
    ys, xs = np.ogrid[:size, :size]
    dist = np.sqrt((ys - size / 2) ** 2 + (xs - size / 2) ** 2)
    # Pila ápex(chico) -> base(grande).
    for s in range(n_slices):
        mask[s] = dist <= (3.0 + s)
    inner = np.linspace(0.5, 6.0, n_slices)
    outer = np.linspace(4.0, 10.0, n_slices)
    centers = np.tile([size / 2, size / 2], (n_slices, 1))

    seg_ab = _FakeSeg(mask, centers, outer, inner)
    assert _base_is_last(seg_ab, list(range(n_slices))) is True

    seg_ba = _FakeSeg(mask[::-1].copy(), centers, outer[::-1].copy(), inner[::-1].copy())
    assert _base_is_last(seg_ba, list(range(n_slices))) is False
    print("[OK] la base se detecta por geometría en ambos sentidos de la pila")


def test_indice_esfericidad_entre_0_y_1_y_menor_en_sistole():
    """El índice tiene que ser positivo y bajar en sístole.

    El eje largo lo fija la extensión de la pila de cortes, que no cambia entre
    gates, mientras que el eje corto se achica en sístole. Por lo tanto el
    índice en ES tiene que ser menor que en ED.
    """
    cube, seg = _make_phantom(ef_target=0.55, seed=5)
    res = analyze_lv_ectb(cube, seg, (PIXEL_MM, PIXEL_MM), SLICE_MM)

    assert res.available, res.reason
    assert res.long_axis_mm > 0.0
    assert 0.0 < res.shape_index_ed < 3.0, res.shape_index_ed
    assert res.shape_index_es < res.shape_index_ed, (res.shape_index_ed, res.shape_index_es)
    assert res.short_axis_es_mm < res.short_axis_ed_mm
    print(f"[OK] esfericidad ED={res.shape_index_ed:.2f} > ES={res.shape_index_es:.2f} "
          f"(eje largo {res.long_axis_mm:.0f} mm)")


def test_ventriculo_mas_corto_es_mas_esferico():
    """Con menos cortes (ventrículo más corto) el índice tiene que subir.

    Es la propiedad que le da sentido clínico: a igual diámetro de eje corto,
    un ventrículo más corto en el eje largo es geométricamente más esférico.
    """
    cube_largo, seg_largo = _make_phantom(n_slices=20, seed=6)
    cube_corto, seg_corto = _make_phantom(n_slices=8, seed=6)

    si_largo = analyze_lv_ectb(cube_largo, seg_largo, (PIXEL_MM, PIXEL_MM), SLICE_MM).shape_index_ed
    si_corto = analyze_lv_ectb(cube_corto, seg_corto, (PIXEL_MM, PIXEL_MM), SLICE_MM).shape_index_ed

    assert si_corto > si_largo, (si_largo, si_corto)
    print(f"[OK] ventriculo corto mas esferico: {si_corto:.2f} vs largo {si_largo:.2f}")


if __name__ == "__main__":
    test_recupera_fevi_conocida()
    test_detecta_ventriculo_deprimido()
    test_ed_gate_y_es_gate_coherentes()
    test_espesor_ed_escala_los_volumenes()
    test_masa_miocardica_usa_densidad()
    test_engrosamiento_positivo()
    test_rechaza_entradas_invalidas()
    test_regresion_qgs()
    test_convert_ef_pct_maneja_unidades()
    test_conversion_baja_respecto_de_qgs()
    test_regresion_desconocida_falla_explicito()
    test_texto_de_ecuacion_declara_unidad()
    test_radios_permiten_reconstruir_la_cota()
    test_plano_valvular_no_toca_la_mitad_lateral()
    test_plano_valvular_con_offset_cero_no_cambia_nada()
    test_plano_valvular_descuenta_volumen_basal()
    test_plano_valvular_es_continuo()
    test_detecta_de_que_lado_esta_la_base()
    test_indice_esfericidad_entre_0_y_1_y_menor_en_sistole()
    test_ventriculo_mas_corto_es_mas_esferico()
    print("\n[TODOS LOS TESTS DE ECTb LV PASARON]")

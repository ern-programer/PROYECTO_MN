"""Validación del ratio S/VD con estudios reales.

Uso:
    python _validate_svd_ratio.py --study <path_to_dcm_or_folder> [--ct <path_to_ct>]

Carga un estudio SPECT (o carpeta con .dcm), reconstruye el volumen,
y permite depositar puntos S/V/D manualmente o usar coordenadas hardcodeadas
para verificar que compute_spect_ratio() produce valores razonables.

Modo interactivo (sin args): abre el panel AMYLO SPECT completo.
Modo batch (--points): usa coordenadas voxel predefinidas y reporta ratios.
"""

from __future__ import annotations

import argparse
import sys
import os

# Asegurar que el módulo esté en el path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def _load_spect_volume(study_path: str):
    """Carga un estudio SPECT y devuelve (volume, spacing_zyx, spect_ds)."""
    import numpy as np
    from pathlib import Path

    p = Path(study_path)
    if p.is_file():
        dcm_files = [p]
    elif p.is_dir():
        dcm_files = sorted(p.glob("*.dcm")) + sorted(p.glob("*"))
        dcm_files = [f for f in dcm_files if f.is_file() and f.suffix.lower() in ('.dcm', '')][:50]
    else:
        raise FileNotFoundError(f"No se encuentra: {study_path}")

    if not dcm_files:
        raise FileNotFoundError(f"No hay archivos DICOM en: {study_path}")

    print(f"[*] Cargando {len(dcm_files)} archivos DICOM desde {study_path}")

    import pydicom
    # Leer primer archivo para metadatos
    ds0 = pydicom.dcmread(str(dcm_files[0]), force=True)
    modality = getattr(ds0, 'Modality', '?')
    print(f"[*] Modalidad: {modality}, StudyDesc: {getattr(ds0, 'StudyDescription', 'N/D')}")

    # Intentar construir volumen 3D desde slices
    # Para SPECT NM, los archivos suelen ser frames de un mismo acquisition
    slices_data = []
    for f in dcm_files[:64]:  # limitar a 64 slices por seguridad
        try:
            ds = pydicom.dcmread(str(f), force=True)
            arr = ds.pixel_array.astype(np.float32)
            # Aplicar slope/intercept si existen
            slope = getattr(ds, 'RescaleSlope', 1.0)
            intercept = getattr(ds, 'RescaleIntercept', 0.0)
            if slope != 1.0 or intercept != 0.0:
                arr = arr * slope + intercept
            slices_data.append(arr)
        except Exception as e:
            print(f"  [!] Error leyendo {f.name}: {e}")

    if not slices_data:
        raise RuntimeError("No se pudo leer ningún slice")

    volume = np.stack(slices_data, axis=0)
    print(f"[*] Volumen shape: {volume.shape}, dtype={volume.dtype}")
    print(f"[*] Range: [{volume.min():.1f}, {volume.max():.1f}], mean={volume.mean():.1f}")

    # Spacing
    try:
        pixel_spacing = getattr(ds0, 'PixelSpacing', None) or [6.8, 6.8]
        slice_thickness = getattr(ds0, 'SliceThickness', pixel_spacing[0])
        spacing_zyx = (float(slice_thickness), float(pixel_spacing[1]), float(pixel_spacing[0]))
    except Exception:
        spacing_zyx = (6.8, 6.8, 6.8)

    print(f"[*] Spacing ZYX: ({spacing_zyx[0]:.3f}, {spacing_zyx[1]:.3f}, {spacing_zyx[2]:.3f}) mm")

    return volume, spacing_zyx, ds0


def _run_batch_validation(volume, spacing_zyx, points_dict: dict[str, tuple[int,int,int]]):
    """Ejecuta compute_spect_ratio con puntos dados e imprime resultados."""
    from core.amyloid_spect import VOISphere, compute_spect_ratio, SvdRatioResult

    print("\n" + "="*60)
    print("VALIDACIÓN RATIO S/VD — MODO BATCH")
    print("="*60)

    # Crear VOIs esféricas
    cz_s, cy_s, cx_s = points_dict.get("S", (32, 32, 32))
    cz_v, cy_v, cx_v = points_dict.get("V", (40, 32, 32))
    cz_d, cy_d, cx_d = points_dict.get("D", (48, 32, 32))

    voi_s = VOISphere(cz=cz_s, cy=cy_s, cx=cx_s, radius_mm=15.0)
    voi_v = VOISphere(cz=cz_v, cy=cy_v, cx=cx_v, radius_mm=10.0)
    voi_d = VOISphere(cz=cz_d, cy=cy_d, cx=cx_d, radius_mm=8.0)

    print(f"\nVOI S (corazón):   center=({cz_s},{cy_s},{cx_s}) r=15mm")
    print(f"VOI V (vértebra):   center=({cz_v},{cy_v},{cx_v}) r=10mm")
    print(f"VOI D (aorta):      center=({cz_d},{cy_d},{cx_d}) r=8mm")

    result = compute_spect_ratio(
        volume=volume,
        spacing_zyx=spacing_zyx,
        voi_heart=voi_s,
        voi_vertebra=voi_v,
        voi_aorta=voi_d,
    )

    # Imprimir resultados formateados
    print(f"\n{'─'*60}")
    print(f"RESULTADO RATIO S/VD")
    print(f"{'─'*60}")
    print(f"  S/√(V×D) = {result.s_vd:.4f}  →  {result.classification}")
    print(f"  S/V      = {result.s_v:.4f}")
    print(f"  S/D      = {result.s_d:.4f}")
    print(f"  V/D      = {result.v_d:.4f}")
    print(f"\n  Cuentas:")
    print(f"    S (corazón):  {result.heart_counts:.1f}  ({result.s_voxels} voxels)")
    print(f"    V (vértebra): {result.vertebra_counts:.1f}  ({result.v_voxels} voxels)")
    print(f"    D (aorta):    {result.aorta_counts:.1f}  ({result.d_voxels} voxels)")
    print(f"\n  Medias (cuentas/voxel):")
    print(f"    S = {result.s_mean:.2f}, V = {result.v_mean:.2f}, D = {result.d_mean:.2f}")
    print(f"\n  Volúmenes:")
    print(f"    S = {result.heart_volume_ml:.2f} mL")
    print(f"    V = {result.vertebra_volume_ml:.2f} mL")
    print(f"    D = {result.aorta_volume_ml:.2f} mL")
    print(f"{'─'*60}")

    # Validación básica de sanidad
    warnings = []
    if result.s_vd <= 0:
        warnings.append("⚠ S/VD ≤ 0 — revisar coordenadas o volumen vacío")
    if result.heart_counts <= 0:
        warnings.append("⚠ Cuentas corazón ≤ 0 — punto S fuera del volumen activo")
    if result.vertebra_counts <= 0:
        warnings.append("⚠ Cuentas vértebra ≤ 0 — punto V fuera del volumen activo")
    if result.aorta_counts <= 0:
        warnings.append("⚠ Cuentas aorta ≤ 0 — punto D fuera del volumen activo")
    if result.s_vd > 10:
        warnings.append("⚠ S/VD > 10 — valor inusualmente alto, revisar")

    if warnings:
        print("\n⚠ ADVERTENCIAS:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("\n✅ Sin advertencias — valores dentro de rangos esperados.")

    return result


def main():
    parser = argparse.ArgumentParser(description="Validación ratio S/VD SINCRO")
    parser.add_argument("--study", "-s", help="Ruta a archivo .dcm o carpeta con estudio SPECT")
    parser.add_argument("--ct", "-c", help="Ruta a CT registrado (opcional)")
    parser.add_argument("--points", "-p",
                        help="Coordenadas S/V/D como 'Z,Y,X:Z,Y,X:Z,Y,X' (ej: 32,32,32:40,32,32:48,32,32)",
                        default=None)
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Abrir panel interactivo completo")
    args = parser.parse_args()

    if args.interactive or not args.study:
        print("[*] Modo interactivo — abriendo panel AMYLO SPECT...")
        from PyQt6.QtWidgets import QApplication
        app = QApplication(sys.argv)
        from ui.amyloid_spect_panel import AmyloidSpectPanel
        dlg = AmyloidSectPanel()
        dlg.exec()
        return

    # Modo batch
    volume, spacing_zyx, ds = _load_spect_volume(args.study)

    # Parsear puntos si se dieron
    points = {}
    if args.points:
        parts = args.points.split(":")
        labels = ["S", "V", "D"]
        for i, part in enumerate(parts):
            if i >= len(labels):
                break
            coords = [int(x.strip()) for x in part.split(",")]
            if len(coords) == 3:
                points[labels[i]] = tuple(coords)
    else:
        # Puntos centrados por defecto (ajustar según shape)
        sz = volume.shape
        cz, cy, cx = sz[0] // 2, sz[1] // 2, sz[2] // 2
        points = {
            "S": (cz, cy, cx),
            "V": (cz + 5, cy - 5, cx),
            "D": (cz + 10, cy + 5, cx),
        }
        print(f"[*] Usando puntos centrados por defecto: {points}")

    result = _run_batch_validation(volume, spacing_zyx, points)

    # También validar HMR si hay puntos A/B
    print("\n" + "="*60)
    print("NOTA: Para validación completa con HMR, usar modo interactivo (-i)")
    print("      y depositar puntos A/B además de S/V/D.")
    print("="*60)


# Alias para evitar error de tipeo en el import condicional
AmyloidSectPanel = None


if __name__ == "__main__":
    # Resolver alias antes de ejecutar
    if AmyloidSectPanel is None:
        def _launch_interactive():
            from PyQt6.QtWidgets import QApplication
            app = QApplication(sys.argv)
            from ui.amyloid_spect_panel import AmyloidSpectPanel
            dlg = AmyloidSpectPanel()
            dlg.exec()

        # Reemplazar main para modo interactivo
        original_main = main
        def patched_main():
            parser = argparse.ArgumentParser(description="Validación ratio S/VD SINCRO")
            parser.add_argument("--study", "-s", default=None)
            parser.add_argument("--ct", "-c", default=None)
            parser.add_argument("--points", "-p", default=None)
            parser.add_argument("--interactive", "-i", action="store_true")
            args = parser.parse_args()
            if args.interactive or not args.study:
                print("[*] Modo interactivo — abriendo panel AMYLO SPECT...")
                _launch_interactive()
            else:
                original_main()
        main = patched_main

    main()

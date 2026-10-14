"""Comparación de FILTROS de FBP sobre el Xeleris 3 (datos sumados no-gated).

Hipótesis: las estrías radiales del transaxial vienen del FBP con rampa cruda
(Ram-Lak) sobre 60 vistas (sub-Nyquist para 64px). Evolution las suaviza con
apodización fuerte. Comparamos, sobre el MISMO sinograma:

  1) actual: Butterworth 0.4/10 pre + iradon rampa (Ram-Lak)
  2) Butterworth 0.4/10 pre + iradon 'hann'
  3) sin pre + iradon 'hann'
  4) sin pre + iradon 'cosine'
  5) Butterworth 0.3/8 (más agresivo) pre + iradon 'shepp-logan'

Mostramos 6 cortes centrales de cada variante, ventaneados por corte, para ver
cuál da un miocardio compacto / anillo limpio como Evolution.

Uso:
    cd "d:\\- PROGRAMACIÓN\\PROYECTO_MN\\mod_SINCRO"
    & ".\\.venv\\Scripts\\python.exe" _xel3_filter_cmp.py
"""
from __future__ import annotations

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.raw_projections import load_raw_projections
from core.raw_reconstruction import (
    ProjectionFilterConfig,
    filter_projections,
    reconstruct_fbp_volume,
)

BASE = r"D:\- GAMMASYS\estudios evolution\estudio prueba evolution xeleris 3\1.2.124.113532.10.1.84.45.20070719.100230.2565043"
# Tomográfico SUMADO (no-gated): mejor SNR que promediar gates.
SUMMED = os.path.join(BASE, "Stress-10sec-1_T_EM001_DS.dcm")
GATED = os.path.join(BASE, "Stress-10sec-1_G_EM_1001_DS.dcm")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_xel3_out")


def recon(proj, angles, pre, fbp_filter):
    p = proj
    if pre is not None:
        p = filter_projections(proj[None], pre)[0] if proj.ndim == 3 else proj
        p = filter_projections(proj, pre)
    return reconstruct_fbp_volume(p, angles, projection_filter=None, fbp_filter_name=fbp_filter)


def central_row(vol, title, cmap="hot"):
    """Devuelve (fig-listo) 6 cortes centrales ventaneados por corte."""
    n = vol.shape[0]
    idxs = np.linspace(n * 0.30, n * 0.70, 6).astype(int)
    return idxs


def reference_sa_geometry(baseline, transverse_study, transverse_header, sa_header):
    from scipy.signal import correlate

    target = transverse_study.cube.sum(axis=0).astype(np.float64)
    template = target - target.mean()
    template_energy = float(np.sum(template ** 2))
    window = np.ones(target.shape, dtype=np.float64)
    padding = 16
    best = None
    summed = baseline.sum(axis=0).astype(np.float64)
    for transpose_xy in (False, True):
        for flip_z in (False, True):
            for flip_y in (False, True):
                for flip_x in (False, True):
                    candidate = summed.swapaxes(1, 2) if transpose_xy else summed
                    candidate = candidate[::(-1 if flip_z else 1), ::(-1 if flip_y else 1), ::(-1 if flip_x else 1)]
                    candidate = np.pad(candidate, ((0, 0), (padding, padding), (padding, padding)))
                    numerator = correlate(candidate, template, mode="valid", method="fft")
                    local_sum = correlate(candidate, window, mode="valid", method="fft")
                    local_energy = correlate(candidate ** 2, window, mode="valid", method="fft") - local_sum ** 2 / template.size
                    scores = numerator / np.sqrt(np.maximum(local_energy * template_energy, 1e-12))
                    position = np.unravel_index(np.argmax(scores), scores.shape)
                    score = float(scores[position])
                    if best is None or score > best[0]:
                        best = (score, int(position[0]), transpose_xy, flip_z, flip_y, flip_x,
                                int(position[1]) - padding, int(position[2]) - padding)

    def frame(header):
        detector = header.DetectorInformationSequence[0]
        orientation = np.asarray(detector.ImageOrientationPatient, dtype=float)
        spacing = np.asarray(header.PixelSpacing, dtype=float)
        basis = np.column_stack((np.cross(orientation[:3], orientation[3:]) * float(header.SpacingBetweenSlices),
                                 orientation[3:] * spacing[0], orientation[:3] * spacing[1]))
        return basis, np.asarray(detector.ImagePositionPatient, dtype=float)

    transverse_basis, transverse_origin = frame(transverse_header)
    sa_basis, sa_origin = frame(sa_header)
    matrix = np.linalg.solve(transverse_basis, sa_basis)
    offset = np.linalg.solve(transverse_basis, sa_origin - transverse_origin)
    offset += np.asarray((best[1], best[6], best[7]))
    return best, matrix, offset


def apply_reference_sa(cube, geometry, matrix, offset, shape):
    from scipy.ndimage import affine_transform

    _, _, transpose_xy, flip_z, flip_y, flip_x = geometry[:6]
    oriented = cube.swapaxes(2, 3) if transpose_xy else cube
    oriented = oriented[:, ::(-1 if flip_z else 1), ::(-1 if flip_y else 1), ::(-1 if flip_x else 1)]
    return np.stack([
        affine_transform(gate, matrix, offset=offset, output_shape=shape, order=1, mode="constant", cval=0.0)
        for gate in oriented
    ])


def compare_gated_filters(study_dir, output_dir):
    import pydicom
    from core.dicom_loader import load
    from core.raw_reconstruction import reconstruct_gated_projection_volume, _apply_post_butterworth
    from core.ectb_lv import ECTbLVConfig, analyze_lv_ectb
    from core.gate_dropout import correct_last_gate_dropout
    from core.segmentation import segment_myocardium

    root = Path(study_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    variants = [
        ("standard_040_ramp", 0.40, "ramp"),
        ("smooth_030_ramp", 0.30, "ramp"),
        ("sharp_050_ramp", 0.50, "ramp"),
        ("standard_040_hann", 0.40, "hann"),
        ("osem4x4_none", None, None),
        ("osem4x4_bw030", 0.30, None),
        ("osem4x4_bw040", 0.40, None),
        ("osem4x4_bw050", 0.50, None),
        ("osem8x4_bw040", 0.40, None),
    ]
    report = {
        "protocol": "FBP and OSEM, fixed DICOM SA geometry and segmentation; no clinical calibration",
        "excluded": "RR, Denoise+, motion correction and additional postfilters",
        "cutoff_units": "fraction of Nyquist", "butterworth_order": 10,
        "stages": {},
    }
    for stage, filename in (("stress", "SGATE_G_1001_DS.dcm"), ("rest", "RGATE_G_1001_DS.dcm")):
        raw = load_raw_projections(str(root / filename))
        projections, dropout = correct_last_gate_dropout(raw.projections)
        baseline = np.stack([
            reconstruct_fbp_volume(gate, raw.angles_deg,
                                   projection_filter=ProjectionFilterConfig("butterworth", 0.40, 10),
                                   fbp_filter_name="ramp")
            for gate in projections
        ])
        prefix = "STRESS" if stage == "stress" else "REST"
        transverse_path = root / f"{prefix}_FBP_1001_DS.dcm"
        sa_path = root / f"{prefix}_FBP_SA_1001_DS.dcm"
        transverse_study = load(str(transverse_path))
        sa_reference = load(str(sa_path))
        geometry, matrix, offset = reference_sa_geometry(
            baseline, transverse_study, pydicom.dcmread(transverse_path, stop_before_pixels=True),
            pydicom.dcmread(sa_path, stop_before_pixels=True),
        )
        print(f"{stage}: alignment NCC={geometry[0]:.3f}, transform={geometry[1:]}", flush=True)
        baseline_sa = apply_reference_sa(baseline, geometry, matrix, offset, sa_reference.cube.shape[1:])
        segmentation = segment_myocardium(baseline_sa)
        header = pydicom.dcmread(root / filename, stop_before_pixels=True)
        pixel_mm = tuple(float(value) for value in header.PixelSpacing)
        if not np.isclose(pixel_mm[0], pixel_mm[1]):
            raise ValueError("This comparison requires isotropic detector pixels")
        baseline_result = analyze_lv_ectb(baseline_sa, segmentation, pixel_mm, pixel_mm[0], ECTbLVConfig())
        if not baseline_result.available:
            raise ValueError(f"{stage}: baseline quantification unavailable: {baseline_result.reason}")
        fixed_gates = (baseline_result.ed_gate - 1, baseline_result.es_gate - 1)
        valid = np.asarray(baseline_result.valid_slices, dtype=int)
        display_slice = int(valid[len(valid) // 2])
        coordinates_y, coordinates_x = np.indices(baseline_sa.shape[-2:])
        distance = np.hypot(
            coordinates_y[None] - segmentation.center_per_slice[:, 0, None, None],
            coordinates_x[None] - segmentation.center_per_slice[:, 1, None, None],
        )
        cavity_mask = distance < (0.5 * segmentation.inner_radius[:, None, None])
        cavity_mask &= np.isin(np.arange(len(distance)), valid)[:, None, None]
        wall_mask = segmentation.mask.copy()
        wall_mask &= np.isin(np.arange(len(distance)), valid)[:, None, None]
        background_mask = (distance > segmentation.outer_radius[:, None, None] + 2.0)
        background_mask &= distance < segmentation.outer_radius[:, None, None] + 5.0
        background_mask &= np.isin(np.arange(len(distance)), valid)[:, None, None]
        if not cavity_mask.any() or not wall_mask.any() or not background_mask.any():
            raise ValueError(f"{stage}: masks do not resolve wall/cavity/background")
        stage_report = {
            "geometry_source": "exported DICOM SA geometry; baseline transaxial alignment fixed across variants",
            "alignment_NCC": geometry[0], "alignment_parameters": list(geometry[1:]),
            "reslice_matrix": matrix.tolist(), "reslice_offset": offset.tolist(),
            "voxel_mm": pixel_mm[0], "dropout_applied": bool(dropout["applied"]),
            "fixed_ED_ES_gates": [gate + 1 for gate in fixed_gates],
            "fixed_display_slice": display_slice, "valid_slices": valid.tolist(), "variants": [],
        }
        figure, axes = plt.subplots(len(variants), 4, figsize=(12, 2.5 * len(variants)), squeeze=False)
        display_max = float(np.percentile(baseline_sa[fixed_gates[0]][wall_mask], 99))
        osem_volumes = {}
        for row, (name, cutoff, ramp_filter) in enumerate(variants):
            print(f"{stage}: {name}", flush=True)
            iterations = 8 if name.startswith("osem8") else 4
            if row == 0:
                sa = baseline_sa
            elif name.startswith("osem"):
                if iterations not in osem_volumes:
                    osem_volumes[iterations] = reconstruct_gated_projection_volume(
                        projections, raw.angles_deg, method="osem", iterations=iterations, subsets=4,
                    )
                volume = np.stack([
                    _apply_post_butterworth(gate, None if cutoff is None else (cutoff, 10), None)
                    for gate in osem_volumes[iterations]
                ])
                sa = apply_reference_sa(volume, geometry, matrix, offset, sa_reference.cube.shape[1:])
            else:
                volume = np.stack([
                    reconstruct_fbp_volume(gate, raw.angles_deg,
                                           projection_filter=ProjectionFilterConfig("butterworth", cutoff, 10),
                                           fbp_filter_name=ramp_filter)
                    for gate in projections
                ])
                sa = apply_reference_sa(volume, geometry, matrix, offset, sa_reference.cube.shape[1:])
            result = analyze_lv_ectb(sa, segmentation, pixel_mm, pixel_mm[0], ECTbLVConfig())
            measurements = []
            for gate in fixed_gates:
                image = sa[gate]
                wall_mean = float(np.mean(image[wall_mask]))
                cavity_mean = float(np.mean(image[cavity_mask]))
                measurements.append({
                    "gate": gate + 1, "wall_mean": wall_mean, "cavity_mean": cavity_mean,
                    "wall_cavity_contrast": (wall_mean - cavity_mean) / max(wall_mean, 1e-12),
                    "background_std_relative": float(np.std(image[background_mask])) / max(wall_mean, 1e-12),
                })
            entry = {
                "name": name, "cutoff": cutoff, "fbp_filter": ramp_filter,
                "method": "osem" if name.startswith("osem") else "fbp",
                "iterations": iterations if name.startswith("osem") else None,
                "subsets": 4 if name.startswith("osem") else None,
                "available": bool(result.available), "reason": result.reason,
                "EF_pct": result.ef_pct if result.available else None,
                "EDV_ml": result.edv_ml if result.available else None,
                "ESV_ml": result.esv_ml if result.available else None,
                "ED_ES_gates": [result.ed_gate, result.es_gate],
                "gate_volumes_ml": result.gate_volumes_ml.tolist(),
                "endo_zero_fraction": result.endo_zero_fraction.tolist(),
                "measurements": measurements, "notes": result.notes,
            }
            stage_report["variants"].append(entry)
            for column, gate in enumerate(fixed_gates):
                panel = axes[row, column]
                panel.imshow(sa[gate, display_slice], cmap="hot", vmin=0, vmax=display_max)
                panel.set_title(f"{name} | {'ED' if column == 0 else 'ES'} fixed gate {gate + 1}", fontsize=8)
                if result.available and display_slice in result.valid_slices:
                    slice_index = result.valid_slices.index(display_slice)
                    theta = np.linspace(0, 2 * np.pi, result.endo_radii_mm.shape[-1] + 1)
                    center_y, center_x = segmentation.center_per_slice[display_slice]
                    for radii, color in ((result.endo_radii_mm, "cyan"), (result.epi_radii_mm, "lime")):
                        values = np.r_[radii[gate, slice_index], radii[gate, slice_index, 0]] / pixel_mm[0]
                        panel.plot(center_x + values * np.cos(theta), center_y + values * np.sin(theta), color=color, linewidth=0.7)
                panel.axis("off")
            curve = axes[row, 2]
            if result.available:
                curve.plot(np.arange(1, len(result.gate_volumes_ml) + 1), result.gate_volumes_ml, "o-")
            curve.set(xlabel="Gate", ylabel="mL", title="Volume, fixed segmentation")
            text = axes[row, 3]
            text.axis("off")
            lines = [f"EF: {result.ef_pct:.1f}%", f"EDV / ESV: {result.edv_ml:.1f} / {result.esv_ml:.1f} mL"]
            for measurement in measurements:
                lines.append(f"Gate {measurement['gate']} contrast: {measurement['wall_cavity_contrast']:.3f}")
                lines.append(f"Relative background SD: {measurement['background_std_relative']:.3f}")
            if result.available:
                lines.append(f"ES zero radii: {100 * result.endo_zero_fraction[result.es_gate - 1]:.1f}%")
            else:
                lines.append(result.reason)
            text.text(0, 0.9, "\n".join(lines), va="top", fontsize=9)
            print(json.dumps(entry, ensure_ascii=True), flush=True)
        figure.suptitle(f"{stage.upper()} FBP / OSEM | fixed DICOM geometry, masks and display | diagnostic only")
        figure.tight_layout()
        figure.savefig(output / f"{stage}_filter_comparison.png", dpi=130)
        plt.close(figure)
        report["stages"][stage] = stage_report
    with (output / "filter_comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True, allow_nan=False)
    print(f"Results: {output}")
    return 0


def compare_osem_filters(study_dir, output_dir):
    import pydicom
    from scipy.ndimage import gaussian_filter
    from core.dicom_loader import load
    from core.gate_dropout import correct_last_gate_dropout
    from core.raw_reconstruction import (
        reconstruct_gated_projection_volume, _apply_post_butterworth, _butterworth_3d,
        _detect_rotation_ccw, _FLIP_X_ON_CCW,
    )

    root = Path(study_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "Real projections, OSEM 4x4 and 8x4; filter functionality, not clinical validation",
        "excluded": "RR, MAP, Denoise+, motion correction and clinical FEVI",
        "quantification_validated": False, "stages": {},
    }
    for stage, filename, prefix in (("stress", "SGATE_G_1001_DS.dcm", "STRESS"),
                                     ("rest", "RGATE_G_1001_DS.dcm", "REST")):
        raw = load_raw_projections(str(root / filename))
        projections, dropout = correct_last_gate_dropout(raw.projections)
        original = projections.copy()
        print(f"{stage}: OSEM 4x4 unfiltered", flush=True)
        baseline = reconstruct_gated_projection_volume(projections, raw.angles_deg, method="osem", iterations=4, subsets=4)
        variants = [("none", baseline)]
        for cutoff in (0.30, 0.40, 0.50):
            variants.append((f"butterworth_{cutoff:.2f}_10", np.stack([
                _apply_post_butterworth(gate, (cutoff, 10), None) for gate in baseline
            ])))
        for kind in ("lowpass", "wiener"):
            print(f"{stage}: {kind}", flush=True)
            variants.append((kind, reconstruct_gated_projection_volume(
                projections, raw.angles_deg, method="osem", iterations=4, subsets=4,
                projection_filter=ProjectionFilterConfig(kind, 0.40, 10),
            )))
        print(f"{stage}: OSEM 8x4", flush=True)
        variants.append(("osem8x4_bw040_10", reconstruct_gated_projection_volume(
            projections, raw.angles_deg, method="osem", iterations=8, subsets=4,
            projection_filter=ProjectionFilterConfig("butterworth", 0.40, 10),
        )))
        variants.append(("additional_gaussian_sigma1", np.stack([
            gaussian_filter(gate, 1.0, mode="constant") for gate in variants[2][1]
        ])))
        variants.append(("additional_butterworth035_5", np.stack([
            np.clip(_butterworth_3d(gate, 0.35, 5), 0.0, None) for gate in variants[2][1]
        ])))
        np.testing.assert_array_equal(projections, original)

        transverse_path = root / f"{prefix}_FBP_1001_DS.dcm"
        sa_path = root / f"{prefix}_FBP_SA_1001_DS.dcm"
        sa_reference = load(str(sa_path))
        fbp = np.stack([reconstruct_fbp_volume(gate, raw.angles_deg,
                       projection_filter=ProjectionFilterConfig("butterworth", 0.40, 10)) for gate in projections])
        geometry, matrix, offset = reference_sa_geometry(
            fbp, load(str(transverse_path)), pydicom.dcmread(transverse_path, stop_before_pixels=True),
            pydicom.dcmread(sa_path, stop_before_pixels=True),
        )
        reference_center = (np.asarray(sa_reference.cube.shape[1:]) - 1) / 2.0
        source_center = matrix @ reference_center + offset
        source_z = baseline.shape[1] - 1 - source_center[0] if geometry[3] else source_center[0]
        display_slice = int(np.clip(round(source_z), 0, baseline.shape[1] - 1))
        ccw = _detect_rotation_ccw(raw.angles_deg)
        flip = True if ccw is None else bool(ccw) == _FLIP_X_ON_CCW
        display_max = float(np.percentile(baseline[:, display_slice], 99.8))
        frequencies = np.meshgrid(*[np.fft.fftfreq(size) / 0.5 for size in baseline.shape[1:]], indexing="ij")
        high = sum(axis ** 2 for axis in frequencies) > 0.5 ** 2
        norm = float(np.linalg.norm(baseline.ravel()))
        figure, axes = plt.subplots(len(variants), 3, figsize=(12, len(variants) * 2.4), squeeze=False)
        entries = []
        for row, (name, volume) in enumerate(variants):
            if volume.shape != baseline.shape or not np.isfinite(volume).all() or np.min(volume) < 0:
                raise ValueError(f"{stage}/{name}: invalid iterative filter output")
            power = np.abs(np.fft.fftn(volume, axes=(1, 2, 3))) ** 2
            entry = {
                "name": name, "shape": list(volume.shape), "finite": True, "nonnegative": True,
                "relative_L2_to_unfiltered": float(np.linalg.norm((volume - baseline).ravel()) / norm),
                "high_frequency_energy_fraction": float(power[:, high].sum() / power.sum()),
                "gate_total_activity": volume.sum(axis=(1, 2, 3)).tolist(),
            }
            entries.append(entry)
            for column, gate in enumerate((0, 3)):
                image = volume[gate, display_slice]
                axes[row, column].imshow(image[:, ::-1] if flip else image, cmap="hot", vmin=0, vmax=display_max)
                axes[row, column].set_title(f"{name} | gate {gate + 1} | TA slice {display_slice}", fontsize=8)
                axes[row, column].axis("off")
            axes[row, 2].axis("off")
            axes[row, 2].text(0, 0.85, f"L2 difference: {entry['relative_L2_to_unfiltered']:.3f}\n"
                f"High-frequency fraction: {entry['high_frequency_energy_fraction']:.5f}\nFinite, nonnegative: yes",
                va="top", fontsize=9)
            print(f"{stage}/{name}: difference={entry['relative_L2_to_unfiltered']:.3f}, nonnegative=True", flush=True)
        figure.suptitle(f"{stage.upper()} | OSEM filters | fixed TA slice and display | no clinical EF")
        figure.tight_layout(rect=(0, 0, 1, 0.985))
        figure.savefig(output / f"{stage}_osem_filter_checks.png", dpi=130)
        plt.close(figure)
        report["stages"][stage] = {
            "dropout_applied": bool(dropout["applied"]), "display_slice": display_slice,
            "display_source": "exported SA center mapped through fixed FBP transaxial alignment; not validated manual LV localization",
            "alignment_NCC": geometry[0], "display_gates": [1, 4],
            "high_frequency_threshold": "radial > 0.5 Nyquist; not a clinical noise estimate", "variants": entries,
        }
    with (output / "osem_filter_checks.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f"Results: {output}", flush=True)
    return 0


def reslice_snapshot_grid(volume, orientation, source_scale, *, output_scale=1.0,
                          source_origin=(0.0, 0.0, 0.0)):
    from core.cardiac_reorientation import reslice_from_vector_gated

    center = (np.asarray(orientation["center"], dtype=float) - source_origin) / source_scale
    sample_scale = float(orientation["sample_scale"]) * output_scale / source_scale
    output_size = int(round(int(orientation["out_size"]) / output_scale))
    return reslice_from_vector_gated(volume, tuple(center), np.asarray(orientation["long_axis"]),
                                    output_size, sample_scale=sample_scale)


def compare_snapshot_matrices(snapshot_dir, output_dir, study_dir):
    import hashlib
    from dataclasses import asdict, replace
    from core.raw_reconstruction import RawReconConfig, reconstruct_raw_gated_pipeline
    from core.ectb_lv import ECTbLVConfig, analyze_lv_ectb
    from core.segmentation import segment_myocardium

    snapshot = Path(snapshot_dir)
    metadata = json.loads((snapshot / "meta.json").read_text(encoding="utf-8"))
    orientation = json.loads((snapshot / "reorient.json").read_text(encoding="utf-8"))
    source = Path(metadata["source_path"])
    if source.parent.resolve() != Path(study_dir).resolve() or not source.is_file():
        raise ValueError("Snapshot does not belong to the requested DICOM directory")
    if metadata.get("schema_version") != 2 or orientation.get("schema_version") != 2:
        raise ValueError("Schema2 snapshot required")
    if metadata["stage"] != orientation["stage"] or metadata["reconstruction_voxel_scale"] != 1.0:
        raise ValueError("Matching stages and a native snapshot are required")
    with (snapshot / "gated_volume.npy").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != metadata["reconstruction_sha256"] or digest != orientation["reconstruction_sha256"]:
        raise ValueError("Reconstruction changed after orientation")
    if metadata.get("rr_model_required"):
        raise ValueError("Saved PSF is not sufficient to reproduce RR")
    if orientation["thickness"] != 1 or orientation.get("voi_semiaxes") is None:
        raise ValueError("Single-slice thickness and saved VOI semiaxes required")
    if not np.allclose(metadata["pixel_spacing"], metadata["pixel_spacing"][0]):
        raise ValueError("Isotropic source spacing required")
    projections = np.load(snapshot / "corrected_projections.npy", allow_pickle=False)
    angles = np.load(snapshot / "angles_deg.npy", allow_pickle=False)
    original = projections.copy()
    saved_volume = np.load(snapshot / "gated_volume.npy", allow_pickle=False)
    saved_sa = np.load(snapshot / "reoriented_gated.npy", allow_pickle=False)
    config_values = dict(metadata["config"])
    for key in ("ungated_filter", "gated_filter"):
        config_values[key] = ProjectionFilterConfig(**config_values[key])
    config = RawReconConfig(**config_values)
    if config.reconstruction_method != "fbp" or (config.gated_method or "fbp") != "fbp":
        raise ValueError("This physical-origin comparison currently supports FBP snapshots only")
    config = replace(config, background_subtract=False, scatter_subtract=False)
    motion = {"applied_shifts_y": np.zeros(projections.shape[1]),
              "applied_shifts_x": np.zeros(projections.shape[1])}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    entries, runs, replay = {}, {}, {}
    quant_config = ECTbLVConfig()
    base, apex = sorted((int(orientation["base_k"]), int(orientation["apex_k"])))
    if not 0 <= base <= apex < int(orientation["out_size"]):
        raise ValueError("Saved slice bounds outside SA grid")
    operations = None

    def postops(cube, ops):
        turns, flip_x, flip_z = ops
        transformed = np.rot90(cube, turns, axes=(-2, -1))
        if flip_x:
            transformed = transformed[..., ::-1]
        if flip_z:
            transformed = transformed[:, ::-1]
        return transformed

    for matrix in (64, 128):
        if projections.shape[-1] != 64:
            raise ValueError("This comparison currently requires an acquired matrix64")
        print(f"{metadata['stage']}: reconstructing {matrix}", flush=True)
        result = reconstruct_raw_gated_pipeline(projections, angles, motion_result=motion,
                                               config=replace(config, reconstruction_matrix=None if matrix == 64 else matrix))
        scale = float(result.voxel_scale)
        origin = np.array([0.0, 0.0, scale - 1.0 if result.flip_x_applied else 0.0])
        center = (np.asarray(orientation["center"]) - origin) / scale
        semiaxes = np.asarray(orientation["voi_semiaxes"]) / scale
        axes = np.ogrid[tuple(slice(0, size) for size in result.gated_volume.shape[1:])]
        mask = sum(((axis - center[index]) / semiaxes[index]) ** 2
                   for index, axis in enumerate(axes)) <= 1.0
        volume = result.gated_volume * mask[None]
        if matrix == 64:
            replay["native_volume_relative_L2"] = float(np.linalg.norm(result.gated_volume - saved_volume)
                                                        / max(np.linalg.norm(saved_volume), 1e-12))
            candidate = reslice_snapshot_grid(volume, orientation, scale)
            candidates = [(turns, flip_x, flip_z) for turns in range(4)
                          for flip_x in (False, True) for flip_z in (False, True)]
            operations = min(candidates, key=lambda ops: np.linalg.norm(postops(candidate, ops) - saved_sa))
            replay["inferred_display_postops"] = list(operations)
            replay["manual_SA_relative_L2"] = float(np.linalg.norm(postops(candidate, operations) - saved_sa)
                                                   / max(np.linalg.norm(saved_sa), 1e-12))
        for grid_scale in (1.0, 0.5):
            name = f"recon{matrix}_sa{int(orientation['out_size'] / grid_scale)}"
            cube = postops(reslice_snapshot_grid(volume, orientation, scale,
                                                output_scale=grid_scale, source_origin=origin), operations)
            start, stop = int(base / grid_scale), int((apex + 1) / grid_scale)
            selected = cube[:, start:stop].copy()
            voxel = float(orientation["voxel_mm"]) * grid_scale
            segmentation = segment_myocardium(selected)
            quant = analyze_lv_ectb(selected, segmentation, (voxel, voxel), voxel, quant_config)
            runs[name] = (selected, segmentation, quant, voxel)
            entries[name] = {
                "available": bool(quant.available), "reason": quant.reason,
                "voxel_mm": voxel, "selected_shape": list(selected.shape),
                "EF_pct": float(quant.ef_pct) if quant.available else None,
                "EDV_ml": float(quant.edv_ml) if quant.available else None,
                "ESV_ml": float(quant.esv_ml) if quant.available else None,
                "ED_ES_gates_one_based": [quant.ed_gate + 1, quant.es_gate + 1],
                "gate_volumes_ml": quant.gate_volumes_ml.tolist(),
                "endo_zero_fraction": quant.endo_zero_fraction.tolist(),
                "valid_slices": list(quant.valid_slices), "notes": quant.notes,
            }
            np.save(output / f"{name}_sa.npy", selected)
            print(name, entries[name]["EF_pct"], entries[name]["EDV_ml"], entries[name]["ESV_ml"], flush=True)
    np.testing.assert_array_equal(projections, original)
    names = list(runs)
    reference_name, fine_name = names[0], names[-1]
    native, native_seg, native_quant, native_voxel = runs[reference_name]
    fine, fine_seg, fine_quant, fine_voxel = runs[fine_name]
    if native_quant.available and fine_quant.available:
        common = [index for index in native_quant.valid_slices
                  if min(fine_quant.valid_slices) <= 2 * index + 0.5 <= max(fine_quant.valid_slices)]
        display = [common[index] for index in np.unique(np.linspace(0, len(common) - 1, min(4, len(common))).astype(int))] if common else []

        def contours(segmentation, quant, voxel, gate, position):
            centers = np.asarray(segmentation.center_per_slice)
            center_y, center_x = [np.interp(position, np.arange(len(centers)), centers[:, axis])
                                  for axis in (0, 1)]
            width = native.shape[-1] if voxel == native_voxel else fine.shape[-1]
            theta = np.linspace(0, 2 * np.pi, quant.endo_radii_mm.shape[-1] + 1)
            lines = []
            for radii in (quant.endo_radii_mm, quant.epi_radii_mm):
                values = np.array([np.interp(position, quant.valid_slices, radii[gate, :, angle])
                                   for angle in range(radii.shape[-1])])
                values = np.r_[values, values[0]]
                lines.append(((center_x - (width - 1) / 2) * voxel + values * np.cos(theta),
                              (center_y - (width - 1) / 2) * voxel + values * np.sin(theta)))
            return lines

        for label, gate in (("ED", native_quant.ed_gate), ("ES", native_quant.es_gate)):
            if not display:
                break
            figure, panels = plt.subplots(3, len(display), figsize=(4 * len(display), 10), squeeze=False)
            for column, index in enumerate(display):
                position = 2 * index + 0.5
                fine_image = (fine[gate, int(np.floor(position))] + fine[gate, int(np.ceil(position))]) / 2
                extent_mm = native.shape[-1] * native_voxel / 2
                for row, image in enumerate((native[gate, index], fine_image, native[gate, index])):
                    panel = panels[row, column]
                    maximum = max(float(np.percentile(native if row != 1 else fine, 99.5)), 1e-12)
                    panel.imshow(image, cmap="hot", vmin=0, vmax=maximum,
                                 extent=(-extent_mm, extent_mm, extent_mm, -extent_mm))
                    if row != 1:
                        for line in contours(native_seg, native_quant, native_voxel, gate, index):
                            panel.plot(*line, color="cyan", linewidth=1.2, label="64")
                    if row != 0:
                        for line in contours(fine_seg, fine_quant, fine_voxel, gate, position):
                            panel.plot(*line, color="lime", linewidth=1.2, linestyle="--", label="128")
                    panel.set_title(f"{('64', '128', 'Overlay')[row]} | SA {base + index + 1}")
                    panel.set_xlabel("mm")
                    panel.set_ylabel("mm")
            figure.suptitle(f"{metadata['stage'].upper()} {label} gate{gate + 1} | cyan64, green128 | diagnostic only\n"
                           "Fixed physical geometry; fine contours interpolated to shared SA planes; no clinical validation")
            figure.tight_layout(rect=(0, 0, 1, 0.94))
            figure.savefig(output / f"contours_{label}.png", dpi=140)
            plt.close(figure)
    report = {"clinical_validation": False, "stage": metadata["stage"],
              "protocol": "Corrected projections reused without double motion/background/scatter; fixed physical axis, center, VOI and SA cell coverage",
              "limitations": ["VOI center assumed equal to saved reslice center; separate VOI center was not recorded",
                              "Display postops inferred from native snapshot; no registration/geometry optimization",
                              "Fixed ECTb diagnostic defaults, not screen FEVI; fixed pixel kernels may have different physical supports",
                              "Saved finer screenshot is not available; this is a new controlled experiment"],
              "reconstruction_sha256": digest, "projection_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
              "native_replay": replay, "orientation": orientation,
              "reconstruction_config": asdict(config), "quantification_config": asdict(quant_config),
              "variants": entries}
    with (output / "matrix_comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def audit_manual_snapshot(snapshot_dir, output_dir):
    import hashlib
    from dataclasses import asdict
    from core.ectb_lv import ECTbLVConfig, analyze_lv_ectb
    from core.segmentation import segment_myocardium

    snapshot = Path(snapshot_dir)
    metadata = json.loads((snapshot / "meta.json").read_text(encoding="utf-8"))
    orientation = json.loads((snapshot / "reorient.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2 or orientation.get("schema_version") != 2:
        raise ValueError("Legacy snapshot lacks provenance/scale; confirm reconstruction and reorientation again")
    if metadata["stage"] != orientation["stage"]:
        raise ValueError("Snapshot stages do not match")
    with (snapshot / "gated_volume.npy").open("rb") as stream:
        volume_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if volume_hash != metadata["reconstruction_sha256"] or volume_hash != orientation["reconstruction_sha256"]:
        raise ValueError("Reconstruction changed after orientation was recorded")
    cube_path = snapshot / "reoriented_gated.npy"
    cube = np.load(cube_path, allow_pickle=False)
    voxel = orientation.get("voxel_mm")
    if cube.ndim != 4 or not np.isfinite(cube).all() or voxel is None or not np.isfinite(voxel) or voxel <= 0:
        raise ValueError("Manual gated cube and finite positive physical voxel spacing are required")
    spacing = metadata.get("pixel_spacing")
    if spacing is None or not np.allclose(spacing, spacing[0]):
        raise ValueError("This diagnostic audit requires known isotropic source pixels")
    base, apex = sorted((int(orientation["base_k"]), int(orientation["apex_k"])))
    if base < 0 or apex >= cube.shape[1]:
        raise ValueError("Manual base/apex outside the saved cube")
    selected = cube[:, base:apex + 1].copy()
    segmentation = segment_myocardium(selected)
    config = ECTbLVConfig()
    result = analyze_lv_ectb(selected, segmentation, (voxel, voxel), voxel, config)
    repeated = analyze_lv_ectb(selected, segmentation, (voxel, voxel), voxel, config)
    repeatable = result.available == repeated.available and np.array_equal(result.gate_volumes_ml, repeated.gate_volumes_ml)
    if not repeatable:
        raise ValueError("Repeated quantification of the same manual cube differs")
    with cube_path.open("rb") as stream:
        cube_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "protocol": "Saved manually approved SA cube, fixed segmentation, two identical ECTb calculations",
        "clinical_validation": False, "stage": metadata["stage"], "bitwise_repeatable": repeatable,
        "quantification_config_source": "Fixed diagnostic ECTb defaults; may differ from user UI configuration",
        "quantification_config": asdict(config), "voxel_mm": voxel,
        "reconstruction_sha256": volume_hash, "manual_cube_sha256": cube_hash,
        "manual_slice_range_zero_based": [base, apex], "available": bool(result.available),
        "reason": result.reason, "notes": result.notes,
        "EF_pct": float(result.ef_pct) if result.available else None,
        "EDV_ml": float(result.edv_ml) if result.available else None,
        "ESV_ml": float(result.esv_ml) if result.available else None,
        "ED_ES_gates": [result.ed_gate, result.es_gate] if result.available else None,
        "gate_volumes_ml": result.gate_volumes_ml.tolist(),
        "gate_counts": result.gate_counts.tolist(),
        "endo_zero_fraction": result.endo_zero_fraction.tolist(),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    display_slices = np.unique(np.linspace(0, selected.shape[1] - 1, min(4, selected.shape[1])).astype(int))
    maximum = max(float(np.percentile(selected, 99.5)), 1e-12)
    figure, axes = plt.subplots(selected.shape[0], len(display_slices),
                                figsize=(3 * len(display_slices), 2.2 * selected.shape[0]), squeeze=False)
    for gate in range(selected.shape[0]):
        for column, slice_index in enumerate(display_slices):
            panel = axes[gate, column]
            panel.imshow(selected[gate, slice_index], cmap="hot", vmin=0, vmax=maximum)
            if result.available and slice_index in result.valid_slices:
                index = result.valid_slices.index(int(slice_index))
                theta = np.linspace(0, 2 * np.pi, result.endo_radii_mm.shape[-1] + 1)
                center_y, center_x = segmentation.center_per_slice[slice_index]
                for radii, color in ((result.endo_radii_mm, "cyan"), (result.epi_radii_mm, "lime")):
                    values = np.r_[radii[gate, index], radii[gate, index, 0]] / voxel
                    panel.plot(center_x + values * np.cos(theta), center_y + values * np.sin(theta), color=color, linewidth=0.7)
            panel.set_title(f"Gate {gate + 1} | manual SA slice {base + int(slice_index) + 1}", fontsize=8)
            panel.axis("off")
    figure.suptitle(f"{metadata['stage'].upper()} | manual snapshot | repeatable, not clinically validated")
    figure.tight_layout()
    figure.savefig(output / "gate_contours.png", dpi=130)
    plt.close(figure)
    with (output / "manual_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"stage": report["stage"], "repeatable": repeatable,
                      "quantification_available": report["available"], "clinical_validation": False}))
    return report


def main():
    parser = argparse.ArgumentParser(description="FBP filter comparison; defaults to the original ungated study")
    parser.add_argument("--gated-study", type=Path)
    parser.add_argument("--osem-checks", action="store_true")
    parser.add_argument("--manual-snapshot", type=Path)
    parser.add_argument("--compare-matrices", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(OUTDIR) / "gated_filter_comparison")
    args = parser.parse_args()
    if args.compare_matrices:
        if args.manual_snapshot is None or args.gated_study is None:
            parser.error("--compare-matrices requires --manual-snapshot and --gated-study")
        compare_snapshot_matrices(args.manual_snapshot, args.output_dir, args.gated_study)
        return 0
    if args.manual_snapshot is not None:
        audit_manual_snapshot(args.manual_snapshot, args.output_dir)
        return 0
    if args.osem_checks:
        if args.gated_study is None:
            parser.error("--osem-checks requires --gated-study")
        return compare_osem_filters(args.gated_study, args.output_dir)
    if args.gated_study is not None:
        return compare_gated_filters(args.gated_study, args.output_dir)
    os.makedirs(OUTDIR, exist_ok=True)
    src = SUMMED if os.path.exists(SUMMED) else GATED
    print("Cargando proyecciones:", os.path.basename(src))
    raw = load_raw_projections(src)
    proj = raw.projections
    if proj.ndim == 4:  # gated -> sumar gates
        proj = proj.sum(axis=0)
    print(f"  proj={proj.shape}  counts={proj.sum():,.0f}  "
          f"angles start={raw.start_angle} step={raw.angular_step} dir={raw.rotation_direction} arc={raw.scan_arc}")

    variants = [
        ("1_actual_Bw0.4_ramp", ProjectionFilterConfig("butterworth", 0.40, 10), "ramp"),
        ("2_Bw0.4_hann", ProjectionFilterConfig("butterworth", 0.40, 10), "hann"),
        ("3_nopre_hann", None, "hann"),
        ("4_nopre_cosine", None, "cosine"),
        ("5_Bw0.3_shepp", ProjectionFilterConfig("butterworth", 0.30, 8), "shepp-logan"),
    ]

    recons = {}
    for name, pre, filt in variants:
        print(f"Reconstruyendo {name} (pre={pre.kind if pre else 'none'} filt={filt})...")
        recons[name] = recon(proj, raw.angles_deg, pre, filt)

    idxs = central_row(next(iter(recons.values())), "")
    rows = len(variants)
    cols = len(idxs)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 1.9))
    for r, (name, _, _) in enumerate(variants):
        vol = recons[name]
        for c, z in enumerate(idxs):
            ax = axes[r, c]
            ax.axis("off")
            sl = vol[z]
            vmax = float(np.percentile(sl, 99.5)) or float(vol.max())
            ax.imshow(sl, cmap="hot", vmin=0, vmax=max(vmax, 1e-6))
            if c == 0:
                ax.set_ylabel(name, fontsize=7)
                ax.axis("on")
                ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"z={z}", fontsize=6)
    fig.suptitle("Comparación filtros FBP - transaxial sumado (ventaneo por corte)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "FCMP_filters.png")
    fig.savefig(out, dpi=95)
    plt.close(fig)
    print("\nGuardado:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

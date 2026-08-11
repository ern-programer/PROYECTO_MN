"""Verifica los ángulos reales por-frame del estudio sumado vs lo que
load_raw_projections está devolviendo. Doble cabezal => los ángulos NO son
uniformes start+step*i; cada cabezal aporta su propio barrido.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pydicom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.raw_projections import load_raw_projections

BASE = r"D:\- GAMMASYS\estudios evolution\estudio prueba evolution xeleris 3\1.2.124.113532.10.1.84.45.20070719.100230.2565043"
SUMMED = os.path.join(BASE, "Stress-10sec-1_T_EM001_DS.dcm")

ds = pydicom.dcmread(SUMMED)
print("Rows/Cols:", ds.Rows, ds.Columns, " NumberOfFrames:", getattr(ds, "NumberOfFrames", None))

# Campos de rotación
for tag_name in ["NumberOfDetectors", "StartAngle", "AngularStep", "ScanArc",
                 "RotationDirection", "NumberOfFramesInRotation", "TypeOfDetectorMotion"]:
    print(f"  {tag_name}: {getattr(ds, tag_name, 'N/A')}")

# Detector Information Sequence
if "DetectorInformationSequence" in ds:
    print("\nDetectorInformationSequence:")
    for i, det in enumerate(ds.DetectorInformationSequence):
        print(f"  det[{i}]: StartAngle={getattr(det,'StartAngle','?')} "
              f"VectorLen? RadialPosition={getattr(det,'RadialPosition','?')}")

# Rotation Information Sequence
if "RotationInformationSequence" in ds:
    print("\nRotationInformationSequence:")
    for i, rot in enumerate(ds.RotationInformationSequence):
        print(f"  rot[{i}]: StartAngle={getattr(rot,'StartAngle','?')} "
              f"AngularStep={getattr(rot,'AngularStep','?')} "
              f"ScanArc={getattr(rot,'ScanArc','?')} "
              f"NumberOfFramesInRotation={getattr(rot,'NumberOfFramesInRotation','?')} "
              f"RotationDirection={getattr(rot,'RotationDirection','?')}")

# Vectores por frame
for tag, name in [((0x0054,0x0090), "AngularViewVector"),
                  ((0x0054,0x0080), "SliceVector"),
                  ((0x0054,0x0070), "TimeSlotVector"),
                  ((0x0054,0x0020), "DetectorVector")]:
    if tag in ds:
        v = np.array(ds[tag].value)
        print(f"\n{name} (len={len(v)}): uniques={np.unique(v)[:20]}")

# Lo que devuelve mi loader
raw = load_raw_projections(SUMMED)
print("\n--- load_raw_projections ---")
print("proj:", raw.projections.shape)
print("start_angle:", raw.start_angle, "step:", raw.angular_step,
      "dir:", raw.rotation_direction, "arc:", raw.scan_arc)
a = np.asarray(raw.angles_deg)
print("angles_deg len:", a.size)
print("angles_deg:", np.round(a, 1))
print("angle span:", a.min(), "->", a.max(), " (max-min)=", a.max()-a.min())
print("diffs unique:", np.unique(np.round(np.diff(a), 2))[:20])

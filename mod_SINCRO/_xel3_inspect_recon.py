"""Inspecciona los DICOM reconstruidos exportados por Xeleris (FBP / IRNC / IRNCRR)
para entender su geometría (frames, filas, cols, pixel spacing) antes de comparar.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pydicom

BASE = r"D:\- GAMMASYS\estudios evolution\estudio prueba evolution xeleris 3\1.2.124.113532.10.1.84.45.20070719.100230.2565043"

patterns = [
    "STRESS_FBP00*_DS.dcm",
    "STRESS_IRNC00*_DS.dcm",
    "STRESS_IRNCRR00*_DS.dcm",
    "STRESS_FBP_SA00*_DS.dcm",
    "STRESS_IRNCRR_SA00*_DS.dcm",
]

for pat in patterns:
    files = sorted(glob.glob(os.path.join(BASE, pat)))
    print(f"\n=== {pat}  ({len(files)} archivos) ===")
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=False)
        arr = ds.pixel_array
        nf = int(getattr(ds, "NumberOfFrames", 1) or 1)
        px = getattr(ds, "PixelSpacing", None)
        sl = getattr(ds, "SliceThickness", None)
        desc = getattr(ds, "SeriesDescription", "")
        print(f"  {os.path.basename(f):32s} shape={arr.shape} frames={nf} "
              f"px={px} thick={sl} max={arr.max()} desc='{desc}'")

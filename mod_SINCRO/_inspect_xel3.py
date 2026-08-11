import os
import numpy as np
import pydicom

root = r"D:\- GAMMASYS\estudios evolution\estudio prueba evolution xeleris 3"
files = []
for dp, _, fs in os.walk(root):
    for f in fs:
        files.append(os.path.join(dp, f))
print("Total archivos:", len(files))
print("=" * 90)
for p in sorted(files):
    try:
        d = pydicom.dcmread(p, force=True)
    except Exception as e:
        print("  ERR", os.path.basename(p), e)
        continue
    mod = getattr(d, "Modality", "")
    sd = str(getattr(d, "SeriesDescription", "") or "")
    nf = int(getattr(d, "NumberOfFrames", 0) or 0)
    gated = getattr(d, "GatedInformationSequence", None) is not None
    rows = getattr(d, "Rows", "?")
    cols = getattr(d, "Columns", "?")
    dur = ""
    try:
        ri = d.RotationInformationSequence[0]
        dur = "dur={:.2f}s arc={} start={} {} nfr={}".format(
            float(ri.ActualFrameDuration) / 1000, ri.ScanArc, ri.StartAngle,
            ri.RotationDirection, ri.NumberOfFramesInRotation)
    except Exception:
        pass
    cnt = ""
    try:
        if mod == "NM":
            cnt = "counts={:,.0f}".format(d.pixel_array.astype(np.float64).sum())
    except Exception:
        pass
    print("{:42s} {:4s} F={:4d} {}x{} gated={} sd='{}' {} {}".format(
        os.path.basename(p)[:40], mod, nf, rows, cols, gated, sd, dur, cnt))

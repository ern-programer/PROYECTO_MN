"""Inspecciona los tags de geometría SPECT relevantes para resolution recovery."""
import sys
import pydicom

f = sys.argv[1]
ds = pydicom.dcmread(f, force=True, stop_before_pixels=True)


def g(tag, default="--"):
    return ds[tag].value if tag in ds else default


def gd(sub, tag, default="--"):
    return sub[tag].value if tag in sub else default


print("Manufacturer:", g(0x00080070))
print("Model:", g(0x00081090))
print("Collimator/grid Name (0018,1180):", g(0x00181180))
print("Collimator Type   (0018,1181):", g(0x00181181))
print("PixelSpacing      (0028,0030):", g(0x00280030))
print("SliceThickness    (0018,0050):", g(0x00180050))

det = ds[0x00540022].value if 0x00540022 in ds else None  # Detector Information Sequence
print("\n--- Detector Information Sequence ---  len:", len(det) if det else 0)
if det:
    d0 = det[0]
    print("  ZoomFactor       (0028,0031):", gd(d0, 0x00280031))
    print("  PixelSpacing     (0028,0030):", gd(d0, 0x00280030))
    print("  FocalDistance    (0018,1182):", gd(d0, 0x00181182))
    print("  DistSrcToDetector(0018,1110):", gd(d0, 0x00181110))
    print("  CollimatorName   (0018,1180):", gd(d0, 0x00181180))
    print("  CollimatorType   (0018,1181):", gd(d0, 0x00181181))
    rp = gd(d0, 0x00181142, None)  # Radial Position within detector seq
    if rp is not None:
        vals = list(rp) if hasattr(rp, "__len__") and not isinstance(rp, str) else [rp]
        print("  RadialPosition   (0018,1142): n=", len(vals), "sample=", vals[:8])

rot = ds[0x00540052].value if 0x00540052 in ds else None  # Rotation Information Sequence
print("\n--- Rotation Information Sequence ---  len:", len(rot) if rot else 0)
if rot:
    r0 = rot[0]
    print("  StartAngle       (0054,0200):", gd(r0, 0x00540200))
    print("  AngularStep      (0018,1144):", gd(r0, 0x00181144))
    print("  ScanArc          (0018,1143):", gd(r0, 0x00181143))
    print("  RotationDirection(0018,1140):", gd(r0, 0x00181140))
    print("  NumberOfFramesInRotation(0054,0053):", gd(r0, 0x00540053))
    print("  TableHeight      (0018,1130):", gd(r0, 0x00181130))
    rp = gd(r0, 0x00181142, None)  # Radial Position per view (la órbita/contorno)
    if rp is not None:
        vals = list(rp) if hasattr(rp, "__len__") and not isinstance(rp, str) else [rp]
        print("  RadialPosition   (0018,1142): n=", len(vals),
              "min=", min(vals) if vals else "--", "max=", max(vals) if vals else "--",
              "sample=", vals[:10])

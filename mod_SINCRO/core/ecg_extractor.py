"""Extracción automática de datos ECG de archivos PDF, SCP-ECG y DICOM waveform."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ECGData:
    """Datos extraídos de un ECG."""
    ritmo: str = ""
    fc: int = 0
    qrs_ms: int = 0
    qt_ms: int = 0
    qtc_ms: int = 0
    bri: bool = False
    brd: bool = False
    marcapasos: bool = False
    observaciones: str = ""
    rr_intervals: list[float] = field(default_factory=list)
    fuente: str = ""  # "pdf", "scp", "dicom", "manual"
    confianza: str = "alta"  # "alta", "media", "baja"
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ritmo": self.ritmo,
            "fc": self.fc,
            "qrs_ms": self.qrs_ms,
            "qt_ms": self.qt_ms,
            "qtc_ms": self.qtc_ms,
            "bri": self.bri,
            "brd": self.brd,
            "marcapasos": self.marcapasos,
            "observaciones": self.observaciones,
            "fuente": self.fuente,
            "confianza": self.confianza,
        }


def extract_from_pdf_text(text: str) -> ECGData:
    """
    Extrae datos ECG de texto PDF.
    Soporta formatos comunes de informes ECG en español e inglés.
    """
    data = ECGData(fuente="pdf", raw_text=text)
    text_upper = text.upper()

    # Ritmo
    if re.search(r'SINUSAL|SINUS|RITMO SINUSAL', text_upper):
        data.ritmo = "Sinusal"
    elif re.search(r'FIBRILACI[OÓ]N AURICULAR|FA\b|AF\b|ATRIAL FIBRILLATION', text_upper):
        data.ritmo = "FA"
    elif re.search(r'MARCAPASOS|PACEMAKER|PM\b|CRT|RESINCRONIZADOR', text_upper):
        data.ritmo = "Marcapasos"
    elif re.search(r'TAQUICARDIA|TACHYCARDIA', text_upper):
        data.ritmo = "Taquicardia"
    elif re.search(r'BRADICARDIA|BRADYCARDIA', text_upper):
        data.ritmo = "Bradicardia"
    else:
        data.ritmo = "No especificado"
        data.confianza = "media"

    # FC (Frecuencia cardíaca)
    fc_patterns = [
        r'(?:FC|HR|HEART RATE|FRECUENCIA)[:\s]+(\d{2,3})',
        r'(\d{2,3})\s*(?:LPM|BPM|LAT/MIN)',
        r'RITMO.*?(\d{2,3})',
    ]
    for pattern in fc_patterns:
        match = re.search(pattern, text_upper)
        if match:
            fc = int(match.group(1))
            if 30 <= fc <= 250:
                data.fc = fc
                break

    # QRS
    qrs_patterns = [
        r'QRS[:\s]+(\d{2,3})\s*(?:MS|MSEC)?',
        r'DURACI[OÓ]N QRS[:\s]+(\d{2,3})',
        r'QRS DURATION[:\s]+(\d{2,3})',
    ]
    for pattern in qrs_patterns:
        match = re.search(pattern, text_upper)
        if match:
            qrs = int(match.group(1))
            if 40 <= qrs <= 300:
                data.qrs_ms = qrs
                break

    # QT
    qt_patterns = [
        r'QT[:\s]+(\d{3})\s*(?:MS|MSEC)?',
        r'INTERVALO QT[:\s]+(\d{3})',
        r'QT INTERVAL[:\s]+(\d{3})',
    ]
    for pattern in qt_patterns:
        match = re.search(pattern, text_upper)
        if match:
            qt = int(match.group(1))
            if 200 <= qt <= 700:
                data.qt_ms = qt
                break

    # QTc (calculado si no viene)
    qtc_patterns = [
        r'QTC[:\s]+(\d{3})',
        r'QT CORREGIDO[:\s]+(\d{3})',
    ]
    for pattern in qtc_patterns:
        match = re.search(pattern, text_upper)
        if match:
            data.qtc_ms = int(match.group(1))
            break

    # Calcular QTc si no viene (Bazett)
    if data.qtc_ms == 0 and data.qt_ms > 0 and data.fc > 0:
        rr_sec = 60.0 / data.fc
        data.qtc_ms = int(data.qt_ms / np.sqrt(rr_sec))

    # BRI (Bloqueo rama izquierda) - evitar falsos positivos con "NO"
    bri_negative = re.search(r'BRI[:\s]+NO|LBBB[:\s]+NO|SIN BRI|NO BRI', text_upper)
    bri_patterns = [
        r'BRI\b(?!.*NO)',
        r'LBBB|LEFT BUNDLE BRANCH BLOCK',
        r'BLOQUEO.*RAMA IZQUIERDA',
        r'BLOQUEO COMPLETO.*IZQUIERDA',
    ]
    if not bri_negative:
        for pattern in bri_patterns:
            if re.search(pattern, text_upper):
                data.bri = True
                break

    # BRD (Bloqueo rama derecha) - evitar falsos positivos con "NO"
    brd_negative = re.search(r'BRD[:\s]+NO|RBBB[:\s]+NO|SIN BRD|NO BRD', text_upper)
    brd_patterns = [
        r'BRD\b(?!.*NO)',
        r'RBBB|RIGHT BUNDLE BRANCH BLOCK',
        r'BLOQUEO.*RAMA DERECHA',
        r'BLOQUEO COMPLETO.*DERECHA',
    ]
    if not brd_negative:
        for pattern in brd_patterns:
            if re.search(pattern, text_upper):
                data.brd = True
                break

    # Marcapasos
    marcapasos_patterns = [
        r'MARCAPASOS',
        r'PACEMAKER',
        r'PM\b',
        r'CRT\b',
        r'RESINCRONIZADOR',
        r'ESTIMULACI[OÓ]N.*VENTRICULAR',
    ]
    for pattern in marcapasos_patterns:
        if re.search(pattern, text_upper):
            data.marcapasos = True
            break

    # Observaciones (primeras 200 chars del texto)
    data.observaciones = text[:200].strip()

    return data


def extract_from_scp_ecg(filepath: str) -> ECGData:
    """
    Extrae datos ECG de archivo SCP-ECG.
    Requiere librería scp-ecg.
    """
    try:
        import scp_ecg
    except ImportError:
        raise ImportError("scp-ecg no instalado. Instalar con: pip install scp-ecg")

    data = ECGData(fuente="scp")

    try:
        with open(filepath, "rb") as f:
            record = scp_ecg.SCPRecord(f.read())

        # Extraer metadatos básicos
        if hasattr(record, "patient_data"):
            # SCP-ECG tiene estructura compleja, simplificamos
            pass

        # Por ahora retornamos estructura básica
        data.confianza = "media"
        data.observaciones = f"SCP-ECG cargado: {filepath}"

    except Exception as exc:
        data.confianza = "baja"
        data.observaciones = f"Error leyendo SCP-ECG: {exc}"

    return data


def extract_from_dicom_waveform(filepath: str) -> ECGData:
    """
    Extrae datos ECG de DICOM waveform.
    Requiere pydicom.
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("pydicom no instalado")

    data = ECGData(fuente="dicom")

    try:
        ds = pydicom.dcmread(filepath)

        # Extraer metadatos básicos del DICOM
        if hasattr(ds, "PatientName"):
            data.observaciones = f"Paciente: {ds.PatientName}"

        # Buscar waveform sequence
        if hasattr(ds, "WaveformSequence"):
            # Análisis básico de waveform
            data.confianza = "media"

        data.confianza = "alta" if hasattr(ds, "WaveformSequence") else "media"

    except Exception as exc:
        data.confianza = "baja"
        data.observaciones = f"Error leyendo DICOM: {exc}"

    return data


# Cache del motor RapidOCR (carga costosa: modelos ONNX).
_RAPIDOCR_ENGINE = None


def _get_rapidocr():
    """Devuelve una instancia (cacheada) de RapidOCR o None si no está disponible.

    RapidOCR (rapidocr-onnxruntime) es OCR 100% pip: NO requiere Tesseract ni
    ningún binario del sistema. Los modelos ONNX vienen en el paquete, por lo que
    es portable a otras PCs con solo `pip install rapidocr-onnxruntime`.
    """
    global _RAPIDOCR_ENGINE
    if _RAPIDOCR_ENGINE is not None:
        return _RAPIDOCR_ENGINE
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return None
    _RAPIDOCR_ENGINE = RapidOCR()
    return _RAPIDOCR_ENGINE


def _pdf_pages_to_numpy(filepath: str, dpi: int = 300) -> list:
    """Renderiza cada página de un PDF a un array numpy RGB usando PyMuPDF (fitz)."""
    import numpy as _np
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF (fitz) requerido para renderizar PDF. Instalar: pip install pymupdf")

    from PIL import Image
    import io as _io

    images = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(filepath)
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(_io.BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append(_np.asarray(img))
    finally:
        doc.close()
    return images


def _ocr_images(images: list) -> str:
    """Aplica RapidOCR a una lista de imágenes (numpy arrays) y concatena el texto."""
    engine = _get_rapidocr()
    if engine is None:
        raise ImportError(
            "OCR no disponible: falta rapidocr-onnxruntime. "
            "Instalar con: pip install rapidocr-onnxruntime (no requiere Tesseract)"
        )

    text_parts = []
    for img in images:
        result, _elapsed = engine(img)
        if result:
            # result = [[box, text, score], ...]
            for item in result:
                if len(item) >= 2 and item[1]:
                    text_parts.append(str(item[1]))
    return "\n".join(text_parts)


def _ocr_pdf(filepath: str) -> str:
    """Extrae texto de un PDF escaneado con OCR (RapidOCR, sin Tesseract)."""
    images = _pdf_pages_to_numpy(filepath, dpi=300)
    return _ocr_images(images)


def _ocr_image_file(filepath: str) -> str:
    """Extrae texto de una imagen (PNG/JPG/etc.) con OCR (RapidOCR, sin Tesseract)."""
    import numpy as _np
    from PIL import Image

    img = Image.open(filepath).convert("RGB")
    return _ocr_images([_np.asarray(img)])


def _extract_pdf_text_digital(filepath: str) -> str:
    """Extrae la capa de texto de un PDF digital (sin OCR). Vacío si es escaneado."""
    text = ""
    # pdfplumber primero (mejor layout), luego pypdf/PyPDF2.
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except ImportError:
        pass

    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError:
            return text

    reader = PdfReader(filepath)
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text:
            text += page_text + "\n"
    return text


def extract_from_pdf_file(filepath: str) -> ECGData:
    """
    Extrae datos ECG de un PDF.

    Estrategia en cascada:
      1. Capa de texto digital (pypdf/pdfplumber) — sin OCR, ideal para PDF generado por equipo.
      2. Si el PDF es escaneado (sin texto) → OCR con RapidOCR (pip puro, sin Tesseract).
    """
    text = _extract_pdf_text_digital(filepath)
    used_ocr = False

    if not text.strip():
        # PDF escaneado → OCR
        try:
            text = _ocr_pdf(filepath)
            used_ocr = True
        except ImportError as exc:
            raise ValueError(f"PDF escaneado sin texto y {exc}")

    if not text.strip():
        raise ValueError("No se pudo extraer texto del PDF ni con OCR.")

    data = extract_from_pdf_text(text)
    if used_ocr:
        # El OCR puede introducir errores → marcar confianza no alta.
        if data.confianza == "alta":
            data.confianza = "media"
    return data


def extract_from_image_file(filepath: str) -> ECGData:
    """
    Extrae datos ECG de una imagen (foto o escaneo: PNG/JPG/BMP/TIFF) vía OCR.
    Lee los valores impresos en el ECG (FC/QRS/QT/QTc/PR/ritmo). Usa RapidOCR.
    """
    try:
        text = _ocr_image_file(filepath)
    except ImportError as exc:
        raise ValueError(f"No se puede leer la imagen: {exc}")

    if not text.strip():
        raise ValueError(
            "No se detectó texto en la imagen. "
            "Asegurate de que el ECG muestre los valores impresos (FC, QRS, QT) legibles."
        )

    data = extract_from_pdf_text(text)
    data.fuente = "imagen"
    # OCR de foto: confianza como mucho media (dependiente de calidad de imagen).
    if data.confianza == "alta":
        data.confianza = "media"
    return data


def extract_ecg(filepath: str) -> ECGData:
    """
    Punto de entrada principal para extracción ECG.
    Detecta formato por extensión y delega al extractor apropiado.
    """
    import os
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return extract_from_pdf_file(filepath)
    elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
        return extract_from_image_file(filepath)
    elif ext in (".scp", ".scp-ecg"):
        return extract_from_scp_ecg(filepath)
    elif ext in (".dcm", ".dicom"):
        return extract_from_dicom_waveform(filepath)
    else:
        raise ValueError(
            f"Formato no soportado: {ext}. Usar .pdf, .png/.jpg (imagen), .scp o .dcm"
        )


def compare_ecg_data(manual: ECGData, extracted: ECGData) -> dict[str, Any]:
    """
    Compara datos ECG manuales vs extraídos y reporta diferencias significativas.
    """
    differences = []

    # Comparar FC
    if manual.fc > 0 and extracted.fc > 0:
        diff_fc = abs(manual.fc - extracted.fc)
        if diff_fc > 10:
            differences.append({
                "field": "fc",
                "manual": manual.fc,
                "extracted": extracted.fc,
                "diff": diff_fc,
                "significant": diff_fc > 20,
            })

    # Comparar QRS
    if manual.qrs_ms > 0 and extracted.qrs_ms > 0:
        diff_qrs = abs(manual.qrs_ms - extracted.qrs_ms)
        if diff_qrs > 10:
            differences.append({
                "field": "qrs_ms",
                "manual": manual.qrs_ms,
                "extracted": extracted.qrs_ms,
                "diff": diff_qrs,
                "significant": diff_qrs > 20,
            })

    # Comparar ritmo
    if manual.ritmo and extracted.ritmo and manual.ritmo != extracted.ritmo:
        differences.append({
            "field": "ritmo",
            "manual": manual.ritmo,
            "extracted": extracted.ritmo,
            "significant": True,
        })

    # Comparar BRI/BRD
    if manual.bri != extracted.bri:
        differences.append({
            "field": "bri",
            "manual": manual.bri,
            "extracted": extracted.bri,
            "significant": True,
        })
    if manual.brd != extracted.brd:
        differences.append({
            "field": "brd",
            "manual": manual.brd,
            "extracted": extracted.brd,
            "significant": True,
        })

    # Comparar marcapasos
    if manual.marcapasos != extracted.marcapasos:
        differences.append({
            "field": "marcapasos",
            "manual": manual.marcapasos,
            "extracted": extracted.marcapasos,
            "significant": True,
        })

    return {
        "has_differences": len(differences) > 0,
        "differences": differences,
        "n_significant": sum(1 for d in differences if d.get("significant")),
        "extracted_confianza": extracted.confianza,
    }

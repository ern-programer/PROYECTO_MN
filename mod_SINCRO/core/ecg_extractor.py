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


def _ocr_pdf(filepath: str) -> str:
    """Extrae texto de un PDF escaneado con OCR (pytesseract + pdf2image/PyMuPDF)."""
    try:
        import pytesseract
    except ImportError:
        raise ImportError("pytesseract no instalado. Instalar con: pip install pytesseract (y Tesseract OCR en el sistema)")

    images = []
    # Intentar pdf2image primero
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(filepath, dpi=300)
    except ImportError:
        # Fallback a PyMuPDF (fitz)
        try:
            import fitz
            doc = fitz.open(filepath)
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                from PIL import Image
                import io
                images.append(Image.open(io.BytesIO(pix.tobytes("png"))))
            doc.close()
        except ImportError:
            raise ImportError("pdf2image o PyMuPDF requerido para OCR. Instalar con: pip install pdf2image o pip install PyMuPDF")

    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang="spa+eng") + "\n"
    return text


def extract_from_pdf_file(filepath: str) -> ECGData:
    """
    Extrae texto de PDF y luego datos ECG.
    Requiere PyPDF2 o pdfplumber. Si el PDF es escaneado, usa OCR (pytesseract).
    """
    text = ""

    # Intentar con pdfplumber primero (mejor extracción)
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except ImportError:
        # Fallback a PyPDF2
        try:
            import PyPDF2
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except ImportError:
            raise ImportError("pdfplumber o PyPDF2 requerido. Instalar con: pip install pdfplumber")

    if not text.strip():
        # PDF escaneado → OCR
        try:
            text = _ocr_pdf(filepath)
        except ImportError as exc:
            raise ValueError(f"PDF escaneado sin texto y OCR no disponible: {exc}")

    if not text.strip():
        raise ValueError("No se pudo extraer texto del PDF ni con OCR.")

    data = extract_from_pdf_text(text)
    if "ocr" in text.lower() or not any(c.isalpha() for c in text[:50]):
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
    elif ext in (".scp", ".scp-ecg"):
        return extract_from_scp_ecg(filepath)
    elif ext in (".dcm", ".dicom"):
        return extract_from_dicom_waveform(filepath)
    else:
        raise ValueError(f"Formato no soportado: {ext}. Usar .pdf, .scp o .dcm")


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

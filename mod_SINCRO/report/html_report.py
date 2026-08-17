# -*- coding: utf-8 -*-
"""SINCRO — Generador de informe HTML autocontenido.

Genera un archivo HTML clínico con:
- CSS inline (sin dependencias externas).
- JS mínimo para tabs, lightbox y toggle de bullseye.
- Imágenes embebidas como base64 data URIs (PNG y GIF).
- Layout responsive, profesional, fácil de leer.

Uso:
    from report.html_report import generate_html_report
    path = generate_html_report(output_html="informe.html", output_dir="output/", ...)
"""
from __future__ import annotations

import base64
import os
from datetime import datetime

import numpy as np


# ============================================================
# Helpers
# ============================================================

def _safe_float(value, ndigits: int = 2) -> str:
    try:
        f = float(value)
    except Exception:
        return "N/D"
    if not np.isfinite(f):
        return "N/D"
    return f"{f:.{int(ndigits)}f}"


def _format_dicom_date(raw: str) -> str:
    val = str(raw or "").strip()
    if len(val) == 8 and val.isdigit():
        return f"{val[6:8]}/{val[4:6]}/{val[0:4]}"
    return val or "N/D"


def _img_to_data_uri(path: str, max_width: int = 900) -> str:
    """Convierte una imagen a data URI (base64 inline), comprimiendo si es necesario.

    Para PNGs grandes los re-escala a max_width px y guarda como JPEG (mucho más
    liviano). Los GIFs se pasan tal cual (no se pueden recomprimir fácilmente).
    """
    if not os.path.exists(path):
        return ""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gif":
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/gif;base64,{data}"
    try:
        from PIL import Image
        img = Image.open(path)
        w, h = img.size
        if w > max_width:
            ratio = max_width / w
            img = img.resize((max_width, int(h * ratio)), Image.LANCZOS)
        import io
        buf = io.BytesIO()
        # Convertir a RGB si tiene canal alfa (JPEG no soporta alpha).
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=78, optimize=True)
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{data}"
    except Exception:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        mime = "image/png" if ext == ".png" else "image/jpeg"
        return f"data:{mime};base64,{data}"


def _img_tag(path: str, alt: str = "", css_class: str = "", max_w: str = "100%") -> str:
    """Genera un <img> con data URI si el archivo existe."""
    uri = _img_to_data_uri(path)
    if not uri:
        return ""
    cls = f' class="{css_class}"' if css_class else ""
    return f'<img src="{uri}" alt="{alt}"{cls} style="max-width:{max_w}; height:auto;" loading="lazy">'


def _gif_tag(path: str, alt: str = "", css_class: str = "") -> str:
    """Genera un <img> para GIF animado con data URI."""
    return _img_tag(path, alt=alt, css_class=css_class, max_w="100%")


# ============================================================
# CSS
# ============================================================

_CSS = """
:root {
  --bg: #0f172a;
  --bg-card: #1e293b;
  --bg-card-alt: #334155;
  --fg: #e2e8f0;
  --fg-muted: #94a3b8;
  --accent: #38bdf8;
  --accent-dark: #0ea5e9;
  --accent-green: #4ade80;
  --accent-yellow: #fbbf24;
  --accent-red: #f87171;
  --border: #475569;
  --radius: 12px;
  --shadow: 0 4px 24px rgba(0,0,0,0.3);
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.6;
  padding: 0;
}

.container { max-width: 1200px; margin: 0 auto; padding: 24px; }

/* Header */
.header {
  background: linear-gradient(135deg, #1a3a5c 0%, #0f172a 100%);
  border-bottom: 3px solid var(--accent);
  padding: 32px 24px;
  text-align: center;
}
.header h1 { font-size: 2rem; color: var(--accent); letter-spacing: 2px; margin-bottom: 4px; }
.header .subtitle { color: var(--fg-muted); font-size: 0.95rem; }
.header .patient-bar {
  display: flex; flex-wrap: wrap; justify-content: center; gap: 16px;
  margin-top: 16px; font-size: 0.85rem;
}
.header .patient-bar .chip {
  background: var(--bg-card); border: 1px solid var(--border);
  padding: 4px 12px; border-radius: 20px; color: var(--fg-muted);
}
.header .patient-bar .chip b { color: var(--fg); }

/* Executive summary */
.exec-summary {
  background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
  border-left: 4px solid var(--accent);
  border-radius: var(--radius);
  padding: 20px 24px;
  margin: 24px 0;
  box-shadow: var(--shadow);
}
.exec-summary h3 { color: var(--accent); margin-bottom: 12px; font-size: 1.1rem; }
.exec-summary p { color: var(--fg-muted); margin-bottom: 8px; font-size: 0.92rem; }

/* FEVI highlight */
.fevi-highlight {
  display: flex; align-items: center; justify-content: center; gap: 24px;
  background: var(--bg-card);
  border: 2px solid var(--accent);
  border-radius: var(--radius);
  padding: 24px;
  margin: 24px auto;
  max-width: 400px;
  box-shadow: var(--shadow);
}
.fevi-highlight .fevi-number {
  font-size: 3.5rem; font-weight: 800; color: var(--accent);
  line-height: 1;
}
.fevi-highlight .fevi-label { color: var(--fg-muted); font-size: 0.9rem; }

/* Tabs */
.tabs { display: flex; gap: 0; border-bottom: 2px solid var(--border); margin-bottom: 24px; }
.tab-btn {
  padding: 12px 24px; background: transparent; border: none;
  color: var(--fg-muted); cursor: pointer; font-size: 0.95rem;
  font-weight: 600; border-bottom: 3px solid transparent;
  transition: all 0.2s;
}
.tab-btn:hover { color: var(--fg); background: rgba(56,189,248,0.05); }
.tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab-panel { display: none; }
.tab-panel.active { display: block; }

/* Tables */
.table-wrap { overflow-x: auto; margin: 16px 0; }
table {
  width: 100%; border-collapse: collapse; font-size: 0.88rem;
  background: var(--bg-card); border-radius: var(--radius); overflow: hidden;
  box-shadow: var(--shadow);
}
thead th {
  background: #1a3a5c; color: white; padding: 10px 14px;
  text-align: left; font-weight: 600; font-size: 0.82rem;
  text-transform: uppercase; letter-spacing: 0.5px;
}
tbody td { padding: 8px 14px; border-bottom: 1px solid var(--border); }
tbody tr:nth-child(even) { background: rgba(255,255,255,0.02); }
tbody tr:hover { background: rgba(56,189,248,0.05); }
td.label { font-weight: 600; color: var(--accent); white-space: nowrap; min-width: 180px; }

/* Cards grid */
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin: 16px 0; }
.card {
  background: var(--bg-card); border-radius: var(--radius); padding: 20px;
  border: 1px solid var(--border); box-shadow: var(--shadow);
}
.card h4 { color: var(--accent); margin-bottom: 8px; font-size: 0.95rem; }
.card .value { font-size: 1.8rem; font-weight: 700; color: var(--fg); }
.card .unit { font-size: 0.85rem; color: var(--fg-muted); }

/* Image gallery */
.gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin: 16px 0; }
.gallery-item {
  background: var(--bg-card); border-radius: var(--radius); overflow: hidden;
  border: 1px solid var(--border); cursor: pointer; transition: transform 0.2s;
  box-shadow: var(--shadow);
}
.gallery-item:hover { transform: scale(1.02); }
.gallery-item img { width: 100%; display: block; }
.gallery-item .caption {
  padding: 10px 14px; font-size: 0.82rem; color: var(--fg-muted);
  border-top: 1px solid var(--border);
}

/* Featured image */
.featured { margin: 24px 0; text-align: center; }
.featured img {
  max-width: 100%; border-radius: var(--radius); box-shadow: var(--shadow);
  border: 1px solid var(--border);
}
.featured .caption {
  margin-top: 8px; font-size: 0.85rem; color: var(--fg-muted);
}

/* Lightbox */
.lightbox {
  display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
  background: rgba(0,0,0,0.92); z-index: 9999; justify-content: center;
  align-items: center; cursor: pointer;
}
.lightbox.open { display: flex; }
.lightbox img { max-width: 95%; max-height: 95%; border-radius: 8px; }
.lightbox .close-btn {
  position: absolute; top: 16px; right: 24px; color: white;
  font-size: 2rem; cursor: pointer; background: none; border: none;
}

/* Bullseye toggle */
.bullseye-toggle { display: flex; gap: 8px; margin: 12px 0; }
.bullseye-toggle button {
  padding: 8px 16px; background: var(--bg-card-alt); border: 1px solid var(--border);
  color: var(--fg-muted); cursor: pointer; border-radius: 8px; font-size: 0.85rem;
  transition: all 0.2s;
}
.bullseye-toggle button.active { background: var(--accent-dark); color: white; border-color: var(--accent); }

/* MIP scale buttons */
.mip-scale-bar { display: flex; gap: 6px; margin: 8px 0 12px; flex-wrap: wrap; }
.mip-scale-btn {
  padding: 6px 14px; background: var(--bg-card-alt); border: 1px solid var(--border);
  color: var(--fg-muted); cursor: pointer; border-radius: 6px; font-size: 0.82rem;
  transition: all 0.2s;
}
.mip-scale-btn:hover { background: rgba(56,189,248,0.1); color: var(--fg); }
.mip-scale-btn.active { background: var(--accent-dark); color: white; border-color: var(--accent); }

/* Interpretation */
.interpretation {
  background: var(--bg-card); border-radius: var(--radius); padding: 20px;
  margin: 16px 0; border: 1px solid var(--border);
}
.interpretation h4 { color: var(--accent); margin-bottom: 12px; }
.interpretation ul { list-style: none; padding: 0; }
.interpretation li {
  padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
  font-size: 0.9rem; color: var(--fg-muted);
}
.interpretation li b { color: var(--fg); }

/* Footer */
.footer {
  text-align: center; padding: 24px; margin-top: 32px;
  border-top: 1px solid var(--border); color: var(--fg-muted); font-size: 0.8rem;
}

/* ECG section */
.ecg-bar {
  display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0;
}
.ecg-chip {
  background: var(--bg-card-alt); border: 1px solid var(--border);
  padding: 6px 14px; border-radius: 8px; font-size: 0.85rem;
}
.ecg-chip b { color: var(--accent); }

/* Responsive */
@media (max-width: 768px) {
  .container { padding: 12px; }
  .header h1 { font-size: 1.4rem; }
  .gallery { grid-template-columns: 1fr; }
  .cards { grid-template-columns: 1fr; }
  .tabs { flex-wrap: wrap; }
  .tab-btn { padding: 10px 14px; font-size: 0.85rem; }
}
"""

# ============================================================
# JS
# ============================================================

_JS = """
// Tabs
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const group = btn.dataset.group;
    const target = btn.dataset.tab;
    document.querySelectorAll(`.tab-btn[data-group="${group}"]`).forEach(b => b.classList.remove('active'));
    document.querySelectorAll(`.tab-panel[data-group="${group}"]`).forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(target).classList.add('active');
  });
});

// Lightbox
const lightbox = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightbox-img');
document.querySelectorAll('.gallery-item, .featured img').forEach(item => {
  item.addEventListener('click', () => {
    const img = item.tagName === 'IMG' ? item : item.querySelector('img');
    if (img) { lightboxImg.src = img.src; lightbox.classList.add('open'); }
  });
});
lightbox.addEventListener('click', () => lightbox.classList.remove('open'));
document.addEventListener('keydown', e => { if (e.key === 'Escape') lightbox.classList.remove('open'); });

// MIP scale buttons
document.querySelectorAll('.mip-scale-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.mip-scale-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const filter = btn.dataset.filter;
    document.querySelectorAll('.mip-img').forEach(img => { img.style.filter = filter; });
  });
});

// Bullseye toggle
document.querySelectorAll('.bullseye-toggle button').forEach(btn => {
  btn.addEventListener('click', () => {
    const group = btn.dataset.group;
    document.querySelectorAll(`.bullseye-toggle button[data-group="${group}"]`).forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const target = btn.dataset.target;
    document.querySelectorAll(`.bullseye-img[data-group="${group}"]`).forEach(img => {
      img.style.display = img.id === target ? 'block' : 'none';
    });
  });
});
"""


# ============================================================
# Generador principal
# ============================================================

def generate_html_report(
    *,
    output_html: str,
    output_dir: str,
    study,
    seg,
    metrics: dict,
    territory: dict,
    processing_params: dict,
    volumes: dict,
    ef: dict,
    stress_rest: dict | None = None,
    perfusion_phase_rows: list | None = None,
) -> str:
    """Genera un informe HTML clínico autocontenido.

    Parameters
    ----------
    output_html : ruta de salida del archivo HTML.
    output_dir : directorio con las imágenes PNG/GIF generadas por el pipeline.
    study : objeto estudio DICOM.
    seg : resultado de segmentación.
    metrics : dict de métricas de fase.
    territory : dict de análisis territorial.
    processing_params : dict de parámetros usados.
    volumes : dict de volúmenes.
    ef : dict de fracción de eyección.
    stress_rest : dict de comparación stress-rest (opcional).
    perfusion_phase_rows : filas de textura GLCM (opcional).

    Returns
    -------
    Ruta del archivo HTML generado.
    """
    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)

    # --- Datos del estudio ---
    phase_label = "Esfuerzo" if "STRESS" in str(getattr(study, "source_path", "")).upper() else (
        "Reposo" if "REST" in str(getattr(study, "source_path", "")).upper() else "Estudio"
    )
    patient_name = str(getattr(study, "patient_name", "") or "").strip() or "N/D"
    patient_id = str(getattr(study, "patient_id", "") or "").strip() or "N/D"
    patient_sex = str(getattr(study, "patient_sex", "") or "").strip() or "N/D"
    study_date = _format_dicom_date(str(getattr(study, "study_date", "") or ""))
    study_desc = str(getattr(study, "study_description", "") or "N/D")
    cube = np.asarray(study.cube)

    # Detectar método de stress/apremio desde la descripción del estudio.
    desc_upper = study_desc.upper()
    source_upper = str(getattr(study, "source_path", "") or "").upper()
    if any(k in desc_upper for k in ("DIPYRIDAMOL", "DIPYRIDAMOLE", "PERSANTIN")):
        stress_method = "DIPYRIDAMOL"
    elif any(k in desc_upper for k in ("ADENOSINA", "ADENOSINE")):
        stress_method = "ADENOSINA"
    elif any(k in desc_upper for k in ("DOBUTAMINA", "DOBUTAMINE")):
        stress_method = "DOBUTAMINA"
    elif any(k in desc_upper + source_upper for k in ("STRESS", "ESFUERZO", "ERGOMETR", "TAPING", "BICI")):
        stress_method = "ESFUERZO"
    else:
        stress_method = ""
    if stress_method:
        header_title = f"PERFUSIÓN MIOCÁRDICA — REPOSO Y {stress_method}"
    else:
        header_title = "PERFUSIÓN MIOCÁRDICA"

    # --- Resumen ejecutivo ---
    exec_html = ""
    try:
        from core.executive_summary import build_executive_summary
        summary = build_executive_summary(
            metrics=metrics, ef=ef, territory=territory,
            volumes=volumes, phase_label=phase_label,
        )
        if summary.get("available"):
            sections = summary.get("sections", [])
            parts = "".join(f"<p><b>{s['title']}.</b> {s['text']}</p>" for s in sections)
            exec_html = f'<div class="exec-summary"><h3>Resumen ejecutivo</h3>{parts}</div>'
    except Exception:
        pass

    # --- FEVI ---
    fevi_html = ""
    ef_pct = ef.get("ef_pct") if ef else None
    if ef_pct is not None and np.isfinite(float(ef_pct)):
        color = "var(--accent-green)" if float(ef_pct) >= 55 else (
            "var(--accent-yellow)" if float(ef_pct) >= 40 else "var(--accent-red)"
        )
        fevi_html = f"""
        <div class="fevi-highlight">
          <div>
            <div class="fevi-number" style="color:{color}">{float(ef_pct):.0f}%</div>
            <div class="fevi-label">FEVI preliminar</div>
          </div>
          <div>
            <div style="font-size:1.1rem; color:var(--fg)">EDV {_safe_float(ef.get('edv_ml'),1)} mL</div>
            <div style="font-size:1.1rem; color:var(--fg)">ESV {_safe_float(ef.get('esv_ml'),1)} mL</div>
            <div style="font-size:0.85rem; color:var(--fg-muted)">Masa {_safe_float(ef.get('myocardial_mass_g'),1)} g</div>
          </div>
        </div>"""

    # --- Métricas cards ---
    psd = _safe_float(metrics.get("phase_sd"), 1)
    bw = _safe_float(metrics.get("bandwidth"), 1)
    ent = _safe_float(metrics.get("entropy_normalized_pct"), 1)
    ai = _safe_float(metrics.get("asynchrony_index"), 1)
    tech_class = str(metrics.get("technical_classification", metrics.get("classification", "N/D")))

    metrics_cards = f"""
    <div class="cards">
      <div class="card"><h4>Phase SD</h4><div class="value">{psd}<span class="unit">°</span></div></div>
      <div class="card"><h4>Bandwidth</h4><div class="value">{bw}<span class="unit">°</span></div></div>
      <div class="card"><h4>Entropy</h4><div class="value">{ent}<span class="unit">%</span></div></div>
      <div class="card"><h4>Asynchrony Index</h4><div class="value">{ai}<span class="unit">%</span></div></div>
    </div>
    <p style="color:var(--fg-muted); font-size:0.82rem; margin-top:8px;">Clase técnica: <b>{tech_class}</b> (orientativa, no diagnóstica)</p>
    """

    # --- Tabla de métricas completa ---
    metrics_rows = [
        ("Phase SD", f"{psd}°"),
        ("Bandwidth", f"{bw}°"),
        ("Entropy Shannon", f"{_safe_float(metrics.get('entropy_shannon_bits'), 3)} bits"),
        ("Entropy normalizada", f"{ent}%"),
        ("Asynchrony Index", f"{ai}%"),
        ("Clase PSD técnica", f"{tech_class}"),
        ("Skewness / Kurtosis", f"{_safe_float(metrics.get('skewness'), 3)} / {_safe_float(metrics.get('kurtosis'), 3)}"),
        ("Peak phase / width", f"{_safe_float(metrics.get('peak_phase'), 1)}° / {_safe_float(metrics.get('peak_width'), 1)}°"),
        ("Latest activation", f"{_safe_float(metrics.get('latest_activation_phase'), 1)}°"),
        ("Volumen miocardio", f"{_safe_float(volumes.get('myocardial_ml'), 2)} mL"),
        ("Volumen cavidad", f"{_safe_float(volumes.get('cavity_ml'), 2)} mL"),
    ]
    if ef and ef.get("available"):
        metrics_rows.extend([
            ("EDV", f"{_safe_float(ef.get('edv_ml'), 2)} mL"),
            ("ESV", f"{_safe_float(ef.get('esv_ml'), 2)} mL"),
            ("FEVI", f"{_safe_float(ef.get('ef_pct'), 1)}%"),
        ])
        if ef.get("myocardial_mass_g") is not None:
            metrics_rows.append(("Masa miocárdica", f"{_safe_float(ef.get('myocardial_mass_g'), 1)} g"))
        if ef.get("thickening_pct") is not None:
            metrics_rows.append(("Engrosamiento ED→ES", f"{_safe_float(ef.get('thickening_pct'), 1)}%"))
    nd = metrics.get("normal_db_eval") or {}
    if nd:
        nd_label = "fuera de referencia" if nd.get("dyssynchrony") else "dentro de referencia"
        metrics_rows.append(("Interpretación vs DB", nd_label))

    metrics_table = _build_table(["Métrica", "Valor"], metrics_rows)

    # --- Territorios coronarios ---
    territory_html = ""
    if territory:
        terr_rows = []
        for t in ("LAD", "LCx", "RCA"):
            d = territory.get(t, {}) or {}
            terr_rows.append((t, f"{_safe_float(d.get('mean'), 1)}°", f"{_safe_float(d.get('std'), 1)}°", str(d.get("n", "N/D"))))
        territory_html = f"<h3 style='color:var(--accent); margin:24px 0 12px;'>Territorios coronarios</h3>" + _build_table(
            ["Territorio", "Fase media", "SD", "n seg."], terr_rows
        )

    # --- Stress-rest ---
    stress_rest_html = ""
    if stress_rest and stress_rest.get("available"):
        deltas = stress_rest.get("deltas", {})
        st = stress_rest.get("stress", {})
        rs = stress_rest.get("rest", {})
        sr_rows = []
        for key, label, unit in [
            ("phase_sd", "Phase SD", "°"), ("bandwidth", "Bandwidth", "°"),
            ("entropy_normalized_pct", "Entropy norm.", "%"),
            ("asynchrony_index", "Asynchrony Idx", "%"),
        ]:
            sr_rows.append((label, f"{_safe_float(st.get(key), 1)}{unit}", f"{_safe_float(rs.get(key), 1)}{unit}", f"{_safe_float(deltas.get(key), 1)}{unit}"))
        stress_rest_html = f"<h3 style='color:var(--accent); margin:24px 0 12px;'>Delta stress-rest</h3>" + _build_table(
            ["Métrica", "Esfuerzo", "Reposo", "Δ"], sr_rows
        )

    # --- Visualizaciones ---
    visual_sections = []

    # Montaje SA (destacado) + GIF cine debajo
    montage = _img_tag(os.path.join(output_dir, "sa_montage.png"), "Montaje clínico SA/HLA/VLA", "featured-img")
    if montage:
        visual_sections.append(f'<div class="featured">{montage}<div class="caption">Montaje clínico: cortes SA/HLA/VLA reorientados, base→ápex.</div></div>')
    montage_gif = _gif_tag(os.path.join(output_dir, "sa_montage_cine.gif"), "Montaje clínico cine")
    if montage_gif:
        visual_sections.append(f'<div class="featured" style="max-width:900px; margin:0 auto;">{montage_gif}<div class="caption">Montaje clínico cine (evolución por gate).</div></div>')

    # Panel funcional grande debajo del montaje cine.
    panel_fn = _img_tag(os.path.join(output_dir, "panel_funcional_gated.png"), "Panel funcional gated", "featured-img")
    if panel_fn:
        visual_sections.append(f'<div class="featured" style="max-width:900px; margin:0 auto;">{panel_fn}<div class="caption">Panel funcional gated: ED/ES + curvas de volumen y fase.</div></div>')

    # MIPs crudas con selector de escala interactivo.
    mip_specs = [("raw_ap_mip.png", "AP (anterior)"), ("raw_oai_mip.png", "OAI 45°"), ("raw_ll_mip.png", "Lat. izquierda")]
    mip_items = ""
    for fname, label in mip_specs:
        uri = _img_to_data_uri(os.path.join(output_dir, fname))
        if uri:
            mip_id = fname.replace(".", "_")
            mip_items += f'''<div class="gallery-item">
  <img src="{uri}" alt="{label}" id="{mip_id}" class="mip-img" style="width:100%; filter: grayscale(1);">
  <div class="caption">{label}</div>
</div>'''
    if mip_items:
        visual_sections.append(f'''<h3 style="color:var(--accent); margin:24px 0 12px;">Proyecciones planares</h3>
<div class="mip-scale-bar">
  <button class="mip-scale-btn active" data-filter="grayscale(1)">Gris</button>
  <button class="mip-scale-btn" data-filter="grayscale(0)">Original</button>
  <button class="mip-scale-btn" data-filter="grayscale(1) sepia(1) hue-rotate(200deg) saturate(3)">Azul</button>
  <button class="mip-scale-btn" data-filter="grayscale(1) sepia(1) hue-rotate(340deg) saturate(4)">Hot</button>
  <button class="mip-scale-btn" data-filter="grayscale(1) invert(1)">Invertido</button>
</div>
<div class="gallery">{mip_items}</div>''')

    # Galería principal - polar fase + histograma lado a lado, luego polar perfusión.
    polar_fase_tag = _img_to_data_uri(os.path.join(output_dir, "polar_map.png"))
    hist_tag = _img_to_data_uri(os.path.join(output_dir, "histograma.png"))
    if polar_fase_tag or hist_tag:
        duo = ""
        if polar_fase_tag:
            duo += f'<div style="flex:1; min-width:0; display:flex; flex-direction:column;"><img src="{polar_fase_tag}" alt="Mapa polar de fase AHA" style="width:100%; border-radius:var(--radius); flex:1; object-fit:fill;"><div style="text-align:center; padding:6px; font-size:0.82rem; color:var(--fg-muted);">Mapa polar de fase AHA (17)</div></div>'
        if hist_tag:
            duo += f'<div style="flex:1; min-width:0; display:flex; flex-direction:column;"><img src="{hist_tag}" alt="Histograma de fase" style="width:100%; border-radius:var(--radius); flex:1; object-fit:fill;"><div style="text-align:center; padding:6px; font-size:0.82rem; color:var(--fg-muted);">Histograma de fase</div></div>'
        visual_sections.append(f'<div style="display:flex; gap:16px; margin:16px 0; align-items:stretch;">{duo}</div>')

    # Polar perfusión + bullseye lado a lado.
    polar_smooth_uri = _img_to_data_uri(os.path.join(output_dir, "polar_perfusion_smooth.png"))
    bullseye_uri = _img_to_data_uri(os.path.join(output_dir, "bullseye_directo.png"))
    if polar_smooth_uri or bullseye_uri:
        duo2 = ""
        if polar_smooth_uri:
            duo2 += f'<div style="flex:1; min-width:0;"><img src="{polar_smooth_uri}" alt="Polar perfusión filtrado" style="width:100%; border-radius:var(--radius);"><div style="text-align:center; padding:6px; font-size:0.82rem; color:var(--fg-muted);">Mapa polar continuo de perfusión (filtrado). Apex en centro, base en borde.</div></div>'
        if bullseye_uri:
            duo2 += f'<div style="flex:1; min-width:0;"><img src="{bullseye_uri}" alt="Bullseye AHA" style="width:100%; border-radius:var(--radius);"><div style="text-align:center; padding:6px; font-size:0.82rem; color:var(--fg-muted);">Bull&#8217;s-eye segmentario AHA (perfusión)</div></div>'
        visual_sections.append(f'<div style="display:flex; gap:16px; margin:16px 0;">{duo2}</div>')

    gallery_specs = [
        ("comparacion_ejes.png", "Comparación original vs reconstruido"),
        ("curva_fevi.png", "Curva FEVI preliminar"),
        ("curva_tac.png", "Curva de actividad por gate"),
    ]
    if stress_rest and stress_rest.get("available"):
        gallery_specs.append(("comparacion_stress_rest.png", "Comparación stress vs rest"))
    gallery_items = ""
    for fname, caption in gallery_specs:
        tag = _img_tag(os.path.join(output_dir, fname), caption, "gallery-img")
        if tag:
            gallery_items += f'<div class="gallery-item">{tag}<div class="caption">{caption}</div></div>'
    if gallery_items:
        visual_sections.append(f'<h3 style="color:var(--accent); margin:24px 0 12px;">Visualizaciones</h3><div class="gallery">{gallery_items}</div>')

    # Guía fase VI con altura doble (para que la tabla sea legible).
    guia_tag = _img_tag(os.path.join(output_dir, "guia_fase_vi.png"), "Guía para fase VI", "featured-img")
    if guia_tag:
        visual_sections.append(f'<div class="featured" style="max-height:1400px;">{guia_tag}<div class="caption">Guía para fase VI: bull\'s-eye doble (fase + perfusión/viabilidad) y tabla segmentaria AHA-17.</div></div>')

    # Vistas 3D (si fueron capturadas).
    td3d_gif = _gif_tag(os.path.join(output_dir, "3d_rotation.gif"), "Reconstrucción 3D rotación")
    if td3d_gif:
        visual_sections.append(f'<div class="featured" style="max-width:700px; margin:0 auto;">{td3d_gif}<div class="caption">Reconstrucción 3D del VI — rotación automática.</div></div>')
    else:
        td_specs = [("3d_anterior.png", "Reconstrucción 3D — Vista anterior")]
        td_items = ""
        for fname, caption in td_specs:
            tag = _img_tag(os.path.join(output_dir, fname), caption, "gallery-img")
            if tag:
                td_items += f'<div class="gallery-item">{tag}<div class="caption">{caption}</div></div>'
        if td_items:
            visual_sections.append(f'<h3 style="color:var(--accent); margin:24px 0 12px;">Reconstrucción 3D</h3><div class="gallery">{td_items}</div>')

    # GIFs animados
    gif_specs = [
        ("polar_cine.gif", "Polar cine gatillado (evolución por gate)"),
        ("sa_montage_cine.gif", "Montaje clínico cine"),
    ]
    gif_items = ""
    for fname, caption in gif_specs:
        tag = _gif_tag(os.path.join(output_dir, fname), caption, "gallery-img")
        if tag:
            gif_items += f'<div class="gallery-item">{tag}<div class="caption">{caption}</div></div>'
    if gif_items:
        visual_sections.append(f'<h3 style="color:var(--accent); margin:24px 0 12px;">Animaciones</h3><div class="gallery">{gif_items}</div>')

    visuals_html = "\n".join(visual_sections)

    # --- Auditoría ---
    proc_params = processing_params or {}
    audit_rows = [
        ("Segmentación", str(seg.method if hasattr(seg, "method") else "N/D")),
        ("Dimensiones", f"{cube.shape[0]} gates × {cube.shape[1]} slices × {cube.shape[2]}×{cube.shape[3]}"),
        ("Voxels miocardio", f"{int(np.count_nonzero(seg.mask)):,}"),
        ("Threshold", _safe_float(proc_params.get("threshold"), 2)),
        ("Sigma", _safe_float(proc_params.get("smooth_sigma"), 1)),
        ("Harmonics", str(proc_params.get("harmonics", "N/D"))),
        ("Estilo visual", str(proc_params.get("visual_style", "N/D"))),
        ("Rotación polar", f"{proc_params.get('polar_rotation_deg', 'N/D')}°"),
    ]
    audit_table = _build_table(["Campo", "Valor"], audit_rows)

    # --- ECG ---
    ecg_html = ""
    ecg_items = []
    if proc_params.get("ecg_ritmo"):
        ecg_items.append(f'<span class="ecg-chip"><b>Ritmo:</b> {proc_params["ecg_ritmo"]}</span>')
    if proc_params.get("ecg_fc"):
        ecg_items.append(f'<span class="ecg-chip"><b>FC:</b> {proc_params["ecg_fc"]} lpm</span>')
    if proc_params.get("ecg_qrs"):
        ecg_items.append(f'<span class="ecg-chip"><b>QRS:</b> {proc_params["ecg_qrs"]} ms</span>')
    if proc_params.get("ecg_qt"):
        ecg_items.append(f'<span class="ecg-chip"><b>QT:</b> {proc_params["ecg_qt"]} ms</span>')
    ecg_flags = []
    if proc_params.get("ecg_bri"):
        ecg_flags.append("BRI")
    if proc_params.get("ecg_brd"):
        ecg_flags.append("BRD")
    if proc_params.get("ecg_marcapasos"):
        ecg_flags.append("Marcapasos/CRT")
    if ecg_flags:
        ecg_items.append(f'<span class="ecg-chip"><b>Conducción:</b> {", ".join(ecg_flags)}</span>')
    if ecg_items:
        ecg_html = f'<h3 style="color:var(--accent); margin:24px 0 12px;">Contexto ECG</h3><div class="ecg-bar">{"".join(ecg_items)}</div>'

    # --- Interpretación ---
    interp_items = [
        "<li><b>polar_map:</b> distribución de fase por segmentos AHA. Patrón/extensión de disincronía.</li>",
        "<li><b>polar_clinico:</b> histograma + bullseye integrados. Lectura rápida estilo estación clínica.</li>",
        "<li><b>polar_perfusion_directa:</b> intensidad perfusional continua (apex→base). Complementa fase.</li>",
        "<li><b>bullseye_directo:</b> resumen segmentario AHA de perfusión/espesor/motilidad.</li>",
        "<li><b>polar_cine:</b> dinámica temporal gate-a-gate. Útil cuando la foto estática no alcanza.</li>",
    ]
    interp_html = f"""
    <div class="interpretation">
      <h4>Guía de interpretación</h4>
      <ul>{"".join(interp_items)}</ul>
      <p style="margin-top:12px; font-size:0.82rem; color:var(--fg-muted);">
        Interpretar siempre en conjunto: fase + perfusión + cine + métricas (PSD/BW/Entropy),
        comparadas contra referencias del mismo software. FEVI en este informe es preliminar.
      </p>
    </div>"""

    # --- Ensamblar HTML ---
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SINCRO — Informe clínico ({phase_label})</title>
<style>{_CSS}</style>
</head>
<body>

<div class="header">
  <h1 style="font-size:2.2rem; letter-spacing:3px;">SINCRO</h1>
  <div style="font-size:1.15rem; color:var(--fg); font-weight:600; margin-top:6px;">{header_title}</div>
  <div class="subtitle">Informe automático ({phase_label})</div>
  <div class="patient-bar">
    <span class="chip"><b>Paciente:</b> {patient_name}</span>
    <span class="chip"><b>ID:</b> {patient_id}</span>
    <span class="chip"><b>Sexo:</b> {patient_sex}</span>
    <span class="chip"><b>Fecha:</b> {study_date}</span>
    <span class="chip"><b>Estudio:</b> {study_desc}</span>
  </div>
</div>

<div class="container">

{exec_html}
{fevi_html}

<div class="tabs">
  <button class="tab-btn active" data-group="main" data-tab="tab-metrics">Métricas</button>
  <button class="tab-btn" data-group="main" data-tab="tab-visual">Visualización</button>
  <button class="tab-btn" data-group="main" data-tab="tab-audit">Auditoría</button>
</div>

<div id="tab-metrics" class="tab-panel active" data-group="main">
  <h3 style="color:var(--accent); margin-bottom:16px;">Métricas principales</h3>
  {metrics_cards}
  {metrics_table}
  {territory_html}
  {stress_rest_html}
</div>

<div id="tab-visual" class="tab-panel" data-group="main">
  {visuals_html}
</div>

<div id="tab-audit" class="tab-panel" data-group="main">
  <h3 style="color:var(--accent); margin-bottom:16px;">Criterios y parámetros</h3>
  {audit_table}
  {ecg_html}
  {interp_html}
</div>

</div>

<div class="container" style="margin-top:32px;">
  <div class="interpretation" style="border-color:var(--border); opacity:0.85;">
    <h4 style="color:var(--fg-muted); font-size:0.9rem;">Referencias y definiciones</h4>
    <ul style="font-size:0.8rem; color:var(--fg-muted);">
      <li><b>AHA (American Heart Association):</b> modelo de 17 segmentos estándar para dividir el ventrículo izquierdo en regiones anatómicas comparables entre estudios y software.</li>
      <li><b>Phase SD (°):</b> desviación estándar de la fase de activación mecánica por segmento. Mayor valor = mayor dispersión temporal = mayor probabilidad de asincronía.</li>
      <li><b>Bandwidth (°):</b> ancho del histograma de fase (percentil 95%). Complementa Phase SD para evaluar dispersión.</li>
      <li><b>Entropy (%):</b> entropía normalizada de Shannon del histograma de fase. Mide desorganización; mayor valor = peor sincronía.</li>
      <li><b>FEVI:</b> fracción de eyección del ventrículo izquierdo (preliminar en SINCRO; no reemplaza software validado para cuantificación).</li>
      <li><b>ECTb:</b> método de contorno elíptico para segmentación del VI y cálculo de volúmenes/FEVI.</li>
      <li><b>Bullseye:</b> representación circular de los17segmentos AHA (ápex en el centro, base en el borde).</li>
      <li><b>Denoise+:</b> filtro propio de SINCRO que aplica denoise bilateral del sinograma + realce por resta para abrir la cavidad y limpiar el fondo.</li>
      <li><b>FBP CLEAR:</b> filtro exclusivo de SINCRO para imágenes gated: denoise Poisson del sinograma + doble recon (nítida + difusa) + resta ponderada. Limpia el ruido de bajo conteo por gate sin perder la cavidad.</li>
      <li><b>NÍTIDA (OmniRes):</b> recuperación de resolución iterativa (OSEM + prior MAP) para nitidez sin perder contraste.</li>
    </ul>
  </div>
</div>

<div class="footer">
  Informe generado automáticamente por SINCRO · {datetime.now().strftime("%d/%m/%Y %H:%M")}<br>
  Resultados orientativos para apoyo clínico y auditoría técnica.
</div>

<div class="lightbox" id="lightbox">
  <button class="close-btn">&times;</button>
  <img id="lightbox-img" src="" alt="Ampliada">
</div>

<script>{_JS}</script>
</body>
</html>"""

    with open(output_html, "wb") as f:
        f.write(html.encode("utf-8"))

    return output_html


def _build_table(headers: list[str], rows: list[tuple]) -> str:
    """Genera una tabla HTML con headers y filas."""
    thead = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = f'<td class="label">{row[0]}</td>' + "".join(f"<td>{c}</td>" for c in row[1:])
        body += f"<tr>{cells}</tr>"
    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr>{thead}</tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>"""

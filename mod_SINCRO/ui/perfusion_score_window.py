"""Panel LCARS de score de perfusión: mapas polares, bull's-eye editable y territorios."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from core.aha_segments import TERRITORY_MAP, bullseye_segment_at, bullseye_segment_layout, bullseye_territory_extent
from core.perfusion_quant import (
    SEGMENT_NAMES, perfusion_scores, score_summary_text, territory_scores,
)
from ui.epar_dialog import EParDialog
from viz.perfusion_score_figure import (
    AMBER, BG, BLUE, PANEL, SCORE_COLORS, SCORE_LABELS, TERRITORY_COLORS, TERRITORY_LABELS, TEXT,
    perfusion_rgba, reversibility_rgba, score_text_color,
)

GRID = BLUE
DEFINITIONS = {
    "SSS": "SSS · Summed Stress Score\nSuma de los scores 0–4 de los 17 segmentos en ESFUERZO (máx. 68).\n"
           "Refleja extensión y severidad del defecto en esfuerzo: isquemia + necrosis.\n"
           "Referencia habitual (scoring visual/base de normales): <4 normal · 4–8 leve · 9–13 moderado · >13 severo.",
    "SRS": "SRS · Summed Rest Score\nSuma de los scores de los 17 segmentos en REPOSO (máx. 68).\n"
           "Refleja el defecto fijo: necrosis/cicatriz o artefacto.",
    "SDS": "SDS · Summed Difference Score = SSS − SRS\n"
           "Refleja la isquemia inducible (defecto reversible).\n"
           "Referencia habitual (scoring visual/base de normales): 0–1 sin isquemia · 2–4 leve · 5–6 moderada · ≥7 severa.",
    "EXT%": "Extensión: % del área del mapa polar con captación <70% del máximo en esfuerzo.\n"
            "Área del mapa, no masa miocárdica.",
    "REV%": "Reversibilidad: % del área con defecto en esfuerzo que mejora ≥10 puntos en reposo.",
}
_PCT_TIP = "\n% miocardio = score / 68 × 100."

STYLE = f"""
QWidget#scoreLcars {{ background:{BG}; color:{TEXT}; font-family:'Arial Narrow','Bahnschrift Condensed','Segoe UI'; }}
QFrame#lcarsBar {{ background:{AMBER}; border-top-left-radius:16px; border-bottom-left-radius:4px; }}
QFrame#lcarsRail {{ background:{PANEL}; border-left:10px solid {BLUE}; border-top-left-radius:18px; border-bottom-left-radius:18px; }}
QFrame#readout {{ background:{PANEL}; border:1px solid #343945; border-radius:10px; }}
QPushButton[lcars="true"] {{ border:none; border-radius:13px; padding:7px 12px; color:#172028;
  font-weight:800; background:{BLUE}; text-align:right; }}
QPushButton[lcars="true"]:checked {{ background:{AMBER}; }}
QPushButton[lcars="true"]:hover {{ background:#9bb6bd; }}
QPushButton#accept {{ background:#90e277; }}
QPushButton#reject {{ background:#a7615e; color:white; }}
QLabel#bigValue {{ color:{AMBER}; font-size:30px; font-weight:800; background:transparent; }}
QLabel#caption {{ color:{BLUE}; font-size:10px; font-weight:800; background:transparent; }}
QLabel#head {{ color:{AMBER}; font-weight:800; background:transparent; }}
QLabel#cell {{ color:{TEXT}; background:transparent; }}
QLabel#totalCell {{ color:{AMBER}; font-weight:800; font-size:13px; background:transparent;
  border-top:1px solid {AMBER}; padding-top:3px; }}
QLabel#status {{ color:#81c7ce; font-weight:700; background:transparent; }}
QLabel#small {{ color:#8a979c; font-size:9px; background:transparent; }}
"""


def _qimage(rgba: np.ndarray) -> QImage:
    arr = np.ascontiguousarray((np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8))
    height, width, _ = arr.shape
    return QImage(arr.data, width, height, 4 * width, QImage.Format.Format_RGBA8888).copy()


class BullseyeView(QWidget):
    """Bull's-eye AHA 17 dibujado con QPainter; opcionalmente editable por clic."""

    segmentActivated = pyqtSignal(int, int)

    def __init__(self, title: str, *, editable: bool = False, parent=None):
        super().__init__(parent)
        self.title = title
        self.editable = editable
        self.image: QImage | None = None
        self.fills: dict[int, QColor] = {}
        self.labels: dict[int, str] = {}
        self.label_colors: dict[int, QColor] = {}
        self.edited: set[int] = set()
        self.show_territories = True
        self.territory_fill = False
        self.footer = ("", "")
        self.info = None
        self.setMinimumSize(180, 210)
        self.setMouseTracking(True)

    def _geometry(self) -> tuple[float, float, float]:
        title_h, footer_h = 22.0, 20.0
        side = max(40.0, min(self.width() - 12.0, self.height() - title_h - footer_h - 8.0))
        return self.width() / 2.0, title_h + 4.0 + side / 2.0, side / 2.0

    @staticmethod
    def _segment_path(item: dict, cx: float, cy: float, radius: float) -> QPainterPath:
        path = QPainterPath()
        r0, r1 = item["r0"] * radius, item["r1"] * radius
        if item["segment"] == 17:
            path.addEllipse(QPointF(cx, cy), r1, r1)
            return path
        start, span = item["theta0"], item["theta1"] - item["theta0"]
        outer = QRectF(cx - r1, cy - r1, 2 * r1, 2 * r1)
        inner = QRectF(cx - r0, cy - r0, 2 * r0, 2 * r0)
        path.arcMoveTo(outer, start)
        path.arcTo(outer, start, span)
        path.arcTo(inner, start + span, -span)
        path.closeSubpath()
        return path

    @staticmethod
    def _label_point(item: dict, cx: float, cy: float, radius: float) -> QPointF:
        if item["segment"] == 17:
            return QPointF(cx, cy)
        r = (item["r0"] + item["r1"]) / 2.0 * radius
        theta = np.radians((item["theta0"] + item["theta1"]) / 2.0)
        return QPointF(cx + r * np.cos(theta), cy - r * np.sin(theta))

    def segment_at(self, pos) -> int:
        cx, cy, radius = self._geometry()
        dx, dy = (pos.x() - cx) / radius, (cy - pos.y()) / radius
        return int(bullseye_segment_at(np.hypot(dx, dy), np.degrees(np.arctan2(dy, dx))))

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(BG))
        title_font = QFont("Arial Narrow", 10, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(AMBER))
        painter.drawText(QRectF(0, 0, self.width(), 22), Qt.AlignmentFlag.AlignCenter, self.title.upper())
        cx, cy, radius = self._geometry()
        disk = QPainterPath()
        disk.addEllipse(QPointF(cx, cy), radius, radius)
        painter.fillPath(disk, QColor("#05070b"))
        if self.image is not None:
            painter.save()
            painter.setClipPath(disk)
            painter.drawImage(QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius), self.image)
            painter.restore()
        layout = bullseye_segment_layout()
        paths = {item["segment"]: self._segment_path(item, cx, cy, radius) for item in layout}
        if self.territory_fill:
            for terr, segs in TERRITORY_MAP.items():
                for seg in segs:
                    painter.fillPath(paths[seg], QColor(TERRITORY_COLORS[terr]).darker(150))
        for seg, color in self.fills.items():
            painter.fillPath(paths[seg], color)
        painter.setPen(QPen(QColor(GRID), 1.0))
        for path in paths.values():
            painter.drawPath(path)
        if self.show_territories:
            for terr, segs in TERRITORY_MAP.items():
                union = QPainterPath()
                for seg in segs:
                    union = union.united(paths[seg])
                painter.setPen(QPen(QColor(TERRITORY_COLORS[terr]), 2.6))
                painter.drawPath(union.simplified())
        label_font = QFont("Arial Narrow", max(7, int(radius / 9)), QFont.Weight.Bold)
        painter.setFont(label_font)
        for item in layout:
            seg = item["segment"]
            text = self.labels.get(seg)
            if not text:
                continue
            point = self._label_point(item, cx, cy, radius)
            box = QRectF(point.x() - 18, point.y() - 10, 36, 20)
            if seg in self.edited:
                painter.setPen(QPen(QColor("#ffffff"), 1.6))
                painter.drawEllipse(point, radius / 11 + 3, radius / 11 + 3)
            painter.setPen(self.label_colors.get(seg, QColor("#f2f2f2")))
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        painter.setFont(title_font)
        footer = QRectF(6, self.height() - 20, self.width() - 12, 18)
        painter.setPen(QColor(AMBER))
        painter.drawText(footer, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.footer[0])
        painter.setPen(QColor("#81c7ce"))
        painter.drawText(footer, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, self.footer[1])

    def mouseMoveEvent(self, event):
        seg = self.segment_at(event.position())
        self.setToolTip(self.info(seg) if seg and self.info else "")
        self.setCursor(Qt.CursorShape.PointingHandCursor if seg and self.editable else Qt.CursorShape.ArrowCursor)

    def mousePressEvent(self, event):
        seg = self.segment_at(event.position())
        if not self.editable or not seg:
            return
        delta = {Qt.MouseButton.LeftButton: 1, Qt.MouseButton.RightButton: -1}.get(event.button(), 0)
        self.segmentActivated.emit(seg, delta)


def _score_text_color(score) -> QColor:
    return QColor(score_text_color(score))


class PerfusionScoreWindow(EParDialog):
    """Mapas Esfuerzo/Reposo/Reversibilidad, scores SSS/SRS/SDS editables y territorios."""

    def __init__(self, parent=None, *, stress_pct, rest_pct=None, stress_img=None, rest_img=None,
                 overrides=None, cmap=None, settings=None, accent: str = AMBER):
        super().__init__(parent, title="SCORE DE PERFUSIÓN · AHA 17", accent=accent,
                         settings=settings, geometry_key="dialogs/perfusion_score_lcars/geometry")
        self.stress_pct = stress_pct
        self.rest_pct = rest_pct or None
        self.has_rest = bool(self.rest_pct)
        self.stress_img = stress_img
        self.rest_img = rest_img if self.has_rest else None
        self.cmap = cmap
        self.edits = {stage: dict((overrides or {}).get(stage) or {}) for stage in ("stress", "rest")}
        self.extent = bullseye_territory_extent(stress_img, self.rest_img) if stress_img is not None else None
        self.result: dict = {}
        self._build()
        self._refresh()
        self.resize(1360 if self.has_rest else 1000, 820)

    def _pill(self, text: str, *, checkable: bool = False, checked: bool = False) -> QPushButton:
        button = QPushButton(text)
        button.setProperty("lcars", True)
        button.setCheckable(checkable)
        button.setChecked(checked)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _legend_chip(self, text: str, color: str, fg: str = "#172028") -> QLabel:
        chip = QLabel(text)
        chip.setStyleSheet(f"background:{color}; color:{fg}; border-radius:9px; padding:3px 8px; font-weight:800;")
        return chip

    def _build(self):
        body = QWidget()
        body.setObjectName("scoreLcars")
        body.setStyleSheet(STYLE)
        self.content_layout.addWidget(body, 1)
        outer = QVBoxLayout(body)
        bar = QFrame()
        bar.setObjectName("lcarsBar")
        bar.setFixedHeight(12)
        outer.addWidget(bar)
        main = QHBoxLayout()
        outer.addLayout(main, 1)

        rail = QFrame()
        rail.setObjectName("lcarsRail")
        rail.setFixedWidth(176)
        rail_l = QVBoxLayout(rail)
        rail_l.setContentsMargins(8, 12, 8, 12)
        self.btn_territories = self._pill("TERRITORIOS", checkable=True, checked=True)
        self.btn_values = self._pill("VALORES %MÁX", checkable=True)
        self.btn_extent = self._pill("EXTENSIÓN", checkable=True)
        self.btn_reference = self._pill("REFERENCIAS", checkable=True, checked=True)
        self.btn_restore = self._pill("RESTAURAR AUTO")
        for button in (self.btn_territories, self.btn_values, self.btn_extent, self.btn_reference):
            button.toggled.connect(self._refresh)
            rail_l.addWidget(button)
        self.btn_restore.clicked.connect(self._restore_auto)
        rail_l.addWidget(self.btn_restore)
        rail_l.addSpacing(10)
        caption = QLabel("ESCALA DE SCORE")
        caption.setObjectName("caption")
        rail_l.addWidget(caption)
        for score, label in SCORE_LABELS.items():
            rail_l.addWidget(self._legend_chip(f"{score}  {label}", SCORE_COLORS[score], _score_text_color(score).name()))
        rail_l.addSpacing(8)
        caption = QLabel("TERRITORIOS")
        caption.setObjectName("caption")
        rail_l.addWidget(caption)
        for terr, color in TERRITORY_COLORS.items():
            rail_l.addWidget(self._legend_chip(f"{TERRITORY_LABELS[terr]} ({terr})", color))
        rail_l.addStretch(1)
        hint = QLabel("Clic izq. +1 · clic der. −1\nclic medio: N/D (excluye)\nPasá el mouse para detalle")
        hint.setObjectName("small")
        rail_l.addWidget(hint)
        main.addWidget(rail)

        grid = QGridLayout()
        grid.setSpacing(6)
        self.view_stress = BullseyeView("Esfuerzo")
        self.view_sss = BullseyeView("SSS · esfuerzo", editable=True)
        self.view_sss.segmentActivated.connect(lambda seg, delta: self._on_segment("stress", seg, delta))
        views = [(self.view_stress, 0, 0), (self.view_sss, 1, 0)]
        self.view_rest = self.view_rev = self.view_srs = self.view_sds = None
        if self.has_rest:
            self.view_rest = BullseyeView("Reposo")
            self.view_rev = BullseyeView("Reversibilidad")
            self.view_srs = BullseyeView("SRS · reposo", editable=True)
            self.view_srs.segmentActivated.connect(lambda seg, delta: self._on_segment("rest", seg, delta))
            self.view_sds = BullseyeView("SDS · diferencia")
            views += [(self.view_rest, 0, 1), (self.view_rev, 0, 2), (self.view_srs, 1, 1), (self.view_sds, 1, 2)]
        for view, row, col in views:
            view.info = self._segment_info
            grid.addWidget(view, row, col)
        main.addLayout(grid, 1)

        self.reference = QFrame()
        self.reference.setObjectName("readout")
        self.reference.setFixedWidth(250)
        ref_l = QVBoxLayout(self.reference)
        head = QLabel("REFERENCIA · AHA 17")
        head.setObjectName("head")
        ref_l.addWidget(head)
        self.ref_numbers = BullseyeView("Segmentos")
        self.ref_numbers.labels = {seg: str(seg) for seg in range(1, 18)}
        self.ref_numbers.show_territories = False
        self.ref_territory = BullseyeView("Territorios coronarios")
        self.ref_territory.territory_fill = True
        self.ref_territory.labels = {seg: TERRITORY_LABELS[t] for t, segs in TERRITORY_MAP.items() for seg in segs}
        for view in (self.ref_numbers, self.ref_territory):
            view.info = self._segment_info
            view.setMinimumSize(200, 210)
            ref_l.addWidget(view)
        names = QLabel("\n".join(f"{seg:>2}. {SEGMENT_NAMES[seg]}" for seg in range(1, 18)))
        names.setObjectName("small")
        ref_l.addWidget(names)
        thresholds = QLabel("Score auto por %máx: ≥70 →0 · 60–69 →1\n50–59 →2 · 40–49 →3 · <40 →4\nSin base de normales: orientativo.")
        thresholds.setObjectName("small")
        ref_l.addWidget(thresholds)
        ref_l.addStretch(1)
        main.addWidget(self.reference)

        readout = QFrame()
        readout.setObjectName("readout")
        readout.setFixedWidth(300)
        read_l = QVBoxLayout(readout)
        self.big = {}
        totals = QGridLayout()
        for col, key in enumerate(("SSS", "SRS", "SDS")):
            cap = QLabel(key)
            cap.setObjectName("caption")
            value = QLabel("—")
            value.setObjectName("bigValue")
            pct = QLabel("")
            pct.setObjectName("small")
            for widget in (cap, value, pct):
                widget.setToolTip(DEFINITIONS[key] + _PCT_TIP)
            totals.addWidget(cap, 0, col)
            totals.addWidget(value, 1, col)
            totals.addWidget(pct, 2, col)
            self.big[key] = (value, pct)
        read_l.addLayout(totals)
        read_l.addSpacing(8)
        head = QLabel("TERRITORIOS")
        head.setObjectName("head")
        read_l.addWidget(head)
        table = QGridLayout()
        columns = ("", "SSS", "SRS", "SDS", "EXT%", "REV%")
        for col, text in enumerate(columns):
            label = QLabel(text)
            label.setObjectName("caption")
            label.setToolTip(DEFINITIONS.get(text, ""))
            table.addWidget(label, 0, col)
        self.cells = {}
        for row, terr in enumerate(list(TERRITORY_MAP) + ["TOTAL"], start=1):
            is_total = terr == "TOTAL"
            name = QLabel(TERRITORY_LABELS.get(terr, terr))
            if is_total:
                name.setObjectName("totalCell")
            else:
                name.setStyleSheet(f"color:{TERRITORY_COLORS[terr]}; font-weight:800; background:transparent;")
            table.addWidget(name, row, 0)
            for col, key in enumerate(columns[1:], start=1):
                cell = QLabel("—")
                cell.setObjectName("totalCell" if is_total else "cell")
                cell.setToolTip(DEFINITIONS[key])
                table.addWidget(cell, row, col)
                self.cells[(terr, key)] = cell
        read_l.addLayout(table)
        note = QLabel("EXT%: área del mapa polar <70% del máximo en esfuerzo.\n"
                      "REV%: área con defecto que mejora ≥10 puntos en reposo.")
        note.setObjectName("small")
        read_l.addWidget(note)
        read_l.addSpacing(8)
        self.status = QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        read_l.addWidget(self.status)
        read_l.addStretch(1)
        accept = self._pill("ACEPTAR Y REVISAR")
        accept.setObjectName("accept")
        accept.clicked.connect(self.accept)
        reject = self._pill("CANCELAR")
        reject.setObjectName("reject")
        reject.clicked.connect(self.reject)
        read_l.addWidget(accept)
        read_l.addWidget(reject)
        main.addWidget(readout)

    def _segment_info(self, seg: int) -> str:
        row = next((r for r in self.result.get("rows", []) if r["segment"] == seg), {})
        terr = next((t for t, segs in TERRITORY_MAP.items() if seg in segs), "")
        lines = [f"{seg}. {SEGMENT_NAMES[seg]} · {TERRITORY_LABELS.get(terr, '')} ({terr})"]
        for stage, label, pct in (("stress", "Esfuerzo", self.stress_pct), ("rest", "Reposo", self.rest_pct)):
            if pct is None:
                continue
            value = pct.get(seg, float("nan"))
            pct_txt = "N/D" if value is None or not np.isfinite(value) else f"{value:.0f}% máx"
            edited = " (corregido)" if row.get(f"{stage}_edited") else ""
            lines.append(f"{label}: {pct_txt} · auto {self._fmt(row.get(f'{stage}_auto'))}"
                         f" · final {self._fmt(row.get(f'{stage}_score'))}{edited}")
        return "\n".join(lines)

    @staticmethod
    def _fmt(value) -> str:
        return "N/D" if value is None else str(value)

    def _on_segment(self, stage: str, seg: int, delta: int):
        row = next(r for r in self.result["rows"] if r["segment"] == seg)
        auto, current = row[f"{stage}_auto"], row[f"{stage}_score"]
        if delta == 0:
            new = (0 if auto is None else auto) if current is None else None
        elif current is None:
            new = 0 if delta > 0 else 4
        else:
            new = (current + delta) % 5
        if new == auto:
            self.edits[stage].pop(seg, None)
        else:
            self.edits[stage][seg] = new
        self._refresh()

    def _restore_auto(self):
        for stage in self.edits:
            self.edits[stage].clear()
        self._refresh()

    def summary_text(self) -> str:
        return score_summary_text({**self.result, "reviewed": True})

    def _refresh(self, *_):
        self.result = perfusion_scores(self.stress_pct, self.rest_pct, self.edits)
        rows = {row["segment"]: row for row in self.result["rows"]}
        show_values = self.btn_values.isChecked()
        show_extent = self.btn_extent.isChecked()
        for stage, view, pct, image in (("stress", self.view_stress, self.stress_pct, self.stress_img),
                                        ("rest", self.view_rest, self.rest_pct, self.rest_img)):
            if view is None:
                continue
            view.image = _qimage(perfusion_rgba(image, self.cmap, show_extent)) if image is not None and self.cmap else None
            view.labels = {seg: f"{pct[seg]:.0f}" for seg in rows if show_values and np.isfinite(pct.get(seg, np.nan))}
            ext = (self.extent or {}).get("TOTAL", {}).get("extent_pct") if stage == "stress" else None
            view.footer = (f"EXT {ext:.0f}%" if ext is not None else "", "")
        if self.view_rev is not None:
            if self.stress_img is not None and self.rest_img is not None:
                self.view_rev.image = _qimage(reversibility_rgba(self.stress_img, self.rest_img, show_extent))
            rev = (self.extent or {}).get("TOTAL", {}).get("reversible_pct")
            self.view_rev.footer = (f"REV {rev:.0f}%" if rev is not None else "", "")
        for stage, view, key in (("stress", self.view_sss, "sss"), ("rest", self.view_srs, "srs")):
            if view is None:
                continue
            view.fills = {seg: QColor(SCORE_COLORS[row[f"{stage}_score"]]) for seg, row in rows.items()}
            view.labels = {seg: self._fmt(row[f"{stage}_score"]) for seg, row in rows.items()}
            view.label_colors = {seg: _score_text_color(row[f"{stage}_score"]) for seg, row in rows.items()}
            view.edited = {seg for seg, row in rows.items() if row[f"{stage}_edited"]}
            total = self.result[key]
            view.footer = (f"{total / 68 * 100:.0f}%" if total is not None else "INCOMPLETO",
                           f"Σ {self._fmt(total)}")
        if self.view_sds is not None:
            diffs = {seg: (None if None in (row["stress_score"], row["rest_score"]) else row["stress_score"] - row["rest_score"])
                     for seg, row in rows.items()}
            self.view_sds.fills = {seg: QColor(SCORE_COLORS[None if d is None else max(0, min(4, d))]) for seg, d in diffs.items()}
            self.view_sds.labels = {seg: self._fmt(d) for seg, d in diffs.items()}
            self.view_sds.label_colors = {seg: _score_text_color(None if d is None else max(0, min(4, d))) for seg, d in diffs.items()}
            sds = self.result["sds"]
            self.view_sds.footer = (f"{sds / 68 * 100:.0f}%" if sds is not None else "", f"Σ {self._fmt(sds)}")
        for view in (self.view_stress, self.view_rest, self.view_rev, self.view_sss, self.view_srs, self.view_sds):
            if view is not None:
                view.show_territories = self.btn_territories.isChecked()
                view.update()
        self.reference.setVisible(self.btn_reference.isChecked())
        for key in ("SSS", "SRS", "SDS"):
            value, pct = self.big[key]
            total = self.result.get(key.lower())
            value.setText(self._fmt(total) if (key == "SSS" or self.has_rest) else "—")
            pct.setText(f"{total / 68 * 100:.0f}% miocardio" if total is not None else "")
        terr_scores = territory_scores(self.result)
        terr_scores["TOTAL"] = {"sss": self.result["sss"], "srs": self.result["srs"], "sds": self.result["sds"]}
        for terr, entry in terr_scores.items():
            for key in ("SSS", "SRS", "SDS"):
                self.cells[(terr, key)].setText(self._fmt(entry.get(key.lower())))
            ext = (self.extent or {}).get(terr, {})
            for key, field in (("EXT%", "extent_pct"), ("REV%", "reversible_pct")):
                value = ext.get(field)
                self.cells[(terr, key)].setText("—" if value is None else f"{value:.0f}")
        self.status.setText(f"{self.result['n_edited']} corrección(es) · se marcará como revisado al aceptar")

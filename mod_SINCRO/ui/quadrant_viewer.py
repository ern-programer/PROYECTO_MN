# -*- coding: utf-8 -*-
"""Visor de cuadrantes para amiloidosis: layouts de 4/8/9/12/16 imágenes.

Cada cuadrante muestra una imagen con rótulo, colormap individual, ventana,
y filtros aplicados. El cuadrante seleccionado tiene borde rojo grueso.

Componentes:
- QuadrantViewer: widget que dibuja la grilla de cuadrantes.
- Selector de layout (combo).
- RangeSlider global + por cuadrante.
- Selector de colormap por cuadrante.
- Filtros de imagen por cuadrante (suavizar, invertir, ecualizar).
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QSizePolicy

from core.amyloid_layouts import Quadrant, Layout


class QuadrantViewer(QWidget):
    """Visor de cuadrantes: grilla de imágenes con selección, colormaps y filtros."""

    quadrantSelected = pyqtSignal(int)  # índice del cuadrante seleccionado
    quadrantLabelEditRequested = pyqtSignal(int)  # click en rótulo para editar

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._layout: Layout | None = None
        self._selected: int = 0
        self._pixmaps: list[QPixmap | None] = []
        self._zoom: float = 1.0
        self._hover_label_idx: int = -1

    def set_layout(self, layout: Layout):
        self._layout = layout
        self._selected = 0
        self._rebuild_pixmaps()
        self.update()

    def select_quadrant(self, idx: int):
        if self._layout is not None and 0 <= idx < self._layout.total():
            self._selected = idx
            self.quadrantSelected.emit(idx)
            self.update()

    def selected_quadrant(self) -> Quadrant | None:
        if self._layout is None:
            return None
        idx = self._selected
        if 0 <= idx < len(self._layout.quadrants):
            return self._layout.quadrants[idx]
        return None

    def _rebuild_pixmaps(self):
        """Reconstruye los pixmaps de todos los cuadrantes."""
        self._pixmaps = []
        if self._layout is None:
            return
        for q in self._layout.quadrants:
            if q.image is not None:
                self._pixmaps.append(self._image_to_qpixmap(q))
            else:
                self._pixmaps.append(None)

    def _image_to_qpixmap(self, q: Quadrant) -> QPixmap:
        """Convierte una imagen 2D o RGB a QPixmap con colormap y ventana."""
        img = np.asarray(q.image, dtype=np.float64)
        if img.ndim == 3:
            # Imagen RGB pre-rendered (H, W, 3) — respetar colores directamente.
            rgb = np.clip(img, 0, 255).astype(np.uint8)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            return QPixmap.fromImage(qimg.copy())
        h, w = img.shape
        # Aplicar ventana.
        p_low = np.percentile(img, q.win_low) if q.win_low > 0 else float(img.min())
        p_high = np.percentile(img, q.win_high) if q.win_high < 100 else float(img.max())
        span = max(p_high - p_low, 1e-8)
        norm = np.clip((img - p_low) / span, 0.0, 1.0)
        # Aplicar filtros.
        norm = self._apply_filters(norm, q.filters)
        # Aplicar colormap simple (gris por defecto).
        rgb = self._apply_colormap(norm, q.cmap)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def _apply_filters(self, img: np.ndarray, filters: list[str]) -> np.ndarray:
        """Aplica filtros a la imagen normalizada (0-1)."""
        out = img.copy()
        for f in filters:
            if f == "invert":
                out = 1.0 - out
            elif f == "equalize":
                from skimage import exposure
                out = exposure.equalize_hist(out)
            elif f == "smooth":
                from scipy.ndimage import gaussian_filter
                out = gaussian_filter(out, sigma=1.0)
        return out

    def _apply_colormap(self, norm: np.ndarray, cmap_name: str) -> np.ndarray:
        """Aplica un colormap simple a la imagen normalizada."""
        h, w = norm.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        if cmap_name == "gray" or cmap_name == "grey":
            val = np.clip(norm * 255, 0, 255).astype(np.uint8)
            rgb[..., 0] = val
            rgb[..., 1] = val
            rgb[..., 2] = val
        elif cmap_name == "hot":
            rgb[..., 0] = np.clip(norm * 255, 0, 255).astype(np.uint8)
            rgb[..., 1] = np.clip((norm - 0.33) * 255 * 3, 0, 255).astype(np.uint8)
            rgb[..., 2] = np.clip((norm - 0.67) * 255 * 3, 0, 255).astype(np.uint8)
        elif cmap_name == "cool":
            rgb[..., 0] = np.clip(norm * 255, 0, 255).astype(np.uint8)
            rgb[..., 1] = np.clip((1.0 - norm) * 255, 0, 255).astype(np.uint8)
            rgb[..., 2] = np.clip(np.ones_like(norm) * 255, 0, 255).astype(np.uint8)
        elif cmap_name == "viridis":
            # Simplified viridis-like.
            rgb[..., 0] = np.clip(norm * 180 + 20, 0, 255).astype(np.uint8)
            rgb[..., 1] = np.clip(norm * 220 + 30, 0, 255).astype(np.uint8)
            rgb[..., 2] = np.clip((1.0 - norm) * 200 + 30, 0, 255).astype(np.uint8)
        else:
            # Fallback: gris.
            val = np.clip(norm * 255, 0, 255).astype(np.uint8)
            rgb[..., 0] = val
            rgb[..., 1] = val
            rgb[..., 2] = val
        return rgb

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#0b1220"))

        if self._layout is None or not self._layout.quadrants:
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Sin imágenes cargadas")
            return

        rows = self._layout.rows
        cols = self._layout.cols
        total = self._layout.total()
        ww, wh = self.width(), self.height()
        margin = 4
        gap = 2

        cell_w = (ww - margin * 2 - gap * (cols - 1)) / max(cols, 1)
        cell_h = (wh - margin * 2 - gap * (rows - 1)) / max(rows, 1)

        for idx in range(total):
            row = idx // cols
            col = idx % cols
            x0 = margin + col * (cell_w + gap)
            y0 = margin + row * (cell_h + gap)
            rect = QRectF(x0, y0, cell_w, cell_h)

            # Borde del cuadrante.
            is_selected = (idx == self._selected)
            border_width = 3 if is_selected else 1
            border_color = QColor("#ff4444") if is_selected else QColor("#334155")
            painter.setPen(QPen(border_color, border_width))
            painter.setBrush(QColor("#0f172a"))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))

            # Imagen.
            if idx < len(self._pixmaps) and self._pixmaps[idx] is not None:
                pix = self._pixmaps[idx]
                img_rect = rect.adjusted(4, 4, -4, -20)  # espacio para rótulo abajo.
                # Escalar pixmap manteniendo aspect ratio.
                pw, ph = pix.width(), pix.height()
                if pw > 0 and ph > 0:
                    scale = min(img_rect.width() / pw, img_rect.height() / ph)
                    draw_w = pw * scale
                    draw_h = ph * scale
                    draw_x = img_rect.x() + (img_rect.width() - draw_w) / 2
                    draw_y = img_rect.y() + (img_rect.height() - draw_h) / 2
                    painter.drawPixmap(QRectF(draw_x, draw_y, draw_w, draw_h), pix, QRectF(pix.rect()))

            # Rótulo.
            if idx < len(self._layout.quadrants):
                q = self._layout.quadrants[idx]
                label = q.label or f"#{idx+1}"
                painter.setPen(QColor("#e2e8f0"))
                painter.drawText(
                    QRectF(x0, y0 + cell_h - 20, cell_w, 18),
                    Qt.AlignmentFlag.AlignCenter,
                    label,
                )
                if idx == self._hover_label_idx:
                    painter.setPen(QColor("#fbbf24"))
                    painter.drawText(
                        QRectF(x0 + cell_w - 18, y0 + cell_h - 20, 14, 14),
                        Qt.AlignmentFlag.AlignCenter,
                        "✎",
                    )

        painter.end()

    def _hit_test(self, x: float, y: float) -> tuple[int, bool]:
        """Devuelve (idx, on_label). idx=-1 si fuera de grilla."""
        if self._layout is None:
            return -1, False
        rows = self._layout.rows
        cols = self._layout.cols
        ww, wh = self.width(), self.height()
        margin = 4
        gap = 2
        cell_w = (ww - margin * 2 - gap * (cols - 1)) / max(cols, 1)
        cell_h = (wh - margin * 2 - gap * (rows - 1)) / max(rows, 1)
        col = int((x - margin) / (cell_w + gap))
        row = int((y - margin) / (cell_h + gap))
        if 0 <= col < cols and 0 <= row < rows:
            idx = row * cols + col
            if idx < self._layout.total():
                y0 = margin + row * (cell_h + gap)
                label_top = y0 + cell_h - 22
                return idx, y >= label_top
        return -1, False

    def mouseMoveEvent(self, event):
        idx, on_label = self._hit_test(event.position().x(), event.position().y())
        new_hover = idx if on_label else -1
        if new_hover != self._hover_label_idx:
            self._hover_label_idx = new_hover
            self.update()
        if on_label:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def leaveEvent(self, event):
        self._hover_label_idx = -1
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if self._layout is None:
            return
        idx, on_label = self._hit_test(event.position().x(), event.position().y())
        if idx < 0:
            return
        self.select_quadrant(idx)
        if on_label:
            self.quadrantLabelEditRequested.emit(idx)

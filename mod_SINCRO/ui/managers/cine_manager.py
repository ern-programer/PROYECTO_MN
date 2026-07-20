"""CineManager - Gestión de cine y visualización."""
from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QMovie, QPixmap


class CineManager:
    """Gestiona cine, previews y visualización."""

    def __init__(self):
        self.preview_pixmaps: dict[str, QPixmap] = {}
        self.preview_movies: dict[str, QMovie] = {}
        self.preview_base_sizes: dict[str, Any] = {}
        self.preview_zoom: dict[str, float] = {}
        self.polar_cine_preview_frames: list[QPixmap] = []
        self.polar_cine_preview_index = 0
        self.polar_cine_playing = False
        self.compare_axes_preview_frames: list[QPixmap] = []
        self.compare_axes_preview_index = 0
        self.compare_axes_playing = False

    def load_preview(self, name: str, path: str) -> bool:
        """Carga una imagen de preview."""
        if not os.path.exists(path):
            return False
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return False
        self.preview_pixmaps[name] = pixmap
        self.preview_base_sizes[name] = pixmap.size()
        return True

    def load_movie(self, name: str, path: str) -> bool:
        """Carga una película GIF."""
        if not os.path.exists(path):
            return False
        movie = QMovie(path)
        if not movie.isValid():
            return False
        self.preview_movies[name] = movie
        return True

    def get_preview(self, name: str) -> QPixmap | None:
        """Obtiene un preview."""
        return self.preview_pixmaps.get(name)

    def get_movie(self, name: str) -> QMovie | None:
        """Obtiene una película."""
        return self.preview_movies.get(name)

    def set_zoom(self, name: str, zoom: float):
        """Establece zoom de un preview."""
        self.preview_zoom[name] = max(0.1, min(5.0, zoom))

    def get_zoom(self, name: str) -> float:
        """Obtiene zoom de un preview."""
        return self.preview_zoom.get(name, 1.0)

    def clear_previews(self):
        """Limpia todos los previews."""
        self.preview_pixmaps.clear()
        self.preview_movies.clear()
        self.preview_base_sizes.clear()
        self.preview_zoom.clear()

    def get_cine_summary(self) -> dict[str, Any]:
        """Retorna resumen del estado del cine."""
        return {
            "n_previews": len(self.preview_pixmaps),
            "n_movies": len(self.preview_movies),
            "polar_cine_frames": len(self.polar_cine_preview_frames),
            "polar_cine_playing": self.polar_cine_playing,
            "compare_axes_frames": len(self.compare_axes_preview_frames),
            "compare_axes_playing": self.compare_axes_playing,
        }


import os  # noqa: E402

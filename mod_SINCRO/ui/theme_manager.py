"""Gestión de temas visuales de la aplicación.

Centraliza la lógica de temas para poder ELEGIR entre:
  - "classic": estilo NATIVO de Qt (como estaba la app antes del QSS). No aplica
	ninguna hoja de estilo; el sistema operativo/Qt decide el look.
  - "modern": hoja de estilo moderna (``ui/theme.qss``), look claro con acento
	azul GammaSync, tarjetas redondeadas, etc.

La preferencia se persiste en ``QSettings("Gammasys", "GammaSync")`` bajo la
clave ``ui/theme``. El default es "classic" (nativo) para no forzar el QSS
mientras el usuario valida el rediseño; el moderno queda disponible como opción
desde el panel de Configuración.

Diseño reversible a propósito: si falta el .qss, el tema moderno cae a nativo
sin romper (se registra el fallo silenciosamente).
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QSettings

# Identificadores estables de tema y su etiqueta legible para la UI.
THEME_CLASSIC = "classic"
THEME_MODERN = "modern"

# Orden e nombres mostrados en el selector del panel de Configuración.
AVAILABLE_THEMES: list[tuple[str, str]] = [
	(THEME_CLASSIC, "Clásico (nativo)"),
	(THEME_MODERN, "Moderno (QSS)"),
]

DEFAULT_THEME = THEME_CLASSIC

_SETTINGS_KEY = "ui/theme"


def _settings() -> QSettings:
	return QSettings("Gammasys", "GammaSync")


def theme_label(theme_id: str) -> str:
	"""Devuelve la etiqueta legible de un id de tema (o el id si es desconocido)."""
	for tid, label in AVAILABLE_THEMES:
		if tid == theme_id:
			return label
	return theme_id


def label_to_theme(label: str) -> str:
	"""Inverso de :func:`theme_label`: de etiqueta legible a id de tema."""
	for tid, lbl in AVAILABLE_THEMES:
		if lbl == label:
			return tid
	return DEFAULT_THEME


def current_theme() -> str:
	"""Lee el tema guardado en QSettings. Si no hay o es inválido, devuelve el default."""
	value = str(_settings().value(_SETTINGS_KEY, DEFAULT_THEME) or DEFAULT_THEME)
	valid = {tid for tid, _ in AVAILABLE_THEMES}
	return value if value in valid else DEFAULT_THEME


def save_theme(theme_id: str) -> None:
	"""Persiste el tema elegido en QSettings."""
	valid = {tid for tid, _ in AVAILABLE_THEMES}
	if theme_id not in valid:
		theme_id = DEFAULT_THEME
	s = _settings()
	s.setValue(_SETTINGS_KEY, theme_id)
	s.sync()


def _qss_path() -> str:
	"""Ruta absoluta al archivo de la hoja de estilo moderna."""
	return os.path.join(os.path.dirname(__file__), "theme.qss")


def load_modern_stylesheet() -> str:
	"""Lee ``ui/theme.qss``. Devuelve "" si no se puede leer (fallback a nativo)."""
	try:
		with open(_qss_path(), encoding="utf-8") as fh:
			return fh.read()
	except OSError:
		return ""


def apply_theme(app, theme_id: str | None = None) -> str:
	"""Aplica un tema a la ``QApplication``.

	- "classic": limpia la hoja de estilo (``setStyleSheet("")``) → look nativo.
	- "modern": carga ``ui/theme.qss``; si falla, cae a nativo.

	Si ``theme_id`` es None, usa el tema guardado en QSettings. Devuelve el id de
	tema efectivamente aplicado (útil para reflejarlo en la UI).
	"""
	if theme_id is None:
		theme_id = current_theme()
	if theme_id == THEME_MODERN:
		qss = load_modern_stylesheet()
		app.setStyleSheet(qss)  # "" si faltó el archivo → nativo, sin romper
		return THEME_MODERN if qss else THEME_CLASSIC
	# Clásico / nativo: sin hoja de estilo.
	app.setStyleSheet("")
	return THEME_CLASSIC

"""SINCRO - ui.managers.pipeline_history

Infraestructura del **flujo ida-y-vuelta** (Fase A): registro lineal de los pasos
del procesamiento + pila de deshacer/rehacer (Ctrl+Z / Ctrl+Shift+Z).

No dibuja nada ni conoce Qt: es lógica pura y testeable. La UI (main_window) le
registra los pasos, le empuja acciones deshacibles y consulta el estado para
pintar la futura barra de pasos (Fase B) y habilitar/deshabilitar Undo/Redo.

Modelo:
- Cada **paso** (StepState) es un nodo lineal con estado VACÍO / VÁLIDO /
  DESACTUALIZADO y una firma (hash de sus inputs). Al recomputar un paso se
  marcan DESACTUALIZADOS todos sus descendientes.
- Cada **acción deshacible** (UndoEntry) guarda dos callables: `undo` restaura el
  estado previo y `redo` re-aplica el posterior. La captura/restauración del
  estado concreto (atributos de la ventana) la resuelve quien registra la acción;
  acá solo se gestiona la pila.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from hashlib import md5
from typing import Callable


class StepStatus(Enum):
	"""Estado de un paso del pipeline."""

	EMPTY = "empty"    # todavía no ejecutado
	VALID = "valid"    # ejecutado y al día
	STALE = "stale"    # desactualizado: un paso anterior cambió


@dataclass
class StepState:
	"""Estado de un paso del pipeline (nodo lineal del DAG)."""

	key: str
	label: str
	order: int
	status: StepStatus = StepStatus.EMPTY
	signature: str = ""
	updated_at: float = 0.0


@dataclass
class UndoEntry:
	"""Una acción deshacible: `undo` restaura el antes, `redo` re-aplica el después."""

	label: str
	undo: Callable[[], None]
	redo: Callable[[], None]
	ts: float = field(default_factory=time.time)


class PipelineHistory:
	"""Registro de pasos + pila de deshacer/rehacer del procesamiento SINCRO."""

	def __init__(self, max_undo: int = 40):
		self._steps: dict[str, StepState] = {}
		self._order: list[str] = []
		self._undo_stack: list[UndoEntry] = []
		self._redo_stack: list[UndoEntry] = []
		self._max_undo = int(max_undo)
		self._listeners: list[Callable[[], None]] = []

	# ------------------------------------------------------------------ pasos
	def register_step(self, key: str, label: str) -> StepState:
		"""Registra un paso (idempotente: si ya existe, devuelve el existente)."""
		existing = self._steps.get(key)
		if existing is not None:
			return existing
		st = StepState(key=key, label=label, order=len(self._order))
		self._steps[key] = st
		self._order.append(key)
		return st

	def steps(self) -> list[StepState]:
		return [self._steps[k] for k in self._order]

	def get(self, key: str) -> StepState | None:
		return self._steps.get(key)

	@staticmethod
	def make_signature(*parts) -> str:
		"""Firma estable (md5) de los inputs de un paso, para detectar cambios."""
		payload = "|".join(str(p) for p in parts)
		return md5(payload.encode("utf-8", "replace")).hexdigest()

	def mark_done(self, key: str, signature: str = "") -> None:
		"""Marca un paso como VÁLIDO con la firma de sus inputs."""
		st = self._steps.get(key)
		if st is None:
			return
		st.signature = str(signature)
		st.status = StepStatus.VALID
		st.updated_at = time.time()
		self._notify()

	def invalidate_from(self, key: str) -> None:
		"""Marca DESACTUALIZADOS el paso `key` y todos los que vienen después."""
		st = self._steps.get(key)
		if st is None:
			return
		for k in self._order[st.order:]:
			s = self._steps[k]
			if s.status != StepStatus.EMPTY:
				s.status = StepStatus.STALE
		self._notify()

	def stale_steps(self) -> list[StepState]:
		return [s for s in self.steps() if s.status == StepStatus.STALE]

	# ------------------------------------------------------------- undo/redo
	def push(self, label: str, undo: Callable[[], None], redo: Callable[[], None]) -> None:
		"""Agrega una acción deshacible y limpia la pila de rehacer."""
		self._undo_stack.append(UndoEntry(label=label, undo=undo, redo=redo))
		if len(self._undo_stack) > self._max_undo:
			self._undo_stack.pop(0)
		self._redo_stack.clear()
		self._notify()

	def can_undo(self) -> bool:
		return bool(self._undo_stack)

	def can_redo(self) -> bool:
		return bool(self._redo_stack)

	def peek_undo_label(self) -> str:
		return self._undo_stack[-1].label if self._undo_stack else ""

	def peek_redo_label(self) -> str:
		return self._redo_stack[-1].label if self._redo_stack else ""

	def undo(self) -> str | None:
		"""Deshace la última acción y devuelve su etiqueta (o None si no hay)."""
		if not self._undo_stack:
			return None
		entry = self._undo_stack.pop()
		try:
			entry.undo()
		finally:
			self._redo_stack.append(entry)
			self._notify()
		return entry.label

	def redo(self) -> str | None:
		"""Rehace la última acción deshecha y devuelve su etiqueta (o None)."""
		if not self._redo_stack:
			return None
		entry = self._redo_stack.pop()
		try:
			entry.redo()
		finally:
			self._undo_stack.append(entry)
			self._notify()
		return entry.label

	def clear(self) -> None:
		"""Vacía las pilas de deshacer/rehacer (p.ej. al cargar otro estudio)."""
		self._undo_stack.clear()
		self._redo_stack.clear()
		self._notify()

	# ------------------------------------------------------------- listeners
	def add_listener(self, fn: Callable[[], None]) -> None:
		"""Suscribe un callback que se dispara ante cualquier cambio de estado."""
		self._listeners.append(fn)

	def _notify(self) -> None:
		for fn in list(self._listeners):
			try:
				fn()
			except Exception:
				pass

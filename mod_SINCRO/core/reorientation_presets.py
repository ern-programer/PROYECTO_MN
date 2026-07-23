"""Presets de orientación para la reorientación cardíaca (Rec/Ref).

Un preset describe cómo dejar los cortes SA/HLA/VLA en la convención de
despliegue deseada. Combina:

- ``flip_ap``: espejar anterior↔posterior el volumen reorientado (plano SA).
- ``invert_ll_handles``: invertir base↔ápex SOLO en la vista lateral izquierda.
- ``post_ops``: lista de operaciones geométricas extra aplicadas al volumen
  reorientado (``rot+90``, ``rot-90``, ``flip-lr``, ``flip-ap``, ``swap-hla-vla``).

Los presets de FÁBRICA cubren los casos clínicos frecuentes (estándar
Xeleris/Odyssey, dextrocardia, sin ajuste). Los presets de USUARIO se guardan
en disco y sirven para equipos con orientaciones no estándar.
"""
from __future__ import annotations

import json
import os
from typing import Any

# Operaciones geométricas válidas en un preset (además de flip_ap/invert_ll).
VALID_OPS = ("rot+90", "rot-90", "flip-lr", "flip-ap", "swap-hla-vla")


def _preset(flip_ap: bool, invert_ll_handles: bool, post_ops: list[str] | None = None,
            description: str = "") -> dict[str, Any]:
    ops = list(post_ops or [])
    return {
        "flip_ap": bool(flip_ap),
        "invert_ll_handles": bool(invert_ll_handles),
        "post_ops": [op for op in ops if op in VALID_OPS],
        "description": str(description),
        "factory": True,
    }


# Presets de fábrica. El estándar es el validado contra Xeleris/Odyssey.
FACTORY_PRESETS: dict[str, dict[str, Any]] = {
    "Estándar (Xeleris/Odyssey)": _preset(
        flip_ap=True, invert_ll_handles=True, post_ops=[],
        description="Convención clínica estándar supino. Validado contra Xeleris 2 y Odyssey.",
    ),
    "Dextrocardia": _preset(
        flip_ap=True, invert_ll_handles=True, post_ops=["flip-lr"],
        description="Corazón a la derecha (situs inversus / dextrocardia): espeja L/R sobre el estándar.",
    ),
    "Sin ajuste (crudo)": _preset(
        flip_ap=False, invert_ll_handles=False, post_ops=[],
        description="Sin espejos ni inversiones. Orientación cruda del reslice (para depuración).",
    ),
}

DEFAULT_PRESET_NAME = "Estándar (Xeleris/Odyssey)"


class ReorientationPresetStore:
    """Carga/guarda presets de reorientación (fábrica + usuario) en disco."""

    def __init__(self, presets_dir: str | None = None):
        if presets_dir is None:
            presets_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "presets",
            )
        os.makedirs(presets_dir, exist_ok=True)
        self.path = os.path.join(presets_dir, "reorientation_presets.json")
        self._user: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return {str(k): self._sanitize(v) for k, v in dict(data).items()}
        except Exception:
            return {}

    def _save(self):
        with open(self.path, "wb") as fh:
            fh.write(json.dumps(self._user, ensure_ascii=False, indent=2).encode("utf-8"))

    @staticmethod
    def _sanitize(raw: dict[str, Any]) -> dict[str, Any]:
        d = dict(raw or {})
        return {
            "flip_ap": bool(d.get("flip_ap", False)),
            "invert_ll_handles": bool(d.get("invert_ll_handles", False)),
            "post_ops": [op for op in list(d.get("post_ops", [])) if op in VALID_OPS],
            "description": str(d.get("description", "")),
            "factory": False,
        }

    # ----------------------------------------------------------- consultas
    def names(self) -> list[str]:
        """Todos los nombres: fábrica primero, luego usuario (ordenado)."""
        return list(FACTORY_PRESETS.keys()) + sorted(self._user.keys())

    def get(self, name: str) -> dict[str, Any] | None:
        if name in FACTORY_PRESETS:
            return dict(FACTORY_PRESETS[name])
        if name in self._user:
            return dict(self._user[name])
        return None

    def is_factory(self, name: str) -> bool:
        return name in FACTORY_PRESETS

    # ----------------------------------------------------------- mutaciones
    def save_user(self, name: str, flip_ap: bool, invert_ll_handles: bool,
                  post_ops: list[str], description: str = "") -> None:
        name = str(name).strip()
        if not name:
            raise ValueError("El nombre del preset no puede estar vacío.")
        if name in FACTORY_PRESETS:
            raise ValueError("No se puede sobrescribir un preset de fábrica; usá otro nombre.")
        self._user[name] = {
            "flip_ap": bool(flip_ap),
            "invert_ll_handles": bool(invert_ll_handles),
            "post_ops": [op for op in list(post_ops) if op in VALID_OPS],
            "description": str(description),
            "factory": False,
        }
        self._save()

    def delete_user(self, name: str) -> bool:
        if name in self._user:
            del self._user[name]
            self._save()
            return True
        return False

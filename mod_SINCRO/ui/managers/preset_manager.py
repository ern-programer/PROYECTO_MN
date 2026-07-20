"""PresetManager - Gestión de presets de procesamiento."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


class PresetManager:
    """Gestiona guardar/cargar presets de procesamiento."""

    def __init__(self, presets_dir: str | None = None):
        if presets_dir is None:
            presets_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "presets"
            )
        self.presets_dir = presets_dir
        os.makedirs(self.presets_dir, exist_ok=True)
        self.presets_path = os.path.join(self.presets_dir, "processing_presets.json")
        self._presets_data = self._load_presets_store()

    def _load_presets_store(self) -> dict:
        """Carga el store de presets desde disco."""
        if not os.path.exists(self.presets_path):
            return {}
        try:
            with open(self.presets_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_presets_store(self):
        """Guarda el store de presets a disco."""
        try:
            with open(self.presets_path, "w", encoding="utf-8") as f:
                json.dump(self._presets_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def save_preset(self, patient: str, name: str, params: dict[str, Any]):
        """Guarda un preset para un paciente."""
        if patient not in self._presets_data:
            self._presets_data[patient] = {}
        self._presets_data[patient][name] = {
            **params,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._save_presets_store()

    def load_preset(self, patient: str, name: str) -> dict[str, Any] | None:
        """Carga un preset de un paciente."""
        return self._presets_data.get(patient, {}).get(name)

    def delete_preset(self, patient: str, name: str) -> bool:
        """Elimina un preset de un paciente."""
        if patient in self._presets_data and name in self._presets_data[patient]:
            del self._presets_data[patient][name]
            if not self._presets_data[patient]:
                del self._presets_data[patient]
            self._save_presets_store()
            return True
        return False

    def list_presets(self, patient: str) -> list[str]:
        """Lista presets de un paciente."""
        return sorted(self._presets_data.get(patient, {}).keys())

    def list_patients(self) -> list[str]:
        """Lista todos los pacientes con presets."""
        return sorted(self._presets_data.keys())

    def get_preset_params(self, patient: str, name: str) -> dict[str, Any] | None:
        """Obtiene parámetros de un preset (sin metadatos)."""
        preset = self.load_preset(patient, name)
        if preset is None:
            return None
        # Remover metadatos internos
        params = dict(preset)
        params.pop("updated_at", None)
        return params

    def export_preset(self, patient: str, name: str, output_path: str) -> str:
        """Exporta un preset a archivo JSON."""
        preset = self.load_preset(patient, name)
        if preset is None:
            raise ValueError(f"Preset no encontrado: {patient}/{name}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(preset, f, indent=2, ensure_ascii=False)
        return output_path

    def import_preset(self, patient: str, name: str, input_path: str):
        """Importa un preset desde archivo JSON."""
        with open(input_path, "r", encoding="utf-8") as f:
            preset = json.load(f)
        self.save_preset(patient, name, preset)

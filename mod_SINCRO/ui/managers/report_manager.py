"""ReportManager - Generación de reportes PDF y exportación."""
from __future__ import annotations

import os
from typing import Any

from core.export_manager import export_all
from core.logging_config import get_logger
from report.report_generator import generate_report


class ReportManager:
    """Gestiona generación de reportes PDF y exportación estructurada."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_pdf_report(
        self,
        study,
        seg,
        metrics: dict,
        territory: dict,
        processing_params: dict,
        volumes: dict,
        ef: dict,
        output_filename: str | None = None,
    ) -> str:
        """Genera reporte PDF completo."""
        if output_filename is None:
            patient_id = getattr(study, "patient_id", "unknown")
            output_filename = f"report_{patient_id}.pdf"

        output_path = os.path.join(self.output_dir, output_filename)

        try:
            generate_report(
                output_pdf=output_path,
                output_dir=self.output_dir,
                study=study,
                seg=seg,
                metrics=metrics,
                territory=territory,
                processing_params=processing_params,
                volumes=volumes,
                ef=ef,
            )
            logger = get_logger()
            logger.log_export("pdf", output_path)
            return output_path
        except Exception as exc:
            logger = get_logger()
            logger.log_error(exc, context={"report_type": "pdf"})
            raise

    def export_structured(
        self,
        study_metadata: dict,
        metrics: dict,
        segmentation_info: dict,
        processing_params: dict,
        robustness: dict | None = None,
        normal_db_eval: dict | None = None,
        qc_info: dict | None = None,
        base_name: str | None = None,
    ) -> dict[str, str]:
        """Exporta resultados a JSON/CSV/Excel."""
        try:
            results = export_all(
                self.output_dir,
                study_metadata,
                metrics,
                segmentation_info,
                processing_params,
                robustness,
                normal_db_eval,
                qc_info,
                base_name,
            )
            logger = get_logger()
            for fmt, path in results.items():
                if path:
                    logger.log_export(fmt, path)
            return results
        except Exception as exc:
            logger = get_logger()
            logger.log_error(exc, context={"report_type": "structured"})
            raise

    def get_report_summary(self) -> dict[str, Any]:
        """Retorna resumen de reportes generados."""
        reports = []
        if os.path.exists(self.output_dir):
            for f in os.listdir(self.output_dir):
                if f.endswith(".pdf"):
                    reports.append({
                        "type": "pdf",
                        "filename": f,
                        "path": os.path.join(self.output_dir, f),
                    })
                elif f.endswith(".json"):
                    reports.append({
                        "type": "json",
                        "filename": f,
                        "path": os.path.join(self.output_dir, f),
                    })
                elif f.endswith(".csv"):
                    reports.append({
                        "type": "csv",
                        "filename": f,
                        "path": os.path.join(self.output_dir, f),
                    })
        return {
            "output_dir": self.output_dir,
            "reports": reports,
            "count": len(reports),
        }

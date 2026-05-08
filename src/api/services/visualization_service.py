import math
from typing import Any, Dict, List, Optional


class VisualizationService:
    """Builds frontend-ready chart data from training results."""

    @staticmethod
    def build(
        task_type: str,
        model_type: str,
        metrics: Dict[str, Any],
        training_time_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        if task_type == "classification":
            return VisualizationService._classification(model_type, metrics, training_time_seconds)
        if task_type == "regression":
            return VisualizationService._regression(model_type, metrics, training_time_seconds)
        if task_type == "clustering":
            return VisualizationService._clustering(model_type, metrics, training_time_seconds)
        return {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _r(value: Any, ndigits: int = 4) -> Any:
        if isinstance(value, float) and not math.isnan(value) and not math.isinf(value):
            return round(value, ndigits)
        return value

    @staticmethod
    def _safe(value: Any) -> Optional[float]:
        """Return None for NaN/Inf so JSON serialization stays valid."""
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    @staticmethod
    def _stat(label: str, value: Any, unit: Optional[str] = None) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"label": label, "value": value}
        if unit:
            entry["unit"] = unit
        return entry

    # ------------------------------------------------------------------
    # Task-specific builders
    # ------------------------------------------------------------------

    @staticmethod
    def _classification(model_type: str, metrics: dict, training_time: Optional[float]) -> dict:
        accuracy  = VisualizationService._r(metrics.get("accuracy", 0))
        f1        = VisualizationService._r(metrics.get("f1", 0))
        precision = VisualizationService._r(metrics.get("precision", 0))
        recall    = VisualizationService._r(metrics.get("recall", 0))

        stats: List[Dict[str, Any]] = []
        if training_time is not None:
            stats.append(VisualizationService._stat("Training Time", round(training_time, 2), "s"))

        return {
            "chart_type": "bar",
            "title": f"{model_type} — Classification",
            "primary_metric": {"label": "F1 Score", "value": f1},
            "chart": {
                "labels": ["Accuracy", "F1", "Precision", "Recall"],
                "datasets": [{"label": "Score", "data": [accuracy, f1, precision, recall]}],
            },
            "stats": stats,
        }

    @staticmethod
    def _regression(model_type: str, metrics: dict, training_time: Optional[float]) -> dict:
        r2   = VisualizationService._r(metrics.get("r2", 0))
        mae  = VisualizationService._r(metrics.get("mae", 0))
        rmse = VisualizationService._r(metrics.get("rmse", 0))
        mse  = VisualizationService._r(metrics.get("mse", 0))

        stats: List[Dict[str, Any]] = [
            VisualizationService._stat("MAE", mae),
            VisualizationService._stat("RMSE", rmse),
            VisualizationService._stat("MSE", mse),
        ]
        if training_time is not None:
            stats.append(VisualizationService._stat("Training Time", round(training_time, 2), "s"))

        return {
            "chart_type": "bar",
            "title": f"{model_type} — Regression",
            "primary_metric": {"label": "R²", "value": r2},
            "chart": {
                "labels": ["R²"],
                "datasets": [{"label": "Score", "data": [r2]}],
            },
            "stats": stats,
        }

    @staticmethod
    def _clustering(model_type: str, metrics: dict, training_time: Optional[float]) -> dict:
        silhouette = VisualizationService._safe(metrics.get("silhouette"))
        if silhouette is not None:
            silhouette = VisualizationService._r(silhouette)

        n_clusters = metrics.get("n_clusters", 0)
        n_noise    = metrics.get("n_noise", 0)

        stats: List[Dict[str, Any]] = [
            VisualizationService._stat("Clusters", n_clusters),
            VisualizationService._stat("Noise Points", n_noise),
        ]
        if training_time is not None:
            stats.append(VisualizationService._stat("Training Time", round(training_time, 2), "s"))

        chart_labels = ["Silhouette Score"] if silhouette is not None else []
        chart_data   = [silhouette] if silhouette is not None else []

        return {
            "chart_type": "bar",
            "title": f"{model_type} — Clustering",
            "primary_metric": {"label": "Silhouette Score", "value": silhouette},
            "chart": {
                "labels": chart_labels,
                "datasets": [{"label": "Score", "data": chart_data}],
            },
            "stats": stats,
        }

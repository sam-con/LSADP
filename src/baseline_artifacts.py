"""Persistent baseline artifact management."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    CANDIDATE_CANONICAL_LEAGUES_FILE,
    CANDIDATE_HISTORY_CURVE_MODELS_FILE,
    CANDIDATE_HISTORY_CURVES_FILE,
    CANDIDATE_HISTORY_ENVIRONMENTS_FILE,
    CANDIDATE_HISTORY_ENVIRONMENT_SEASONS_FILE,
    CANDIDATE_HISTORY_LIBRARY_METADATA_FILE,
    CANDIDATE_CURVES_FILE,
    CANDIDATE_MARKET_CALIBRATION_FILE,
    CANDIDATE_METADATA_FILE,
    CANDIDATE_MODEL_PARAMETERS_FILE,
    CANDIDATE_MODEL_VALIDATION_FILE,
    CANDIDATE_REPLACEMENT_FILE,
    PRODUCTION_CANONICAL_LEAGUES_FILE,
    PRODUCTION_HISTORY_CURVE_MODELS_FILE,
    PRODUCTION_HISTORY_CURVES_FILE,
    PRODUCTION_HISTORY_ENVIRONMENTS_FILE,
    PRODUCTION_HISTORY_ENVIRONMENT_SEASONS_FILE,
    PRODUCTION_HISTORY_LIBRARY_METADATA_FILE,
    PRODUCTION_CURVES_FILE,
    PRODUCTION_MARKET_CALIBRATION_FILE,
    PRODUCTION_METADATA_FILE,
    PRODUCTION_MODEL_PARAMETERS_FILE,
    PRODUCTION_MODEL_VALIDATION_FILE,
    PRODUCTION_REPLACEMENT_FILE,
)
from src.models import CanonicalArtifacts, ConfigError


class CanonicalArtifactManager:
    """Manage candidate and production canonical-model artifacts."""

    def __init__(
        self,
        curves_path: Path,
        replacement_path: Path,
        market_calibration_path: Path,
        model_parameters_path: Path,
        validation_path: Path,
        canonical_config_path: Path,
        metadata_path: Path,
        history_position_environments_path: Path | None = None,
        history_environment_seasons_path: Path | None = None,
        history_curve_models_path: Path | None = None,
        history_curves_path: Path | None = None,
        history_library_metadata_path: Path | None = None,
    ) -> None:
        self.curves_path = curves_path
        self.replacement_path = replacement_path
        self.market_calibration_path = market_calibration_path
        self.model_parameters_path = model_parameters_path
        self.validation_path = validation_path
        self.canonical_config_path = canonical_config_path
        self.metadata_path = metadata_path
        base_dir = curves_path.parent
        self.history_position_environments_path = history_position_environments_path or (base_dir / "position_scoring_environments.csv")
        self.history_environment_seasons_path = history_environment_seasons_path or (base_dir / "position_environment_seasons.csv")
        self.history_curve_models_path = history_curve_models_path or (base_dir / "position_curve_models.csv")
        self.history_curves_path = history_curves_path or (base_dir / "fitted_position_curves.csv")
        self.history_library_metadata_path = history_library_metadata_path or (base_dir / "history_library_metadata.json")

    @classmethod
    def production(cls) -> "CanonicalArtifactManager":
        return cls(
            curves_path=PRODUCTION_CURVES_FILE,
            replacement_path=PRODUCTION_REPLACEMENT_FILE,
            market_calibration_path=PRODUCTION_MARKET_CALIBRATION_FILE,
            model_parameters_path=PRODUCTION_MODEL_PARAMETERS_FILE,
            validation_path=PRODUCTION_MODEL_VALIDATION_FILE,
            canonical_config_path=PRODUCTION_CANONICAL_LEAGUES_FILE,
            metadata_path=PRODUCTION_METADATA_FILE,
            history_position_environments_path=PRODUCTION_HISTORY_ENVIRONMENTS_FILE,
            history_environment_seasons_path=PRODUCTION_HISTORY_ENVIRONMENT_SEASONS_FILE,
            history_curve_models_path=PRODUCTION_HISTORY_CURVE_MODELS_FILE,
            history_curves_path=PRODUCTION_HISTORY_CURVES_FILE,
            history_library_metadata_path=PRODUCTION_HISTORY_LIBRARY_METADATA_FILE,
        )

    @classmethod
    def candidate(cls) -> "CanonicalArtifactManager":
        return cls(
            curves_path=CANDIDATE_CURVES_FILE,
            replacement_path=CANDIDATE_REPLACEMENT_FILE,
            market_calibration_path=CANDIDATE_MARKET_CALIBRATION_FILE,
            model_parameters_path=CANDIDATE_MODEL_PARAMETERS_FILE,
            validation_path=CANDIDATE_MODEL_VALIDATION_FILE,
            canonical_config_path=CANDIDATE_CANONICAL_LEAGUES_FILE,
            metadata_path=CANDIDATE_METADATA_FILE,
            history_position_environments_path=CANDIDATE_HISTORY_ENVIRONMENTS_FILE,
            history_environment_seasons_path=CANDIDATE_HISTORY_ENVIRONMENT_SEASONS_FILE,
            history_curve_models_path=CANDIDATE_HISTORY_CURVE_MODELS_FILE,
            history_curves_path=CANDIDATE_HISTORY_CURVES_FILE,
            history_library_metadata_path=CANDIDATE_HISTORY_LIBRARY_METADATA_FILE,
        )

    def validate(self) -> None:
        required_paths = [
            self.curves_path,
            self.replacement_path,
            self.market_calibration_path,
            self.model_parameters_path,
            self.validation_path,
            self.canonical_config_path,
            self.metadata_path,
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise ConfigError(
                "Canonical model artifacts are missing. Build or validate a candidate model from the Development page. "
                f"Missing: {', '.join(missing)}"
            )
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        required_metadata = {
            "selected_model_name",
            "selected_utility_transform",
            "selected_weight_power",
            "generated_timestamp",
            "model_version",
            "canonical_environments",
            "validation_complete",
        }
        missing_metadata = sorted(required_metadata - set(metadata))
        if missing_metadata:
            raise ConfigError(
                "Canonical metadata is malformed or incompatible. Missing fields: "
                f"{', '.join(missing_metadata)}"
            )

    def load(self) -> CanonicalArtifacts:
        self.validate()
        return CanonicalArtifacts(
            curves=pd.read_csv(self.curves_path),
            replacement=pd.read_csv(self.replacement_path),
            market_calibration=pd.read_csv(self.market_calibration_path),
            model_parameters=pd.read_csv(self.model_parameters_path),
            validation=pd.read_csv(self.validation_path),
            canonical_config=json.loads(self.canonical_config_path.read_text(encoding="utf-8")),
            metadata=json.loads(self.metadata_path.read_text(encoding="utf-8")),
            history_position_environments=pd.read_csv(self.history_position_environments_path)
            if self.history_position_environments_path.exists()
            else pd.DataFrame(),
            history_environment_seasons=pd.read_csv(self.history_environment_seasons_path)
            if self.history_environment_seasons_path.exists()
            else pd.DataFrame(),
            history_curve_models=pd.read_csv(self.history_curve_models_path)
            if self.history_curve_models_path.exists()
            else pd.DataFrame(),
            history_curves=pd.read_csv(self.history_curves_path)
            if self.history_curves_path.exists()
            else pd.DataFrame(),
            history_library_metadata=json.loads(self.history_library_metadata_path.read_text(encoding="utf-8"))
            if self.history_library_metadata_path.exists()
            else {},
        )

    def save(
        self,
        curves: pd.DataFrame,
        replacement: pd.DataFrame,
        market_calibration: pd.DataFrame,
        model_parameters: pd.DataFrame,
        validation: pd.DataFrame,
        canonical_config: dict[str, Any],
        metadata: dict[str, Any],
        history_position_environments: pd.DataFrame | None = None,
        history_environment_seasons: pd.DataFrame | None = None,
        history_curve_models: pd.DataFrame | None = None,
        history_curves: pd.DataFrame | None = None,
        history_library_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.curves_path.parent.mkdir(parents=True, exist_ok=True)
        curves.to_csv(self.curves_path, index=False)
        replacement.to_csv(self.replacement_path, index=False)
        market_calibration.to_csv(self.market_calibration_path, index=False)
        model_parameters.to_csv(self.model_parameters_path, index=False)
        validation.to_csv(self.validation_path, index=False)
        self.canonical_config_path.write_text(json.dumps(canonical_config, indent=2, sort_keys=True), encoding="utf-8")
        self.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
        if history_position_environments is not None:
            history_position_environments.to_csv(self.history_position_environments_path, index=False)
        if history_environment_seasons is not None:
            history_environment_seasons.to_csv(self.history_environment_seasons_path, index=False)
        if history_curve_models is not None:
            history_curve_models.to_csv(self.history_curve_models_path, index=False)
        if history_curves is not None:
            history_curves.to_csv(self.history_curves_path, index=False)
        if history_library_metadata is not None:
            self.history_library_metadata_path.write_text(
                json.dumps(history_library_metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    def describe(self) -> dict[str, Any]:
        artifacts = self.load()
        return {
            "curve_rows": len(artifacts.curves),
            "replacement_rows": len(artifacts.replacement),
            "market_calibration_rows": len(artifacts.market_calibration),
            "model_parameter_rows": len(artifacts.model_parameters),
            "validation_rows": len(artifacts.validation),
            "history_position_environment_rows": len(artifacts.history_position_environments),
            "history_curve_rows": len(artifacts.history_curves),
            "metadata": artifacts.metadata,
        }

    def promote_from(self, candidate_manager: "CanonicalArtifactManager") -> None:
        """Promote a validated candidate model into this artifact set."""

        candidate_manager.load()

        self.curves_path.parent.mkdir(parents=True, exist_ok=True)
        for source, target in [
            (candidate_manager.curves_path, self.curves_path),
            (candidate_manager.replacement_path, self.replacement_path),
            (candidate_manager.market_calibration_path, self.market_calibration_path),
            (candidate_manager.model_parameters_path, self.model_parameters_path),
            (candidate_manager.validation_path, self.validation_path),
            (candidate_manager.canonical_config_path, self.canonical_config_path),
            (candidate_manager.metadata_path, self.metadata_path),
        ]:
            shutil.copy2(source, target)
        for source, target in [
            (candidate_manager.history_position_environments_path, self.history_position_environments_path),
            (candidate_manager.history_environment_seasons_path, self.history_environment_seasons_path),
            (candidate_manager.history_curve_models_path, self.history_curve_models_path),
            (candidate_manager.history_curves_path, self.history_curves_path),
            (candidate_manager.history_library_metadata_path, self.history_library_metadata_path),
        ]:
            if source.exists():
                shutil.copy2(source, target)

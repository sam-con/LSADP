"""Persistent baseline artifact management."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    BASELINE_CURVES_FILE,
    BASELINE_METADATA_FILE,
    BASELINE_MODEL_FILE,
    BASELINE_REPLACEMENT_FILE,
    CANDIDATE_CANONICAL_LEAGUES_FILE,
    CANDIDATE_CURVES_FILE,
    CANDIDATE_MARKET_CALIBRATION_FILE,
    CANDIDATE_METADATA_FILE,
    CANDIDATE_MODEL_PARAMETERS_FILE,
    CANDIDATE_MODEL_VALIDATION_FILE,
    CANDIDATE_REPLACEMENT_FILE,
    PRODUCTION_CANONICAL_LEAGUES_FILE,
    PRODUCTION_CURVES_FILE,
    PRODUCTION_MARKET_CALIBRATION_FILE,
    PRODUCTION_METADATA_FILE,
    PRODUCTION_MODEL_PARAMETERS_FILE,
    PRODUCTION_MODEL_VALIDATION_FILE,
    PRODUCTION_REPLACEMENT_FILE,
)
from src.models import BaselineArtifacts, CanonicalArtifacts, ConfigError


class BaselineArtifactManager:
    """Load, validate, save, and describe baseline artifacts."""

    def __init__(
        self,
        curves_path: Path = BASELINE_CURVES_FILE,
        replacement_path: Path = BASELINE_REPLACEMENT_FILE,
        model_path: Path = BASELINE_MODEL_FILE,
        metadata_path: Path = BASELINE_METADATA_FILE,
    ) -> None:
        self.curves_path = curves_path
        self.replacement_path = replacement_path
        self.model_path = model_path
        self.metadata_path = metadata_path

    def load(self) -> BaselineArtifacts:
        self.validate()
        curves = pd.read_csv(self.curves_path)
        replacement = pd.read_csv(self.replacement_path)
        model = pd.read_csv(self.model_path)
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return BaselineArtifacts(curves=curves, replacement=replacement, model=model, metadata=metadata)

    def validate(self) -> None:
        missing = [
            str(path)
            for path in [self.curves_path, self.replacement_path, self.model_path, self.metadata_path]
            if not path.exists()
        ]
        if missing:
            raise ConfigError(
                "Baseline artifacts are missing. Build / refresh the baseline model from the Development page. "
                f"Missing: {', '.join(missing)}"
            )
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        required_metadata = {
            "baseline_1qb_league_id",
            "baseline_superflex_league_id",
            "historical_seasons",
            "generated_timestamp",
            "model_version",
        }
        missing_metadata = sorted(required_metadata - set(metadata))
        if missing_metadata:
            raise ConfigError(
                "Baseline metadata is malformed or incompatible. Missing fields: "
                f"{', '.join(missing_metadata)}"
            )

    def save(
        self,
        curves: pd.DataFrame,
        replacement: pd.DataFrame,
        model: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> None:
        self.curves_path.parent.mkdir(parents=True, exist_ok=True)
        curves.to_csv(self.curves_path, index=False)
        replacement.to_csv(self.replacement_path, index=False)
        model.to_csv(self.model_path, index=False)
        self.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def describe(self) -> dict[str, Any]:
        artifacts = self.load()
        return {
            "curve_rows": len(artifacts.curves),
            "replacement_rows": len(artifacts.replacement),
            "model_rows": len(artifacts.model),
            "metadata": artifacts.metadata,
        }


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
    ) -> None:
        self.curves_path = curves_path
        self.replacement_path = replacement_path
        self.market_calibration_path = market_calibration_path
        self.model_parameters_path = model_parameters_path
        self.validation_path = validation_path
        self.canonical_config_path = canonical_config_path
        self.metadata_path = metadata_path

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
    ) -> None:
        self.curves_path.parent.mkdir(parents=True, exist_ok=True)
        curves.to_csv(self.curves_path, index=False)
        replacement.to_csv(self.replacement_path, index=False)
        market_calibration.to_csv(self.market_calibration_path, index=False)
        model_parameters.to_csv(self.model_parameters_path, index=False)
        validation.to_csv(self.validation_path, index=False)
        self.canonical_config_path.write_text(json.dumps(canonical_config, indent=2, sort_keys=True), encoding="utf-8")
        self.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def describe(self) -> dict[str, Any]:
        artifacts = self.load()
        return {
            "curve_rows": len(artifacts.curves),
            "replacement_rows": len(artifacts.replacement),
            "market_calibration_rows": len(artifacts.market_calibration),
            "model_parameter_rows": len(artifacts.model_parameters),
            "validation_rows": len(artifacts.validation),
            "metadata": artifacts.metadata,
        }

    def promote_from(self, candidate_manager: "CanonicalArtifactManager") -> None:
        """Promote a validated candidate model into this artifact set."""

        candidate_artifacts = candidate_manager.load()
        candidate_score = float(candidate_artifacts.metadata.get("selected_model_score", float("inf")))

        try:
            production_artifacts = self.load()
        except ConfigError:
            production_artifacts = None

        if production_artifacts is not None:
            production_score = float(production_artifacts.metadata.get("selected_model_score", float("inf")))
            if candidate_score > production_score * 1.02:
                raise ConfigError(
                    "Candidate model materially underperforms the current production model and cannot be promoted."
                )

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

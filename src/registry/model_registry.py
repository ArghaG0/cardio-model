import json
from pathlib import Path
from datetime import datetime
from src.config.config import REGISTRY_DIR

class ModelRegistry:
    def __init__(self, registry_dir: Path = REGISTRY_DIR):
        self.registry_dir = registry_dir
        self.registry_file = self.registry_dir / "registry.json"
        self._initialize_registry()

    def _initialize_registry(self):
        """Creates the registry directory and JSON file if they don't exist."""
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_file.exists():
            initial_state = {
                "active_version": None,
                "models": {}
            }
            self._save_registry(initial_state)

    def _load_registry(self) -> dict:
        """Reads the registry from disk."""
        with open(self.registry_file, 'r') as f:
            return json.load(f)

    def _save_registry(self, data: dict):
        """Writes the registry to disk."""
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=4)

    def get_active_version(self):
        """Returns the currently active production version."""
        registry = self._load_registry()
        return registry.get("active_version")

    def get_model(self, version: str) -> dict:
        """Returns the metadata for a specific model version."""
        registry = self._load_registry()
        return registry["models"].get(version)

    def load_active_model(self):
        """
        Resolves the active version, locates its artifact, and returns the loaded pipeline.
        Raises ValueError if no active version or missing artifact.
        """
        active_version = self.get_active_version()
        if not active_version:
            raise ValueError("No active model version is configured in the registry.")
            
        model_meta = self.get_model(active_version)
        if not model_meta:
            raise ValueError(f"Metadata for active version {active_version} not found.")
            
        artifact_path = Path(model_meta.get("artifact_path", ""))
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {artifact_path}")
            
        # SECURITY NOTICE: Joblib/Pickle must only be loaded from trusted registry locations
        import joblib
        pipeline = joblib.load(artifact_path)
        return pipeline, active_version, model_meta.get("name")

    def register_model(self, name: str, artifact_path: str, cv_metrics: dict = None, 
                       test_metrics: dict = None, gate_results: dict = None) -> str:
        """
        Registers a new model version in the registry.
        (Skeleton for future implementation)
        """
        registry = self._load_registry()
        
        # Determine new version
        versions = [int(v.replace("v", "")) for v in registry["models"].keys() if v.startswith("v")]
        next_v_num = max(versions) + 1 if versions else 1
        new_version = f"v{next_v_num}"
        
        registry["models"][new_version] = {
            "name": name,
            "version": new_version,
            "artifact_path": str(artifact_path),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "cv_metrics": cv_metrics or {},
            "test_metrics": test_metrics or {},
            "gate_results": gate_results or {},
            "promotion_status": "candidate"
        }
        
        self._save_registry(registry)
        return new_version

    def validate_artifact(self, version: str) -> bool:
        """
        Validates that a specific version's artifact exists, can be loaded,
        and is a valid Scikit-Learn Pipeline.
        Raises specific errors if validation fails.
        """
        registry = self._load_registry()
        if version not in registry["models"]:
            raise ValueError(f"Version {version} not found in registry.")
            
        model_meta = registry["models"][version]
        artifact_path = Path(model_meta.get("artifact_path", ""))
        
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {artifact_path}")
            
        import joblib
        try:
            pipeline = joblib.load(artifact_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load artifact using joblib: {e}")
            
        if not hasattr(pipeline, "named_steps") or "preprocessor" not in pipeline.named_steps or "model" not in pipeline.named_steps:
            raise ValueError("Artifact is not a valid Pipeline containing 'preprocessor' and 'model' steps.")
            
        return True

    def promote_model(self, version: str):
        """
        Promotes a specific model version to production.
        Marks the previous active version as superseded.
        Validates artifact before promotion.
        """
        # 1. Validate artifact first
        self.validate_artifact(version)
        
        registry = self._load_registry()
        old_active = registry.get("active_version")
        if old_active and old_active in registry["models"]:
            registry["models"][old_active]["promotion_status"] = "superseded"
            
        registry["active_version"] = version
        registry["models"][version]["promotion_status"] = "promoted"
        self._save_registry(registry)

    def reject_model(self, version: str):
        """Marks a candidate as rejected."""
        registry = self._load_registry()
        if version in registry["models"]:
            registry["models"][version]["promotion_status"] = "rejected"
            self._save_registry(registry)

    def rollback(self, previous_version: str):
        """
        Rolls back the active model to a previous version.
        Validates the previous version's artifact before making changes.
        """
        # 1. Validate artifact first
        self.validate_artifact(previous_version)
        
        registry = self._load_registry()
        old_active = registry.get("active_version")
        if old_active and old_active in registry["models"]:
            registry["models"][old_active]["promotion_status"] = "superseded"
            
        registry["active_version"] = previous_version
        registry["models"][previous_version]["promotion_status"] = "promoted"
        self._save_registry(registry)


"""Model library service for loading and querying curated models."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QuantizationInfo:
    """Information about a specific quantization."""
    quant: str
    filename: str
    file_size_mb: int
    min_vram_mb: int


@dataclass
class CuratedModel:
    """Information about a curated pre-tested model."""
    id: str
    name: str
    description: str
    repo_id: str
    filename_template: str
    parameters: float
    context_window: int
    category: str
    tags: List[str]
    recommended_quant: Dict[str, str]
    tested: bool
    default: bool
    performance: Dict[str, str]
    quantizations: List[QuantizationInfo]
    warning: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CuratedModel":
        """Create CuratedModel from dictionary."""
        # Convert quantizations list
        quants = [QuantizationInfo(**q) for q in data.get("quantizations", [])]
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            repo_id=data["repo_id"],
            filename_template=data["filename_template"],
            parameters=data["parameters"],
            context_window=data["context_window"],
            category=data["category"],
            tags=data["tags"],
            recommended_quant=data["recommended_quant"],
            tested=data["tested"],
            default=data["default"],
            performance=data["performance"],
            quantizations=quants,
            warning=data.get("warning")
        )


class ModelLibrary:
    """
    Curated library of pre-tested models with metadata.
    
    Loads from:
    - chorus_engine/data/curated_models.json (git tracked, official)
    - data/user_models.json (gitignored, user's custom models)
    
    Provides:
    - Pre-tested model list
    - Model categories (creative, technical, balanced, advanced)
    - Quantization recommendations
    - Download links
    - Usage notes
    """
    
    # Model categories
    CATEGORIES = {
        "balanced": {
            "name": "Balanced",
            "description": "Good at both conversation and technical tasks",
            "icon": "⚖️"
        },
        "creative": {
            "name": "Creative",
            "description": "Best for storytelling, roleplay, creative writing",
            "icon": "🎨"
        },
        "technical": {
            "name": "Technical",
            "description": "Strong at coding, analysis, structured tasks",
            "icon": "🔧"
        },
        "advanced": {
            "name": "Advanced",
            "description": "Large models for maximum performance",
            "icon": "🚀"
        }
    }
    
    def __init__(self):
        """Load models from JSON files."""
        self.curated_models: List[CuratedModel] = []
        self.user_models: List[CuratedModel] = []
        
        # Load curated models
        self._load_curated_models()
        
        # Load user models (optional)
        self._load_user_models()
        
        logger.info(f"ModelLibrary initialized: {len(self.curated_models)} curated, {len(self.user_models)} user")
    
    def _load_curated_models(self) -> None:
        """Load official pre-tested models from git-tracked JSON."""
        try:
            # Path to curated models JSON
            json_path = Path(__file__).parent.parent / "data" / "curated_models.json"
            
            if not json_path.exists():
                logger.warning(f"Curated models file not found: {json_path}")
                return
            
            with open(json_path) as f:
                data = json.load(f)
            
            # Parse models
            for model_data in data.get("models", []):
                try:
                    model = CuratedModel.from_dict(model_data)
                    self.curated_models.append(model)
                except Exception as e:
                    logger.error(f"Failed to parse curated model: {e}")
            
            logger.info(f"Loaded {len(self.curated_models)} curated models")
            
        except Exception as e:
            logger.error(f"Failed to load curated models: {e}", exc_info=True)
    
    def _load_user_models(self) -> None:
        """Load user's custom models from gitignored JSON (optional)."""
        try:
            json_path = Path("data/user_models.json")
            
            if not json_path.exists():
                # This is expected for most users
                return
            
            with open(json_path) as f:
                data = json.load(f)
            
            # Parse models
            for model_data in data.get("models", []):
                try:
                    model = CuratedModel.from_dict(model_data)
                    self.user_models.append(model)
                except Exception as e:
                    logger.error(f"Failed to parse user model: {e}")
            
            logger.info(f"Loaded {len(self.user_models)} user models")
            
        except Exception as e:
            logger.error(f"Failed to load user models: {e}")
    
    def get_curated_models(self, include_user: bool = False) -> List[CuratedModel]:
        """
        Get list of pre-tested models.
        
        Args:
            include_user: Whether to include user's custom models
            
        Returns:
            List of CuratedModel objects
        """
        models = self.curated_models.copy()
        
        if include_user:
            models.extend(self.user_models)
        
        return models
    
    def get_model_info(self, model_id: str) -> Optional[CuratedModel]:
        """
        Get detailed info for specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            CuratedModel or None if not found
        """
        # Check curated models
        for model in self.curated_models:
            if model.id == model_id:
                return model
        
        # Check user models
        for model in self.user_models:
            if model.id == model_id:
                return model
        
        return None
    
    def get_default_model(self) -> Optional[CuratedModel]:
        """
        Get the default recommended model.
        
        Returns:
            Default CuratedModel or None
        """
        for model in self.curated_models:
            if model.default:
                return model
        
        # Fallback to first model
        return self.curated_models[0] if self.curated_models else None
    
    def search_models(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        max_params: Optional[float] = None,
        tested_only: bool = False,
        include_user: bool = False
    ) -> List[CuratedModel]:
        """
        Search curated models by criteria.
        
        Args:
            query: Search query (matches name, description, tags)
            category: Filter by category (balanced, creative, technical, advanced)
            max_params: Maximum model size in billions
            tested_only: Only include tested models
            include_user: Include user's custom models
            
        Returns:
            List of matching CuratedModel objects
        """
        models = self.get_curated_models(include_user=include_user)
        
        # Filter by tested
        if tested_only:
            models = [m for m in models if m.tested]
        
        # Filter by category
        if category:
            models = [m for m in models if m.category == category]
        
        # Filter by max params
        if max_params:
            models = [m for m in models if m.parameters <= max_params]
        
        # Search by query
        if query:
            query_lower = query.lower()
            models = [
                m for m in models
                if query_lower in m.name.lower()
                or query_lower in m.description.lower()
                or any(query_lower in tag.lower() for tag in m.tags)
            ]
        
        return models
    
    def filter_by_vram(
        self,
        available_vram_mb: int,
        models: Optional[List[CuratedModel]] = None
    ) -> List[CuratedModel]:
        """
        Filter models that will fit in available VRAM.
        
        Args:
            available_vram_mb: Available VRAM in MB
            models: Models to filter (default: all curated)
            
        Returns:
            List of models that will fit
        """
        if models is None:
            models = self.curated_models
        
        fitting_models = []
        
        for model in models:
            # Check if any quantization will fit
            for quant in model.quantizations:
                if quant.min_vram_mb <= available_vram_mb:
                    fitting_models.append(model)
                    break
        
        return fitting_models
    
    def get_recommended_quantization(
        self,
        model_id: str,
        available_vram_mb: int
    ) -> Optional[QuantizationInfo]:
        """
        Get recommended quantization for a model given available VRAM.
        
        Args:
            model_id: Model identifier
            available_vram_mb: Available VRAM in MB
            
        Returns:
            Recommended QuantizationInfo or None
        """
        model = self.get_model_info(model_id)
        if not model:
            return None
        
        # Find best quantization that fits
        # Prefer higher quality if it fits
        best_quant = None
        
        for quant in sorted(model.quantizations, key=lambda q: q.min_vram_mb, reverse=True):
            if quant.min_vram_mb <= available_vram_mb:
                best_quant = quant
                break
        
        # If none fit, return lowest (for error message)
        if not best_quant and model.quantizations:
            best_quant = min(model.quantizations, key=lambda q: q.min_vram_mb)
        
        return best_quant
    
    def get_categories(self) -> Dict[str, Dict[str, str]]:
        """
        Get all model categories with metadata.
        
        Returns:
            Dict of category definitions
        """
        return self.CATEGORIES.copy()
    
    def get_models_by_category(self, category: str) -> List[CuratedModel]:
        """
        Get all models in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of models in that category
        """
        return [m for m in self.curated_models if m.category == category]
    
    def reload(self) -> None:
        """Reload models from disk (useful after updates)."""
        self.curated_models = []
        self.user_models = []
        self._load_curated_models()
        self._load_user_models()
        logger.info("ModelLibrary reloaded")

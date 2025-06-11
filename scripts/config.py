"""
config.py
"""
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class ModelType(str, Enum):
    RIDGE = "Ridge"
    RANDOM_FOREST = "Random Forest"


class VisualType(str, Enum):
    ICON = "Icon Image"
    PHOTO = "Photo Image"


CATEGORICAL_FEATURE_LIST = [
    "Species",
    "Gender",
    "Personality",
    "Hobby",
    "Style List",
    "Color List",
    "Favorite Song",
    "Default Umbrella",
    "Wallpaper",
    "Flooring",
    "Version Added",
    "Pocket Camp Theme",
    "Meta Tags"
]

# TODO: I will add Furniture List later


@dataclass
class VisualFeatureConfig:
    name: VisualType
    pca: Optional[int] = None


@dataclass
class ModelSettings:
    model: ModelType
    alpha: Optional[float] = None
    n_estimators: Optional[int] = None
    max_depth: Optional[int] = None
    random_seed: Optional[int] = None


@dataclass
class ModelConfig:
    categorical_settings: List[str]
    visual_settings: List[VisualFeatureConfig]
    model_settings: ModelSettings

    def validate(self):
        if self.model_settings.model == ModelType.RIDGE and self.model_settings.alpha is None:
            raise ValueError("Ridge model requires an alpha value.")
        if self.model_settings.model == ModelType.RANDOM_FOREST:
            if self.model_settings.n_estimators is None or self.model_settings.max_depth is None:
                raise ValueError("Random Forest requires n_estimators and max_depth.")
        if len(self.categorical_settings) == 0 and len(self.visual_settings) == 0:
            raise ValueError("No features selected.")
# All of the variables which are written in config.yaml and specifying their types

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_augmentation: bool
    params_image_size: list
    params_batch_size: int
    params_include_top: bool
    params_epochs: int
    params_classes: int
    params_weights: str
    params_learning_rate: float
    params_activation: str
    params_freeze_all: bool
    params_freeze_till: int


@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_augmentation: bool
    params_image_size: list
    params_batch_size: int
    params_epochs: int
    params_learning_rate: float
    params_activation: str
    
    
@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
    params_learning_rate: float
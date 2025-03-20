from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.utils.common import read_yaml, create_directories
from src.logger import CustomLogger
from src.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig, EvaluationConfig
from pathlib import Path
import os

class ConfigurationManager:
    def __init__(
        self,
        logger: CustomLogger,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):
        self.logger = logger
        self.config = read_yaml(config_filepath, logger=self.logger)
        self.params = read_yaml(params_filepath, logger=self.logger)

        create_directories([self.config.data_artifacts_root], logger=self.logger)
        create_directories([self.config.model_artifacts_root], logger=self.logger)




    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir], logger=self.logger)

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config




    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir], logger=self.logger)

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_augmentation=self.params.AUGMENTATION,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_include_top=self.params.INCLUDE_TOP,
            params_epochs=self.params.EPOCHS,
            params_classes=self.params.CLASSES,
            params_weights=self.params.WEIGHTS,
            params_learning_rate=self.params.LEARNING_RATE,
            params_activation=self.params.ACTIVATION,
            params_freeze_all=self.params.FREEZE_ALL,
            params_freeze_till=self.params.FREEZE_TILL,
        )
        return prepare_base_model_config
    
    
    
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "kidney-ct-scan-image")
        
        create_directories([training.root_dir], logger=self.logger)
        
        prepare_training_model_config = TrainingConfig(
            root_dir = Path(training.root_dir),
            trained_model_path = Path(training.trained_model_path),
            updated_base_model_path = Path(prepare_base_model.updated_base_model_path),
            training_data = Path(training_data),
            
            params_augmentation = self.params.AUGMENTATION,
            params_image_size = self.params.IMAGE_SIZE,
            params_batch_size = self.params.BATCH_SIZE,
            params_epochs = self.params.EPOCHS,
            params_learning_rate = self.params.LEARNING_RATE,
            params_activation=self.params.ACTIVATION
        )
        return prepare_training_model_config



    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model=self.config.training.best_trained_model_path,
            training_data=self.config.training.data_path,
            mlflow_uri=self.config.evaluation.mlflow_uri,
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_learning_rate=self.params.LEARNING_RATE
                
        )
        
        return eval_config
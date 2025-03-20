import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from src.logger import CustomLogger
from src.entity.config_entity import EvaluationConfig
from src.utils.common import save_json
from dotenv import load_dotenv

load_dotenv()

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
        
    def test_generator(self, test_generator_logger: CustomLogger):
    
        try:
            
            
            ########## kwargs start ########## 
            
            datagenerator_kwargs = dict(
                rescale=1./255,
                validation_split=0.30
            )
            
            dataflow_kwargs = dict(
                target_size = self.config.params_image_size[:-1],
                batch_size = self.config.params_batch_size,
                interpolation = "bilinear"
            )
            
            test_specific_kwargs = dict(
                directory = self.config.training_data,
                subset="validation",
                shuffle = False,
                class_mode = "categorical"
            )
            
            
            ########## kwargs end ########## 
            

                    
            ########## test datagenerator start ########## 
            test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
            
            self.test_generator = test_datagenerator.flow_from_directory(
                **test_specific_kwargs,
                **dataflow_kwargs
            )
            
            test_generator_logger.save_logs(msg=f"Generated Validation generator successfully with these \ndatagenerator_kwargs: {datagenerator_kwargs},\ndataflow_kwargs: {test_specific_kwargs, dataflow_kwargs}", log_level='info')
            
            ########## test datagenerator end ##########
            
        except Exception as e:
            test_generator_logger.save_logs(msg=f"Error in creating train and validation Generators. Error: {e}",log_level="error")
            raise e
        
        
    @staticmethod
    def load_model(path: Path, logger: CustomLogger) -> tf.keras.Model:
        try:
            model = tf.keras.models.load_model(path)
            logger.save_logs(msg=f"Model loaded from {path} successfully.", log_level="info")
        except Exception as e:
            logger.save_logs(msg=f"Failed to load model from {path} with Error: {e}", log_level="error")
            raise e
        
        return model
    
    def evaluate(self, evaluation_logger:CustomLogger):
        self.model = self.load_model(self.config.path_of_model, evaluation_logger)
            
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
            )
        
        self.test_generator(test_generator_logger=evaluation_logger)
        self.score = self.model.evaluate(self.test_generator)
        self.save_score(evaluation_logger)
        
    def save_score(self, save_score_logger: CustomLogger):
        scores = {"loss": self.score[0], "accuracy":self.score[1]}
        save_json(path=Path("scores.json"), data=scores, logger=save_score_logger)
        
    def log_into_mlflow(self, mlflow_logger: CustomLogger):
        try:
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            with mlflow.start_run():
                mlflow.log_params(self.config.all_params)
                mlflow.log_metrics(
                    {"loss": self.score[0], "accuracy":self.score[1]}
                )
                # Model Registry does not work with file store
                if tracking_url_type_store != "file":
                    
                    # Register the mode
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # Please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    
                    mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
                else:
                    mlflow.keras.log_model(self.model, "model")
        
            mlflow_logger.save_logs(msg="MLFlow Logging Successful", log_level="info")
        
        except Exception as e:
            mlflow_logger.save_logs(msg=f"MLFlow Logger failed to execute. Error: {e}", log_level="error")
                    
    
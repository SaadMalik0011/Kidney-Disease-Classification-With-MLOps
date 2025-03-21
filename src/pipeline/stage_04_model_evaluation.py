from src.config.configuration import ConfigurationManager
from src.models.model_evaluation import Evaluation
from src.logger import create_log_path, CustomLogger
import logging
from tensorflow.keras import callbacks
import tensorflow as tf
import os
from pathlib import Path
from src.all_logs import (pipeline_logger, configuration_logger,
                          evaluation_logger, mlflow_logger) 



STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(
        self,
        configuration_logger: CustomLogger,
        evaluation_logger: CustomLogger,
        mlflow_logger: CustomLogger
    ):
        
        self.configuration_logger = configuration_logger
        self.evaluation_logger = evaluation_logger 
        self.mlflow_logger = mlflow_logger
        
    
    def main(self):
        config = ConfigurationManager(self.configuration_logger)
        eval_config = config.get_evaluation_config()
        
        evaluation = Evaluation(eval_config)
        evaluation.evaluate(self.evaluation_logger)
        # evaluation.log_into_mlflow(self.mlflow_logger) # only uncomment it when doing experimentation.
        
        
if __name__ == "__main__":
    
    try:
        pipeline_logger().save_logs(msg=f"\n\n>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<",log_level="info")
        obj = ModelEvaluationPipeline(
            configuration_logger=configuration_logger(),
            evaluation_logger=evaluation_logger(),
            mlflow_logger=mlflow_logger()
        )
        obj.main()
        pipeline_logger().save_logs(msg=f">>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<\n\nx============x", log_level="info")
    
    except Exception as e:
        pipeline_logger().save_logs(msg=f"Error in {STAGE_NAME}. Error: {e}", log_level="error")
        raise e
    
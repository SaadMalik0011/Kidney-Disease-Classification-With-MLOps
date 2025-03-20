import logging
from src.logger import CustomLogger, create_log_path
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.pipeline.stage_03_model_training import ModelTrainingPipeline
from src.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline
from tensorflow.keras import callbacks
import tensorflow as tf
import os
from pathlib import Path
from src.callbacks import callbacks_list
from src.all_logs import (pipeline_logger, configuration_logger,
                          download_data_logger, extract_data_logger, data_logger, 
                          download_model_logger, prepare_full_model_logger, update_base_model_logger, 
                          get_base_model_logger, train_valid_generator_logger, train_logger, 
                          evaluation_logger, mlflow_logger) 

 




STAGE_NAME = "Data Ingestion stage"

try:
	pipeline_logger().save_logs(
		msg=f">>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<",
		log_level="info",
	)
	obj = DataIngestionTrainingPipeline(
		download_data_logger=download_data_logger(),
		extract_data_logger=extract_data_logger(),
		data_logger=data_logger(),
		configuration_logger=configuration_logger(),
	)
	obj.main()
	pipeline_logger().save_logs(msg=f">>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<\n\nx============x",log_level="info")
except Exception as e:
	pipeline_logger().save_logs(msg=f"Error in {STAGE_NAME}. Error: {e}", log_level="error")
	raise e



STAGE_NAME = "Prepare Base Model stage"
  
try:
	pipeline_logger().save_logs(msg=f">>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<",log_level="info")
	obj = PrepareBaseModelTrainingPipeline(
		download_model_logger=download_model_logger(),
		prepare_full_model_logger=prepare_full_model_logger(),
		update_base_model_logger=update_base_model_logger(),
		configuration_logger=configuration_logger(),
	)
	obj.main()
	pipeline_logger().save_logs(msg=f">>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<\n\nx============x", log_level="info")
except Exception as e:
	pipeline_logger().save_logs(msg=f"Error in {STAGE_NAME}. Error: {e}", log_level="error")
	raise e


STAGE_NAME = "Model Training Stage"

try:
	pipeline_logger().save_logs(msg=f">>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<",log_level="info")
	obj = ModelTrainingPipeline(
		configuration_logger=configuration_logger(),
		get_base_model_logger=get_base_model_logger(),
		train_valid_generator_logger=train_valid_generator_logger(),
		train_logger=train_logger(),
		callbacks_list=callbacks_list()
	)
	obj.main()
	pipeline_logger().save_logs(msg=f">>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<\n\nx============x", log_level="info")

except Exception as e:
	pipeline_logger().save_logs(msg=f"Error in {STAGE_NAME}. Error: {e}", log_level="error")
	raise e



STAGE_NAME = "Model Evaluation Stage"

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

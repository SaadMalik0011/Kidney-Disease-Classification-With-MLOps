import logging
from src.logger import CustomLogger, create_log_path
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from src.pipeline.stage_03_model_training import ModelTrainingPipeline
from tensorflow.keras import callbacks
import tensorflow as tf
import os
from pathlib import Path
  
pipeline_log_file_path = create_log_path("pipeline")
pipeline_logger = CustomLogger(logger_name="pipeline", log_filename=pipeline_log_file_path)
pipeline_logger.set_log_level(level=logging.INFO)
 

configuration_log_file_path = create_log_path("configuration")
configuration_logger = CustomLogger(logger_name="configuration", log_filename=configuration_log_file_path)
configuration_logger.set_log_level(level=logging.INFO)

  
############################################################################
  
# path to save the log files
download_log_file_path = create_log_path("data/download_dataset")
download_data_logger = CustomLogger(logger_name="download_dataset", log_filename=download_log_file_path)
download_data_logger.set_log_level(level=logging.INFO)
 

extract_log_file_path = create_log_path("data/extract_dataset")
extract_data_logger = CustomLogger(logger_name="extract_dataset", log_filename=extract_log_file_path)
extract_data_logger.set_log_level(level=logging.INFO)
 

data_log_file_path = create_log_path("data/data_ingestion")
data_logger = CustomLogger(logger_name="data_ingestion", log_filename=data_log_file_path)
data_logger.set_log_level(level=logging.INFO)
 

############################################################################

download_log_file_path = create_log_path("model/download_model")
download_model_logger = CustomLogger(logger_name="download_model", log_filename=download_log_file_path)
download_model_logger.set_log_level(level=logging.INFO)
 

prepare_model_log_file_path = create_log_path("model/prepare_model")
prepare_full_model_logger = CustomLogger(logger_name="prepare_model", log_filename=prepare_model_log_file_path)
prepare_full_model_logger.set_log_level(level=logging.INFO)


update_model_log_file_path = create_log_path("model/update_model")
update_base_model_logger = CustomLogger(logger_name="update_model", log_filename=update_model_log_file_path)
update_base_model_logger.set_log_level(level=logging.INFO)
 

############################################################################  

get_base_model_log_file_path = create_log_path("training/get_base_model")
get_base_model_logger = CustomLogger(logger_name="get_base_model", log_filename=get_base_model_log_file_path)
get_base_model_logger.set_log_level(level=logging.INFO)


train_valid_generator_log_file_path = create_log_path("training/train_valid_generator")
train_valid_generator_logger = CustomLogger(logger_name="train_valid_generator", log_filename=train_valid_generator_log_file_path)
train_valid_generator_logger.set_log_level(level=logging.INFO)


train_log_file_path = create_log_path("training/train")
train_logger = CustomLogger(logger_name="train", log_filename=train_log_file_path)
train_logger.set_log_level(level=logging.INFO)


		############### Callbacks Start ###############
  
# Callbacks
# Early Stopping
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Learning Rate Scheduler
def scheduler(epoch, lr):
	if epoch < 10:
		return lr
	else:
		return lr * tf.math.exp(-0.1).numpy()

lr_scheduler = callbacks.LearningRateScheduler(scheduler)

# Model Checkpoint
model_checkpoint = callbacks.ModelCheckpoint(
	Path("models/training/best_model.h5"), monitor='val_loss', save_best_only=True, verbose=1
)

# ReduceLROnPlateau
reduce_lr = callbacks.ReduceLROnPlateau(
	monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

# TensorBoard
tensorboard = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# TerminateOnNaN
terminate_on_nan = callbacks.TerminateOnNaN()

# ProgbarLogger
progbar_logger = callbacks.ProgbarLogger()

# CSV Logger
csv_logger = callbacks.CSVLogger('models/training/training_log.csv', append=True)



callbacks_list = [
	early_stopping,
	lr_scheduler,
	model_checkpoint,
	reduce_lr,
	tensorboard,
	terminate_on_nan,
	#progbar_logger,
	csv_logger
 ]

  
  
		############### Callbacks End ###############


############################################################################  


STAGE_NAME = "Data Ingestion stage"

try:
	pipeline_logger.save_logs(
		msg=f">>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<",
		log_level="info",
	)
	obj = DataIngestionTrainingPipeline(
		download_data_logger=download_data_logger,
		extract_data_logger=extract_data_logger,
		data_logger=data_logger,
		configuration_logger=configuration_logger,
	)
	obj.main()
	pipeline_logger.save_logs(msg=f">>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<\n\nx============x\n\n",log_level="info")
except Exception as e:
	pipeline_logger.save_logs(msg=f"Error in {STAGE_NAME}. Error: {e}", log_level="error")
	raise e



STAGE_NAME = "Prepare Base Model stage"
  
try:
	pipeline_logger.save_logs(msg=f">>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<",log_level="info")
	obj = PrepareBaseModelTrainingPipeline(
		download_model_logger=download_model_logger,
		prepare_full_model_logger=prepare_full_model_logger,
		update_base_model_logger=update_base_model_logger,
		configuration_logger=configuration_logger,
	)
	obj.main()
	pipeline_logger.save_logs(msg=f">>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<\n\nx============x\n\n", log_level="info")
except Exception as e:
	pipeline_logger.save_logs(msg=f"Error in {STAGE_NAME}. Error: {e}", log_level="error")
	raise e


STAGE_NAME = "Model Training Stage"

try:
	pipeline_logger.save_logs(msg=f">>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<",log_level="info")
	obj = ModelTrainingPipeline(
		configuration_logger=configuration_logger,
		get_base_model_logger=get_base_model_logger,
		train_valid_generator_logger=train_valid_generator_logger,
		train_logger=train_logger,
		callbacks_list=callbacks_list
	)
	obj.main()
	pipeline_logger.save_logs(msg=f">>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<\n\nx============x\n\n", log_level="info")

except Exception as e:
	pipeline_logger.save_logs(msg=f"Error in {STAGE_NAME}. Error: {e}", log_level="error")
	raise e

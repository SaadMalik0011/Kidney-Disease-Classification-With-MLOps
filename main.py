import logging
from src.logger import CustomLogger, create_log_path
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline


pipeline_log_file_path = create_log_path("pipeline")
pipeline_logger = CustomLogger(
    logger_name="pipeline", log_filename=pipeline_log_file_path
)
pipeline_logger.set_log_level(level=logging.INFO)


configuration_log_file_path = create_log_path("configuration")
configuration_logger = CustomLogger(
    logger_name="configuration", log_filename=configuration_log_file_path
)
configuration_logger.set_log_level(level=logging.INFO)

############################################################################

# path to save the log files
download_log_file_path = create_log_path("data/download_dataset")
download_data_logger = CustomLogger(
    logger_name="download_dataset", log_filename=download_log_file_path
)
download_data_logger.set_log_level(level=logging.INFO)

extract_log_file_path = create_log_path("data/extract_dataset")
extract_data_logger = CustomLogger(
    logger_name="extract_dataset", log_filename=extract_log_file_path
)
extract_data_logger.set_log_level(level=logging.INFO)

data_log_file_path = create_log_path("data/data_ingestion")
data_logger = CustomLogger(
    logger_name="data_ingestion", log_filename=data_log_file_path
)
data_logger.set_log_level(level=logging.INFO)

############################################################################

download_log_file_path = create_log_path("model/download_model")
download_model_logger = CustomLogger(
    logger_name="download_model", log_filename=download_log_file_path
)
download_model_logger.set_log_level(level=logging.INFO)

prepare_model_log_file_path = create_log_path("model/prepare_model")
prepare_full_model_logger = CustomLogger(
    logger_name="prepare_model", log_filename=prepare_model_log_file_path
)
prepare_full_model_logger.set_log_level(level=logging.INFO)

update_model_log_file_path = create_log_path("model/update_model")
update_base_model_logger = CustomLogger(
    logger_name="update_model", log_filename=update_model_log_file_path
)
update_base_model_logger.set_log_level(level=logging.INFO)


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
    pipeline_logger.save_logs(
        msg=f">>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<\n\nx============x",
        log_level="info",
    )
except Exception as e:
    pipeline_logger.save_logs(
        msg=f"Error in {STAGE_NAME}. Error: {e}", log_level="error"
    )
    raise e


STAGE_NAME = "Prepare Base Model stage"

try:
    pipeline_logger.save_logs(
        msg=f">>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<",
        log_level="info",
    )
    obj = PrepareBaseModelTrainingPipeline(
        download_model_logger=download_model_logger,
        prepare_full_model_logger=prepare_full_model_logger,
        update_base_model_logger=update_base_model_logger,
        configuration_logger=configuration_logger,
    )
    obj.main()
    pipeline_logger.save_logs(
        msg=f">>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<\n\nx============x",
        log_level="info",
    )
except Exception as e:
    pipeline_logger.save_logs(
        msg=f"Error in {STAGE_NAME}. Error: {e}", log_level="error"
    )
    raise e

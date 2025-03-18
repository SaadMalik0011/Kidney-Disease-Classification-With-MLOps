import logging
from src.logger import CustomLogger, create_log_path
from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion stage"

# path to save the log files
download_log_file_path = create_log_path("download_dataset")
download_data_logger = CustomLogger(
    logger_name="download_dataset", log_filename=download_log_file_path
)
download_data_logger.set_log_level(level=logging.INFO)

extract_log_file_path = create_log_path("extract_dataset")
extract_data_logger = CustomLogger(
    logger_name="extract_dataset", log_filename=extract_log_file_path
)
extract_data_logger.set_log_level(level=logging.INFO)

data_log_file_path = create_log_path("data_ingestion")
data_logger = CustomLogger(
    logger_name="data_ingestion", log_filename=data_log_file_path
)
data_logger.set_log_level(level=logging.INFO)

pipeline_log_file_path = create_log_path("pipeline")
pipeline_logger = CustomLogger(
    logger_name="pipeline", log_filename=pipeline_log_file_path
)
pipeline_logger.set_log_level(level=logging.INFO)


try:
    pipeline_logger.save_logs(
        msg=f">>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<",
        log_level="info",
    )
    obj = DataIngestionTrainingPipeline(
        download_data_logger=download_data_logger,
        extract_data_logger=extract_data_logger,
        data_logger=data_logger,
    )
    obj.main()
    pipeline_logger.save_logs(
        msg=f">>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<\n\nx============x",
        log_level="info",
    )
except Exception as e:
    pipeline_logger.save_logs(msg=f"Error in {STAGE_NAME} {e}", log_level="error")
    raise e

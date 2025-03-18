from src.config.configuration import ConfigurationManager
from src.data.data_ingestion import DataIngestion
from src.logger import create_log_path, CustomLogger
import logging

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def __init__(
        self,
        download_data_logger: CustomLogger,
        extract_data_logger: CustomLogger,
        data_logger: CustomLogger,
    ):
        self.download_data_logger = download_data_logger
        self.extract_data_logger = extract_data_logger
        self.data_logger = data_logger

    def main(self):
        config = ConfigurationManager(self.data_logger)
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file(download_logger=self.download_data_logger)
        data_ingestion.extract_zip_file(extract_logger=self.extract_data_logger)


if __name__ == "__main__":
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

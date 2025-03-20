from src.config.configuration import ConfigurationManager
from src.data.data_ingestion import DataIngestion
from src.logger import create_log_path, CustomLogger
import logging
from src.all_logs import (pipeline_logger, configuration_logger,
                          download_data_logger, extract_data_logger, data_logger) 


STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def __init__(
        self,
        download_data_logger: CustomLogger,
        extract_data_logger: CustomLogger,
        data_logger: CustomLogger,
        configuration_logger: CustomLogger,
    ):
        self.download_data_logger = download_data_logger
        self.extract_data_logger = extract_data_logger
        self.data_logger = data_logger
        self.configuration_logger = configuration_logger

    def main(self):
        config = ConfigurationManager(self.configuration_logger)
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file(download_logger=self.download_data_logger)
        data_ingestion.extract_zip_file(extract_logger=self.extract_data_logger)


if __name__ == "__main__":

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
        pipeline_logger().save_logs(
            msg=f">>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<\n\nx============x",
            log_level="info",
        )
    except Exception as e:
        pipeline_logger().save_logs(
            msg=f"Error in {STAGE_NAME}. Error: {e}", log_level="error"
        )
        raise e

from src.config.configuration import ConfigurationManager
from src.models.prepare_base_model import PrepareBaseModel
from src.logger import create_log_path, CustomLogger
import logging

STAGE_NAME = "Prepare Base Model Stage"


class PrepareBaseModelTrainingPipeline:
    def __init__(
        self,
        download_model_logger: CustomLogger,
        prepare_full_model_logger: CustomLogger,
        update_base_model_logger: CustomLogger,
        configuration_logger: CustomLogger,
    ):
        self.download_model_logger = download_model_logger
        self.prepare_full_model_logger = prepare_full_model_logger
        self.update_base_model_logger = update_base_model_logger
        self.configuration_logger = configuration_logger

    def main(self):
        config = ConfigurationManager(self.configuration_logger)
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model(
            download_base_model_logger=self.download_model_logger
        )
        prepare_base_model.update_base_model(
            prepare_full_model_logger=self.prepare_full_model_logger,
            update_base_model_logger=self.update_base_model_logger,
        )


if __name__ == "__main__":
    # path to save the log files
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

    configuration_log_file_path = create_log_path("configuration")
    configuration_logger = CustomLogger(
        logger_name="configuration", log_filename=configuration_log_file_path
    )
    configuration_logger.set_log_level(level=logging.INFO)

    pipeline_log_file_path = create_log_path("pipeline")
    pipeline_logger = CustomLogger(
        logger_name="pipeline", log_filename=pipeline_log_file_path
    )
    pipeline_logger.set_log_level(level=logging.INFO)

    try:
        pipeline_logger.save_logs(
            msg=f"\n\n>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<",
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

from src.config.configuration import ConfigurationManager
from src.models.prepare_base_model import PrepareBaseModel
from src.logger import create_log_path, CustomLogger
import logging
from src.all_logs import (pipeline_logger, configuration_logger,
                          download_model_logger, prepare_full_model_logger, update_base_model_logger) 



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

    try:
        pipeline_logger().save_logs(
            msg=f"\n\n>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<",
            log_level="info",
        )
        obj = PrepareBaseModelTrainingPipeline(
            download_model_logger=download_model_logger(),
            prepare_full_model_logger=prepare_full_model_logger(),
            update_base_model_logger=update_base_model_logger(),
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

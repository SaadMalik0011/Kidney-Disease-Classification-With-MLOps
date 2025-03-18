from src.constants import *
from src.utils.common import read_yaml, create_directories
from src.logger import CustomLogger
from src.entity.config_entity import DataIngestionConfig


class ConfigurationManager:
    def __init__(
        self,
        logger: CustomLogger,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):
        self.logger = logger
        self.config = read_yaml(config_filepath, logger=self.logger)
        self.params = read_yaml(params_filepath, logger=self.logger)

        create_directories([self.config.data_artifacts_root], logger=self.logger)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir], logger=self.logger)

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config

import os
import zipfile
import gdown
from src.logger import CustomLogger
from src.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self, download_logger: CustomLogger) -> str:
        """
        Fetch data from the url
        """

        try:
            root_dir = self.config.root_dir
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs(root_dir, exist_ok=True)
            download_logger.save_logs(
                msg=f"Downloading data from {dataset_url} to {zip_download_dir} successfully",
                log_level="info",
            )

            file_id = dataset_url.split("/")[-2]
            prefix = prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix + file_id, zip_download_dir)

            download_logger.save_logs(
                msg=f"Data downloaded from {dataset_url} into file {zip_download_dir} successfully",
                log_level="info",
            )

        except Exception as e:
            download_logger.save_logs(
                msg=f"Error in downloading data from {dataset_url} into file {zip_download_dir}",
                log_level="error",
            )
            raise e

    def extract_zip_file(self, extract_logger: CustomLogger) -> str:
        """
        zip file path: str
        Extracts the zip file into the data directory
        Function returns None
        """

        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)

        extract_logger.save_logs(
            msg=f"Data extracted from {self.config.local_data_file} into {unzip_path} Successfully",
            log_level="info",
        )

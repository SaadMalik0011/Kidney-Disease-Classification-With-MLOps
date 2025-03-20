import logging
from src.logger import CustomLogger, create_log_path

GENERAL_LOGS_PATH = "0_general_logs/"
DATA_LOGS_PATH = "1_data/"
MODEL_LOGS_PATH = "2_model/"
TRAINING_LOGS_PATH = "3_training/"
EVALUATION_LOGS_PATH = "4_evaluation/"

def pipeline_logger():
    pipeline_log_file_path = create_log_path(GENERAL_LOGS_PATH+"pipeline")
    pipeline_logger = CustomLogger(logger_name="pipeline", log_filename=pipeline_log_file_path)
    pipeline_logger.set_log_level(level=logging.INFO)
    return pipeline_logger

def configuration_logger():
    configuration_log_file_path = create_log_path(GENERAL_LOGS_PATH+"configuration")
    configuration_logger = CustomLogger(logger_name="configuration", log_filename=configuration_log_file_path)
    configuration_logger.set_log_level(level=logging.INFO)
    return configuration_logger


def download_data_logger():
    download_log_file_path = create_log_path(DATA_LOGS_PATH+"download_dataset")
    download_data_logger = CustomLogger(logger_name="download_dataset", log_filename=download_log_file_path)
    download_data_logger.set_log_level(level=logging.INFO)
    return download_data_logger


def extract_data_logger():
    extract_log_file_path = create_log_path(DATA_LOGS_PATH+"extract_dataset")
    extract_data_logger = CustomLogger(logger_name="extract_dataset", log_filename=extract_log_file_path)
    extract_data_logger.set_log_level(level=logging.INFO)
    return extract_data_logger

def data_logger():
    data_log_file_path = create_log_path(DATA_LOGS_PATH+"data_ingestion")
    data_logger = CustomLogger(logger_name="data_ingestion", log_filename=data_log_file_path)
    data_logger.set_log_level(level=logging.INFO)
    return data_logger

def download_model_logger():
    download_log_file_path = create_log_path(MODEL_LOGS_PATH+"download_model")
    download_model_logger = CustomLogger(logger_name="download_model", log_filename=download_log_file_path)
    download_model_logger.set_log_level(level=logging.INFO)
    return download_model_logger

def prepare_full_model_logger():
    prepare_model_log_file_path = create_log_path(MODEL_LOGS_PATH+"prepare_model")
    prepare_full_model_logger = CustomLogger(logger_name="prepare_model", log_filename=prepare_model_log_file_path)
    prepare_full_model_logger.set_log_level(level=logging.INFO)
    return prepare_full_model_logger

def update_base_model_logger():
    update_model_log_file_path = create_log_path(MODEL_LOGS_PATH+"update_model")
    update_base_model_logger = CustomLogger(logger_name="update_model", log_filename=update_model_log_file_path)
    update_base_model_logger.set_log_level(level=logging.INFO)
    return update_base_model_logger




def get_base_model_logger():
    get_base_model_log_file_path = create_log_path(TRAINING_LOGS_PATH+"get_base_model")
    get_base_model_logger = CustomLogger(logger_name="get_base_model", log_filename=get_base_model_log_file_path)
    get_base_model_logger.set_log_level(level=logging.INFO)
    return get_base_model_logger

def train_valid_generator_logger():
    train_valid_generator_log_file_path = create_log_path(TRAINING_LOGS_PATH+"train_valid_generator")
    train_valid_generator_logger = CustomLogger(logger_name="train_valid_generator", log_filename=train_valid_generator_log_file_path)
    train_valid_generator_logger.set_log_level(level=logging.INFO)
    return train_valid_generator_logger

def train_logger():
    train_log_file_path = create_log_path(TRAINING_LOGS_PATH+"train")
    train_logger = CustomLogger(logger_name="train", log_filename=train_log_file_path)
    train_logger.set_log_level(level=logging.INFO)
    return train_logger


def evaluation_logger():
    evaluation_logger_log_file_path = create_log_path(EVALUATION_LOGS_PATH+"evaluation_logger")
    evaluation_logger = CustomLogger(logger_name="evaluation_logger", log_filename=evaluation_logger_log_file_path)
    evaluation_logger.set_log_level(level=logging.INFO)
    return evaluation_logger


def mlflow_logger():
    mlflow_logger_log_file_path = create_log_path(EVALUATION_LOGS_PATH+"mlflow_logger")
    mlflow_logger = CustomLogger(logger_name="mlflow_logger", log_filename=mlflow_logger_log_file_path)
    mlflow_logger.set_log_level(level=logging.INFO)
    return mlflow_logger
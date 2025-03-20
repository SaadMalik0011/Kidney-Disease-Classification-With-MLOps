from src.config.configuration import ConfigurationManager
from src.models.train_model import Training
import tensorflow as tf
from src.callbacks import callbacks_list

from src.all_logs import (pipeline_logger, configuration_logger, 
                          get_base_model_logger, train_valid_generator_logger, train_logger) 
from src.logger import CustomLogger


STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(
        self,
        configuration_logger: CustomLogger,
        get_base_model_logger: CustomLogger,
        train_valid_generator_logger: CustomLogger,
        train_logger: CustomLogger,
        callbacks_list: list
    ):
    
        self.configuration_logger = configuration_logger
        self.get_base_model_logger = get_base_model_logger
        self.train_valid_generator_logger = train_valid_generator_logger
        self.train_logger = train_logger
        self.callbacks_list = callbacks_list
    
    def main(self):
        config = ConfigurationManager(self.configuration_logger)
        training_config = config.get_training_config()
        
        training = Training(config=training_config)
        training.get_base_model(get_base_model_logger=self.get_base_model_logger)
        training.train_valid_generator(train_valid_generator_logger=self.train_valid_generator_logger)
        training.train(train_logger=self.train_logger, callback_list=self.callbacks_list)
        

if __name__ == "__main__":


    ############################## ERROR CORRECTION START ##########################

    tf.config.run_functions_eagerly(True)
    print("Eager Execution:", tf.executing_eagerly())  # Check if enabled

    ############################## ERROR CORRECTION END ##########################

    
    try:
        pipeline_logger().save_logs(msg=f"\n\n>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<",log_level="info")
        obj = ModelTrainingPipeline(
            configuration_logger=configuration_logger(),
            get_base_model_logger=get_base_model_logger(),
            train_valid_generator_logger=train_valid_generator_logger(),
            train_logger=train_logger(),
            callbacks_list=callbacks_list()
        )
        obj.main()
        pipeline_logger().save_logs(msg=f">>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<\n\nx============x", log_level="info")
    
    except Exception as e:
        pipeline_logger().save_logs(msg=f"Error in {STAGE_NAME}. Error: {e}", log_level="error")
        raise e

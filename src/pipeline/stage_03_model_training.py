from src.config.configuration import ConfigurationManager
from src.models.train_model import Training
from src.logger import create_log_path, CustomLogger
import logging
from tensorflow.keras import callbacks
import tensorflow as tf
import os
from pathlib import Path

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
    # path to save the log files
    get_base_model_log_file_path = create_log_path("training/get_base_model")
    get_base_model_logger = CustomLogger(logger_name="get_base_model", log_filename=get_base_model_log_file_path)
    get_base_model_logger.set_log_level(level=logging.INFO)


    train_valid_generator_log_file_path = create_log_path("training/train_valid_generator")
    train_valid_generator_logger = CustomLogger(logger_name="train_valid_generator", log_filename=train_valid_generator_log_file_path)
    train_valid_generator_logger.set_log_level(level=logging.INFO)
    

    train_log_file_path = create_log_path("training/train")
    train_logger = CustomLogger(logger_name="train", log_filename=train_log_file_path)
    train_logger.set_log_level(level=logging.INFO)


    configuration_log_file_path = create_log_path("configuration")
    configuration_logger = CustomLogger(logger_name="configuration", log_filename=configuration_log_file_path)
    configuration_logger.set_log_level(level=logging.INFO)

    pipeline_log_file_path = create_log_path("pipeline")
    pipeline_logger = CustomLogger(
        logger_name="pipeline", log_filename=pipeline_log_file_path
    )
    pipeline_logger.set_log_level(level=logging.INFO)

    ############################## ERROR CORRECTION START ##########################
    tf.config.run_functions_eagerly(True)
    print("Eager Execution:", tf.executing_eagerly())  # Check if enabled


    ############## callbacks start ################
    # Callbacks
    # Early Stopping
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Learning Rate Scheduler
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1).numpy()

    lr_scheduler = callbacks.LearningRateScheduler(scheduler)

    # Model Checkpoint
    model_checkpoint = callbacks.ModelCheckpoint(
        Path("models/training/best_model.h5"), monitor='val_loss', save_best_only=True, verbose=1
    )

    # ReduceLROnPlateau
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )

    # TensorBoard
    tensorboard = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # TerminateOnNaN
    terminate_on_nan = callbacks.TerminateOnNaN()

    # ProgbarLogger
    progbar_logger = callbacks.ProgbarLogger()

    # CSV Logger
    csv_logger = callbacks.CSVLogger(Path('models/training/training_log.csv'), append=True)



    callbacks_list = [
        early_stopping,
        lr_scheduler,
        model_checkpoint,
        reduce_lr,
        tensorboard,
        terminate_on_nan,
        #progbar_logger,
        csv_logger
    ]
    
    ############## callbacks end ################
    
    try:
        pipeline_logger.save_logs(msg=f"\n\n>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<",log_level="info")
        obj = ModelTrainingPipeline(
            configuration_logger=configuration_logger,
            get_base_model_logger=get_base_model_logger,
            train_valid_generator_logger=train_valid_generator_logger,
            train_logger=train_logger,
            callbacks_list=callbacks_list
        )
        obj.main()
        pipeline_logger.save_logs(msg=f">>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<\n\nx============x", log_level="info")
    
    except Exception as e:
        pipeline_logger.save_logs(msg=f"Error in {STAGE_NAME}. Error: {e}", log_level="error")
        raise e

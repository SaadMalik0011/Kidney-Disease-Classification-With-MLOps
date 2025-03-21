import tensorflow as tf
from src.logger import CustomLogger
import math
from pathlib import Path
from src.entity.config_entity import TrainingConfig
import os

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def get_base_model(self, get_base_model_logger: CustomLogger):
        try:
            self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
            self.model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
                )

            
            get_base_model_logger.save_logs(msg=f"Model loaded from {self.config.updated_base_model_path} successfully.", log_level="info")
        
        except Exception as e:
            get_base_model_logger.save_logs(msg=f"Error loading model from {self.config.updated_base_model_path}. Error: {str(e)}", log_level="error")
            raise e
        
    def train_valid_generator(self, train_valid_generator_logger: CustomLogger):
    
        try:
            
            
            ########## kwargs start ########## 
            
            datagenerator_kwargs = dict(
                rescale=1./255,
                validation_split=0.2
            )
            
            dataflow_kwargs = dict(
                target_size = self.config.params_image_size[:-1],
                batch_size = self.config.params_batch_size,
                interpolation = "bilinear"
            )
            
            val_specific_kwargs = dict(
                directory = self.config.training_data,
                subset="validation",
                shuffle = False,
                class_mode = "categorical"
            )
            
            train_specific_kwargs = dict(
                directory = self.config.training_data,
                subset = "training",
                shuffle = True,
                class_mode = "categorical"
            )
            
            augmentation_kwargs = dict(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)
            
            ########## kwargs end ########## 
            

                    
            ########## validation datagenerator start ########## 
            valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
            
            self.valid_generator = valid_datagenerator.flow_from_directory(
                **val_specific_kwargs,
                **dataflow_kwargs
            )
            
            train_valid_generator_logger.save_logs(msg=f"Generated Validation generator successfully with these \ndatagenerator_kwargs: {datagenerator_kwargs},\ndataflow_kwargs: {val_specific_kwargs, dataflow_kwargs}", log_level='info')
            
            ########## validation datagenerator end ##########
            
            ########## training datagenerator start ##########
            if self.config.params_augmentation:
                train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**augmentation_kwargs, **datagenerator_kwargs)
                

            else:
                train_datagenerator = valid_datagenerator
                
            self.train_generator = train_datagenerator.flow_from_directory(
                **train_specific_kwargs,
                **dataflow_kwargs
                )
            
            if self.config.params_augmentation:
                train_valid_generator_logger.save_logs(msg=f"Generated Train generator with AUGMENTATION: True successfully with these \naugmentation_kwargs: {augmentation_kwargs},\ndatagenerator_kwargs: {datagenerator_kwargs}, \ndataflow_kwargs: {train_specific_kwargs, dataflow_kwargs}", log_level='info')
            else:
                train_valid_generator_logger.save_logs(msg=f"Generated Train generator with AUGMENTATION: False successfully with these \ndatagenerator_kwargs: {datagenerator_kwargs}, \ndataflow_kwargs: {train_specific_kwargs, dataflow_kwargs}", log_level='info')
            
            ########## training datagenerator end ##########
        except Exception as e:
            train_valid_generator_logger.save_logs(msg=f"Error in creating train and validation Generators. Error: {e}",log_level="error")
            raise e

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model, logger: CustomLogger):
        try:
            model.save(path)
            logger.save_logs(msg=f"Model saved at {path} successfully", log_level="info")
        except Exception as e:
            logger.save_logs(msg=f"Error in saving model at {path}. Error: {e}", log_level="error")
            raise e
        
        
    def train(self, train_logger: CustomLogger, callback_list: list = None):
        self.steps_per_epoch = math.ceil(self.train_generator.samples // self.train_generator.batch_size)
        self.validation_steps = math.ceil(self.valid_generator.samples // self.valid_generator.batch_size)
        self.model.summary(show_trainable=True, expand_nested=True)
        try:
            self.model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_steps=self.validation_steps,
                validation_data=self.valid_generator,
                callbacks=callback_list
            )
            train_logger.save_logs(msg="model.fit successfully",log_level="info")
            
        except Exception as e:
            train_logger.save_logs(msg=f"Error in model.fit. Error: {e}",log_level="error")     
            raise e       
        
        self.save_model(path=self.config.trained_model_path, model=self.model, logger=train_logger)
        
        try:
            # Define source and destination paths
            source_file = "models/training/best_model.h5"
            destination_file = "models/deployed_model/best_model.h5"

            # Open source file in read mode and destination file in write mode
            with open(source_file, 'rb') as src, open(destination_file, 'wb') as dst:
                dst.write(src.read())

            train_logger.save_logs(msg=f"Model copied to {destination_file} successfully", log_level="info")
        except Exception as e:
            train_logger.save_logs(msg=f"Model failed to copy to {destination_file}. Error: {e}", log_level="error")
            raise e
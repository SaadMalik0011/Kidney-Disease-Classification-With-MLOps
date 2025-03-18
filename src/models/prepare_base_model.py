import tensorflow as tf
from src.logger import CustomLogger
from pathlib import Path
from src.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self, download_base_model_logger: CustomLogger):
        try:
            self.model = tf.keras.applications.vgg16.VGG16(
                include_top=self.config.params_include_top,
                weights=self.config.params_weights,
                input_tensor=None,
                input_shape=self.config.params_image_size,
                pooling=None,
                classes=self.config.params_classes,
                classifier_activation=self.config.params_activation,
            )
            self.save_model(
                path=self.config.base_model_path,
                model=self.model,
                logger=download_base_model_logger,
            )

            download_base_model_logger.save_logs(
                msg="Base model downloaded successfully.", log_level="info"
            )

        except Exception as e:
            download_base_model_logger.save_logs(
                msg=f"Error in downloading base model. Error: {e}", log_level="error"
            )
            raise e

    @staticmethod
    def _prepare_full_model(
        model,
        classes,
        freeze_all,
        freeze_till,
        learning_rate,
        activation,
        logger: CustomLogger,
    ):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        if activation == "softmax":
            prediction = tf.keras.layers.Dense(units=64, activation="relu")(flatten_in)
            prediction = tf.keras.layers.Dense(units=classes, activation=activation)(
                prediction
            )
            logger.save_logs(
                msg=f"Model Architecture constructed successfully with {activation} activation function and {classes} classes.",
                log_level="info",
            )

        elif activation == "sigmoid":
            prediction = tf.keras.layers.Dense(units=64, activation="relu")(flatten_in)
            prediction = tf.keras.layers.Dense(units=1, activation=activation)(
                flatten_in
            )
            logger.save_logs(
                msg=f"Model Architecture constructed successfully with {activation} activation function and {classes} classes.",
                log_level="info",
            )
        else:
            logger.save_logs(
                msg=f"Activation function {activation} not supported. Error: Correct the last layer activation function OR last layer neurons.",
                log_level="error",
            )
            raise ValueError(
                f"Activation function {activation} not supported. Error: Correct the last layer activation function OR last layer neurons."
            )

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

        if activation == "softmax":
            full_model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            )
            logger.save_logs(
                msg="Model compiled successfully with optimizer: SGD, loss: SparseCategoricalCrossentropy and metrics: SparseCategoricalAccuracy.",
                log_level="info",
            )
        elif activation == "sigmoid":
            full_model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()],
            )
            logger.save_logs(
                msg="Model compiled successfully with optimizer: SGD, loss: BinaryCrossentropy and metrics: BinaryAccuracy.",
                log_level="info",
            )
        else:
            logger.save_logs(
                msg=f"Activation function {activation} not supported. Error: Only choose activation function from 'softmax' or 'sigmoid'.",
                log_level="error",
            )
            raise ValueError(
                f"Activation function {activation} not supported. Error: Only choose activation function from 'softmax' or 'sigmoid'."
            )

        full_model.summary(show_trainable=True, expand_nested=True)
        return full_model

    def update_base_model(
        self,
        prepare_full_model_logger: CustomLogger,
        update_base_model_logger: CustomLogger,
    ):
        try:
            self.full_model = self._prepare_full_model(
                model=self.model,
                classes=self.config.params_classes,
                freeze_all=self.config.params_freeze_all,
                freeze_till=self.config.params_freeze_till,
                learning_rate=self.config.params_learning_rate,
                activation=self.config.params_activation,
                logger=prepare_full_model_logger,
            )

            self.save_model(
                path=self.config.updated_base_model_path,
                model=self.full_model,
                logger=update_base_model_logger,
            )

            update_base_model_logger.save_logs(
                msg="Base model updated successfully.", log_level="info"
            )
        except Exception as e:
            update_base_model_logger.save_logs(
                msg=f"Error in updating base model. Error: {e}", log_level="error"
            )
            raise e

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model, logger: CustomLogger):
        try:
            model.save(path)
            logger.save_logs(
                msg=f"Model saved at {path} successfully.", log_level="info"
            )
        except Exception as e:
            logger.save_logs(
                msg=f"Error in saving model at {path}. Error: {e}", log_level="error"
            )
            raise e

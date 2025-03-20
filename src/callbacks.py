import os
import tensorflow as tf
from tensorflow.keras import callbacks
from pathlib import Path


def callbacks_list():
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
    return callbacks_list
"""
1D ResNet Model for ECG Classification

This module contains a ResNet (Residual Network) architecture adapted for 1D time series data.
ResNet uses "shortcut connections" to allow gradients to flow more easily through the network,
making it possible to train much deeper models effectively.
"""

import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dense, Dropout, 
                                     BatchNormalization, Add, Activation, GlobalAveragePooling1D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras


# Custom callback to track learning rate during training
class LearningRateTracker(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.learning_rates = []
    
    def on_epoch_end(self, epoch, logs=None):
        lr = float(self.model.optimizer.learning_rate)
        self.learning_rates.append(lr)
        if epoch % 5 == 0:  # Print every 5 epochs to avoid clutter
            print(f"Epoch {epoch + 1}: Learning rate = {lr:.2e}")


def resnet_block(x, filters, kernel_size, strides=1):
    """
    Simplified ResNet block with fewer parameters for ECG time series.
    
    Args:
        x: Input tensor
        filters: Number of filters in the convolutional layers
        kernel_size: Size of the convolutional kernel
        strides: Stride for the first convolutional layer
    
    Returns:
        Output tensor after applying the ResNet block
    """
    shortcut = x

    # First component
    x = Conv1D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second component
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # Adjust shortcut if needed
    if strides > 1 or shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def create_resnet_model(input_shape, n_classes):
    """
    Creates a 1D ResNet model optimized for ECG time series classification.
    
    Args:
        input_shape: Tuple representing the input shape (sequence_length, channels)
        n_classes: Number of output classes
    
    Returns:
        Keras model
    """
    inputs = Input(shape=input_shape)

    # Initial convolution - smaller filters for ECG signals
    x = Conv1D(32, 7, strides=2, padding='same')(inputs)  # Reduced from 64 to 32 filters
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)  # More aggressive downsampling

    # Simplified residual blocks - fewer blocks and smaller filters
    # Stage 1: 32 filters, 2 blocks instead of 3
    for _ in range(2):
        x = resnet_block(x, 32, 3)

    # Stage 2: 64 filters, 2 blocks instead of 3
    for i in range(2):
        x = resnet_block(x, 64, 3, strides=2 if i == 0 else 1)

    # Stage 3: 128 filters, 2 blocks instead of 3 (removed 256-filter stage)
    for i in range(2):
        x = resnet_block(x, 128, 3, strides=2 if i == 0 else 1)

    # Classification head - simplified
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)  # Reduced from 256 to 64 neurons
    x = Dropout(0.3)(x)  # Reduced dropout for faster convergence
    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def get_resnet_training_config():
    """
    Returns the optimal training configuration for the ResNet model.
    
    Returns:
        Dictionary containing training parameters and callbacks
    """
    # ResNet can handle higher learning rates due to residual connections and batch normalization
    initial_lr = 0.001
    
    # Learning rate scheduler for ResNet - more aggressive scheduling due to ResNet's robustness
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,   # More aggressive reduction (30% of current LR)
        patience=3,   # Shorter patience due to ResNet's faster convergence
        min_lr=1e-6,  # Slightly higher minimum LR
        verbose=0,
        cooldown=1    # Shorter cooldown for more responsive scheduling
    )
    
    # Learning rate tracker for ResNet
    lr_tracker = LearningRateTracker()
    
    # Define callbacks for training with adjusted patience
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint('resnet_model.keras', save_best_only=True, monitor='val_loss', verbose=1)
    
    return {
        'initial_lr': initial_lr,
        'optimizer': Adam(learning_rate=initial_lr),
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
        'callbacks': [early_stopping, model_checkpoint, lr_scheduler, lr_tracker],
        'lr_tracker': lr_tracker,
        'custom_objects': None
    }


def compile_resnet_model(model, config=None):
    """
    Compiles the ResNet model with optimal settings.
    
    Args:
        model: Keras model to compile
        config: Optional configuration dictionary (uses default if None)
    
    Returns:
        Compiled model and training configuration
    """
    if config is None:
        config = get_resnet_training_config()
    
    model.compile(
        optimizer=config['optimizer'],
        loss=config['loss'],
        metrics=config['metrics']
    )
    
    return model, config

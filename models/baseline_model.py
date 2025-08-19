"""
Baseline STFT-CNN-RNN Model for ECG Classification

This module contains the baseline model architecture that combines:
1. Short-Time Fourier Transform (STFT) for time-frequency analysis
2. 2D CNN layers for feature extraction from spectrograms
3. RNN (LSTM) for temporal sequence modeling
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, LSTM, Dense, Dropout, BatchNormalization, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras


# Define a custom Keras Layer for the STFT operation
class STFTLayer(Layer):
    def __init__(self, frame_length=256, frame_step=128, fft_length=256, **kwargs):
        super(STFTLayer, self).__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

    def call(self, inputs):
        # Squeeze the channel dimension, as tf.signal.stft expects a 2D or 3D tensor.
        x = tf.squeeze(inputs, axis=-1)

        # Perform STFT
        stfts = tf.signal.stft(x,
                               frame_length=self.frame_length,
                               frame_step=self.frame_step,
                               fft_length=self.fft_length)

        # Calculate the magnitude and apply a log scale for stability.
        x = tf.abs(stfts)
        x = tf.math.log(x + 1e-6)

        # Add the channel dimension back for the subsequent Conv2D layers.
        return tf.expand_dims(x, axis=-1)

    def get_config(self):
        config = super(STFTLayer, self).get_config()
        config.update({
            'frame_length': self.frame_length,
            'frame_step': self.frame_step,
            'fft_length': self.fft_length,
        })
        return config


# Custom layer for dynamic reshaping
class ReshapeForRNN(Layer):
    def __init__(self, **kwargs):
        super(ReshapeForRNN, self).__init__(**kwargs)

    def call(self, inputs):
        # Get the shape dynamically during execution
        shape = tf.shape(inputs)
        # Reshape to (batch_size, timesteps, features)
        return tf.reshape(inputs, (shape[0], shape[1], shape[2] * shape[3]))

    def get_config(self):
        return super(ReshapeForRNN, self).get_config()


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


def create_baseline_model(input_shape, n_classes):
    """
    Creates the baseline model with STFT, CNN, and RNN components.
    
    Args:
        input_shape: Tuple representing the input shape (sequence_length, channels)
        n_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=input_shape)

    # 1. Custom STFT Layer
    stft_output = STFTLayer()(inputs)

    # 2. CNN Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(stft_output)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # 3. CNN Block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool1)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # 4. Reshape for RNN using custom layer
    reshaped = ReshapeForRNN()(pool2)

    # 5. RNN Layer
    lstm = LSTM(units=80, return_sequences=False)(reshaped)
    dropout = Dropout(0.4)(lstm)

    # 6. Output Layer
    outputs = Dense(n_classes, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def get_baseline_training_config():
    """
    Returns the optimal training configuration for the baseline model.
    
    Returns:
        Dictionary containing training parameters and callbacks
    """
    # Baseline model prefers lower learning rate due to STFT complexity and RNN sensitivity
    initial_lr = 0.00001
    
    # Learning rate schedulers
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduce LR by half when plateau detected
        patience=2,   # Wait 2 epochs before reducing (ECG patterns can be complex)
        min_lr=1e-7,  # Minimum learning rate
        verbose=0,
        cooldown=1    # Wait 1 epochs after LR reduction before monitoring again
    )
    
    # Learning rate tracker
    lr_tracker = LearningRateTracker()
    
    # Early stopping with higher patience due to LR scheduling
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
    
    # Save the best model found during training
    model_checkpoint = ModelCheckpoint('baseline_model.keras', save_best_only=True, monitor='val_loss', verbose=1)
    
    return {
        'initial_lr': initial_lr,
        'optimizer': Adam(learning_rate=initial_lr),
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
        'callbacks': [early_stopping, model_checkpoint, lr_scheduler, lr_tracker],
        'lr_tracker': lr_tracker,
        'custom_objects': {'STFTLayer': STFTLayer, 'ReshapeForRNN': ReshapeForRNN}
    }


def compile_baseline_model(model, config=None):
    """
    Compiles the baseline model with optimal settings.
    
    Args:
        model: Keras model to compile
        config: Optional configuration dictionary (uses default if None)
    
    Returns:
        Compiled model and training configuration
    """
    if config is None:
        config = get_baseline_training_config()
    
    model.compile(
        optimizer=config['optimizer'],
        loss=config['loss'],
        metrics=config['metrics']
    )
    
    return model, config

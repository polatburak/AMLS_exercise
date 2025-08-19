"""
Training utilities for ECG classification models

This module contains common training functions and utilities that can be used
with different model architectures for ECG time series classification.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_data(X_train, X_val, y_train, y_val, max_len=8000, n_classes=4):
    """
    Preprocesses ECG data for training.
    
    Args:
        X_train: Training data (list of arrays)
        X_val: Validation data (list of arrays)
        y_train: Training labels
        y_val: Validation labels
        max_len: Maximum sequence length for padding/truncation
        n_classes: Number of classes for one-hot encoding
    
    Returns:
        Tuple of (X_train_processed, X_val_processed, y_train_processed, y_val_processed)
    """
    print("Preprocessing ECG data...")
    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of validation samples: {len(X_val)}")
    print("-" * 30)

    # Pad/truncate sequences to a fixed length
    X_train_padded = pad_sequences(X_train, maxlen=max_len, dtype='float32', padding='post', truncating='post')
    X_val_padded = pad_sequences(X_val, maxlen=max_len, dtype='float32', padding='post', truncating='post')

    # Reshape for 1D CNN input, which expects a "channels" dimension
    X_train_reshaped = np.expand_dims(X_train_padded, axis=-1)
    X_val_reshaped = np.expand_dims(X_val_padded, axis=-1)

    print("Shape of X_train after padding and reshape:", X_train_reshaped.shape)
    print("Shape of X_val after padding and reshape:", X_val_reshaped.shape)
    print("-" * 30)

    # One-hot encode the labels for categorical cross-entropy
    y_train_encoded = to_categorical(y_train, num_classes=n_classes)
    y_val_encoded = to_categorical(y_val, num_classes=n_classes)

    print("Shape of y_train_encoded:", y_train_encoded.shape)
    print("Shape of y_val_encoded:", y_val_encoded.shape)
    print("\nSample of original labels:", y_train.values[:5] if hasattr(y_train, 'values') else y_train[:5])
    print("Sample of one-hot encoded labels:\n", y_train_encoded[:5])

    # Define the input shape for the models
    input_shape = X_train_reshaped.shape[1:]
    print(f"\nInput shape for models: {input_shape}")
    
    return X_train_reshaped, X_val_reshaped, y_train_encoded, y_val_encoded, input_shape


def train_model(model, X_train, y_train, X_val, y_val, config, epochs=50, batch_size=16, model_name="Model"):
    """
    Trains a model with the given configuration.
    
    Args:
        model: Keras model to train
        X_train: Training data
        y_train: Training labels (one-hot encoded)
        X_val: Validation data
        y_val: Validation labels (one-hot encoded)
        config: Training configuration dictionary
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_name: Name of the model (for logging)
    
    Returns:
        Training history
    """
    print(f"\n{'='*50}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*50}")
    
    print(f"Initial learning rate: {config['initial_lr']}")
    
    # Train the model
    print(f"Starting training for the {model_name}...")
    
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=config['callbacks'],
        verbose=1
    )
    
    print(f"\n{model_name} training complete.")
    if 'lr_tracker' in config and config['lr_tracker']:
        print(f"Final learning rate: {config['lr_tracker'].learning_rates[-1]:.2e}")
    
    return history


def evaluate_model(model_path, X_val, y_val_true, class_names, model_name="Model", custom_objects=None):
    """
    Evaluates a trained model and displays results.
    
    Args:
        model_path: Path to the saved model file
        X_val: Validation data
        y_val_true: True validation labels (not one-hot encoded)
        class_names: List of class names
        model_name: Name of the model (for display)
        custom_objects: Dictionary of custom objects needed to load the model
    
    Returns:
        Dictionary containing evaluation results
    """
    print(f"\n{'-'*40}")
    print(f"EVALUATING: {model_name}")
    print(f"{'-'*40}")
    
    try:
        # Load model
        if custom_objects:
            model = load_model(model_path, custom_objects=custom_objects)
        else:
            model = load_model(model_path)
        
        # Make predictions
        y_pred_probs = model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val_true, y_pred)
        
        # Store results
        results = {
            "accuracy": accuracy,
            "predictions": y_pred,
            "probabilities": y_pred_probs,
            "parameters": model.count_params()
        }
        
        print(f" {model_name} - Validation Accuracy: {accuracy:.4f}")
        print(f"   Parameters: {model.count_params():,}")
        
        # Classification report
        print(f"\nClassification Report for {model_name}:")
        print(classification_report(y_val_true, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_val_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return results
        
    except Exception as e:
        print(f"Failed to evaluate {model_name}: {str(e)}")
        return None


def generate_test_predictions(model_path, X_test, metadata, custom_objects=None, output_file='predictions.csv'):
    """
    Generates predictions for test data using a trained model.
    
    Args:
        model_path: Path to the saved model file
        X_test: Test data (list of arrays)
        metadata: Dataset metadata containing class information
        custom_objects: Dictionary of custom objects needed to load the model
        output_file: Name of the output CSV file
    
    Returns:
        Array of predictions
    """
    print("Loading and preprocessing test data...")
    
    # Preprocess test data (same as training data)
    MAX_LEN = 8000
    X_test_padded = pad_sequences(X_test, maxlen=MAX_LEN, dtype='float32', padding='post', truncating='post')
    X_test_reshaped = np.expand_dims(X_test_padded, axis=-1)
    print(f"Test data preprocessed. Shape: {X_test_reshaped.shape}")
    
    # Load model and generate predictions
    print("Loading model and generating predictions...")
    if custom_objects:
        model = load_model(model_path, custom_objects=custom_objects)
    else:
        model = load_model(model_path)
    
    test_predictions_probs = model.predict(X_test_reshaped)
    test_predictions = np.argmax(test_predictions_probs, axis=1)
    
    print(f"Predictions generated for {len(test_predictions)} test samples")
    
    # Create submission file
    print("Creating submission file...")
    submission_df = pd.DataFrame({
        'id': range(len(test_predictions)),
        'class': test_predictions
    })
    
    # Save predictions
    np.savetxt(output_file, test_predictions, fmt='%d')
    print(f"Predictions saved to '{output_file}'")
    
    # Display prediction statistics
    print(f"\n📈 PREDICTION STATISTICS:")
    print(f"Total predictions: {len(test_predictions)}")
    print(f"Class distribution:")
    class_counts = np.bincount(test_predictions)
    for class_idx, count in enumerate(class_counts):
        if class_idx < len(metadata['class_names']):
            class_name = metadata['class_names'][class_idx]
            percentage = (count / len(test_predictions)) * 100
            print(f"  Class {class_idx} ({class_name}): {count} samples ({percentage:.1f}%)")
    
    return test_predictions

"""
Models package for ECG classification

This package contains model architectures and training utilities for ECG time series classification.
"""

from .baseline_model import (
    create_baseline_model, 
    get_baseline_training_config, 
    compile_baseline_model,
    STFTLayer,
    ReshapeForRNN,
    LearningRateTracker
)

from .resnet_model import (
    create_resnet_model,
    get_resnet_training_config,
    compile_resnet_model,
    resnet_block
)

from .training_utils import (
    preprocess_data,
    train_model,
    evaluate_model,
    generate_test_predictions
)

__all__ = [
    # Baseline model
    'create_baseline_model',
    'get_baseline_training_config', 
    'compile_baseline_model',
    'STFTLayer',
    'ReshapeForRNN',
    'LearningRateTracker',
    
    # ResNet model
    'create_resnet_model',
    'get_resnet_training_config',
    'compile_resnet_model',
    'resnet_block',
    
    # Training utilities
    'preprocess_data',
    'train_model',
    'evaluate_model',
    'generate_test_predictions'
]

import struct
import numpy as np
import matplotlib.pyplot as plt

def read_binary_file(path):
    signals = []
    with open(path, "rb") as file:
        while True:
            size_bytes = file.read(4)
            if not size_bytes:
                break
            length = struct.unpack('i', size_bytes)[0]
            data = file.read(length * 2)
            signal = struct.unpack(f'{length}h', data)
            signals.append(np.array(signal))
    return signals

# Plot training history
def plot_history(history, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_name} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.show()

# Plot training history with learning rate
def plot_history_with_lr(history, model_name, lr_tracker=None):
    """
    Plot training history including learning rate changes
    """
    if lr_tracker is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_name} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Plot learning rate if available
    if lr_tracker is not None:
        ax3.plot(lr_tracker.learning_rates, 'r-', linewidth=2, label='Learning Rate')
        ax3.set_title(f'{model_name} - Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')  # Log scale for better visualization
        ax3.legend()
        ax3.grid(True)
        
        # Add annotations for significant LR changes
        for i in range(1, len(lr_tracker.learning_rates)):
            if lr_tracker.learning_rates[i] < lr_tracker.learning_rates[i-1] * 0.9:  # Significant drop
                ax3.annotate(f'LR reduced\n{lr_tracker.learning_rates[i]:.1e}', 
                           xy=(i, lr_tracker.learning_rates[i]), 
                           xytext=(i, lr_tracker.learning_rates[i] * 3),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                           fontsize=8, ha='center')
    
    plt.tight_layout()
    plt.show()
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import random


# Config and setup
# Update this to the path containing the animals folder from Kaggle (in this case, it's raw-img).
# Example: "C:/Users/YourName/Downloads/animals-10/raw-img"
dataset_path = "C:/Users/codep/OneDrive/Desktop/DeskSub/unistuff/cecs456/Project/archive/raw-img" 

BATCH_SIZE = 32
IMG_HEIGHT = 150
IMG_WIDTH = 150
EPOCHS = 10
SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


try:
    tf.config.experimental.enable_op_determinism()
    print("Op Determinism Enabled")
except AttributeError:
    print("Warning: enable_op_determinism not found (requires TensorFlow 2.9+)")

def main():
    print(f"TensorFlow Version: {tf.__version__}")
    
    data_dir = pathlib.Path(dataset_path)
    if not data_dir.exists():
        print(f"Error: Directory not found at {data_dir}")
        return

    # Data Loading
    print("Loading Training Data...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    print("Loading Validation Data...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")

    # Optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Model Design
    print("Building Model...")
    model = models.Sequential([
        # Rescaling: Normalize pixel values to be between 0 and 1
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Classification Head
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5), # Regularization to reduce overfitting
        layers.Dense(len(class_names), activation='softmax') # Output layer
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    model.summary()

    # Training
    print("Starting Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Evaluation and report
    print("Generating Reports...")
    
    # A. Plot Accuracy/Loss Curves
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(EPOCHS)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('training_results.png')
    print("Saved training_results.png")

    # B. Confusion Matrix
    # Extract labels and predictions from the validation set
    y_true = []
    y_pred = []

    print("Running predictions on validation set for Confusion Matrix...")
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    # Generate Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")

    # Print Classification Report (Precision, Recall, F1-Score)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    main()
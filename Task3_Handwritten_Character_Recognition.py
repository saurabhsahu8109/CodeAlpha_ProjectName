# =============================================================================
# TASK 3: Handwritten Character Recognition
# CodeAlpha Machine Learning Internship
# =============================================================================
# Dataset Source: Scikit-learn Digits Dataset (8×8 pixel handwritten digits)
#   — A compact, fully offline subset equivalent to MNIST.
#
# For the FULL MNIST dataset (28×28), download from:
#   URL : http://yann.lecun.com/exdb/mnist/  (original site)
#         https://www.kaggle.com/datasets/hojjatk/mnist-dataset  (Kaggle mirror)
#   Code: from tensorflow.keras.datasets import mnist
#         (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# For EMNIST (letters & digits):
#   URL : https://www.kaggle.com/datasets/crawford/emnist
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

# TensorFlow / Keras — used for CNN model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 60)
print("     TASK 3: HANDWRITTEN CHARACTER RECOGNITION")
print("     CodeAlpha ML Internship")
print("=" * 60)

# =============================================================================
# STEP 1 — LOAD DATASET
# =============================================================================
print("\n[INFO] Loading Digits dataset (sklearn built-in, 8×8 pixels, 10 classes)...")
digits = load_digits()
X_raw = digits.data          # shape: (1797, 64)
y     = digits.target        # 0-9 digit labels
images = digits.images       # shape: (1797, 8, 8)

print(f"Total samples : {X_raw.shape[0]}")
print(f"Image size    : 8 × 8 pixels (64 features)")
print(f"Classes       : {digits.target_names}")

# Normalise pixel values to [0, 1]
X_norm = X_raw / 16.0

# Reshape for CNN: (samples, height, width, channels)
X_cnn = X_norm.reshape(-1, 8, 8, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_cnn, y, test_size=0.2, random_state=42, stratify=y
)

# One-hot encode labels for categorical cross-entropy
y_train_oh = keras.utils.to_categorical(y_train, num_classes=10)
y_test_oh  = keras.utils.to_categorical(y_test,  num_classes=10)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# =============================================================================
# STEP 2 — VISUALISE SAMPLE IMAGES
# =============================================================================
fig, axes = plt.subplots(2, 10, figsize=(14, 3))
fig.suptitle('Sample Handwritten Digits (one per class)', fontsize=11, fontweight='bold')
for digit in range(10):
    idx = np.where(y == digit)[0][0]
    for row, cmap in enumerate(['gray', 'Blues']):
        ax = axes[row, digit]
        ax.imshow(images[idx], cmap=cmap)
        ax.set_title(str(digit), fontsize=9)
        ax.axis('off')
plt.tight_layout()
plt.savefig('task3_sample_digits.png', dpi=150, bbox_inches='tight')
print("[INFO] Sample digits plot saved as 'task3_sample_digits.png'")

# =============================================================================
# STEP 3 — BUILD CNN MODEL
# =============================================================================
print("\n[INFO] Building CNN model...")

def build_cnn():
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(8, 8, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Classifier head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

cnn_model = build_cnn()
cnn_model.summary()

# =============================================================================
# STEP 4 — TRAIN THE MODEL
# =============================================================================
print("\n[INFO] Training CNN model...")
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
]

history = cnn_model.fit(
    X_train, y_train_oh,
    epochs=60,
    batch_size=32,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

# =============================================================================
# STEP 5 — EVALUATION
# =============================================================================
print("\n[INFO] Evaluating model...")
y_pred_prob = cnn_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

test_loss, test_acc = cnn_model.evaluate(X_test, y_test_oh, verbose=0)
print(f"\nTest Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test Loss     : {test_loss:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

# =============================================================================
# STEP 6 — VISUALISATIONS
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Task 3 — Handwritten Character Recognition', fontsize=13, fontweight='bold')

# Training history
ax = axes[0]
ax.plot(history.history['accuracy'],     label='Train Acc')
ax.plot(history.history['val_accuracy'], label='Val Acc')
ax.set_title('Model Accuracy over Epochs')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
ax.legend(); ax.grid(True, alpha=0.3)

# Loss history
ax = axes[1]
ax.plot(history.history['loss'],     label='Train Loss', color='tomato')
ax.plot(history.history['val_loss'], label='Val Loss',   color='darkorange')
ax.set_title('Model Loss over Epochs')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(); ax.grid(True, alpha=0.3)

# Confusion Matrix
ax = axes[2]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
            xticklabels=range(10), yticklabels=range(10))
ax.set_title(f'Confusion Matrix\n(Acc = {test_acc:.3f})')
ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('task3_cnn_results.png', dpi=150, bbox_inches='tight')
print("[INFO] Results plot saved as 'task3_cnn_results.png'")

# --- Show a few predictions ---
fig, axes = plt.subplots(2, 10, figsize=(14, 3))
fig.suptitle('CNN Predictions on Test Set (green=correct, red=wrong)', fontsize=10)
for i in range(10):
    ax_top = axes[0, i]; ax_bot = axes[1, i]
    img = X_test[i].reshape(8, 8)
    correct = (y_pred[i] == y_test[i])
    color = 'green' if correct else 'red'
    ax_top.imshow(img, cmap='gray'); ax_top.axis('off')
    ax_bot.text(0.5, 0.5, f"P:{y_pred[i]}\nA:{y_test[i]}", ha='center', va='center',
                fontsize=9, color=color, fontweight='bold')
    ax_bot.axis('off')
plt.tight_layout()
plt.savefig('task3_predictions.png', dpi=150, bbox_inches='tight')
print("[INFO] Predictions plot saved as 'task3_predictions.png'")
print("\n[DONE] Task 3 — Handwritten Character Recognition complete.")

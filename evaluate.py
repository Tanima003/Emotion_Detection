import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# =========================
# Paths
# =========================

test_path = r"C:\Python.csv\archive\test"

model_path = "model/emotion_model.keras"

# =========================
# Settings
# =========================

IMG_SIZE = (48, 48)
BATCH_SIZE = 64

# =========================
# Load Model
# =========================

model = load_model(model_path)

print("Model loaded successfully!")

# =========================
# Test Data
# =========================

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# =========================
# Emotion Names
# =========================

class_names = list(test_data.class_indices.keys())

print("\nEmotion Classes:")
print(class_names)

# =========================
# Predictions
# =========================

print("\nGenerating predictions...")

predictions = model.predict(test_data)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = test_data.classes

# =========================
# Accuracy
# =========================

accuracy = accuracy_score(
    true_classes,
    predicted_classes
)

print("\n==============================")
print("MODEL ACCURACY")
print("==============================")

print(
    "Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

# =========================
# Classification Report
# =========================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================\n")

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_names
)

print(report)

# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)

# =========================
# Draw Confusion Matrix
# =========================

plt.figure(figsize=(9, 7))

plt.imshow(cm, interpolation="nearest")

plt.title("Facial Emotion Recognition - Confusion Matrix")

plt.colorbar()

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.xlabel("Predicted Emotion")

plt.ylabel("Actual Emotion")

# Add numbers inside matrix

for i in range(len(class_names)):

    for j in range(len(class_names)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "model/confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved at:")

print("model/confusion_matrix.png")
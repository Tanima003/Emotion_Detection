import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D,MaxPooling2D,BatchNormalization,Flatten,Dense,Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

train_path = r"C:\Python.csv\archive\train"
test_path = r"C:\Python.csv\archive\test"

IMG_SIZE = (48, 48)
BATCH_SIZE = 64
NUM_CLASSES = 7

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.10,
    horizontal_flip=True,
    validation_split=0.20
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

validation_data = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)
test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\nClass Indices:")
print(train_data.class_indices)

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
        input_shape=(48, 48, 1)
    ),
    BatchNormalization(),
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    BatchNormalization(),
    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    BatchNormalization(),
    Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.30),

    Flatten(),

    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.50),

    Dense(128, activation="relu"),
    Dropout(0.30),

    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

os.makedirs("model", exist_ok=True)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=8,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "model/best_emotion_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

print("\n==============================")
print("CNN EMOTION MODEL TRAINING")
print("==============================\n")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=40,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]
)

model = tf.keras.models.load_model(
    "model/best_emotion_model.keras"
)

print("\n==============================")
print("TESTING MODEL")
print("==============================\n")

test_loss, test_accuracy = model.evaluate(test_data)

print("\nTest Accuracy:", round(test_accuracy * 100, 2), "%")

model.save("model/emotion_model.keras")

print("\n================================")
print("MODEL TRAINING COMPLETED")
print("================================")
print("Model saved at:")
print("model/emotion_model.keras")
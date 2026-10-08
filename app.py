from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import cv2
import os
import base64
import io

app = Flask(__name__)

# Load model
model = load_model("model/emotion_model.keras")

# Emotion names
emotions = [
    "Angry",
    "Disgusted",
    "Fearful",
    "Happy",
    "Neutral",
    "Sad",
    "Surprised"
]

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Create static folder
os.makedirs("static", exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/camera")
def camera():
    return render_template("camera.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template(
            "index.html",
            error="Please select an image."
        )

    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "index.html",
            error="Please select an image."
        )

    # Open image
    image = Image.open(file)

    # Fix mobile photo orientation
    image = ImageOps.exif_transpose(image)

    # Convert to RGB
    image = image.convert("RGB")

    image_array = np.array(image)

    # Save image
    image_path = "static/uploaded_image.jpg"
    image.save(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    # Improve contrast
    gray_equalized = cv2.equalizeHist(gray)

    # Detect face
    faces = face_cascade.detectMultiScale(
        gray_equalized,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20)
    )

    # Try original grayscale
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        )

    if len(faces) == 0:
        return render_template(
            "index.html",
            error="No face detected. Please upload a clear face image."
        )

    # Select largest face
    faces = sorted(
        faces,
        key=lambda face: face[2] * face[3],
        reverse=True
    )

    x, y, w, h = faces[0]

    # Crop face
    face = gray[y:y+h, x:x+w]

    # Resize
    face = cv2.resize(
        face,
        (48, 48)
    )

    # Normalize
    face = face.astype("float32") / 255.0

    # Channel dimension
    face = np.expand_dims(
        face,
        axis=-1
    )

    # Batch dimension
    face = np.expand_dims(
        face,
        axis=0
    )

    # Prediction
    predictions = model.predict(
        face,
        verbose=0
    )[0]

    max_index = np.argmax(predictions)

    detected_emotion = emotions[max_index]

    confidence = round(
        float(predictions[max_index]) * 100,
        2
    )

    # All probabilities
    percentages = {}

    for i in range(len(emotions)):
        percentages[emotions[i]] = round(
            float(predictions[i]) * 100,
            2
        )

    return render_template(
        "index.html",
        emotion=detected_emotion,
        confidence=confidence,
        percentages=percentages,
        image_path=image_path
    )


# Camera prediction
@app.route("/predict_camera", methods=["POST"])
def predict_camera():

    try:

        data = request.json

        if not data or "image" not in data:
            return jsonify({
                "error": "No camera image received."
            })

        # Remove base64 header
        image_data = data["image"].split(",")[1]

        # Decode image
        image_bytes = base64.b64decode(image_data)

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        image = image.convert("RGB")

        image_array = np.array(image)

        # Convert to grayscale
        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        # Improve contrast
        gray_equalized = cv2.equalizeHist(gray)

        # Detect face
        faces = face_cascade.detectMultiScale(
            gray_equalized,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        )

        # Try original grayscale
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )

        if len(faces) == 0:
            return jsonify({
                "error": "No face detected."
            })

        # Select largest face
        faces = sorted(
            faces,
            key=lambda face: face[2] * face[3],
            reverse=True
        )

        x, y, w, h = faces[0]

        # Crop face
        face = gray[y:y+h, x:x+w]

        # Resize
        face = cv2.resize(
            face,
            (48, 48)
        )

        # Normalize
        face = face.astype("float32") / 255.0

        # Channel dimension
        face = np.expand_dims(
            face,
            axis=-1
        )

        # Batch dimension
        face = np.expand_dims(
            face,
            axis=0
        )

        # Prediction
        predictions = model.predict(
            face,
            verbose=0
        )[0]

        max_index = np.argmax(predictions)

        detected_emotion = emotions[max_index]

        confidence = round(
            float(predictions[max_index]) * 100,
            2
        )

        percentages = {}

        for i in range(len(emotions)):
            percentages[emotions[i]] = round(
                float(predictions[i]) * 100,
                2
            )

        return jsonify({
            "emotion": detected_emotion,
            "confidence": confidence,
            "percentages": percentages
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)
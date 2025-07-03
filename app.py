from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model("model.h5")
print(model.output_shape)  # Check the output shape of the model

# Replace with your actual class labels
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # match your model's input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        img = request.files["image"]
        if img:
            img_path = os.path.join("static", img.filename)
            img.save(img_path)

            processed = preprocess_image(img_path)
            pred = model.predict(processed)
            print("Prediction array:", pred)
            print("Prediction shape:", pred.shape)
            print("Argmax result:", np.argmax(pred))
            print("Class names length:", len(class_names))

            label = class_names[np.argmax(pred)]
            confidence = np.max(pred)

            prediction = f"{label} ({confidence*100:.2f}%)"
            image_path = img_path

    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)

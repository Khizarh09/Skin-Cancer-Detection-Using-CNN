Skin Disease Classification using Deep Learning
===============================================

Overview:
---------
This project is a deep learning-based image classification system that detects 7 types of skin diseases using a Convolutional Neural Network (CNN). The model has been trained on a labeled image dataset and deployed using a Flask web application, allowing users to upload an image and receive a predicted class along with confidence.

Skin Conditions Classified:
---------------------------
- akiec: Actinic keratoses
- bcc: Basal cell carcinoma
- bkl: Benign keratosis-like lesions
- df: Dermatofibroma
- mel: Melanoma
- nv: Melanocytic nevi
- vasc: Vascular lesions

Tech Stack:
-----------
- Python
- TensorFlow / Keras
- NumPy / Matplotlib
- Flask (for deployment)
- Google Colab (for training)
- HTML/CSS (basic UI)

How it Works:
-------------
1. The CNN model is trained on 224x224 resized images from the dataset.
2. The model consists of multiple convolutional and max-pooling layers followed by dense layers.
3. After training, the model is saved as 'model.h5'.
4. A Flask web app is created where users can upload an image.
5. The image is preprocessed and fed to the model for prediction.
6. The result is displayed with predicted class and confidence score.

Instructions:
-------------
- To retrain the model: Run the training script in a Jupyter Notebook or Colab.
- To run the Flask app:
  1. Place `model.h5` in the same directory as the Flask script.
  2. Run the app: `python app.py`
  3. Open `http://127.0.0.1:5000/` in your browser.
- Upload any skin image to test the prediction live.

Results:
--------
The model achieves good accuracy on both training and test data. Accuracy and loss plots are generated after training to visualize performance over epochs.

Repository Contents:
--------------------
- `model.h5`              : Trained CNN model
- `app.py`                : Flask web server
- `templates/index.html` : Web interface
- `static/`               : Folder to store uploaded images
- `README.txt`            : Project summary (this file)
- `training_script.ipynb`: Model training and evaluation code

Author:
-------
This project was built by khizar hayyat as part of a machine learning portfolio.  
For collaboration or freelance work, connect via GitHub, LinkedIn, or Upwork.


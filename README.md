# Plant Leaf Disease Detection using Deep Learning 🌿

## Project Overview
This project uses Deep Learning and a Convolutional Neural Network (CNN) to classify plant leaf images into healthy and diseased categories. A Streamlit web application allows users to upload a leaf image and view the model's prediction.

## Supported Classes
- Pepper — Bacterial Spot
- Pepper — Healthy
- Potato — Early Blight
- Potato — Healthy
- Tomato — Early Blight
- Tomato — Healthy

## Technologies Used
- Python
- Deep Learning
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Streamlit

## Project Files
- `Plant_Leaf_Disease_Detection.ipynb` — Jupyter Notebook containing the project workflow.
- `app.py` — Streamlit application.
- `plant_disease_weighted_cnn.keras` — Trained CNN model.
- Dataset — Plant leaf images used for training and testing.

## How to Run the Project

### 1. Install Required Libraries

```bash
python -m pip install streamlit tensorflow opencv-python numpy matplotlib pillow
```

### 2. Keep the Files Together

Ensure that `app.py` and `plant_disease_weighted_cnn.keras` are in the same project folder.

### 3. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

### 4. Use the Application
1. Upload a leaf image in JPG, JPEG, or PNG format.
2. Click **Predict Disease**.
3. View the predicted class and confidence score.

## Project Objective
The objective of this project is to use deep learning for image classification and identify selected plant leaf health and disease categories from uploaded images.

## Future Scope
- Add more plant species and disease categories.
- Improve model accuracy through further training.
- Deploy the application online.

## Disclaimer
This project is developed for educational purposes. Predictions may be incorrect and should not replace expert agricultural advice.

## Author
**Vaishnavi Dhumal**

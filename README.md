# 🍎 Fruit Image Classification Using CNN

A multi-class image classification project using a **Convolutional Neural Network (CNN)** built with TensorFlow and Keras. The model classifies fruit images into four categories: Apple, Banana, Mango, and Orange.

## 📌 Project Overview

This project demonstrates an end-to-end deep learning workflow, including:

- Image dataset preparation and organization
- Image preprocessing and augmentation
- CNN model development
- Model training and validation
- Performance evaluation
- Confusion matrix visualization
- Prediction on new fruit images

## 🎯 Classification Categories

The model recognizes four fruit categories:

| Class | Label |
|---|---|
| Apple | 0 |
| Banana | 1 |
| Mango | 2 |
| Orange | 3 |

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- Jupyter Notebook

## 📁 Project Structure

```text
Fruit-Classification-CNN/
│
├── dataset/
│   ├── origin/
│   │   ├── apple/
│   │   ├── banana/
│   │   ├── mango/
│   │   └── orange/
│   │
│   └── prepared/
│       ├── train/
│       ├── validation/
│       └── test/
│
├── models/
│   └── fruit_cnn_model.keras
│
├── predict/
│   └── mango.jpg
│
├── notebooks/
│   ├── 01_Fruit_Dataset_Collection.ipynb
│   ├── 02_Download_Select_Organize_Fruits360.ipynb
│   ├── Image Generator New(Multi-class).ipynb
│   ├── New Fit (Multi-class).ipynb
│   └── Predict new.ipynb
│
└── README.md
```

*Adjust the notebook paths if your repository stores them outside the `notebooks/` directory.*

## 📊 Dataset Preparation

The dataset contains images of four fruit categories.

Original images are organized into separate class folders inside `dataset/origin/`.

The dataset is divided using stratified sampling:

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

Stratification helps maintain the class distribution across all three splits.

## 🔄 Image Preprocessing and Augmentation

Images are resized to **128 × 128 pixels** and converted to RGB format.

Pixel values are normalized to the range of 0–1 during model training.

Training images are augmented using:

- Rotation
- Zoom
- Brightness adjustment
- Height shifting
- Width shifting
- Horizontal flipping

Augmentation is applied only to the training dataset to prevent data leakage into validation and testing.

## 🧠 CNN Architecture

The model uses a Sequential CNN architecture.

| Layer | Configuration |
|---|---|
| Input | 128 × 128 × 3 |
| Conv2D | 32 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 64 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Dropout | 0.25 |
| Flatten | — |
| Dense | 128 neurons, ReLU |
| Dropout | 0.5 |
| Output | 4 neurons, Softmax |

### Model Configuration

- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Evaluation Metric:** Accuracy
- **Batch Size:** 32
- **Maximum Epochs:** 20
- **Input Size:** 128 × 128 × 3

## ⚙️ Training Strategy

Three callbacks are used to improve training:

1. **ModelCheckpoint:** Saves the model with the best validation accuracy.
2. **EarlyStopping:** Stops training when validation loss stops improving.
3. **ReduceLROnPlateau:** Reduces the learning rate when validation loss plateaus.

The trained model is saved as:

```text
models/fruit_cnn_model.keras
```

## 📈 Model Evaluation

The model is evaluated using the independent test dataset.

Evaluation includes:

- Test loss
- Test accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### Initial Test Result

| Metric | Result |
|---|---:|
| Test Loss | 0.3440 |
| Test Accuracy | 77.78% |

*These are the results from the current training run, not a guaranteed final performance.*

## 🔍 Prediction on New Images

The trained model can classify new fruit photographs.

1. Place a new fruit image inside the `predict/` directory.
2. Open `Predict new.ipynb`.
3. Specify the image filename.
4. Run the prediction function.

Example:

```python
IMAGE_PATH = PREDICT_DIR / "mango.jpg"
result = predict_image(IMAGE_PATH)
```

The prediction function returns:

- Predicted fruit category
- Prediction confidence
- Probability for each class
- Original image visualization
- Class probability chart

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/rahat-codes/CNN-model-Image-Binary-Classification.git
```

### 2. Navigate to the project directory

```bash
cd CNN-model-Image-Binary-Classification
```

*Update the repository URL and directory name if this project is stored in a different repository.*

### 3. Install dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn pillow jupyter tqdm
```

### 4. Run the notebooks

Execute the notebooks in this order:

1. Dataset collection and organization (if needed).
2. Image generation and augmentation.
3. CNN model training and evaluation.
4. Prediction on new images.

## 🔮 Future Improvements

- Increase dataset size and diversity.
- Experiment with deeper CNN architectures.
- Apply transfer learning using MobileNetV2 or EfficientNet.
- Improve generalization on real-world fruit photographs.
- Deploy the trained model through a web application.
- Compare different optimizers and hyperparameters.

## 👨‍💻 Author

**Rahat Md Mahadi Hasan**

AI and Computer Engineering Student  
Tongmyong University, Busan, South Korea

- GitHub: https://github.com/rahat-codes
- LinkedIn: https://linkedin.com/in/rahat-mahadi-hasan

---

⭐ If you find this project interesting, consider giving the repository a star!
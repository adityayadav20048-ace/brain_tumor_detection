# 🧠 Brain Tumor Detection using Deep Learning with Synthetic MRI Generation

---

## 📌 Overview

This project presents an end-to-end deep learning pipeline for **brain tumor detection from MRI scans** using **MobileNetV2 transfer learning**.

To enhance dataset size and diversity, a **Generative Adversarial Network (GAN)** is utilized to generate synthetic MRI images across four tumor classes.

🎯 **Objective:**
Improve classification accuracy by combining **real medical images** with **GAN-generated synthetic data**.

---

## 🚀 Key Features

* 🧠 Brain tumor classification from MRI scans
* 🧪 Synthetic MRI dataset generation using GAN
* 🔁 Transfer learning with MobileNetV2
* 🔄 Data augmentation for better generalization
* 📊 Visualization of training performance metrics
* 🧩 Modular pipeline for training & inference
* 🌐 Streamlit-based web interface for predictions

---

## 🧬 Tumor Classes Detected

The model classifies MRI images into:

* Glioma Tumor
* Meningioma Tumor
* Pituitary Tumor
* No Tumor

---

## 📁 Project Structure

```
brain-tumor-detection/
│
├── Training/                  # Real MRI training dataset
├── Testing/                   # Real MRI testing dataset
├── synthetic_dataset/         # GAN-generated images
│
├── gan_models/                # Saved GAN models
├── gan_preview/               # GAN training preview images
│
├── mri_gan_generator.py       # GAN training script
├── brain_tumor_training.py    # Model training (real + synthetic)
├── brain_tumor_streamlit.py   # Streamlit web app
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* OpenCV
* Matplotlib
* PIL
* Streamlit

---

## 🤖 Deep Learning Models

### 🔹 GAN Model (Synthetic Data Generation)

* **Generator:** Deep Convolutional Generator
* **Discriminator:** CNN-based classifier
* **Latent Space:** 128-dimensional noise vector

---

### 🔹 Classification Model

Transfer Learning using **MobileNetV2**, with:

* Global Average Pooling
* Dense Layer
* Dropout
* Softmax Output Layer

---

## 📊 Dataset

The project uses a publicly available **Brain MRI dataset** sourced from Kaggle.

* Contains MRI images across four classes
* Used strictly for academic and research purposes

📌 **Note:**
Due to size limitations, the dataset is not included in this repository. Please download it from the link below.

---

## 🔗 Dataset Source

Dataset: Brain Tumor MRI Dataset (Kaggle)
👉 https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

### Dataset Classes:

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

---

## 🧪 Synthetic Data Enhancement

To improve model generalization and robustness:

* GAN is used to generate additional MRI images
* Synthetic images are combined with real data
* Helps reduce overfitting and improve accuracy

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Training the GAN

```bash
python mri_gan_generator.py
```

✔️ This will:

* Train GAN models for each class
* Save trained generators
* Generate synthetic MRI images

📂 Output Directory:

```
synthetic_dataset/
```

---

## 🏋️ Training the Classification Model

```bash
python brain_tumor_training.py
```

✔️ Training uses:

* Real MRI dataset
* GAN-generated synthetic dataset

---

## 🌐 Running the Streamlit App

```bash
streamlit run brain_tumor_streamlit.py
```

📌 Upload an MRI image to predict tumor type.

---

## 📈 Model Performance

Training includes:

* Early Stopping
* Learning Rate Scheduling
* Data Augmentation

📊 Evaluation Metrics:

* Accuracy
* Loss Curves
* Classification Report

---

## 🔮 Future Improvements

* Upgrade to StyleGAN for higher-quality synthetic images
* Deploy as a full-scale web application
* Add tumor segmentation for localization
* Expand dataset for better performance

---

## 👨‍💻 Author

**Aditya Kumar**
B.Tech CSE
KIIT University

---

## 📄 License

This project is intended for **educational and research purposes only**.

---

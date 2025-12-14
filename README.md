# CIFAR100_Using_Transfer_Learning
This repository applies pre-trained CNN models (ResNet50, VGG16, MobileNetV2) to classify images in the CIFAR-100 dataset, comparing performance across architectures for fine-grained object recognition.
# CIFAR-100 Image Classification Using Transfer Learning

## 📌 Project Overview

This project showcases the application of **transfer learning** to solve a **real-world, large-scale image classification problem** using the CIFAR-100 dataset. It is designed as a **portfolio-ready machine learning project**, demonstrating end-to-end skills including data preprocessing, model adaptation, fine-tuning, evaluation, and comparative analysis of deep learning architectures.

The work highlights practical experience with **pre-trained CNNs**, model optimization, and performance trade-off analysis—key skills relevant to roles in **Machine Learning, Computer Vision, and AI Engineering**.

---

## 🧩 Business Problem

Modern organizations in retail, e-commerce, media, and robotics manage massive volumes of image data that must be accurately classified and organized. Manual image categorization is time-consuming, error-prone, and not scalable.

The challenge addressed in this project is:

* How to efficiently classify images into **many fine-grained categories (100 classes)**
* While minimizing training time and computational cost
* And achieving acceptable accuracy using **limited labeled data**

---

## 💼 Business Impact

By leveraging transfer learning with pre-trained deep neural networks:

* **Reduced training time and resource usage** compared to training models from scratch
* Enabled **scalable image classification pipelines** suitable for production use
* Improved **search, recommendation, and inventory organization** through automated image understanding
* Demonstrated how model selection directly impacts accuracy and deployment feasibility

This approach is directly applicable to business use cases such as product categorization, visual content moderation, and automated quality inspection.

---

## 🧠 Models Used

* **ResNet50**
* **VGG16**
* **MobileNetV2**

All models are initialized with **ImageNet pre-trained weights** and customized for CIFAR-100 classification.

---

## 🔄 Project Flow

### 1. Data Loading and Preprocessing

* Loaded the **CIFAR-100 dataset** using TensorFlow/Keras.
* Applied **model-specific preprocessing functions** for ResNet50, VGG16, and MobileNetV2.
* Scaled pixel values and adjusted inputs to match the requirements of each architecture.

### 2. Model Preparation

* Loaded pre-trained models **without top (classification) layers**.
* Added custom fully connected layers to support **100 output classes**.
* Initially **froze base model layers** to preserve learned feature representations.
* Compiled models using:

  * Optimizer: `Adam`
  * Loss Function: `Sparse Categorical Crossentropy`
  * Metric: `Accuracy`

### 3. Fine-Tuning and Training

* Demonstrated **fine-tuning** by unfreezing the top layers of the ResNet50 model.
* Trained the fine-tuned model for **10 epochs** to improve task-specific performance.
* Monitored training and validation metrics throughout the process.

### 4. Model Evaluation

* Evaluated all trained models on the **held-out test dataset**.
* Compared classification performance using **accuracy** as the primary metric.

### 5. Results Comparison

* Analyzed and compared the effectiveness of ResNet50, VGG16, and MobileNetV2 for CIFAR-100 classification.
* Highlighted trade-offs between model complexity, accuracy, and generalization ability.

---

## 📊 Model Performance

| Model       | Test Accuracy |
| ----------- | ------------- |
| ResNet50    | 45%           |
| VGG16       | 31%           |
| MobileNetV2 | 28%           |

---

## ✅ Key Takeaways

* **ResNet50 outperformed other models**, benefiting from its depth and residual connections.
* **VGG16 achieved moderate results** but struggled with fine-grained classification.
* **MobileNetV2 prioritized efficiency**, resulting in lower accuracy on this complex dataset.
* Transfer learning proved to be an effective approach for multi-class image classification with limited data.

---

## 🚀 Future Improvements

* Apply data augmentation to improve generalization.
* Experiment with different fine-tuning strategies and learning rates.
* Try advanced architectures such as EfficientNet or Vision Transformers.
* Evaluate additional metrics such as top-5 accuracy and confusion matrices.

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib

---

## 📂 Dataset

* **CIFAR-100**: 60,000 color images (32×32) across 100 classes.

---

## 📌 Conclusion

This project demonstrates how transfer learning can be effectively applied to complex image classification problems. By comparing multiple pre-trained architectures, it provides practical insights into model selection and fine-tuning strategies for real-world computer vision applications.

## 📜 How to Run the Notebook
- Open Jupyter or Google Colab
  
- Run all cells in order

## 🧑‍💻 Author
Faheemunnisa Syeda

- 📧 Contact: [syedafaheem56@gmail.com]

- 🔗 GitHub: [https://github.com/syedafaheem7/]

- 🔗 linkedln: [https://www.linkedin.com/in/faheem-unnisa-s-6270888b/]

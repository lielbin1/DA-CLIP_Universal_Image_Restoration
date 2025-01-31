# Multi-Degradation Image Restoration with Extended DA-CLIP  

This repository contains the implementation of an **extension to DA-CLIP** for restoring images affected by **multiple simultaneous degradation types**. Our project explores the challenge of **multi-degradation image restoration**, specifically addressing a limitation highlighted in the **DA-CLIP paper**, where the original model supports only single-degradation scenarios.  

This work is built on the article **"Controlling Vision-Language Models for Multi-Task Image Restoration"**, available on **[arXiv](https://arxiv.org/abs/2310.01018)**. We extend the methods presented in this work to handle images with **multiple simultaneous degradations**. Full credit to the **original authors** of DA-CLIP, whose official GitHub repository can be found [here](https://github.com/Algolzw/daclip-uir/blob/main/da-clip/README.md).  

This project was conducted as part of the **Generative Models in AI** course, focusing on **enhancing generative techniques** for image restoration tasks.  

---

## 📌 Project Overview  

### 🔹 Motivation  
The DA-CLIP framework effectively restores **single-degradation images**, but real-world degraded images often contain **multiple noise sources**. Our approach extends DA-CLIP to handle **exactly two types of degradation per image**, generating additional **degradation embeddings** and modifying the **U-Net** model to process this additional information.  

### 🔹 Methodology  
1. **Input Processing**  
   - **LLQ (Lower-Quality) images** with two simultaneous degradations.  
   - Extraction of **Content Embedding** from DA-CLIP’s **Image Encoder**.  
   - Extraction of **two Degradation Embeddings** from DA-CLIP’s **Image Controller**.  

2. **Model Architecture**  
   - The **three embeddings** are **concatenated** and passed into a modified **U-Net**.  
   - The model is trained using **Mean Squared Error (MSE) loss**, the same loss function used in the original DA-CLIP paper.  

3. **Training Setup**  
   - Trained on **10,755 images** containing **24 different degradation combinations**.  
   - Used **the same datasets as DA-CLIP**, integrating additional degradations (noisy, blurry, JPEG-compressed).  
   - Trained for **7 epochs** due to computational constraints.  

### 🔹 Results  
- The model showed **some success** in reducing motion blur but **struggled with low-light noise**.  
- Performance was **inconsistent**, sometimes returning the **same degraded image** or generating **pure noise**.  
- Due to the **small dataset size and limited training time**, the model **did not generalize well** across all degradations.  

---

## 🔧 Installation & Setup  

### 📌 Prerequisites  
Ensure you have **Python 3.8+** installed along with the following dependencies:  

```bash
pip install torch torchvision transformers numpy matplotlib

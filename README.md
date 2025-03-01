# Deepfake-PRNU-Shield

Deepfake-PRNU-Shield is a deep learning-based project designed to detect deepfake images using Convolutional Neural Networks (CNNs) and enhance robustness against adversarial attacks using Photo Response Non-Uniformity (PRNU) noise-based defense.

## Features
- **Deepfake Detection**: CNN-based model trained to classify real and fake images.
- **Adversarial Attack (FGSM)**: Generates adversarial examples to fool the model.
- **PRNU-Based Defense**: Detects adversarial attacks using sensor noise analysis.
- **Gradio UI**: Provides a user-friendly interface for testing images.

## Dataset
The dataset consists of real and fake images stored in Google Drive. The dataset is loaded and preprocessed before training.

## Installation & Setup
Run the following commands in Google Colab to set up the environment:

```python
# Install dependencies
!pip install tensorflow keras numpy matplotlib scikit-image pywavelets gradio
```

## Model Training
Train the CNN model using the dataset

## FGSM Adversarial Attack
Generate adversarial examples using the Fast Gradient Sign Method (FGSM):

```python
import numpy as np

def fgsm_attack(image, epsilon, gradient):
    perturbation = epsilon * np.sign(gradient)
    adversarial_image = np.clip(image + perturbation, 0, 1)
    return adversarial_image
```

## PRNU-Based Defense
Enhance robustness against adversarial images using PRNU:

```python
from skimage.restoration import denoise_wavelet

def extract_prnu(image):
    return denoise_wavelet(image, method='BayesShrink', mode='soft')
```

## Gradio UI
Deploy a web-based interface using Gradio:

```python
import gradio as gr

def classify_image(image):
    # Implement image classification
    return "Real"  # Example output

demo = gr.Interface(fn=classify_image, inputs="image", outputs="label")
demo.launch()
```

## Repository Structure
```
Deepfake-PRNU-Shield/
│── dataset/
│── models/
│── adversarial_examples/
│── defense/
│── UI/
│── README.md
│── requirements.txt
```




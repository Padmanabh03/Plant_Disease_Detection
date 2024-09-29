# Plant Disease Detection and Diagnosis

This repository contains code and models for detecting and diagnosing plant diseases using a deep learning-based image classification model. The model is trained on a dataset of plant leaves affected by various diseases, and it can accurately classify multiple types of plant diseases based on the images of the affected leaves.

![Confusion Matrix](./CNF_Plant_Disease.png)

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

In agriculture, plant diseases can cause a significant loss in crop yield and quality. This project aims to leverage deep learning techniques to automatically detect plant diseases from leaf images. The model has been trained using MobileNet-V2, a lightweight convolutional neural network, to ensure both accuracy and efficiency in real-time applications.

## Dataset

The dataset used in this project consists of several plant disease categories, including:

- Pepper__bell___Bacterial_spot
- Pepper__bell___healthy
- Potato__Early_blight
- Potato__Late_blight
- Potato__healthy
- Tomato__Bacterial_spot
- Tomato__Early_blight
- Tomato__Late_blight
- Tomato__Leaf_Mold
- Tomato__Septoria_leaf_spot
- Tomato__Spider_mites_Two_spotted_spider_mite
- Tomato__Target_Spot
- Tomato__Tomato_Yellow_Leaf_Curl_Virus
- Tomato__Tomato_mosaic_virus
- Tomato__healthy

The dataset was split into training and validation sets, and the model was trained using this data to classify images into the correct disease category.

## Model Architecture

The model is based on the MobileNet-V2 architecture, which is optimized for mobile and embedded vision applications. It is pretrained on ImageNet and fine-tuned on the plant disease dataset to achieve high accuracy while maintaining a lightweight architecture suitable for deployment on devices with limited computational resources.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/plant-disease-detection.git
    cd plant-disease-detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset and place it in the appropriate directory.

4. Run the Jupyter notebook or Python script:
    - For notebook:
      ```bash
      jupyter notebook Plant_Disease_Detection_and_Diagnosis.ipynb
      ```
    - For script:
      ```bash
      python plant_disease_detection_and_diagnosis.py
      ```

## Usage

Once the model is trained, you can test it using images of plant leaves by running the provided Python script. The script will load the model, perform inference, and classify the input images into the correct disease categories.

Example:

```bash
python plant_disease_detection_and_diagnosis.py --image_path path_to_image

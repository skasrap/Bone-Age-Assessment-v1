# Bone Age Prediction Using Integrated Deep Neural Networks

This repository contains the implementation of a deep learning model to predict bone age from hand radiographs. The project utilizes ResNet-50 for image feature extraction and a Shallow Neural Network for integrating gender feature.

## Table of Contents
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Validation](#training-and-validation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Obtaining the Pre-trained Model](#Obtaining-the-Pre-trained-Model)
- [Usage](#usage)
- [Contributing](#contributing)

## Dataset
### RSNA 2017 Bone Age Assessment Dataset
This project uses the RSNA 2017 Bone Age Assessment dataset, which contains hand radiographs of pediatric patients along with their corresponding bone age labels. The dataset is provided by the Radiological Society of North America (RSNA) and is available on the [Kaggle platform](https://www.kaggle.com/kmader/rsna-bone-age)
In the project we start by exploring the dataset, which includes radiographic images, gender, and corresponding bone age labels. To ensure the robustness of our model, we perform a stratified split to create training and validation datasets with similar label distributions.

## Model Architecture
Our model architecture consists of two main components:
1. **ResNet-50**: Used for extracting features from the radiographic images.
2. **Shallow Neural Network**: Used to process additional features such as gender.

These components are then combined and passed through two linear layers to predict bone age.

## Training and Validation
To facilitate training, we define a custom dataset class that feeds data to the dataloaders for training, validation, and testing. Conditional statements ensure the correct processing for each dataset type.

## Hyperparameter Tuning
We use [Weights and Biases (wandb)](https://wandb.ai/) sweeps to automatically tune the hyperparameters. This includes defining a sweep configuration and running multiple experiments to find the optimal settings. The sweep configuration is saved to the cloud for easy access and reproducibility.

## Evaluation
After training, we load the best model and the optimal hyperparameters obtained from the sweeps. Keep in mind that we have provided a set of suitable hyperparameters hard coded in the scripts. Feel free to change them to fing more suitable ones!

## Prediction

To make predictions on the test dataset, run the following command:

```bash
python scripts/predict.py
```

## Obtaining the Pre-trained Model

To make predictions, you need the pre-trained model file (`best_model.pth`). 

### Option 1: Train the Model

You can train the model using the provided training script. After training, the model file will be saved as `best_model.pth` in the project root directory.

Run the training script:
```bash
python scripts/train.py
```

### Option 2: Download the Pre-trained Model
You can download the pre-trained model from the following link:
[Download best_model.pth](https://drive.google.com/file/d/1Vwlq2tLmv3_EOinPvB0hgcfvMWF_RtzP/view?usp=sharing)

Place the downloaded best_model.pth file in the project root directory.

## Usage
To use this repository, follow these steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/skasrap/bone-age-assessment-v1.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Prepare the dataset and configure the paths in the script.
4. Run the training script:
    ```bash
    python scripts/train.py
    ```
5. (Optional) Use wandb sweeps for hyperparameter tuning.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

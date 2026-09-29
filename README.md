# Recyclable Waste Classification with CNNs and ResNet18

## Dataset

The original image dataset used in this project is the
[Drinking Waste Classification dataset on Kaggle](https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification).

The dataset contains images of four recyclable drinking-waste categories:

- Aluminium cans (AluCan)
- Glass
- HDPE milk bottles (HDPEM)
- PET bottles

The original image files are not redistributed in this repository. The CSV files contain the dataset partitions used for training, validation and testing.

A computer vision project for classifying recyclable drinking waste using PyTorch.

The project compares a custom convolutional neural network with a pre-trained ResNet18 model and investigates the effect of class imbalance, additional real-world data collection, and data augmentation on classification performance.

## Classes

The model classifies images into four recyclable waste categories:

- AluCan
- Glass
- HDPEM
- PET

The original dataset was highly imbalanced, with substantially fewer HDPEM samples than the other three classes.

## Project Overview

The project was completed in three main stages:

### 1. Data Preparation and Exploration

The original dataset contained 3,100 labelled images:

| Class | Images |
|---|---:|
| AluCan | 1,000 |
| Glass | 1,000 |
| HDPEM | 100 |
| PET | 1,000 |

The data was split using a stratified 60/20/20 split:

- 60% training
- 20% validation
- 20% testing

Stratification was used to maintain approximately the same class proportions in each split.

### 2. Image Classification

Two classification systems were developed.

#### Custom CNN

A convolutional neural network was implemented from scratch using PyTorch.

The architecture included:

- 3 convolutional layers
- 16, 32 and 64 filters
- 3×3 kernels
- Max pooling
- ReLU activation
- Fully connected layer with 128 units
- Dropout
- Four-class output layer

Final test performance:

| Metric | Result |
|---|---:|
| Accuracy | 83.55% |
| Macro-F1 | 71.31% |

The model performed reasonably well on the majority classes but struggled with the minority HDPEM class.

#### ResNet18

A ResNet18 model pre-trained on ImageNet was used through transfer learning.

The pre-trained feature-extraction layers were frozen and the final classification layer was replaced with a four-class output layer.

Final test performance:

| Metric | Result |
|---|---:|
| Accuracy | 93.06% |
| Macro-F1 | 91.58% |

ResNet18 substantially outperformed the custom CNN, particularly on the minority HDPEM class.

## Class Imbalance

The original training distribution was:

| Class | Training Images |
|---|---:|
| AluCan | 600 |
| Glass | 600 |
| HDPEM | 60 |
| PET | 600 |

This imbalance had a noticeable impact on minority-class performance.

The custom CNN achieved only 25% recall on HDPEM, while ResNet18 achieved 80%.

## Additional Real-World Data

To improve minority-class performance, 150 additional HDPEM images were captured using a webcam and real milk bottles.

The photographs varied:

- Object orientation
- Camera position
- Background
- Lighting

This increased the number of real HDPEM training images from:

`60 → 210`

The validation and test sets remained unchanged so results could still be compared directly with the original models.

## Data Augmentation

The remaining class imbalance was addressed using `torchvision.transforms`.

The following transformations were applied to additional HDPEM training samples:

- Random horizontal flipping
- Random rotation up to ±15°
- Brightness variation
- Contrast variation

390 additional augmented HDPEM training samples were generated dynamically.

This produced a balanced training distribution:

| Class | Training Samples |
|---|---:|
| AluCan | 600 |
| Glass | 600 |
| HDPEM | 600 |
| PET | 600 |

Total balanced training samples: **2,400**

Validation and test images were not augmented.

## Final Results

The ResNet18 classification head was trained further using the balanced dataset.

| Metric | Before Balancing | After Balancing |
|---|---:|---:|
| Accuracy | 93.06% | 93.71% |
| Macro-F1 | 91.58% | 92.88% |
| AluCan Recall | 93.0% | 93.0% |
| Glass Recall | 94.0% | 93.5% |
| HDPEM Recall | 80.0% | 90.0% |
| PET Recall | 93.5% | 95.0% |

The largest improvement occurred in the minority HDPEM class:

**HDPEM recall increased from 80% to 90%.**

Macro-F1 improved more than overall accuracy, indicating that balancing the dataset primarily improved performance across the individual classes rather than simply increasing the number of total correct predictions.

## Error Analysis

Confusion matrices were used to examine classification errors before and after balancing.

Several errors became less common after retraining, including:

- HDPEM → AluCan
- HDPEM → Glass
- PET → AluCan
- PET → Glass

The most common error remained confusion between aluminium cans and glass images.

## Technologies

- Python
- PyTorch
- TorchVision
- ResNet18
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Pillow
- OpenCV
- Jupyter Notebook

## Project Structure

```text
waste-classification-cnn-resnet18/
│
├── notebook.ipynb
├── README.md
├── requirements.txt
│
└── generated_images/
    └── HDPEM/
        ├── Image001.png
        ├── Image002.png
        └── ...

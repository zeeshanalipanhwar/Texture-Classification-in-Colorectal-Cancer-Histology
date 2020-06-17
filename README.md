[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/zeeshanalipnhwr/Texture-Classification-in-Colorectal-Cancer-Histology/blob/master/LICENSE)

<embed width=200 height=200
    src="https://github.com/zeeshanalipnhwr/Texture-Classification-in-Colorectal-Cancer-Histology/blob/master/Images/Samples/01_TUMOR/1A11_CRC-Prim-HE-07_022.tif_Row_601_Col_151.tif"
    type="image/tiff"
    negative=yes>

# Texture Classification in Colorectal Cancer Histology Images
Classification of textures in colorectal cancer histology images.

# Dataset
This data set represents a collection of textures in histological images of human colorectal cancer. Each example is a 150 x 150 x 3 RGB image of one of 8 classes. Histological samples are fully anonymized images of formalin-fixed paraffin-embedded human colorectal adenocarcinomas (primary tumors) from a pathology archive (Institute of Pathology, University Medical Center Mannheim, Heidelberg University, Mannheim, Germany).

Download: [Kather_texture_2016_image_tiles_5000.zip](https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip?download=1). Google Drive Link to the dataset: [link 2](https://drive.google.com/file/d/1auURSHx5iolWqoaD6UTnRyen-B_E0CTK/view?usp=sharing).

Following is a sample training Tissue image with its correcsponding ground truth segmentation mask.
![Train0](https://github.com/zeeshanalipnhwr/Texture-Classification-in-Colorectal-Cancer-Histology/blob/master/Images/Train0.JPG)

The dataset contains *5000* images.

**Tissue images shapes**: *150x150x3*

# Data Spliting
The data is split into training, validation and testing sets with *60:20:20* ratio having 3k:1k:1k images for each set respectively.

# Data Augmentation
Following three augmentations are applied on the training and validation images and their correcsponding ground truth segmentation masks using a custom data augmenter:
1. Rotations of angles *90*, *180*, *270* degrees.
2. Horizontal flips
3. Vertical flips

These augmentations were applied on *10%* of the training and *20%* of the validation data.

# Requirements
- python version 3.x
- tensorflow version 1.x

# Project Structure

    .
    ├── Colab Notebooks       # Interactive notebooks containing the steps followed for training, testing, and predictions
    ├── Configs               # Configuration files for respective models
    ├── Images                # Screenshots or images needed for better presentation of README.md file
    ├── Models                # Complete implementations of models of the project
    │   ├── ResNet32.py          # ResNet32 standard model
    ├── Training Plots        # Training and validation performance graphs for loss, accuracy, and f1 scores
    ├── Utils                 # Files that include custom functionalities needed for this project
    ├── README.md             # A complete overview of this directory
    ├── predict.py            # Functions to predict a class for an image
    ├── test.py               # Functions to test a model performance on any test data
    └── train.py              # Functions to train a model with simple or augmented data


# Model Diagrams
## 1. ResNet32

# Model Summaries
Go to the colab notebooks in the Colab Notebooks directory for each model to view the detailed model summary.

# Performance Measures

## Accuracy
It is defined as <img src="https://render.githubusercontent.com/render/math?math=accuracy = \frac{TP%2BTN}{TP%2BFP%2BTN%2BFN}">.

## F1 Score (or Dice Score)
F1 Score is defined as the harmonic mean of precision and recall as <img src="https://render.githubusercontent.com/render/math?math=F_1=\frac{2}{\frac{1}{precision}%2B\frac{1}{recall}}"> where <img src="https://render.githubusercontent.com/render/math?math=precision=\frac{TP}{TP%2BFP}"> and <img src="https://render.githubusercontent.com/render/math?math=recall=\frac{TP}{TP%2BFN}">.

# Quantitatvie Results
| Model | Accuracy | Precision | Recall | F1 Score |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| ResNet32 | **None** | **None** | None | None |

# Qualitative Results
Following is the test tissue image with its ground truth segmentation mask that I show the qualitative results of my models on.
![Test Image For Qualitative Results](https://github.com/zeeshanalipnhwr/Texture-Classification-in-Colorectal-Cancer-Histology/blob/master/Images/test.JPG)

## 1. ResNet32
Not implimented yet.

# Replication Instructions
Use the colab notebooks in the Colab Notebooks directory for training, testing, and predictions on different models.

# Pretrained Models
- [ResNet32_basic.model](https://drive.google.com/file/d/notimplimentedyet)

# Instructions to load a pretrained model
Either use the colab notebooks in the Colab Notebooks directory for predictions on respective models, or follow the following steps using your console.
## 1. Clone this repository to your current directory

    git clone https://github.com/zeeshanalipnhwr/Texture-Classification-in-Colorectal-Cancer-Histology
    mv Texture-Classification-in-Colorectal-Cancer-Histology Texture_Classification_in_Colorectal_Cancer_Histology

## 2. Create a model

```python
# import all the models and their respective configuration files
from Texture_Classification_in_Colorectal_Cancer_Histology.Models import ResNet32
from Texture_Classification_in_Colorectal_Cancer_Histology.Configs import Configs
```

```python
# create a model of your choice among the above availabe models
model = load_model()
```

```python
# optionally view the created model summary
model.summary()
```

## 3. Load the respective pretrained-model weights

```python
model.load_weights("ResNet32_basic.model")
```

## 4. Make prediction for a sample on the network

```python
from Texture_Classification_in_Colorectal_Cancer_Histology.Utils import display
import numpy as np
import cv2

print_statements = False # do you need to see the print results blow?

# load a sample image
image_path = "drive/My Drive/sample_tissue_image.tif"
sample_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
sample_image = np.array(sample_image, dtype="float") / 255.0
sample_image = np.expand_dims(sample_image, axis=0)
if print_statements: print ("sample_image shape:", sample_image.shape)
```

## License
This project is licensed under the terms of the [MIT License](https://github.com/zeeshanalipnhwr/Texture-Classification-in-Colorectal-Cancer-Histology/blob/master/LICENSE).

## Acknowledgements
This project structure followed guidlines from [DongjunLee/hb-base](https://github.com/DongjunLee/hb-base) repository.

The [./.github/CONTRIBUTING.md](https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras/blob/master/.github/CONTRIBUTING.md) was adapted from a basic template for [contributing guidelines](https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62).

The [./.github/PULL_REQUEST_TEMPLATE.md](https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras/blob/master/.github/PULL_REQUEST_TEMPLATE.md) is taken from [TalAter/open-source-templates](https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras/blob/master/.github/PULL_REQUEST_TEMPLATE.md).

## Author
`Maintainer` [Zeeshan Ali](https://github.com/zeeshanalipnhwr) (zapt1860@gmail.com)

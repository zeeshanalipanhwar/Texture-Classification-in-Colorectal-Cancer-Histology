[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/zeeshanalipnhwr/Texture-Classification-in-Colorectal-Cancer-Histology/blob/master/LICENSE)

# Texture Classification in Colorectal Cancer Histology
Classification of textures in colorectal cancer histology images.

# Dataset
This data set represents a collection of textures in histological images of human colorectal cancer. Each example is a 150 x 150 x 3 RGB image of one of 8 classes. Histological samples are fully anonymized images of formalin-fixed paraffin-embedded human colorectal adenocarcinomas (primary tumors) from a pathology archive (Institute of Pathology, University Medical Center Mannheim, Heidelberg University, Mannheim, Germany).

Download: [Kather_texture_2016_image_tiles_5000.zip](https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip?download=1). Google Drive Link to the dataset: [link 2](https://drive.google.com/file/d/1auURSHx5iolWqoaD6UTnRyen-B_E0CTK/view?usp=sharing).

Following are some sample training Tissue images with their correcsponding class labels.
![samples](https://github.com/zeeshanalipnhwr/Texture-Classification-in-Colorectal-Cancer-Histology/blob/master/Images/samples.JPG)

The dataset contains *5000* images.

**Tissue images shapes**: *150x150x3*

**Reference:** Kather, J. N., Zöllner, F. G., Bianconi, F., Melchers, S. M., Schad, L. R., Gaiser, T., … Weis, C.-A. (2016). Collection of textures in colorectal cancer histology [Data set]. Zenodo. http://doi.org/10.5281/zenodo.53169

# Data Spliting
The data is split into 60% training, 20% validation and 20% testing sets containing 3k, 1k, and 1k images for each set respectively.

# Data Augmentation
Following affine transformations were applied to the images:
- Random horizontal and vertical flips

# Requirements
- python version 3.x
- tensorflow version 1.x

# Project Structure

    .
    ├── Colab Notebooks       # Interactive notebooks for training, testing, and predictions on the data
    ├── Configs               # Configuration files for respective models
    ├── Images                # Screenshots or images needed for better presentation of README.md file
    ├── Models                # Complete implementations of models of the project
    │   └── ResNet34.py          # ResNet34 standard model
    ├── Training Plots        # Training and validation performance graphs for loss, accuracy, and f1 scores
    ├── Utils                 # Files that include custom functionalities needed for this project
    ├── LICENCE               # MIT license
    └── README.md             # A complete overview of this directory


# Model Diagrams
## 1. ResNet34
<img src="https://github.com/zeeshanalipnhwr/Texture-Classification-in-Colorectal-Cancer-Histology/blob/master/Images/resnet34.png">

# Model Summaries
Go to the colab notebooks in the Colab Notebooks directory for each model to view the detailed model summary.

# Performance Measures

## Accuracy
It is defined as <img src="https://render.githubusercontent.com/render/math?math=accuracy = \frac{TP%2BTN}{TP%2BFP%2BTN%2BFN}">.

## F1 Score
F1 Score is defined as the harmonic mean of precision and recall as <img src="https://render.githubusercontent.com/render/math?math=F_1=\frac{2}{\frac{1}{precision}%2B\frac{1}{recall}}"> where <img src="https://render.githubusercontent.com/render/math?math=precision=\frac{TP}{TP%2BFP}"> and <img src="https://render.githubusercontent.com/render/math?math=recall=\frac{TP}{TP%2BFN}">.

# Quantitatvie Results
| Model | Accuracy | Precision | Recall | F1 Score |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| ResNet34 | 0.84 | 0.87 | 0.84 | 0.84 |

# Qualitative Results
The qualitative results of my models are shown on the test images.

## 1. ResNet34
Not implemented yet.

# Replication Instructions
Use the colab notebooks in the [Colab Notebooks](https://github.com/zeeshanalipanhwar/Texture-Classification-in-Colorectal-Cancer-Histology/tree/master/Colab%20Notebooks) directory for training, testing, and predictions on different models.

# Pretrained Models
- [ResNet34_basic.model](https://drive.google.com/file/d/1yFMnJefgXs0pgjwBaPIRmS5dJtmv0RSR/view?usp=sharing)

# Instructions to load a pretrained model
Either use the colab notebooks in the Colab Notebooks directory for predictions on respective models, or follow the following steps using your console.
## 1. Clone this repository to your current directory

    git clone https://github.com/zeeshanalipnhwr/Texture-Classification-in-Colorectal-Cancer-Histology
    mv Texture-Classification-in-Colorectal-Cancer-Histology TCCCH

## 2. Create a model

```python
# import all the models and their respective configuration files
from TCCCH.Models import ResNet34
from TCCCH.Configs import Configs
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
model.load_weights("ResNet34_basic.model")
```

## 4. Make prediction for a sample on the network

```python
from TCCCH.Utils import display
import numpy as np
import cv2

print_statements = False # do you need to see the print results blow?

# load a sample image
image_path = "drive/My Drive/sample_tissue_image.tif"
sample_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
sample_image = np.array(sample_image, dtype="float") / 255.0
if print_statements: print ("sample_image shape:", sample_image.shape)

# Make predictions
display(sample_image)
print ("Predicted class:", np.argmax(model.predict(sample_image), axis=-1))
```

## License
This project is licensed under the terms of the [MIT License](https://github.com/zeeshanalipnhwr/Texture-Classification-in-Colorectal-Cancer-Histology/blob/master/LICENSE).

## Acknowledgements
This project structure followed guidlines from [DongjunLee/hb-base](https://github.com/DongjunLee/hb-base) repository.

The [./.github/CONTRIBUTING.md](https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras/blob/master/.github/CONTRIBUTING.md) was adapted from a basic template for [contributing guidelines](https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62).

The [./.github/PULL_REQUEST_TEMPLATE.md](https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras/blob/master/.github/PULL_REQUEST_TEMPLATE.md) is taken from [TalAter/open-source-templates](https://github.com/zeeshanalipnhwr/Semantic-Segmentation-Keras/blob/master/.github/PULL_REQUEST_TEMPLATE.md).

## Author
`Maintainer` [Zeeshan Ali](https://github.com/zeeshanalipnhwr) (zapt1860@gmail.com)

import cv2
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def load(image_path): #Load an image from a file path
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def display(image): # Show image
    plt.figure(figsize = (5, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def display_samples(images, labels):
    # Visualize some examples from the dataset.
    classes = labels
    num_classes = len(classes)
    samples_per_class = 3
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(labels == cls)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(images[idx])
            plt.axis('off')
            if i == 0: plt.title(cls)
    plt.show()

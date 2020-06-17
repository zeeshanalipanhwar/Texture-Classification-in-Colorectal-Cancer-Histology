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
    
def visualize_samples(images, labels, classes):
    # Visualize some examples from the dataset.
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

def plot_training_losses(H, configurations):
    # plot the training and validation losses
    N = np.arange(0, configurations.EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="Training")
    plt.plot(N, H.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("/content/drive/My Drive/tccch_resnet32_loss_plot.png")
    plt.show()

def plot_training_accuracies(H, configurations):
    # plot the training and validation accuracies
    N = np.arange(0, configurations.EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["accuracy"], label="Training")
    plt.plot(N, H.history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("/content/drive/My Drive/tccch_resnet32_accuracy_plot.png")
    plt.show()

def plot_training_f1_scores(H, configurations):
    # plot the training and validation f1_scores
    N = np.arange(0, configurations.EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["f1_score"], label="Training")
    plt.plot(N, H.history["val_f1_score"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("/content/drive/My Drive/tccch_resnet32_f1_score_plot.png")
    plt.show()

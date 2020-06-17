from glob import glob
import numpy as np
import cv2

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def load(image_path): #Load an image from a file path
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

# load the data set
def load_data(configurations):
    files_path = "./" + configurations.DATASETNAME
    images, labels = [], []

    for dir_path in glob(files_path+'/*'):
        label = dir_path.split('_')[-1]
        print ("loading images of class label: {}..".format(label))
        for image_path in glob(dir_path+'/*'):
            image = load(image_path)
            images.append(image)
            labels.append(label)
        print ("Done!")
        
    images = np.array(images, dtype="float") / 255.0
    labels = np.array(labels)

    return images, labels

def train_valididation_test_split(images, labels, sizes=[3000, 1000, 1000]):
    # train_test_ratio is the split ratio for the input data to be split into train and test sets
    train_test_ratio = sizes[2] / (sum(sizes))
    # split the data set to training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=train_test_ratio, random_state=42)

    # train_valid_ratio is the split ratio for the train set to be split into train and validation sets
    train_valid_ratio = sizes[1] / (sizes[0] + sizes[1])
    # split the training set to training and validation sets
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)

    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

def labels_binarizer(Y_train, Y_valid, Y_test):
    # update each image label to its binary reporesentation
    Y_train = LabelBinarizer().fit_transform(Y_train)
    Y_train = Y_train.argmax(axis=-1)

    # update each image label to its binary reporesentation
    Y_valid = LabelBinarizer().fit_transform(Y_valid)
    Y_valid = Y_valid.argmax(axis=-1)
    
    # update each image label to its binary reporesentation
    Y_test = LabelBinarizer().fit_transform(Y_test)
    Y_test = Y_test.argmax(axis=-1)

    return Y_train, Y_valid, Y_test

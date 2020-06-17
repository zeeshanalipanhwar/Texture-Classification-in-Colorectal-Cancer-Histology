from glob import glob
import numpy as np
import cv2

def load(image_path): #Load an image from a file path
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def load_data(files_path):
    # load the data set
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

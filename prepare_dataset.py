import os
import matplotlib.pyplot as plt
import skimage.io as skimage_io
import skimage.transform as skimage_transform
import random as r
import numpy as np
import datetime
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.utils import plot_model

def prepareDataset(datasetPath):

    trainSetX = []
    trainSetY = []
    valSetX = []
    valSetY = []
    testSetX = []
    base_dir = os.getcwd()
    trainImagesPath = os.path.join(base_dir, datasetPath, 'train', "images")
    trainMasksPath = os.path.join(base_dir, datasetPath, 'train', "labels")
    trainSetFolder = os.scandir(trainImagesPath)

    for tile in trainSetFolder:
        imagePath = tile.path
        trainSetX.append(imagePath)


    r.shuffle(trainSetX)
    for trainExample in trainSetX:
        maskPath = os.path.join(trainMasksPath, os.path.basename(trainExample))
        trainSetY.append(maskPath)

    valImagesPath = os.path.join(base_dir, datasetPath, 'val', "images")
    valSetXFolder = os.scandir(valImagesPath)
    for tile in valSetXFolder:
        imagePath = tile.path
        valSetX.append(imagePath)

    valMasksPath = os.path.join(base_dir, datasetPath, 'val', "labels")
    valSetYFolder = os.scandir(valMasksPath)
    for tile in valSetYFolder:
        maskPath = tile.path
        valSetY.append(maskPath)

    return trainSetX, trainSetY, valSetX, valSetY


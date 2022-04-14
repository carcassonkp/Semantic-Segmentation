import os
from prepare_dataset import prepareDataset
from generate_data import trainGenerator,load_single_img
from model import unetCustom
import splitfolders
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
input_folder = "dataset"
import datetime
import numpy as np
import random


# splitfolders.ratio(input_folder, output="dataset", seed=1337, ratio=(.8, .2), group_prefix=None)
classes = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static',
          'dynamic', 'ground', 'road', 'sidewalk', 'parking',
          'rail track', 'building', 'wall', 'fence', 'guard rail',
          'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light',
          'traffic sign', 'vegetation', 'terrain', 'sky', 'person',
          'rider', 'car', 'truck', 'bus', 'caravan',
          'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']


batch_size = 8
inputSize = (256, 256)
inputChannels = 3
num_classes = len(classes)
trainSetX, trainSetY, valSetX, valSetY = prepareDataset(datasetPath=input_folder)

val_generator = trainGenerator(batch_size, valSetX, valSetY, dict(), inputSize, inputChannels,
                                 inputSize, num_classes)

model = unetCustom(num_classes)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
model.summary()
print(model.input_shape)

# --------- LOAD PRE-TRAINED MODEL---------
Nval = len(valSetX)
validationSteps = np.ceil(Nval / batch_size)
model.load_weights('saved/')
model.summary()
# --------- MODEL EVALUATION ---------
# test_loss, test_acc = model.evaluate(val_generator, steps=validationSteps, verbose=1)

# --------- SHOW SEGMENTATION ---------

# Load single image

test_img_number = 40
test_img, ground_truth = load_single_img(valSetX=valSetX, valSetY=valSetY, index=test_img_number,
                                         numClasses=num_classes)
test_img_input = np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img = np.argmax(prediction, axis=3)[0, :, :]

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img)
plt.show()
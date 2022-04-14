import os
from prepare_dataset import prepareDataset
from generate_data import trainGenerator
from model import unetCustom
import splitfolders
import tensorflow as tf
from tensorflow import keras
input_folder = "dataset"
import datetime
import numpy as np

# splitfolders.ratio(input_folder, output="dataset", seed=1337, ratio=(.8, .2), group_prefix=None)
classes = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static',
          'dynamic', 'ground', 'road', 'sidewalk', 'parking',
          'rail track', 'building', 'wall', 'fence', 'guard rail',
          'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light',
          'traffic sign', 'vegetation', 'terrain', 'sky', 'person',
          'rider', 'car', 'truck', 'bus', 'caravan',
          'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']

augmentation_args = dict(
    horizontal_flip=True
    # ,width_shift_range=range(256)
    # ,height_shift_range=range(256)
)
batch_size = 8
inputSize = (256, 256)
inputChannels = 3
num_classes = len(classes)
trainSetX, trainSetY, valSetX, valSetY = prepareDataset(datasetPath=input_folder)


train_generator = trainGenerator(batch_size, trainSetX, trainSetY, augmentation_args, inputSize, inputChannels,
                                 inputSize, num_classes)
val_generator = trainGenerator(batch_size, valSetX, valSetY, dict(), inputSize, inputChannels,
                                 inputSize, num_classes)

model = unetCustom(num_classes)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
model.summary()
print(model.input_shape)

# --------- CALLBACKS ---------
# tensorboard
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
# checkpoint

base_dir = os.getcwd()

save_callback = keras.callbacks.ModelCheckpoint(
    filepath=base_dir,
    save_weights_only=True, monitor='val_loss',
    save_best_only=True,
)
# decreases lr if accuracy does not increase
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
Ntrain = len(trainSetX)
stepsPerEpoch = np.ceil(Ntrain / batch_size)
Nval = len(valSetX)
validationSteps = np.ceil(Nval / batch_size)

LOAD_MODEL = True # disable if it is first time training
if LOAD_MODEL:
    # save_dir = os.getcwd()
    # save_dir = os.path.join(base_dir, 'saved')
    model.load_weights('saved/')
    model.summary()
    # --------- MODEL EVALUATION ---------
    test_loss, test_acc = model.evaluate(val_generator, steps=validationSteps, verbose=1)

TRAIN_MODEL = False
if TRAIN_MODEL:
    history1 = model.fit(train_generator,
                         steps_per_epoch=stepsPerEpoch,
                         verbose=1,
                         epochs=100,
                         validation_data=val_generator,
                         validation_steps=validationSteps,
                         callbacks=[tensorboard_callback
                                    , save_callback
                                    , lr_scheduler,
                                    ])



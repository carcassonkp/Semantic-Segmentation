import numpy as np
from PIL import Image
import random as r
from tensorflow.keras.utils import to_categorical



# def normalizeMask(mask, num_class=2):
#     # mask = mask / 255
#     new_mask = np.zeros(mask.shape + (num_class,))
#     for i in range(num_class):
#         new_mask[mask == i, i] = 1.
#     return new_mask

def getMaskChannels(tile, input_size):
    channel0 = Image.open(tile)
    channel0 = channel0.resize(input_size)
    channel0 = np.array(channel0)
    return channel0


def getImageChannels(tile, input_size):
    channel0 = Image.open(tile)
    channel0 = channel0.resize(input_size)
    channel0 = np.array(channel0)
    # normalize
    channel0 = channel0 / 255.
    return channel0


def augmentImage(image, inputSize, mask, aug_dict):
    # if 'width_shift_range' in aug_dict:
    #     cropx = r.sample(aug_dict['width_shift_range'], 1)[0]
    # else:
    #     cropx = (int)((image[0].shape[1] - inputSize[1]) / 2)
    # if 'height_shift_range' in aug_dict:
    #     cropy = r.sample(aug_dict['height_shift_range'], 1)[0]
    # else:
    #     cropy = (int)((image[0].shape[0] - inputSize[0]) / 2)
    if 'horizontal_flip' in aug_dict and aug_dict['horizontal_flip']:
        do_horizontal_flip = r.sample([False, True], 1)[0]
    else:
        do_horizontal_flip = False

    # mask = mask[cropy:cropy + inputSize[0], cropx:cropx + inputSize[1]]
    # image = image[cropy:cropy + inputSize[0], cropx:cropx + inputSize[1]]

    if do_horizontal_flip:
        mask = mask[:, ::-1]
        image = image[:, ::-1]

    return image, mask


def trainGenerator(batch_size, trainSetX, trainSetY, aug_dict, inputSize=(256, 256), inputChannels=1,
                   maskSize=(256, 256), numClasses=2):
    if batch_size > 0:
        while 1:
            iTile = 0
            nBatches = int(np.ceil(len(trainSetX) / batch_size))
            for batchID in range(nBatches):
                images = np.zeros(((batch_size,) + inputSize + (inputChannels,)))  # 1 channel
                masks = np.zeros(((batch_size,) + maskSize))
                iTileInBatch = 0
                while iTileInBatch < batch_size:
                    if iTile < len(trainSetX):
                        # Get images and masks and resize/normalize them
                        image = getImageChannels(trainSetX[iTile], inputSize)
                        mask = getMaskChannels(trainSetY[iTile], inputSize)

                        # Do augmentations
                        image, mask = augmentImage(image, inputSize, mask, aug_dict)

                        # Add new image/mask to batch
                        images[iTileInBatch] = image
                        masks[iTileInBatch] = mask

                        iTile = iTile + 1
                        iTileInBatch = iTileInBatch + 1
                    else:
                        images = images[0:iTileInBatch, :, :, :]
                        masks = masks[0:iTileInBatch, :, :]
                        break
                # Convert masks to one hot encoding

                masks = to_categorical(masks, num_classes=numClasses)
                yield images, masks


def load_single_img(valSetX, valSetY, index, inputSize=(256, 256), numClasses=2):
    image = getImageChannels(valSetX[index], inputSize)
    mask = getMaskChannels(valSetY[index], inputSize)
    # mask = to_categorical(mask, num_classes=numClasses)
    return image, mask

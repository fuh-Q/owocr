"""
Convolutional network that does real-world character recognition
Mitchell Vitez 2020

Input: images of characters from the chars74k dataset
Output: a character in the set "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

Dataset from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k
"""

from __future__ import annotations

import glob
import os
import random

import cv2 as cv
import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard  # type: ignore
from tensorflow.keras.layers import Activation, BatchNormalization, Convolution2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.models import Model, Sequential  # type: ignore
from tensorflow.keras.optimizers import schedules, RMSprop  # type: ignore
from tensorflow.keras.regularizers import L1, L2, L1L2  # type: ignore

from typing import Generator, List, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from tensorflow import Tensor
    from tensorflow.keras.preprocessing.image import NumpyArrayIterator  # type: ignore

# convolution is the process of applying filters to reduce the amount of neurons that need to be connected
# you slide a kernel (mask) over an image, and do some averaging on each chunk of the image
# because the data is labelled, the network will learn "for x outcome, the pixels in x region should average to n"

# a gradient is the curvature depicting how good or bad a prediction is, the lower down the curve, the better
# basically what training is doing is finding the lowest point along the whole curvature
# because it's more complex than just a parabola-like shape

# a feature map maps elements of target objects to neuron activations
# a sample is an image from the training set
# a batch size is how many samples are processed before the model's weights are adjusted
# a step is when the weights are updated after `batch_size` amount of samples have been processed
# an epoch is one complete pass through the training set

data_path = "dataset/English/Img/GoodImg/Bmp/*"
# data_path = "dataset/English/Img/BadImag/Bmp/*"

sample_count              = 7705
# sample_count              = 4798
sample_shape              = (36, 24, 1)     # 12x18 image, 3 channels
batch_size                = 128          # how many samples to run through before updating weights
num_classes               = 10 + 26 + 26 # 10 digits, 26 capital letters, 26 lowercase letters
val_split                 = 0.2          # save 20% of the data for validation

# logger + visualization
log                       = os.path.join(os.getcwd(), "logs")
tensorboard               = TensorBoard(log_dir=log, histogram_freq=1, write_images=True)

# train-validation split
# train_size                = (1 - val_split) * sample_count
# val_size                  =       val_split * sample_count

# # number of steps per epoch is dependent on batch size
# steps_per_epoch           = math.floor(train_size / batch_size)
# val_steps_per_epoch       = math.floor(val_size / batch_size)

epochs = 10

# learning rate (after each step when the weights are updated, determine by how much to jump across the gradient to try correcting)
# if too large it'll overshoot the optimal minimum loss, if too low it'll take forever to train
# we're using PiecewiseConstantDecay, this will lower and refine the learning rate as we take more steps
# here, we're learning at a rate of 0.1 for the first 20% of epochs, 0.01 for the next 20%, and 0.001 after that
# values                    = [0.1, 0.01, 0.001]
# boundaries                = [math.floor(epochs * steps_per_epoch * 0.2), math.floor(epochs * steps_per_epoch * 0.4)]
# lr_schedule               = schedules.PiecewiseConstantDecay(boundaries, values)

images = np.zeros(shape=(sample_count, *sample_shape))
labels = np.zeros(shape=(sample_count,), dtype=int)
sample_num = 0

print("Loading Chars74K dataset...")

for sample in sorted(glob.glob(data_path)):
    for filename in glob.glob(sample + "/*"):
        resized = cv.resize(cv.imread(filename), dsize=sample_shape[:-1][::-1])
        resized = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        resized = np.expand_dims(resized, axis=-1)
        images[sample_num] = resized
        labels[sample_num] = int(sample[-2:]) - 1
        sample_num += 1

print("Splitting data...")

split: List[NDArray[np.uint8]] = train_test_split(images, labels, test_size=val_split)
images, images_test, labels, labels_test = split
del split

print("Preprocessing...")

def random_crop(img: Tensor, crop_x: int, crop_y: int) -> Tensor:
    """
    Takes an image and a crop size, and returns a randomly cropped slice of the original image

    Sourced from https://jkjung-avt.github.io/keras-image-cropping/
    """
    height, width, _ = img.shape
    x = np.random.randint(0, width - crop_x + 1)
    y = np.random.randint(0, height - crop_y + 1)
    return img[y:(y+crop_y), x:(x+crop_x), :]

def crop_generator(
    batches: NumpyArrayIterator,
    crop_x: int,
    crop_y: int
) -> Generator[Tuple[NDArray[np.float64], NDArray[np.float64]], None, None]:
    """
    Takes a Keras ImageGen iterator and generates random crops from the image batches generated by the original iterator

    Sourced from https://jkjung-avt.github.io/keras-image-cropping/
    """
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_y, crop_x, batch_x.shape[-1]))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], crop_x, crop_y)

        yield (batch_crops, batch_y)

# pad images with zeroes for preprocessing
# paddings = tf.constant([[0, 0], [4, 4], [4, 4], [0, 0]])
# images = tf.pad(images, paddings, mode="CONSTANT")

# data generator for training data
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=val_split,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    rotation_range=5,
    zoom_range=0.1,
    rescale=1/255,
    # preprocessing_function=add_noise
)

# generate training and validation batches
train_batches = train_generator.flow(images, labels, batch_size=batch_size, subset="training")
validation_batches = train_generator.flow(images, labels, batch_size=batch_size, subset="validation")

train_batches = crop_generator(train_batches, sample_shape[1], sample_shape[0])
validation_batches = crop_generator(validation_batches, sample_shape[1], sample_shape[0])

# data generator for testing data
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    # preprocessing_function=add_noise,
    rescale=1/255
)

# generate test batches
test_batches = test_generator.flow(images_test, labels_test, batch_size=batch_size)

print(f"{images.shape[0]} sample images of size {images.shape[1]}x{images.shape[2]}")
print(f"{labels.shape[0]} labels")
assert images.shape[0] == labels.shape[0]

layers = \
  [ Input(shape=sample_shape)

  , Convolution2D(16, 3, 3)
  , BatchNormalization()
  , Activation("relu")

  , Convolution2D(32, 2, 1, kernel_regularizer=L2(0.001))
  , BatchNormalization()
  , Activation("relu")

  , Convolution2D(64, 2, 1, kernel_regularizer=L2(0.001))
  , BatchNormalization()
  , Activation("relu")

  , Convolution2D(128, 1, 1, kernel_regularizer=L2(0.001))
  , BatchNormalization()
  , Activation("relu")
  , MaxPooling2D(pool_size=(2, 2))

  , Flatten()
  , Dense(1024, activation="relu", kernel_regularizer=L2(0.001))
  , Dropout(0.5)
  , Dense(512, activation="relu")
  , Dropout(0.5)
  , Dense(62, activation="softmax")
  ]

if input("load saved model? [y/n] > ") in ("1", "true", "y", "ye", "yes"):
    model = tf.keras.models.load_model("model.keras")
else:
    model: Model = Sequential()
    for layer in layers:
        model.add(layer)
    epochs = 35

    model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

model.fit(
    train_batches,
    epochs=epochs,
    callbacks=[tensorboard],
    batch_size=batch_size,
    steps_per_epoch=100,
    validation_data=validation_batches,
    validation_steps=25,
)

model.summary()

model.evaluate(test_batches, verbose=1)
predictions = model.predict(images_test)

import string
CHARS = string.digits + string.ascii_uppercase + string.ascii_lowercase

for num in range(3):
    i = random.randint(0, images_test.shape[0])
    print("\nselected a random image")
    cv.imwrite(f"image-{num}.png", images_test[i])
    print("actual:", CHARS[labels_test[i]])
    chance = predictions[i][np.argmax(predictions[i])]
    print("confidence: {:0.2f}%".format(chance * 100))
    print("prediction:", CHARS[np.argmax(predictions[i])])

if input("save model? [y/n] > ").lower() in ("1", "true", "y", "ye", "yes"):
    model.save("model.keras", overwrite=True)
    # model.save_weights("model.weights.h5", overwrite=True)
    model.export("model")

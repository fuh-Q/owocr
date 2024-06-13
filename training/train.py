from __future__ import annotations

import random

import cv2 as cv
import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import mnist  # type: ignore
from tensorflow.keras.regularizers import L2  # type: ignore
from tensorflow.keras.models import Model, Sequential  # type: ignore
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Convolution2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)

from sklearn.model_selection import train_test_split

from typing import Generator, List, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from tensorflow import Tensor
    from tensorflow.keras.preprocessing.image import NumpyArrayIterator  # type: ignore

    from numpy.typing import NDArray

input_shape = (32, 32, 1)
val_split = 0.2
batch_size = 128
num_classes = 0

print("Loading MNIST dataset...")

num_classes += 10
((images_mnist, labels_mnist), (images_test_mnist, labels_test_mnist)) = mnist.load_data()

images = np.vstack((images_mnist, images_test_mnist))
labels = np.hstack((labels_mnist, labels_test_mnist))

# print("Loading Kaggle A-Z dataset...")

# num_classes += 26
# images_kaggle = []
# labels_kaggle = []

# with open("dataset/a_z-handwritten.csv", "r") as f:
#     for line in f.readlines():
#         line = [int(i) for i in line.split(",")]
#         label, image = line[0], np.array(line[1:], dtype=np.uint8)

#         image = image.reshape((28, 28))

#         images_kaggle.append(image)
#         labels_kaggle.append(label + 10)  # first 10 labels are the numbers in the MNIST set

# images_kaggle = np.array(images_kaggle, dtype=np.uint8)
# labels_kaggle = np.array(labels_kaggle, dtype=int)

# images = np.vstack((images, images_kaggle))
# labels = np.hstack((labels, labels_kaggle))

images = [cv.resize(img, dsize=input_shape[:-1][::-1]) for img in images]
images = np.expand_dims(images, axis=-1)

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
    horizontal_flip=False,
    vertical_flip=False,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    rotation_range=5,
    zoom_range=0.1,
    rescale=1/255,
)

# generate training and validation batches
train_batches = train_generator.flow(images, labels, batch_size=batch_size, subset="training")
validation_batches = train_generator.flow(images, labels, batch_size=batch_size, subset="validation")

train_batches = crop_generator(train_batches, input_shape[1], input_shape[0])
validation_batches = crop_generator(validation_batches, input_shape[1], input_shape[0])

# data generator for testing data
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

# generate test batches
test_batches = test_generator.flow(images_test, labels_test, batch_size=batch_size)

print(f"{images.shape[0]} sample images of size {images.shape[1]}x{images.shape[2]}")
print(f"{labels.shape[0]} labels")
assert images.shape[0] == labels.shape[0]

layers = \
  [ Input(shape=input_shape)

  , Convolution2D(16, 3, 3)
  , BatchNormalization()
  , Activation("relu")

  , Convolution2D(32, 2, 1, kernel_regularizer=L2(0.001))
  , BatchNormalization()
  , Activation("relu")

  , Convolution2D(64, 2, 1, kernel_regularizer=L2(0.001))
  , BatchNormalization()
  , Activation("relu")

  , MaxPooling2D(pool_size=(2, 2))

  , Flatten()
  , Dense(1024, activation="relu")
  , Dropout(0.5)
  , Dense(num_classes, activation="softmax")
  ]

if input("load saved model? [y/n] > ") in ("1", "true", "y", "ye", "yes"):
    model = tf.keras.models.load_model("model.keras")
else:
    model: Model = Sequential()
    for layer in layers:
        model.add(layer)
    epochs = 7

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(
    train_batches,
    epochs=epochs,
    batch_size=batch_size,
    steps_per_epoch=69,
    validation_data=validation_batches,
    validation_steps=14,
)

model.summary()

model.evaluate(test_batches, verbose=1)
predictions = model.predict(images_test[100:, :, :, :])

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

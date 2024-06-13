from pathlib import Path

import cv2 as cv
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, load_model  # type: ignore

input_shape = (32, 32, 1)
model: Model = load_model("model.keras")
model.summary()

path = Path.cwd() / input("enter path to image (relative) > ")
image = cv.imread(str(path))
image = cv.resize(image, dsize=input_shape[:-1][::-1])
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
image = np.expand_dims(image, axis=-1)
predictions = model.predict(tf.convert_to_tensor([image]))

import string
CHARS = string.digits + string.ascii_uppercase + string.ascii_lowercase

chance = predictions[0][np.argmax(predictions[0])]
print("confidence: {:0.2f}%".format(chance * 100))
print("prediction:", CHARS[np.argmax(predictions[0])])

cv.imwrite("resized.png", image)

print(predictions[0])

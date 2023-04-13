import cv2
import numpy as np
import tensorflow as tf
import sys


if len(sys.argv) != 3:
    sys.exit("Usage: python test.py model_file test_image")

image = cv2.imread(sys.argv[2])
image.resize((35, 35, 3))
test_data = np.array([image])

model = tf.keras.models.load_model(sys.argv[1])
prediction = model.predict(test_data).tolist()[0]
print(prediction)
print(prediction.index(max(prediction)))

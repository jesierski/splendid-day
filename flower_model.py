import pathlib
import random
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential

import PIL
import PIL.Image

"""[get flower training data from tensorflow]
"""
def get_training_flowers():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                   fname='flower_photos', 
                                   untar=True)
    data_dir = pathlib.Path(data_dir)
    batch_size = 32
    img_height = 180
    img_width = 180

    X_train = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    return X_train

"""[classify uploaded file according to used model]
"""
def classify(picture):
    loaded_model = keras.models.load_model('models/model_b2')
    img = picture.resize((180,180))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 
    predictions = loaded_model.predict(img_array)
    score = predictions[0]
    class_names = get_training_flowers().class_names
    flower_type = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    return flower_type
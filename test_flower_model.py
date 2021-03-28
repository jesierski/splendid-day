import pathlib
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential

import PIL
import PIL.Image

import recognition
from flower_model import classify, get_training_flowers

def test_classify():
    '''Predictions result is a string'''
    picture = PIL.Image.open('static/images/sunflower_14.jpg')
    answer = classify(picture)
    assert type(answer)==str
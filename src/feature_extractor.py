import keras
from keras import backend as K
from keras.preprocessing import image
from keras.models import Model

import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def get_model():
    vgg_model = keras.applications.VGG16(include_top=True, weights='imagenet')
    vgg_model.layers.pop()
    vgg_model.layers.pop()

    inp = vgg_model.input
    out = vgg_model.layers[-1].output

    model = Model(inp, out)
    return model

def get_features(model, cropped_image):
    x = image.img_to_array(cropped_image)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.vgg16.preprocess_input(x)
    features = model.predict(x)
    return features

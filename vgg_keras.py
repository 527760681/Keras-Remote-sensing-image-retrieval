from PIL import Image
from keras import models, Model, optimizers
from keras.applications.densenet import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
import utils
import numpy as np
from keras.models import load_model
classes = 21
image_size=224

def vgg(): #finetuning
    # load weight without 3 FC
    base_model = VGG16(include_top=False, weights='imagenet',classes=classes,input_shape=(image_size, image_size, 3))
    model=base_model.output

    #retrain 3 fc layers
    model = Flatten(name='flatten')(model)
    model = Dense(4096, activation='relu', name='fc1')(model)
    model = Dropout(rate=0.5)(model)
    model = Dense(4096, activation='relu', name='fc2')(model)
    model = Dropout(rate=0.5)(model)
    predictions = Dense(classes, activation='softmax', name='predictions')(model)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model
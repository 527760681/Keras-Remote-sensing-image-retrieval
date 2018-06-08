import keras
from PIL import Image
from keras import models, Model, optimizers
from keras.applications.densenet import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator

import utils
import numpy as np
from keras.models import load_model
import vgg_keras

classes = 21
image_size = 224
log_filepath = 'tb_log'


def train():  # finetuning
    # input tensor
    X_train, X_test, y_train, y_test = utils.load_data()

    # Data Augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # load model
    model = vgg_keras.vgg()

    # Compilation
    sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    tb_cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False,
                                        embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    # Train
    # model.fit(X_train,y_train,epochs=50,batch_size=32)

    # Fits the model on batches with real-time data augmentation
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                        steps_per_epoch=len(X_train) / 32, epochs=50, callbacks=[tb_cb])
    # Evaluate
    score = model.evaluate(X_test, y_test, batch_size=32)
    print(score)

    # Save Model
    model.save('vgg16.h5')
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')


def val(model_path):  # val
    model = vgg_keras.vgg()

    model.load_weights(model_path)

    image_path = r'QQ图片20180608144003.png'

    # Load img and resize
    img = Image.open(image_path)
    img = img.resize((224, 224))
    x = np.array(img)
    # keep the shape same as domel.output (256,256,3,1)
    x = np.expand_dims(x, axis=0)

    # Predict
    preds = model.predict(x)
    print(np.argmax(preds))

val('vgg16.h5')
# train()

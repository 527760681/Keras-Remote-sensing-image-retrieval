import csv
import h5py
import keras
from keras import optimizers, Model
import keras.backend as K
from utils import get_lines_count, get_topK, distance, load_data, single_data_gen, batch_data_gen
import numpy as np
from DenseNet import DenseNet
import collections


def train(image_size, classes,
          csv_train_path, csv_test_path,
          batch_size, epochs, lr,
          log_filepath, model_path):
    '''

    :param image_size:
    :param classes: total number of classes.
    :param csv_train_path: path that contains train image data
    :param csv_test_path: path that contains test image data
    :param batch_size:
    :param epochs:total training step
    :param lr:learning rate ,default in paper is 1e-4
    :param log_filepath:tensorflow callback log path
    :param model_path:path saving the model structure and weight

        model structure is stored in model.png
        optimizer can be changed in
        ` RMS = optimizers.RMSprop(lr=lr)
        ` model.compile(optimizer=RMS, loss='categorical_crossentropy', metrics=['accuracy'])

    '''

    # # load model
    model = DenseNet((image_size, image_size, 3), classes=classes)

    # Compilation
    RMS = optimizers.RMSprop(lr=lr)
    model.compile(optimizer=RMS, loss='categorical_crossentropy', metrics=['accuracy'])
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=0, write_graph=True, write_images=False)

    # Train
    total_num_train = get_lines_count(csv_train_path)
    print('total number of train set is %d' % (total_num_train))
    total_num_test = get_lines_count(csv_test_path)
    print('total number of test set is %d' % (total_num_test))

    # Fits the model on batches and validate in real-time
    model.fit_generator(batch_data_gen(csv_train_path, batch_size, classes, image_size),
                        steps_per_epoch=total_num_train // batch_size,
                        # total number of train set divided by batch_size
                        epochs=epochs,
                        validation_data=batch_data_gen(csv_test_path, batch_size, classes, image_size),
                        validation_steps=total_num_test // batch_size,
                        callbacks=[tb_cb])

    # Save Model
    print('saving the model')
    model.save(model_path)
    K.clear_session()

    # plot model in png file
    print('ploting model in png file')
    from keras.utils import plot_model
    plot_model(model, to_file=r'model.png')  # maybe raise exception,please install graphviz,pydot


def validate(image_size, classes,
             model_path, csv_val_path, csv_result_path):
    '''
    This is an implement of image classification using pre-trained model
    :param image_size:
    :param classes:total number of classes.
    :param model_path:h5 file which saved the training result
    :param csv_val_path:path that contains validation image data
    :param csv_result_path:path to save validate result
    this function saves result in csv_result_path,which contains 'image path','label','prediction' of
    '''
    # load model and weights
    print('load model and weights')
    model = DenseNet((image_size, image_size, 3), classes=classes)
    model.load_weights(model_path)

    right_num = 0
    total_num = 0

    writer = csv.writer(open(csv_result_path, 'a', newline='', encoding='utf8'))  # csv writer for saving the results

    # write header
    header = ['image path', 'label', 'prediction']
    writer.writerow(header)

    gen = single_data_gen(csv_val_path, classes, image_size)  # single image generator

    print('predicting')
    while 1:
        try:
            # get the raw data
            data, label, img_path = gen.__next__()
            data = np.expand_dims(data, axis=0)

            # get the prediction
            pred = model.predict(data)

            if np.argmax(pred) == np.argmax(label):
                right_num += 1

            result_list = list()
            result_list.append(img_path)
            result_list.append(np.argmax(label))
            result_list.append(np.argmax(pred))
            writer.writerow(result_list)

            total_num += 1
        except StopIteration:
            print(str(right_num / total_num))
            break


def index(image_size, classes,
          model_path, csv_imageLib_path, index_file
          ):
    '''
    This is an implement of feature extraction.The DenseNet's avg_pool layer is used to extract
    features.'img_path,label,features' are saved into index_file.
    :param image_size:
    :param classes:
    :param model_path:
    :param csv_imageLib_path:path to image which feature are going to be extracted
    :param index_file:h5py index file

    '''

    # load model's avg_pool layer for feature extraction and load weights
    base_model = DenseNet((image_size, image_size, 3), classes=classes)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    print('loading model')
    model.load_weights(model_path, by_name=True)

    gen = single_data_gen(csv_imageLib_path, classes, image_size)
    img_paths = []
    preds = []
    labels = []

    while 1:
        try:
            # get the raw data
            data, label, img_path = gen.__next__()
            data = np.expand_dims(data, axis=0)

            # get the feature
            pred = model.predict(data)

            label = np.argmax(label)

            img_paths.append(img_path)
            labels.append(label)
            preds.append(pred)

            print('extracing features from %s ' % (img_path))
        except StopIteration:
            break
    print('Writing features information to the file')

    img_paths_encode = []
    for word in img_paths:
        img_paths_encode.append(word.encode())

    h5f = h5py.File(index_file, 'w')
    h5f.create_dataset('img_paths_encode', data=img_paths_encode)
    h5f.create_dataset('labels', data=labels)
    h5f.create_dataset('preds', data=preds)
    h5f.close()
    print('done!')


def retrieval(image_size,classes,
              model_path,target_path,index_file,retrieval_result_file):
    '''
    This function use Euclidean distance to find out the image which is similar as target image.
    :param image_size:
    :param classes:
    :param model_path:
    :param target_path:need to be a csv file which formatted as 'image_path,category',only one image should be in this file
    :param index_file:
    :param retrieval_result_file:file to save the retrieval result
    '''

    # load model's avg_pool layer for feature extraction and load weights
    base_model = DenseNet((image_size, image_size, 3), classes=classes)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    print('loading model')
    model.load_weights(model_path, by_name=True)

    # load target image
    target, target_label = load_data(target_path, classes, image_size)
    target_label = np.argmax(target_label)

    print('load target feature')
    pre = model.predict(target)

    distant_dict = collections.OrderedDict()

    h5f = h5py.File(index_file, 'r')
    img_paths_encode = h5f['img_paths_encode'][:]
    labels = h5f['labels'][:]
    preds = h5f['preds'][:]
    h5f.close()

    preds_dict = {}

    for i, img_path in enumerate(img_paths_encode):
        img_path = img_path.decode()
        preds_dict[img_path] = {'pred': preds[i],
                                'label': labels[i]}
    print('Calculating the distance')
    for img_path in preds_dict.keys():
        pred = preds_dict[img_path]['pred']
        label = preds_dict[img_path]['label']

        dis = distance(pred, pre)
        distant_dict[img_path] = {'dis': dis, 'label': label}

    print('ranking')
    dict = sorted(distant_dict.items(), key=lambda d: d[1]['dis'])

    print('writing the result to %s'%(retrieval_result_file))
    get_topK(100, dict, target_label,retrieval_result_file)
    print('done!')

if __name__ == '__main__':
    lr = 1e-5
    epochs = 70
    batch_size = 16
    classes = 38
    image_size = 224
    model_path = 'pattern.h5'
    log_filepath = 'pattern_log'
    csv_train_path = r'PatternNet_train.csv'
    csv_test_path = r'PatternNet_test.csv'
    csv_val_path = r'PatternNet_test.csv'
    csv_result_path = r'PatternNet_result.csv'
    csv_imageLib_path = r'PatternNet_test.csv'
    index_file = r'PatternNet_index.h5'
    target_path = r'target.csv'
    retrieval_result_file = r'retrieval_result.txt'

    # train(image_size, classes,
    #       csv_train_path, csv_test_path,
    #       batch_size, epochs, lr,
    #       log_filepath, model_path)

    # validate(image_size,classes,
    #          model_path,csv_val_path,csv_result_path)

    # index(image_size, classes,
    #       model_path, csv_imageLib_path, index_file)

    retrieval(image_size,classes,
              model_path,target_path,index_file,retrieval_result_file)

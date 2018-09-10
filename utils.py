'''
this file contains some functions that used in train_and_val.py
'''
from PIL import Image
import numpy as np
import os
import csv
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle


def write_csv(image_folder, csv_train_path, csv_test_path):
    '''
    write the image path and category in the folder to a file
    default split 30% of all data into test csv
    :param image_folder:contains one subfolder for each category
    :param csv_train_path:path to save the train data
    :param csv_test_path:path to save the test data
    the csv is formatted by follow
        image_path,category
    such as
        .\dataset\EuroSAT\AnnualCrop\AnnualCrop_1.jpg,0
    '''
    writer_train = csv.writer(open(csv_train_path, 'a', newline='', encoding='utf8'))
    writer_test = csv.writer(open(csv_test_path, 'a', newline='', encoding='utf8'))

    count = 0  # for split test and train data
    current_category = 0

    for floder in os.listdir(image_folder):
        floder = os.path.join(image_folder, floder)
        for file in os.listdir(floder):
            print('writing down the ' + file)
            abs_path = os.path.join(floder, file)

            if count in [1, 2, 3]:
                writer_test.writerow([abs_path, current_category])
                count += 1
            elif count == 9:
                writer_train.writerow([abs_path, current_category])
                count = 0
            else:
                writer_train.writerow([abs_path, current_category])
                count += 1
        current_category += 1


def process_single(img_path, img_size):
    '''
    function to process single image,contains 'convert to rgb' and 'resize'
    :param img_path:single image path
    :param img_size:target image size
    :return: numpy array of input image
    '''
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = img.resize((img_size, img_size), Image.ANTIALIAS)
    img_array = np.array(img)
    return img_array


def batch_data_gen(csv_path, batch_size, num_classes, img_size):
    '''
    generator that yield shuffled data fit the batch size
    :param csv_path:path that contains input image data
    :param batch_size:target batch size
    :param num_classes:total number of classes.
    :param img_size:target image size
    :return:numpy ndarray (data,label),note that label is one-hot formation.
    '''
    content = []  # list for shuffle
    while 1:
        X = []
        Y = []

        for img_path, current_category in csv.reader(open(csv_path, 'r', encoding='utf8')):
            content.append((img_path, current_category))

        content = shuffle(content)  # do shuffle

        count = 0  # variable used to count the batch size
        for img_path, current_category in content:
            img_array = process_single(img_path, img_size)
            X.append(img_array)
            Y.append(current_category)
            count += 1
            if count == batch_size:
                count = 0
                data = np.stack(X, axis=0)
                label = np.stack(Y, axis=0)
                label = to_categorical(label, num_classes=num_classes)
                yield data, label
                X = []
                Y = []



def single_data_gen(csv_path, num_classes, img_size):
    '''
    generator that yield single image data
    :param csv_path:path that contains input image data
    :param num_classes:total number of classes.
    :param img_size:target image size
    :return:data, label, img_path;note that label is one-hot formation.
    '''
    while 1:
        for img_path, current_category in csv.reader(open(csv_path, 'r', encoding='utf8')):
            data = process_single(img_path, img_size)
            label = to_categorical(current_category, num_classes=num_classes)
            yield data, label, img_path
        raise StopIteration


def load_data(csv_path, num_classes, img_size):
    '''
    this function is writing for small dataset that can be loaded into memory directly
    :param csv_path:path that contains input image data
    :param num_classes:total number of classes.
    :param img_size:target image size
    :return:numpy ndarray (data,label),note that label is one-hot formation.
    '''
    X = []
    Y = []
    for img_path, current_category in csv.reader(open(csv_path, 'r', encoding='utf8')):
        img_array = process_single(img_path, img_size)
        X.append(img_array)
        Y.append(current_category)

    data = np.stack(X, axis=0)
    label = np.stack(Y, axis=0)
    label = to_categorical(label, num_classes=num_classes)

    return data, label


def distance(featureA, featureB):
    '''
    Euclidean distance of two feature
    :param featureA:
    :param featureB:
    :return:Euclidean distance (float)
    '''
    featureA = featureA.flatten()
    featureB = featureB.flatten()
    return np.sqrt(np.sum(np.square(featureA - featureB)))


def get_topK(k, dict, target_label,retrieval_result_file):
    '''
    get the top K images ranked by distance
    :param k:number of returned image
    :param dict:
        dict = {'image_path':image_path,
                content:{
                    'dis':distance,
                    'label':label
                }}
    :param target_label:the label of target image (int)
    '''
    f = open(retrieval_result_file,'a')

    num_right = 0
    num_total = 0
    for image_path, content in dict:
        distance = content['dis']
        label = content['label']

        result = image_path + ';  dis:' + str(distance) + '; label:' + str(label)+'\n'
        f.write(result)

        if target_label == label: num_right += 1

        num_total += 1

        if num_total == k: break

    correct = '正确率是' + str(num_right / num_total)
    f.write(correct)
    print(correct)

    f.close()

def get_lines_count(filename):
    '''
    Gets the count of lines in filename
    :return:count (int)
    '''
    count = 0
    fp = open(filename, "r", encoding='utf-8')
    while 1:
        buffer = fp.read(8 * 1024 * 1024)
        if not buffer:
            break
        count += buffer.count('\n')
    fp.close()
    return count

if __name__ == '__main__':
    image_folder = r'PatternNet'
    csv_train_path = r'PatternNet_train.csv'
    csv_test_path = r'PatternNet_test.csv'
    img_size = 224

    write_csv(image_folder, csv_train_path, csv_test_path)


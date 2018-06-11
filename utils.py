from PIL import Image
import numpy
import os
import csv
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

landuse_image_path = r'.\UCMerced_LandUse\Images'
csv_path = r'.\UCMerced_LandUse\class.txt'

img_size = 224
def write_csv():
    floders = os.listdir(landuse_image_path)

    f = open(csv_path, 'a', newline='', encoding='utf8')
    writer = csv.writer(f)

    # print(floders)
    class_num = 0
    for floder in floders:
        floder = os.path.join(landuse_image_path, floder)  # r'.\UCMerced_LandUse\Images\beach'
        files = os.listdir(floder)
        for file in files:
            abs_path = os.path.join(floder, file)  # r'.\UCMerced_LandUse\Images\tenniscourt\tenniscourt69.tif'
            # print(abs_path)
            writer.writerow([abs_path, class_num])
        class_num += 1
        # print(class_num)

    f.close()


def load_data():
    X=[]
    Y=[]
    reader = csv.reader(open(csv_path,'r',encoding='utf8'))
    for img_path,class_num in reader:
        img = Image.open(img_path)
        img = img.resize((img_size, img_size))
        img_array = numpy.array(img)
        X.append(img_array)
        Y.append(class_num)

    data=numpy.stack(X,axis=0)
    label = numpy.stack(Y,axis=0)

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=42)
    y_train= to_categorical(y_train,num_classes=21)
    y_test = to_categorical(y_test, num_classes=21)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()

# resize_img()
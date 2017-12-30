import numpy as np
import pandas as pd
import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator

import os
from os.path import join
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2

base_dir = 'C:/Users/Evan/Documents/GitHub/Data/Doggos'
test_dir = join(base_dir, 'test')

df_test = pd.read_csv(join(base_dir, 'sample_submission.csv'))

x_test = []
img_width, img_height = 500, 375

for f in tqdm(df_test['id'].values):
    img = cv2.imread(base_dir + '/test/{}.jpg'.format(f))
    img_alter = cv2.resize(img, (img_width, img_height))
    img_alter = img_alter / 255.

    x_test.append(img_alter)

#x_test = np.array(x_test, np.float32)

x_test[1:4]

model = load_model(join(base_dir, 'first_cnn.h5'))
preds = model.predict(x_test)
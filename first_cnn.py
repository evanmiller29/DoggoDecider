# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 09:55:45 2017

@author: Evan
"""

import os
import pandas as pd
import numpy as np

from os.path import join
from tqdm import tqdm
import cv2

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import functions as fn
from sklearn.model_selection import train_test_split

data_dir = 'C:/Users/Evan/Documents/GitHub/Data/Doggos'
train_raw_dir = join(data_dir, 'train_raw')
test_dir = join(data_dir, 'test')

create_dirs = False

df_train = pd.read_csv(join(data_dir, 'labels.csv'))
df_test = pd.read_csv(join(data_dir, 'sample_submission.csv'))

targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse=True)
one_hot_labels = np.asarray(one_hot)

im_size = 90
x_train = []
y_train = []
x_test = []

i = 0
for f, breed in tqdm(df_train.values):
    img = cv2.imread(train_raw_dir + '/{}.jpg'.format(f))
    label = one_hot_labels[i]
    x_train.append(cv2.resize(img, (im_size, im_size)))
    y_train.append(label)
    i += 1

for f in tqdm(df_test['id'].values):
    img = cv2.imread(test_dir + '/{}.jpg'.format(f))
    x_test.append(cv2.resize(img, (im_size, im_size)))

y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255.
x_test = np.array(x_test, np.float32) / 255.

print(x_train_raw.shape)
print(y_train_raw.shape)
print(x_test.shape)

num_class = y_train_raw.shape[1]

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=1)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(im_size, im_size, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.add(Dense(num_class, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)



























df_train['file'] = df_train['id']  + '.jpg'

df_test = pd.read_csv(join(data_dir, 'sample_submission.csv'))

targets_series = pd.Series(df_train['breed'])
NUM_CLASSES = targets_series.unique().shape[0]

doggo_types = set(targets_series)

if create_dirs:

    fn.create_train_valid_dirs(doggo_types, data_dir)

os.chdir(data_dir)

img_width, img_height = 500, 375

train_data_dir = 'train'
validation_data_dir = 'validation'
nb_train_samples = 7143
nb_validation_samples = 3079
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling

valid_path = join(data_dir, 'validation')
train_path = join(data_dir, 'train')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes= list(doggo_types),
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes= list(doggo_types),
    class_mode='categorical')

model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save('first_cnn.h5')


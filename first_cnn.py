# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 09:55:45 2017

@author: Evan
"""

import os
import pandas as pd

from os.path import join

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import functions as fn

data_dir = 'C:/Users/Evan/Documents/GitHub/Data/Doggos'
create_dirs = False

df_train = pd.read_csv(join(data_dir, 'labels.csv'))
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
nb_train_samples = 2000
nb_validation_samples = 800
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
model.add(Dense(NUM_CLASSES, activation = 'softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling

valid_path = join(os.getcwd(), 'validation')
train_path = join(os.getcwd(), 'train')


test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes = list(doggo_types),
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes = list(doggo_types),
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
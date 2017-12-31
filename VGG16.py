import os
import pandas as pd
import numpy as np

from os.path import join
from tqdm import tqdm
import cv2

from keras.applications.vgg16 import preprocess_input, VGG16
from keras.preprocessing import image

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression

data_dir = 'C:/Users/Evan/Documents/GitHub/Data/Doggos'
sub_dir = 'F:/Nerdy Stuff/Kaggle submissions/Doggos'
train_raw_dir = join(data_dir, 'train_raw')
test_dir = join(data_dir, 'test')

df_train = pd.read_csv(join(data_dir, 'labels.csv'))
df_test = pd.read_csv(join(data_dir, 'sample_submission.csv'))

targets_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(targets_series, sparse=True)
one_hot_labels = np.asarray(one_hot)

y_train_label = targets_series

im_size = 90
y_train= []
x_train = []
x_test = []

i = 0

for f, breed in tqdm(df_train.values):

    img = cv2.imread(train_raw_dir + '/{}.jpg'.format(f))
    img = cv2.resize(img, (im_size, im_size))
    img = image.img_to_array(img)
    img_pre = preprocess_input(img.copy())
    x_train.append(img_pre)

    label = one_hot_labels[i]
    y_train.append(label)
    i += 1

for f in tqdm(df_test['id'].values):

    img = cv2.imread(test_dir + '/{}.jpg'.format(f))
    img = cv2.resize(img, (im_size, im_size))
    img = image.img_to_array(img.copy())

    img_pre = preprocess_input(img)

    x_test.append(img_pre)

y_train_raw = np.array(y_train, np.uint8)
x_train_raw = np.array(x_train, np.float32) / 255.
x_test = np.array(x_test, np.float32) / 255.

num_class = y_train_raw.shape[1]

print(x_train_raw.shape)
print(y_train_raw.shape)

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=1)

vgg_bottleneck = VGG16(weights='imagenet', include_top=False, pooling='avg')

preds_vgg16 = vgg_bottleneck.predict(x_test)
train_preds_vgg16 = vgg_bottleneck.predict(X_train)
valid_preds_vgg16 = vgg_bottleneck.predict(X_valid)

logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=1234)
logreg.fit(train_preds_vgg16, (Y_train * range(num_class)).sum(axis=1))

train_probs = logreg.predict_proba(train_preds_vgg16)
valid_probs = logreg.predict_proba(valid_preds_vgg16)

train_preds = logreg.predict(train_preds_vgg16)
valid_preds = logreg.predict(valid_preds_vgg16)

print('Validation VGG LogLoss {}'.format(log_loss(Y_valid, valid_probs)))
print('Validation VGG Accuracy {}'.format(accuracy_score((Y_valid * range(num_class)).sum(axis=1), valid_preds)))

print('Train VGG LogLoss {}'.format(log_loss(Y_train, train_probs)))
print('Train VGG Accuracy {}'.format(accuracy_score((Y_train * range(num_class)).sum(axis=1), train_preds)))

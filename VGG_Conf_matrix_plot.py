#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 10:45:56 2020

@author: Shofi
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
#import utils
import os
import PIL
#%matplotlib inline

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers

from IPython.display import SVG, Image
from livelossplot.tf_keras import PlotLossesCallback
import tensorflow as tf
print("Tensorflow version:", tf.__version__)

img_size = 96
batch_size = 64

datagen_train = ImageDataGenerator(horizontal_flip=True)

train_generator = datagen_train.flow_from_directory("daisee/train/",
                                                    target_size=(img_size,img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = datagen_validation.flow_from_directory("daisee/test/",
                                                    target_size=(img_size,img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)

test_generator = datagen_validation.flow_from_directory("daisee/evaluation/",
                                                    target_size=(img_size,img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)

# Initialising the CNN
model = Sequential()

## 4 conv net layers, 3 dense layers (2 fully connected, 1 softmax) 
# 1 - Convolution
model.add(Conv2D(64,(3,3), padding='same', input_shape=(img_size, img_size,1)))
model.add(Conv2D(64,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #downsampling/ shrink the height and width dimension by factor of 2  
model.add(Dropout(0.8))

# 2nd Convolution layer
model.add(Conv2D(128,(3,3), padding='same'))
model.add(Conv2D(128,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.8))

# 3rd Convolution layer
model.add(Conv2D(256,(3,3), padding='same'))
model.add(Conv2D(256,(3,3), padding='same'))
#model.add(Conv2D(256,(3,3), padding='same'))
#model.add(Conv2D(256,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.8))

# 4th Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(Conv2D(512,(3,3), padding='same'))
#model.add(Conv2D(512,(3,3), padding='same'))
#model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.8))

# 5th Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(Conv2D(512,(3,3), padding='same'))
#model.add(Conv2D(512,(3,3), padding='same'))
#model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.8))


# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.7))

# Fully connected layer 2nd layer
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.7))

# Fully connected layer 2nd layer
model.add(Dense(1000, name = 'Dense_1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.7))

model.add(Dense(3, activation='softmax'))

model.get_layer('Dense_1').kernel_regularizer = regularizers.l1(0.0001) #0.0001
opt = Adam(lr=0.0005)
#opt=SGD(lr=0.02, momentum = 0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


model.load_weights('eng_model_weights_serv.h5')
tic = datetime.now()
evaluate = model.evaluate_generator(test_generator, steps = test_generator.n // test_generator.batch_size, verbose =1)
time = datetime.now() - tic
print('Processing time {}'.format(time))

# assigning label names to the corresponding indexes
labels = {0:'Very_Engaged', 1:'Not_Engaged', 2:'Normally_Engaged'}

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import PIL
# import cv2

prediction = []
original = []
image = []
count = 0
for i in os.listdir('./evaluation'):
  for item in os.listdir(os.path.join('./evaluation',i)):
    #code to open the image
    img= PIL.Image.open(os.path.join('./evaluation',i,item))
    #resizing the image to (48x48)
    img = img.resize((96,96))
    #appending image to the image list
    image.append(img)
    #converting image to array
    img = np.asarray(img, dtype= np.float32)
    #normalizing the image
    img = img / 255
    #reshaping the image in to a 4D array
    img = img.reshape(-1,img_size,img_size,1)
    #making prediction of the model
    predict = model.predict(img)
    #getting the index corresponding to the highest value in the prediction
    predict = np.argmax(predict)
    #appending the predicted class to the list
    prediction.append(labels[predict])
    #appending original class to the list
    original.append(i)
  
# visualizing the results
import random
fig=plt.figure(figsize = (100,100))
for i in range(20):
    j = random.randint(0,len(image))
    fig.add_subplot(20,1,i+1)
    plt.xlabel("Prediction -" + prediction[j] +"   Original -" + original[j])
    plt.imshow(image[j],'gray')
    
fig.tight_layout()
plt.show()
plt.savefig('VGG_results.png')

# classification report
print(classification_report(np.asarray(original), np.array(prediction)))

# plot confusion matrix

plt.figure(figsize=(20,20))
cm = confusion_matrix(np.asarray(original), np.asarray(prediction))
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Original')
ax.set_title('Confusion_matrix')

plt.savefig('VGG_confusionMatrix.png')


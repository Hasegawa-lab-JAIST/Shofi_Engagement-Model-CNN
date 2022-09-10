#!/usr/bin/env python
# coding: utf-8

# <h2 align=center> Facial Expression Recognition</h2>

#  

# ### Task 1: Import Libraries

# In[ ]:


#import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import os
#import PIL
import tensorflow as tf
import time

from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, GlobalAveragePooling2D#, Input,Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential#, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau#, EarlyStopping
from tensorflow.keras.utils import plot_model
#from sklearn.model_selection import KFold, StratifiedKFold

#from IPython.display import SVG, Image
from livelossplot.tf_keras import PlotLossesCallback
print("Tensorflow version:", tf.__version__)


# ### Task 2: Plot Sample Images

#samples = utils.datasets.fer.plot_example_images(plt).show()

for expression in os.listdir("train/"):
    if not expression.startswith('.'):  
        print(str(len(os.listdir("train/" + expression))) + " " + expression + " images")

# ### Task 3: Generate Training and Validation Batches
img_size = 48 #set image 
batch_size = 50 #can change to see if the it gives better result
'''
datagen_train = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)
datagen_validation = ImageDataGenerator(horizontal_flip=True)

featurewise_center=True,
                featurewise_std_normalization=True,
                
kf = KFold(n_splits=5)
skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)

auxilary function for getting model name in each of the k iterations
def get_model_name(k):
    return 'model_'+str+'.h5'
'''
#datagen_train = ImageDataGenerator(horizontal_flip=True)
datagen_train = ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                shear_range=0.3,
                zoom_range=0.3,
                horizontal_flip=True)
datagen_validation = ImageDataGenerator(horizontal_flip=False)

#load images from training directory
train_generator = datagen_train.flow_from_directory("train/",
                                                    target_size=(img_size,img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

validation_generator = datagen_train.flow_from_directory("test/",
                                                    target_size=(img_size,img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

test_generator = datagen_train.flow_from_directory("evaluation/",
                                                    target_size=(img_size,img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

# ###  Create CNN Model

# Initialising the CNN
model = Sequential()

model.add(Conv2D(192,(3,3), padding='same', input_shape=(img_size, img_size,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(192,(1,1)))
model.add(Activation('relu'))

model.add(Conv2D(192,(3,3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(96,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(96,(1,1)))
model.add(Conv2D(96,(1,1)))

model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
model.add(Dropout(0.3))

model.add(Conv2D(32,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

# Fully connected layer 2nd layer
model.add(Dense(512, name = 'convy'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

'''
model.add(Conv2D(3,(1,1), name = 'convy'))
model.add(GlobalAveragePooling2D())
'''
model.add(Dense(3, activation='softmax'))

model.get_layer('convy').kernel_regularizer = regularizers.l2(0.0001) #0.0001
opt = Adam(lr=0.0001)#0.0005
#opt = SGD(lr=0.01, decay=0, momentum=0.01)#decay=1e-6, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ### Visualize Model Architecture

plot_model(model, to_file='model_murshed2019.png', show_shapes=True, show_layer_names=True)
#Image('model_serv.png',width=400, height=200)

tic = time.time()

# ### Train and Evaluate Model

epochs = 300
steps_per_epoch = train_generator.n//train_generator.batch_size #number of samples/batch size
#n number of samples in train_generator // (floor division) by batch_size
validation_steps = validation_generator.n//validation_generator.batch_size

#make checkpoint and save it when it reach the best accuracy during the training
checkpoint = ModelCheckpoint("weights_murshed2019.h5", monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=1)#make check point during training and save the weight

#learning rate schedule: reduce learning rate 0.1 every 2 epochs 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.00001, mode='auto')
#early stopping
#early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

#use livelossplot to monitor the accuracy during the training 
#callbacks = [PlotLossesCallback(), checkpoint, reduce_lr, early_stopping] 
callbacks = [PlotLossesCallback(), checkpoint, reduce_lr] 

history = model.fit(
    x=train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data = validation_generator,
    validation_steps = validation_steps,
    callbacks=callbacks
)

# print duration
toc = time.time()

tictoc = 1000*(toc-tic)
tictocC = str(round(tictoc,3))

plt.savefig('Accuracy_murshed2019.png')
 
  
# ### Represent Model as JSON String

model_json = model.to_json()
with open("model_murshed2019.json", "w") as json_file:
    json_file.write(model_json)




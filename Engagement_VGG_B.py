#import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import os
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout,Flatten, Conv2D, GlobalAveragePooling2D#, Input
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Sequential#, Model, 
from tensorflow.keras.optimizers import Adam#, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau#, EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers

#from IPython.display import SVG, Image
from livelossplot.tf_keras import PlotLossesCallback
import tensorflow as tf
print("Tensorflow version:", tf.__version__)


# ### Plot Sample Images

#samples = utils.datasets.fer.plot_example_images(plt).show()

for expression in os.listdir("train/"):
    if not expression.startswith('.'):  
        print(str(len(os.listdir("train/" + expression))) + " " + expression + " images")



# ###  Generate Training and Validation Batches
img_size = 48 #image size same as input
batch_size = 500 #can change to see if the it gives better result

#data Augmentation
'''
datagen_train = ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)
datagen_validation = ImageDataGenerator(horizontal_flip=True)
'''
#datagen_train = ImageDataGenerator(horizontal_flip=True)
datagen_validation = ImageDataGenerator(horizontal_flip=False)

datagen_train = ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                shear_range=0.3,
                zoom_range=0.3,
                horizontal_flip=True)

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

# ###  Create CNN Model

# Initialising the CNN
model = Sequential()

## 4 conv net layers, 3 dense layers (2 fully connected, 1 softmax) 
# 1 - Convolution
model.add(Conv2D(64,(3,3), padding='same', input_shape=(img_size, img_size,1)))
model.add(Conv2D(64,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #downsampling/ shrink the height and width dimension by factor of 2  
#model.add(Dropout(0.5))

# 2nd Convolution layer
model.add(Conv2D(128,(3,3), padding='same'))
model.add(Conv2D(128,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.3))

# 3rd Convolution layer
model.add(Conv2D(256,(3,3), padding='same'))
model.add(Conv2D(256,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))

# 4th Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.3))

# 5th Convolution layer
model.add(Conv2D(512,(3,3), padding='same'))
model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))


# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(4096))#256
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.25))

# Fully connected layer 2nd layer
model.add(Dense(4096))#4096
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.25))

# Fully connected layer 2nd layer
model.add(Dense(1000, name = 'convy'))#512
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.25))
'''
model.add(Conv2D(3,(1,1), name = 'convy'))
model.add(GlobalAveragePooling2D())
'''

model.add(Dense(3, activation='softmax'))

model.get_layer('convy').kernel_regularizer = regularizers.l1(0.0001) #0.0001
opt = Adam(lr=0.0001)
#opt=SGD(lr=0.02, momentum = 0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ### Visualize Model Architecture

plot_model(model, to_file='model_vggB.png', show_shapes=True, show_layer_names=True)
#Image('model_serv.png',width=400, height=200)

tic = time.time()

# ### Train and Evaluate Model

epochs = 300
steps_per_epoch = train_generator.n//train_generator.batch_size #number of samples/batch size
#n number of samples in train_generator // (floor division) by batch_size
validation_steps = validation_generator.n//validation_generator.batch_size

#make checkpoint and save it when it reach the best accuracy during the training
checkpoint = ModelCheckpoint("weights_vggB.h5", monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=1)#make check point during training and save the weight

#learning rate schedule: reduce learning rate 0.1 every 2 epochs 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.00001, mode='auto')

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

print("Training and Evaluation:" + tictocC+" ms")


plt.savefig('Accuracy_vggB.png')
 
  
# ### Represent Model as JSON String

model_json = model.to_json()
with open("model_vggB.json", "w") as json_file:
    json_file.write(model_json)




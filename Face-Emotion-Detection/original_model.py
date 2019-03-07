# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 13:55:11 2018

@author: Xuan Liu
"""
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

img_width, img_height = 350, 350

train_data_dir = 'E:/Graduate Study Material/ML/Project/training'
validation_data_dir = 'E:/Graduate Study Material/ML/Project/validation'
predict_data_dir = 'E:/Graduate Study Material/ML/Project/test'

nb_train_samples = 11085
nb_validation_samples = 1233
nb_predict_samples = 1380
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

# Build models
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
model.add(Dense(8))
model.add(Activation('sigmoid'))

# Configure learning
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Augmentation configuration for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Augmentation configuration for testing
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Building train, validation, predict generator to train, validate and predict
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    seed=1111)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    color_mode='grayscale',
    batch_size=batch_size,
    seed=1111)

predict_generator = test_datagen.flow_from_directory(
    predict_data_dir,
    target_size=(img_width, img_height),
    class_mode=None,
    seed=1111)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.evaluate_generator(
        validation_generator, 
        steps=nb_validation_samples // batch_size)

predict=model.predict_generator(
        predict_generator,
        steps=1)

# Save predicted results
np.savetxt("predict.csv", predict, delimiter=",")

# Save the model in a HDF5 file 'model.h5'
model.save('model.h5')

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import TensorBoard
import pathlib
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import cv2
import torch
import random as rn

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_height = 100
img_width = 100
channel = 3

train = ImageDataGenerator(rescale = 1/255,
                           shear_range = 0.1,
                           zoom_range = 0.1,
                           validation_split = 0.2,
                           rotation_range = 10,
                           width_shift_range = 0.1,
                           height_shift_range = 0.1,
                           horizontal_flip = True
                           )
validation = ImageDataGenerator(rescale = 1/255,
                                shear_range = 0.1,
                                zoom_range = 0.1,
                                validation_split = 0.2,
                                rotation_range = 10,
                                width_shift_range = 0.1,
                                height_shift_range = 0.1,
                                horizontal_flip = True
                                )
train_dataset = train.flow_from_directory('C:/Users/angel/Downloads/AER850-Fall23-main (1)/AER850-Fall23-main/Project2Data/Data/Train',
                                          target_size = (img_width, img_height),
                                          batch_size = 32,
                                          class_mode = 'categorical',
                                          subset='training')
validation_dataset = validation.flow_from_directory('C:/Users/angel/Downloads/AER850-Fall23-main (1)/AER850-Fall23-main/Project2Data/Data/Validation',
                                          target_size = (img_width, img_height),
                                          batch_size = 32,
                                          class_mode = 'categorical',
                                          subset='validation')

train_dataset.class_indices
train_dataset.classes
root = pathlib.Path('C:/Users/angel/Downloads/AER850-Fall23-main (1)/AER850-Fall23-main/Project2Data/Data/Train')
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
print (classes)


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters = 32,kernel_size = 3, strides=(1,1),activation = 'relu', input_shape =(100, 100, 3)),
                                    tf.keras.layers.Conv2D(filters = 32,kernel_size = 3, strides=(1,1),activation = 'relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Dropout(0.05),
                                    tf.keras.layers.Conv2D(filters = 64,kernel_size = 3, strides=(1,1),activation = 'relu'), 
                                    tf.keras.layers.Conv2D(filters = 64,kernel_size = 3, strides=(1,1),activation = 'relu'), 
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Dropout(0.1),
                                    tf.keras.layers.Conv2D(filters = 128,kernel_size = 3, strides=(1,1),activation = 'relu'), 
                                    tf.keras.layers.Conv2D(filters = 128,kernel_size = 3, strides=(1,1),activation = 'relu'), 
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Dropout(0.15),
                                    tf.keras.layers.Conv2D(filters = 256,kernel_size = 3, strides=(1,1),activation = 'relu'), 
                                    tf.keras.layers.Conv2D(filters = 256,kernel_size = 3, strides=(1,1),activation = 'relu'), 
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Dropout(0.20),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512,activation = 'relu'),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(256,activation = 'relu'),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(4,activation = 'softmax'),
                                    ])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model_fit = model.fit(train_dataset,
                      validation_data = validation_dataset,
                      epochs = 10,
                      validation_split=0.33
                      )
loss = model_fit.history['loss']    
val_loss = model_fit.history['val_loss'] 
epochs = range(1, len(loss)+1)                               
plt.plot(epochs, loss, 'y', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = model_fit.history['accuracy']    
val_acc = model_fit.history['val_accuracy']                         
plt.plot(epochs, acc, 'y', label = 'Training acc')
plt.plot(epochs, val_acc, 'r', label = 'Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save('crack_classifier')
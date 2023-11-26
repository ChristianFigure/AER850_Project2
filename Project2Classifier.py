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
new_model = tf.keras.models.load_model('crack_classifier')
img_height = 100
img_width = 100
channel = 3



# dir_path = 'C:/Users/angel/Downloads/AER850-Fall23-main (1)/AER850-Fall23-main/Project2Data/Data/Test/Small'
# for i in os.listdir(dir_path):
#     img = cv2.imread(dir_path+'//'+i)
#     img = cv2.resize(img,(100,100))
#     X = image.img_to_array(img)
#     X = np.expand_dims(X/255, axis = 0)
#     images = np.vstack([X])
#     val = new_model.predict(images)
#     print(val)
#     val = np.transpose(val)
    
#     val_0 = ''.join(map(str,val[0]))
#     val_1 = ''.join(map(str,val[1]))
#     val_2 = ''.join(map(str,val[2]))
#     val_3 = ''.join(map(str,val[3])) 
#     cv2.putText(img,'Large:'+val_0, (0,10), cv2.FONT_ITALIC , 0.3, (0,255,0), 1)
#     cv2.putText(img,'Medium:'+val_1, (0,20), cv2.FONT_ITALIC , 0.3, (0,255,0), 1)
#     cv2.putText(img,'None:'+val_2, (0,30), cv2.FONT_ITALIC , 0.3, (0,255,0), 1)
#     cv2.putText(img,'Small:'+val_3, (0,40), cv2.FONT_ITALIC , 0.3, (0,255,0), 1)
#     plt.imshow(img)
#     plt.show()
    

img = cv2.imread(r'C:\Users\angel\Downloads\AER850-Fall23-main (1)\AER850-Fall23-main\Project2Data\Data\Test\Medium\Crack__20180419_06_19_09,915.bmp')
img = cv2.resize(img,(img_width, img_height))
X = image.img_to_array(img)
X = np.expand_dims(X/255, axis = 0)
images = np.vstack([X])
val = new_model.predict(images)
print(val)
val = np.transpose(val)
val_0 = ''.join(map(str,val[0]*100))
val_1 = ''.join(map(str,val[1]*100))
val_2 = ''.join(map(str,val[2]*100))
val_3 = ''.join(map(str,val[3]*100)) 
cv2.putText(img,'Large:'+val_0, (0,10), cv2.FONT_ITALIC , 0.3, (0,255,0), 1)
cv2.putText(img,'Medium:'+val_1, (0,20), cv2.FONT_ITALIC , 0.3, (0,255,0), 1)
cv2.putText(img,'None:'+val_2, (0,30), cv2.FONT_ITALIC , 0.3, (0,255,0), 1)
cv2.putText(img,'Small:'+val_3, (0,40), cv2.FONT_ITALIC , 0.3, (0,255,0), 1)
plt.imshow(img)
plt.show()

img = cv2.imread(r'C:\Users\angel\Downloads\AER850-Fall23-main (1)\AER850-Fall23-main\Project2Data\Data\Test\Large\Crack__20180419_13_29_14,846.bmp')
img = cv2.resize(img,(img_width, img_height))
X = image.img_to_array(img)
X = np.expand_dims(X/255, axis = 0)
images = np.vstack([X])
val = new_model.predict(images)
print(val)
val = np.transpose(val)
val_0 = ''.join(map(str,val[0]*100))
val_1 = ''.join(map(str,val[1]*100))
val_2 = ''.join(map(str,val[2]*100))
val_3 = ''.join(map(str,val[3]*100)) 
cv2.putText(img,'Large:'+val_0, (0,10), cv2.FONT_ITALIC , 0.3, (0,255,0), 1)
cv2.putText(img,'Medium:'+val_1, (0,20), cv2.FONT_ITALIC , 0.3, (0,255,0), 1)
cv2.putText(img,'None:'+val_2, (0,30), cv2.FONT_ITALIC , 0.3, (0,255,0), 1)
cv2.putText(img,'Small:'+val_3, (0,40), cv2.FONT_ITALIC , 0.3, (0,255,0), 1)
plt.imshow(img)
plt.show()





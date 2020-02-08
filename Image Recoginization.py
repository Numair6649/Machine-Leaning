import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.datasets import fashion_mnist
(train_image,train_label),(test_image,test_label) = fashion_mnist.load_data()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.imshow(train_image[0])
plt.colorbar()
plt.show()

class_names = ['T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandals','Shirts','Sneakers',
               'bags','Ankle Boot']

import matplotlib

plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_image[i],cmap = matplotlib.cm.binary)
    plt.xlabel(class_names[train_label[i]])
plt.show()    
    
train_image = train_image.reshape((60000, 784))
test_image = test_image.reshape((10000, 784))
                                  
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(criterion = 'entropy',max_depth = 9)
dtf.fit(train_image,train_label)

dtf.score(train_image,train_label)
dtf.score(test_image,test_label)







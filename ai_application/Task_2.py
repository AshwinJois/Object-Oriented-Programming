# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:53:29 2021

@author: ashwi
"""

import tensorflow as tf
import sys
sys.path.append('G:/ICS_Sylabus/Object_Oriented_Programming/repo_ai_application')
from Task_1 import Dataset_Generation
from Task_1 import Network_layers

mnist = tf.keras.datasets.mnist # 28x28 images of handwritten digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()


dataset_mnist = Dataset_Generation(x_train,y_train, x_test, y_test,10)

Network_mnist = Network_layers(x_train,y_train, x_test, y_test,10)

Network_mnist.layers()
Network_mnist.compile_fit_model()

"""
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('C:/Users/ashwi/OneDrive/Desktop/Travel March 2021/models', 
                                          compile=True)

prediction = model.predict([x_test])
print(prediction)

print(np.argmax(prediction[1000]))
plt.imshow(x_test[1000])
"""

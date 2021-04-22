# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:53:29 2021

@author: ashwi
"""

import tensorflow as tf
import sys
sys.path.append('enter the path of the file Task_1')
from Task_1 import Dataset_Generation
from Task_1 import Network_layers

mnist = tf.keras.datasets.mnist # 28x28 images of handwritten digits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()

dataset_mnist = Dataset_Generation(x_train,y_train, x_test, y_test,10) # 10 refers to the last layer computing softmax probabilities for 10 classes
Network_mnist = Network_layers(x_train,y_train, x_test, y_test,10)

Network_mnist.layers()
Network_mnist.compile_fit_model()



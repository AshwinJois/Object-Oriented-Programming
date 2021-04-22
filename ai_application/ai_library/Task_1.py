# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 12:38:26 2021

@author: ashwi
"""
import tensorflow as tf
import datetime, os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

class Dataset_Generation:
    
    def __init__(self,x,y,a,b,output):
        self.x = x # train_dataset
        self.y = y # validation_dataset
        self.a = a # Used for MNIST
        self.b = b # Used for MNIST 
        self.output = output # Output classes of custom generated data is 3 = Circle, Rectangle, Triangle
                             # Output classes of MNIST is 10 = 0 to 9    
     
class Network_layers(Dataset_Generation):  
    
    def layers(self):        
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) 
        self.model.add(tf.keras.layers.Dense(self.output, activation=tf.nn.softmax))  
        
    def compile_fit_model(self):
        self.model.compile(loss = 'sparse_categorical_crossentropy',
                      optimizer = 'adam',
                      metrics = ['accuracy']) 
        
        logdir = os.path.join("G:/ICS_Sylabus/Object_Oriented_Programming/tensorboard", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)              
        
        callbacks = [tensorboard_callback,
            tf.keras.callbacks.ModelCheckpoint(
                filepath='C:/Users/ashwi/OneDrive/Desktop/Travel March 2021/models',
                save_best_only=True,  # Saves model if `loss` has improved.
                monitor="loss",
                verbose=1,
            )
        ]
               
        try:        
            hist = self.model.fit(self.x, validation_data = self.y, epochs=29, callbacks=callbacks, shuffle=True) # For Custom Generated Data
        
        except:   
            hist = self.model.fit(self.x, self.y, validation_data = (self.a, self.b),epochs = 30, callbacks=callbacks, shuffle=True) # For MNIST dataset
        
        hist.history        

        f = open('C:/Users/ashwi/OneDrive/Desktop/Travel March 2021/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        return


train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)            
train_dataset = train.flow_from_directory('G:/My Projects/computer-vision projects/basedata/train/', target_size = (64,64), batch_size = 3, class_mode = 'binary')
validation_dataset = validation.flow_from_directory('G:/My Projects/computer-vision projects/basedata/validation/', target_size = (64,64), batch_size = 3, class_mode = 'binary')

train_dataset.class_indices

datset = Dataset_Generation(train_dataset,validation_dataset,None,None,3) # 3 refers to the last layer computing softmax probabilities for 3 classes
Architecture = Network_layers(train_dataset,validation_dataset,None,None,3)
Architecture.layers()
Architecture.compile_fit_model()




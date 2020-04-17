import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, Flatten, Dropout, Activation, MaxPool2D
from tensorflow.keras import Model
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
import time
import copy
import os


class DQNModel(Model):
    def __init__(self, layer_num1 = 32, layer_num2 = 5):
        super(DQNModel, self).__init__()
        self.d1 = Dense(layer_num1, kernel_regularizer=tf.keras.regularizers.l1())
        self.d2 = Dense(layer_num2, kernel_regularizer=tf.keras.regularizers.l1())        

    def call(self, x):
        y = self.d1(x)
        y = self.d2(y)
        return y

class MyNet:
    def __init__(self, layer_num1 = 32, layer_num2 = 5, checkpoint_path = "training_1/cp.ckpt"):
        self.history = None
        self.acc = None
        self.val_acc = None
        self.loss = None
        self.val_loss = None

        self.model = DQNModel(layer_num1 = 32, layer_num2 = 5)
        self.model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.05),\
            loss = 'mse', \
            metrics = ['accuracy'])

        # 先不使用断点续训
        #加入断点续训
        self.checkpoint_save_path = checkpoint_path
        self.checkpoint_save_dir = os.path.dirname(checkpoint_path)
        if os.path.exists(self.checkpoint_save_path + '.index'):
            print('---------------load weights----------------')
            self.model.load_weights(self.checkpoint_save_path)

        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = self.checkpoint_save_path,
            save_weights_only = True,
            save_best_only = True,
            verbose = 1
        )
        

    def __call__(self, x):
        return self.model.predict(x)    #可否换成其他

    def fit(self, x, y, batch_size = 32, epochs = 5, validation_split = 0.2, validation_freq = 20):
        
        self.history = self.model.fit(x, y, batch_size, epochs, verbose=0, \
            validation_split = validation_split, validation_freq= validation_freq, callbacks = [self.cp_callback])
        #self.model.summary()

        self.acc = self.history.history['accuracy']
        self.val_acc = self.history.history['val_accuracy']
        self.loss = self.history.history['loss']
        self.val_loss = self.history.history['val_loss']

        return np.mean(np.array(self.loss))

    def clone(self):
        '''返回当前模型的深度拷贝对象
        '''
        new_net = MyNet(layer_num2 = 5)
        #new_net.load_weights(new_net.checkpoint_save_path)
        #self.model.save_weights('./weights/behaviour_w')
        #new_net.model.load_weights('./weights/behaviour_w')
        new_net.history = self.history
        new_net.acc = self.acc
        new_net.val_acc = self.val_acc
        new_net.loss = self.loss
        new_net.val_loss = self.val_loss


        return new_net

    def draw(self):
        #绘制loss和acc图线
        '''
        self.acc = self.history.history['sparse_categorical_accuracy']
        self.val_acc = self.history.history['val_sparse_categorical_accuracy']
        self.loss = self.history.history['loss']
        self.val_loss = self.history.history['val_loss']
        '''

        plt.subplot(1, 2, 1)
        plt.plot(self.acc, label = 'Training Accuracy')
        plt.plot(self.val_acc, label = 'Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.loss, label = 'Training Loss')
        plt.plot(self.val_loss, label = 'Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()
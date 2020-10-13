import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, ReLU, ELU, LeakyReLU, Flatten, Dense, AveragePooling2D

import numpy as np
np.random.seed(666)
tf.random.set_seed(666)

options = {
    'a_filter_size': [(3,3), (5,5), (7,7), (13,13)],
    'a_include_BN': [True, False],
    'a_output_channels': [32, 64, 128, 256],
    'activation_type': [ReLU, ELU, LeakyReLU],
    'b_filter_size': [(3,3), (5,5), (7,7), (13,13)],
    'b_include_BN': [True, False],
    'b_output_channels': [32, 64, 128, 256],
    'include_pool': [True, False],
    'pool_type': [MaxPool2D, AveragePooling2D]
    }

class Organism:
    def __init__(self, chromosome={}, phase=0):
      self.phase = phase
      self.chromosome = chromosome
    
    def build_model(self):
      if self.phase == 0:
        keras.backend.clear_session()
        
        inputs = Input(shape=(32,32,3))
        x = Conv2D(filters=self.chromosome['a_output_channels'],
                    kernel_size=self.chromosome['a_filter_size'],
                    use_bias=self.chromosome['a_include_BN'])(inputs)
        if self.chromosome['a_include_BN']:
            x = BatchNormalization()(x)
        x = self.chromosome['activation_type']()(x)
        if self.chromosome['include_pool']:
            x = self.chromosome['pool_type']()(x)

        x = Conv2D(filters=self.chromosome['b_output_channels'],
                    kernel_size=self.chromosome['b_filter_size'],
                    use_bias=self.chromosome['b_include_BN'])(x)
        if self.chromosome['b_include_BN']:
            x = BatchNormalization()(x)
        x = self.chromosome['activation_type']()(x)

        x = Flatten()(x)
        x = Dense(10, activation='softmax')(x)

        self.model = tf.keras.Model(inputs=[inputs], outputs=[x])
        self.model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
      else:
        print('Phase under construction')
    
    def fitnessFunction(self, train_ds, test_ds):
      self.model.fit(train_ds,
                      epochs=5,
                      verbose=1)
      _, self.fitness = self.model.evaluate(test_ds, verbose=0)
    
    def crossover(self, partner):
      child = Organism(chromosome={}, phase=0)
      endpoint = np.random.randint(low=0, high=len(self.chromosome))
      print(endpoint)
      for idx, key in enumerate(self.chromosome):
        if idx <= endpoint:
          child.chromosome[key] = self.chromosome[key]
        else:
          child.chromosome[key] = partner.chromosome[key]
      return child
    
    def mutation(self, mutationRate):
      for idx, key in enumerate(self.chromosome):
        if np.random.rand() <= mutationRate:
          self.chromosome[key] = options[key][np.random.randint(len(options[key]))]
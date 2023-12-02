import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

kernel_size = 3

class Network():
    
    def build(self):
        # kernal_size = 3
        inputs = layers.Input(shape=(32,32,3)) 
        
        x = self.conv_block(inputs, num_filters=16, kernel_size=kernel_size)
        x = self.conv_block(x, num_filters=32, kernel_size=kernel_size)
        x = self.conv_block(x, num_filters=64, kernel_size=kernel_size)
        x = self.conv_block(x, num_filters=128, kernel_size=kernel_size) 
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(100, activation="softmax")(x)
        
        model = Model(inputs=inputs, outputs=outputs, name="CNN")
        return model
    
    def conv_block(self, block_input, num_filters, kernel_size, strides=1  , padding="same"):
        x = block_input 
        
        x = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding=padding, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x) 
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        return x
    
        

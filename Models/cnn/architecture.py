import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

kernal_size = 3

class Network():
    
    def build(self):
        # kernal_size = 3
        inputs = layers.Input(shape=(32,32,3)) 
        
        
        x = layers.Conv2D(64, 3, activation='relu', kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(l=0.01), padding='same')(inputs)
        x = self.conv_block(x, num_filters=64, kernel_size=3)
        
        x = layers.Conv2D(128, 2, activation='relu', padding='same')(x)
        x = layers.Conv2D(128, 2, activation='relu', padding='same')(x)     
        x = self.conv_block(x, num_filters=128, kernel_size=2)
        

        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = self.conv_block(x, num_filters=256, kernel_size=3)
        
        
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
    
        

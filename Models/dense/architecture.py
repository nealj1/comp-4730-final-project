import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, BatchNormalization, Dropout, UpSampling2D




class Network():

    def build(self):
        inputs = layers.Input(shape=(32,32,3)) 
        x = layers.Flatten()(inputs)
        x = layers.Dense(512, activation='relu')(x)
        #x = Dropout(0.25)(x)
        x = BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        #x = Dropout(0.25)(x)
        x = BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        #x = Dropout(0.25)(x)
        x = BatchNormalization()(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(100, activation="softmax")(x)
        
        model = Model(inputs=inputs, outputs=outputs, name="Dense")
        
        return model

n = Network()
m = n.build()

m.summary()

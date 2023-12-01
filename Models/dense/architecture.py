import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model




class Network():

    def build(self):
        inputs = layers.Input(shape=(32,32,3)) 
        # x = layers.Flatten()(inputs)
        x = layers.Dense(16)(inputs)
        x = layers.Dense(32)(x)
        x = layers.Dense(64)(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(100, activation="softmax")(x)
        
        model = Model(inputs=inputs, outputs=outputs, name="Dense")
        
        return model

n = Network()
m = n.build()

m.summary()

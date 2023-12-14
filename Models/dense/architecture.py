import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, BatchNormalization, Dropout, UpSampling2D

class Network():

    def build(self):
        # Input layer: expects input shape of 32x32 pixels with 3 color channels (RGB)
        inputs = layers.Input(shape=(32,32,3))
        
        # Dropout parameter
        dropout = 0.5 
        
        # Flatten layer: converts the 2D input images into 1D arrays.
        x = layers.Flatten()(inputs)
         # First Dense layer: fully connected layer with 512 neurons and ReLU activation.
        x = layers.Dense(512, activation='relu')(x)
        # Dropout layer
        x = Dropout(dropout)(x)
        # Batch normalization layer
        x = BatchNormalization()(x)
        # Second Dense layer
        x = layers.Dense(256, activation='relu')(x)
        # Dropout layer
        x = Dropout(dropout)(x)
        # Batch normalization layer
        x = BatchNormalization()(x)
        # Third Dense layer
        x = layers.Dense(128, activation='relu')(x)
        # Dropout layer
        x = Dropout(dropout)(x)
        # Batch normalization layer
        x = BatchNormalization()(x)
        # Flatten layer
        x = layers.Flatten()(x)
        # Output layer
        outputs = layers.Dense(100, activation="softmax")(x)
        
        # Creating the model: defining its input and output layers.
        model = Model(inputs=inputs, outputs=outputs, name="Dense")
        
        # The method returns the constructed model.
        return model

# Creating an instance of the Network class.
n = Network()

# Building the model using the build method.
m = n.build()

# Printing the summary of the model.
m.summary()

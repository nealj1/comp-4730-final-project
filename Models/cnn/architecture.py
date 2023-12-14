import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# Kernel size
kernel_size = 3

class Network():
    
    def build(self):
        # Input layer: expects input shape of 32x32 pixels with 3 color channels (RGB)
        inputs = layers.Input(shape=(32,32,3))
         
        # Sequentially adding convolutional blocks with increasing number of filters
        conv_layer_output = self.conv_block(inputs, num_filters=16, kernel_size=kernel_size)
        conv_layer_output = self.conv_block(conv_layer_output, num_filters=32, kernel_size=kernel_size)
        conv_layer_output = self.conv_block(conv_layer_output, num_filters=64, kernel_size=kernel_size)
        conv_layer_output = self.conv_block(conv_layer_output, num_filters=128, kernel_size=kernel_size)
        
        # Flatten layer: converts the 2D output of the last conv_block into a 1D array 
        conv_layer_output = layers.Flatten()(conv_layer_output)
        # Dense layer: fully connected layer with 512 neurons and ReLU activation
        conv_layer_output = layers.Dense(512, activation='relu')(conv_layer_output)
        # Batch normalization layer: normalizes the activations from the previous layer
        conv_layer_output = layers.BatchNormalization()(conv_layer_output)
        # Dropout layer
        conv_layer_output = layers.Dropout(0.3)(conv_layer_output)
        # Output layer
        outputs = layers.Dense(100, activation="softmax")(conv_layer_output)
        
        # Creating the model: defining its input and output layers
        model = Model(inputs=inputs, outputs=outputs, name="CNN")
        
        # The method returns the constructed model.
        return model
    
    # Function to create a convolutional block, which is a common pattern in CNNs
    def conv_block(self, block_input, num_filters, kernel_size, strides=1  , padding="same"):
        layer_out = block_input 
        # Conv2D layer: applies convolution operation with specified filters, kernel size, strides, and padding
        layer_out = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding=padding, activation="relu")(layer_out)
        # MaxPooling2D layer: reduces the spatial dimensions (height and width) of the input volume
        layer_out = layers.MaxPooling2D(pool_size=(2, 2))(layer_out)
        # Batch normalization layer 
        layer_out = layers.BatchNormalization()(layer_out)
        # Dropout layer
        layer_out = layers.Dropout(0.4)(layer_out)
        # returns a transformed version of its input after applying a series of layers
        return layer_out
    
        

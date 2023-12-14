import tensorflow as tf
from tensorflow.keras.applications import ResNet50 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, BatchNormalization, Dropout, UpSampling2D
from keras.models import Model

class Network():
    
    def build(self, num_unfreeze_layers=0):
        # Define the input layer with the shape of your data
        input_layer = tf.keras.Input(shape=(32, 32, 3))
        
        # Dropout parameter
        dropout = 0.5

        # Initialize the ResNet50 model pre-trained on ImageNet, excluding the top (final fully connected) layers
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Initially setting all layers of the base model to non-trainable
        for layer in base_model.layers:
            layer.trainable = False
        
        # Make BatchNormalization layers trainable
        for layer in base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
                
        # Upsampling: Increase the spatial dimensions of the input to match those expected by ResNet50
        x = UpSampling2D(size=(7, 7), interpolation='bilinear')(input_layer)

        # Feed the upsampled input into the base model
        x =  base_model(x)
        
        # Global Average Pooling: Convert the 3D feature maps to 1D feature vectors
        x = GlobalAveragePooling2D()(x)
        
        # Dropout layer: sets parameter above
        x = Dropout(dropout)(x)
        
         # Dense layer: fully connected layer with 256 neurons and ReLU activation
        x = Dense(256, activation='relu')(x)
        
        # Batch normalization layer
        x = BatchNormalization()(x)
        
        # Output layer: Dense layer with 100 neurons, using softmax activation for multi-class classification
        output = Dense(100, activation='softmax')(x)
        
        # Constructing the model by defining its input and output layers
        model = Model(inputs=input_layer, outputs=output)
        
        # The method returns the constructed model.
        return model
        

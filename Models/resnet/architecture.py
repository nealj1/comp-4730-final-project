import tensorflow as tf
from tensorflow.keras.applications import ResNet50 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, BatchNormalization, Dropout, UpSampling2D
from keras.models import Model

class Network():
    
    def build(self, num_unfreeze_layers=0):
        # Define the input layer with the shape of your data
        input_layer = tf.keras.Input(shape=(32, 32, 3))

        # Initialize the ResNet50 model
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        for layer in base_model.layers:
            layer.trainable = False
            
        for layer in base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
                
        
        x = UpSampling2D(size=(7, 7), interpolation='bilinear')(input_layer)

        x =  base_model(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.50)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        output = Dense(100, activation='softmax')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        
        return model
        

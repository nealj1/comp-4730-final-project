import tensorflow as tf
from keras.applications import ResNet50 
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


class Network():
    
    def build(self,  num_unfreeze_layers=0):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32,32,3))
        for layer in base_model.layers:
            layer.trainable = False
        
        
        if num_unfreeze_layers > 0:
            for layer in base_model.layers[-num_unfreeze_layers:]:
                layer.trainable = True        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)
        output = Dense(100, activation="softmax")(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        
        return model
    
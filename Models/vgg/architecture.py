import tensorflow as tf
from tensorflow import keras

from keras.applications.vgg16 import VGG16
from keras import layers, Model

class Network():
    
    def build(self, freeze_base_model=True):
        base_model = VGG16(include_top=False, input_shape=(32, 32, 3))
        if freeze_base_model:
            for layer in base_model.layers:
                layer.trainable = False                
        x = layers.Flatten()(base_model.output)
        x = layers.Dense(512)(x)

        output = layers.Dense(100, activation='softmax')(x)
        
        model = Model(inputs=base_model.inputs, outputs=output, name="VGG16-CIFAR100" )
        return model
    
    
n = Network()

model = n.build()
model.summary()
        
        
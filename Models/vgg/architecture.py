import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

class Network():

    def __init__(self, input_shape=(32, 32, 3), num_classes=100, weight_decay=0.01):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weight_decay = weight_decay
    
    def build(self):
        inputs = layers.Input(shape=self.input_shape)
        x = inputs

        kernel_size = 3

        filter_sizes = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

        for i, filters in enumerate(filter_sizes):
            x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu', 
                              kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            
            x = layers.BatchNormalization()(x)

            if i % 2 == 1:
                x = layers.Dropout(0.4 if filters < 512 else 0.5)(x)

            if filters in [64, 128, 256, 512] and (i+1) < len(filter_sizes) and filter_sizes[i+1] != filters:
                x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name="Custom_VGG")
        
        # The method returns the constructed model.
        return model
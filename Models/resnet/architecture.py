import tensorflow as tf
from tensorflow.keras import layers, Model

class Network():
    
    def residual_block(self, x, filters, kernel_size=3, stride=1):
        shortcut = x
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)

        if stride != 1 or shortcut.shape[-1] != filters:
            # Apply a 1x1 convolution to the shortcut if the shapes do not match
            shortcut = layers.Conv2D(filters, 1, strides=stride)(shortcut)

        x = layers.add([x, shortcut])  # Element-wise sum
        x = layers.ReLU()(x)
        return x
    
    def build(self):
        inputs = layers.Input(shape=(32, 32, 3))
        x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

        num_filters = 64
        for i, num_blocks in enumerate([3, 4, 5 ,6]):
            for _ in range(num_blocks):
                x = self.residual_block(x, num_filters)
            num_filters *= 2

        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(100, activation='softmax')(x)

        model = Model(inputs, outputs)
        return model
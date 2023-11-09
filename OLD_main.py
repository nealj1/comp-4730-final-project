import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import tensorflow as tf

class AiCnnModel:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.num_classes = 10
        self.batch_size = 32
        self.model = None

    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

    def preprocess_data(self):
        self.y_train = tf.keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, self.num_classes)
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

    def build_model(self, layers):
        self.model = Sequential(layers)
        self.model.summary()

    def compile_and_train(self, optimizer, epochs):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=epochs,
            validation_data=(self.x_test, self.y_test),
            shuffle=True
        )

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

model1_layers = [
    # First Convolutional Layer: 32 filters, 3x3 size, 'same' padding, input shape based on x_train
    Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]),
    # ReLU activation for non-linearity
    Activation('relu'),
    # Second Convolutional Layer: 32 filters, 3x3 size
    Conv2D(32, (3, 3)),
    # ReLU activation
    Activation('relu'),
    # Max-pooling with 2x2 pool size
    MaxPooling2D(pool_size=(2, 2)),
    # Dropout layer with a rate of 0.25 for regularization
    Dropout(0.25),
    # Flatten the feature maps for the fully connected layers
    Flatten(),
    # First Fully Connected Layer: 256 neurons
    Dense(256),
    # ReLU activation
    Activation('relu'),
    # Another dropout layer with a rate of 0.5
    Dropout(0.5),
    # Output Layer: 10 neurons for 10 classes
    Dense(10),
    
    Activation('softmax')  # Softmax activation for multiclass classification
]


model2_layers = [
    Conv2D(32, (3, 3), padding='same', strides=(2, 2), input_shape=x_train.shape[1:]),
    Activation('relu'),
    Conv2D(32, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.5),
    Conv2D(64, (3, 3), padding='same'),
    Activation('relu'),
    Conv2D(64, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(512),
    Activation('relu'),
    Dropout(0.5),
    Dense(10),
    Activation('softmax')
]

model3_layers = [
    Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]),
    Activation('relu'),
    Conv2D(32, (3, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), padding='same'),
    Activation('relu'),
    Conv2D(64, (3, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(512),
    Dense(10),
    Activation('softmax')
]

model1 = AiCnnModel(x_train, y_train, x_test, y_test)
model1.load_data()
model1.preprocess_data()
model1.build_model(model1_layers)
model1.compile_and_train(tf.keras.optimizers.RMSprop(learning_rate=0.0005), epochs=5)

'''
model2 = AiCnnModel(x_train, y_train, x_test, y_test)
model2.load_data()
model2.preprocess_data()
model2.build_model(model2_layers)
model2.compile_and_train(tf.keras.optimizers.RMSprop(lr=0.0005), epochs=15)

model3 = AiCnnModel(x_train, y_train, x_test, y_test)
model3.load_data()
model3.preprocess_data()
model3.build_model(model3_layers)
model3.compile_and_train(tf.keras.optimizers.RMSprop(lr=0.0005), epochs=15)
'''
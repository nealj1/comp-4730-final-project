import tensorflow as tf
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout

class AiModel:
    def __init__(self, num_classes=100, batch_size=32):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.model = None

    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar100.load_data()
        print(f'{self.y_train[0]}')

    def preprocess_data(self):
        self.y_train = tf.keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, self.num_classes)
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

    def build_model(self, layers):
        self.model = Sequential(layers)
        print(self.model.summary())

    def compile_and_train(self, optimizer, epochs):
        self.model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'] )

        modelInformation = self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, 
                       epochs=epochs,validation_data=(self.x_test, self.y_test), shuffle=True )
        return(modelInformation.history)

if __name__ == '__main__':
    cifar100_model = AiModel(num_classes=100, batch_size=32)
    cifar100_model.load_data()
    cifar100_model.preprocess_data()
    epochs = 2

    model_1 = [
        # The first convolutional layer with 32 filters, a kernel size of (3, 3), 'same' padding
        Conv2D(32, (3, 3), padding='same', input_shape=cifar100_model.x_train.shape[1:]),
        # Applies the Rectified Linear Unit (ReLU) activation function after the first convolutional layer.
        Activation('relu'),
        # Second convolutional layer with 32 filters and a kernel size of (3, 3)
        Conv2D(32, (3, 3)),
        # Applies ReLU activation after the second convolutional layer.
        Activation('relu'),
        # Performs max pooling with a pool size of (2, 2) to downsample the spatial dimensions.
        MaxPooling2D(pool_size=(2, 2)),
        # Introduces dropout with a rate of 0.25, which helps prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.
        Dropout(0.25),
        # Flattens the input, converting it into a one-dimensional array before feeding it into the dense layers.
        Flatten(),
        # Fully connected layer with 512 units.
        Dense(512),
        # Applies ReLU activation after the first dense layer.
        Activation('relu'),
        # Introduces dropout with a rate of 0.5 after the first dense layer.
        Dropout(0.5),
        # Fully connected layer with units equal to the number of classes in the CIFAR-100 dataset.
        Dense(cifar100_model.num_classes),
        # Applies the softmax activation function to produce class probabilities for multiclass classification.
        Activation('softmax')
    ]

    cifar100_model.build_model(model_1)
    cifar100_model.compile_and_train(tf.keras.optimizers.RMSprop(learning_rate=0.0005), epochs=epochs)


'''
    model_2 = [
        Conv2D(32, (3, 3), padding='same', input_shape=cifar100_model.x_train.shape[1:]),
        Activation('relu'),
        Conv2D(32, (3, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(512),
        Activation('relu'),
        Dropout(0.5),
        Dense(cifar100_model.num_classes),
        Activation('softmax')
    ]

    cifar100_model.build_model(model_2)
    cifar100_model.compile_and_train(tf.keras.optimizers.RMSprop(learning_rate=0.0005), epochs=epochs)

    model_3 = [
        Conv2D(32, (3, 3), padding='same', input_shape=cifar100_model.x_train.shape[1:]),
        Activation('relu'),
        Conv2D(32, (3, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(512),
        Activation('relu'),
        Dropout(0.5),
        Dense(cifar100_model.num_classes),
        Activation('softmax')
    ]

    cifar100_model.build_model(model_3)
    cifar100_model.compile_and_train(tf.keras.optimizers.RMSprop(learning_rate=0.0005), epochs=epochs)
'''

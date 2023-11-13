import tensorflow as tf
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout

class CIFARModel:
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
    cifar100_model = CIFARModel(num_classes=100, batch_size=32)
    cifar100_model.load_data()
    cifar100_model.preprocess_data()

    model_layers = [
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

    cifar100_model.build_model(model_layers)
    cifar100_model.compile_and_train(tf.keras.optimizers.RMSprop(learning_rate=0.0005), epochs=5)

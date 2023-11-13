import tensorflow as tf
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
import matplotlib.pyplot as plt

class CIFARModel:
    def __init__(self, num_fine_classes=100, num_coarse_classes=20, batch_size=32):
        self.num_fine_classes = num_fine_classes
        self.num_coarse_classes = num_coarse_classes
        self.batch_size = batch_size
        self.fine_model = None
        self.coarse_model = None

    def load_data(self):
        (self.x_train, self.y_train_fine), (self.x_test, self.y_test_fine) = cifar100.load_data(label_mode='fine')
        (_, self.y_train_coarse), (_, self.y_test_coarse) = cifar100.load_data(label_mode='coarse')

    def plot_history(history):
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def preprocess_data(self):
        # Preprocess fine-grained labels
        self.y_train_fine = tf.keras.utils.to_categorical(self.y_train_fine, self.num_fine_classes)
        self.y_test_fine = tf.keras.utils.to_categorical(self.y_test_fine, self.num_fine_classes)
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

        # Preprocess coarse-grained labels
        self.y_train_coarse = tf.keras.utils.to_categorical(self.y_train_coarse, self.num_coarse_classes)
        self.y_test_coarse = tf.keras.utils.to_categorical(self.y_test_coarse, self.num_coarse_classes)

    def build_fine_model(self, layers):
        self.fine_model = Sequential(layers)
        self.fine_model.summary()

    def build_coarse_model(self, layers):
        self.coarse_model = Sequential(layers)
        self.coarse_model.summary()

    def compile_and_train(self, model, x_train, y_train, x_test, y_test, optimizer, epochs):
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True
        )

if __name__ == '__main__':
    cifar100_model = CIFARModel(num_fine_classes=100, num_coarse_classes=20, batch_size=32)
    cifar100_model.load_data()
    cifar100_model.preprocess_data()

    # Define model layers for fine-grained classification
    fine_model_layers = [
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
        Dense(cifar100_model.num_fine_classes),
        Activation('softmax')
    ]

    # Define model layers for coarse-grained classification
    coarse_model_layers = [
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=cifar100_model.x_train.shape[1:]),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),

        Dense(cifar100_model.num_coarse_classes, activation='softmax')
    ]

    cifar100_model.build_fine_model(fine_model_layers)
    cifar100_model.build_coarse_model(coarse_model_layers)

    # You can train both models separately
    cifar100_model.compile_and_train(cifar100_model.fine_model, cifar100_model.x_train, cifar100_model.y_train_fine, cifar100_model.x_test, cifar100_model.y_test_fine, tf.keras.optimizers.RMSprop(learning_rate=0.0005), epochs=5)
    cifar100_model.compile_and_train(cifar100_model.coarse_model, cifar100_model.x_train, cifar100_model.y_train_coarse, cifar100_model.x_test, cifar100_model.y_test_coarse, tf.keras.optimizers.RMSprop(learning_rate=0.0005), epochs=5)

        # Train both models and store the history
    fine_history = cifar100_model.fine_model.fit(
        cifar100_model.x_train,
        cifar100_model.y_train_fine,
        batch_size=cifar100_model.batch_size,
        epochs=5,
        validation_data=(cifar100_model.x_test, cifar100_model.y_test_fine),
        shuffle=True
    )

    coarse_history = cifar100_model.coarse_model.fit(
        cifar100_model.x_train,
        cifar100_model.y_train_coarse,
        batch_size=cifar100_model.batch_size,
        epochs=5,
        validation_data=(cifar100_model.x_test, cifar100_model.y_test_coarse),
        shuffle=True
    )

    # Plot the training history for both models
    CIFARModel.plot_history(fine_history)
    CIFARModel.plot_history(coarse_history)



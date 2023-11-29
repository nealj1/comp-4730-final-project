import tensorflow as tf
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA

class CIFARModel:
    def __init__(self, num_classes=100, batch_size=32):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.model = None

    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar100.load_data()

    def preprocess_data(self):
        self.y_train = tf.keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, self.num_classes)
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

    def apply_pca(self, n_components):
        # Flatten the images for PCA
        x_train_flat = self.x_train.reshape((self.x_train.shape[0], -1))

        # Apply PCA
        pca = PCA(n_components=n_components)
        self.x_train_pca = pca.fit_transform(x_train_flat)

    def build_model(self, layers):
        self.model = Sequential(layers)
        self.model.summary()

    def compile_and_train(self, optimizer, epochs):
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        datagen.fit(self.x_train)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        self.model.fit(
            datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
            steps_per_epoch=len(self.x_train) // self.batch_size,
            epochs=epochs,
            validation_data=(self.x_test, self.y_test),
            shuffle=True
        )
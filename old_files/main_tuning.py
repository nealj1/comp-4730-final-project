import tensorflow as tf
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
import numpy as np

class CIFARModel:
    def __init__(self, num_classes=100, batch_size=32):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar100.load_data()

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

    def evaluate(self):
        return self.model.evaluate(self.x_test, self.y_test)

def grid_search():
    # Define a list of hyperparameter configurations to test
    hyperparameter_configs = [
    {"learning_rate": 0.01, "batch_size": 32, "num_layers": 3},
    {"learning_rate": 0.01, "batch_size": 64, "num_layers": 3},
    {"learning_rate": 0.01, "batch_size": 128, "num_layers": 3},
    {"learning_rate": 0.01, "batch_size": 32, "num_layers": 4},
    {"learning_rate": 0.01, "batch_size": 64, "num_layers": 4},
    {"learning_rate": 0.01, "batch_size": 128, "num_layers": 4},
    
    {"learning_rate": 0.001, "batch_size": 32, "num_layers": 3},
    {"learning_rate": 0.001, "batch_size": 64, "num_layers": 3},
    {"learning_rate": 0.001, "batch_size": 128, "num_layers": 3},
    {"learning_rate": 0.001, "batch_size": 32, "num_layers": 4},
    {"learning_rate": 0.001, "batch_size": 64, "num_layers": 4},
    {"learning_rate": 0.001, "batch_size": 128, "num_layers": 4},
    
    {"learning_rate": 0.0001, "batch_size": 32, "num_layers": 3},
    {"learning_rate": 0.0001, "batch_size": 64, "num_layers": 3},
    {"learning_rate": 0.0001, "batch_size": 128, "num_layers": 3},
    {"learning_rate": 0.0001, "batch_size": 32, "num_layers": 4},
    {"learning_rate": 0.0001, "batch_size": 64, "num_layers": 4},
    {"learning_rate": 0.0001, "batch_size": 128, "num_layers": 4},
        # Add more configurations here
    ]

    results = []

    for config in hyperparameter_configs:
        model = CIFARModel(num_classes=100, batch_size=config["batch_size"])
        model.load_data()
        model.preprocess_data()
        
        model_layers = [
            Conv2D(32, (3, 3), padding='same', input_shape=model.x_train.shape[1:]),
            Activation('relu'),
            Conv2D(32, (3, 3)),
            Activation('relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(512),
            Activation('relu'),
            Dropout(0.5),
            Dense(model.num_classes),
            Activation('softmax')
        ]
        
        model.build_model(model_layers)
        
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=config["learning_rate"])
        model.compile_and_train(optimizer, epochs=1)
        
        # Evaluate the model and store the results
        evaluation_result = model.evaluate()
        results.append({"config": config, "evaluation_result": evaluation_result})

    # After testing all configurations, analyze and compare the results
    for result in results:
        config = result["config"]
        evaluation_result = result["evaluation_result"]
        print(f"Configuration: {config}, Evaluation Result: {evaluation_result}")

if __name__ == '__main__':
    grid_search()

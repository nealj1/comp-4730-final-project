from optparse import OptionParser
import os
import traceback

parser = OptionParser()
parser.add_option("-m", dest="model_name")  # architecture to choose
parser.add_option("-s", dest="session")  # new session to create directories and save model

(options, args) = parser.parse_args()

if options.model_name is not None:
    model_name = options.model_name
else:
    print("Error: missing model name. supply with -m to select a model")

model_save_path = ""
if options.session is not None:
    session = int(options.session)
    model_save_path = f"train_{model_name}_Session_{session}"
    os.makedirs(model_save_path, exist_ok=True)
else:
    print("Error: missing session name. supply with -s for session number (int)")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import importlib

from tensorflow import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator


def preprocess_data(x_train, y_train, x_test, y_test):
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    return x_train, y_train, x_test, y_test


try:
    Network = getattr(importlib.import_module("Models." + model_name + ".architecture"), "Network")
except ModuleNotFoundError:
    traceback.print_exc()
    print("Model name does not exist.")
    exit()

network = Network()
model = network.build()

model.summary()
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)
datagen = ImageDataGenerator(
    # horizontal_flip=True,
    # vertical_flip=True,
    # shear_range=0.2,
    # brightness_range=[0.5, 1.5],
    # rotation_range=10,
    # zoom_range=0.2
)
early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
# Create augmented data generator
train_generator = datagen.flow(x_train, y_train, batch_size=64)
testing_generator = datagen.flow(x_test, y_test, batch_size=64)


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics='accuracy')

history = model.fit(x_train, y_train, epochs=50, validation_data=testing_generator, batch_size=128, shuffle=True, callbacks=[early_stopping])

training_loss = history.history["loss"]
testing_loss = history.history["val_loss"]
training_accuracy = history.history["accuracy"]
testing_accuracy = history.history["val_accuracy"]

epochs = range(1, len(training_loss) + 1)

# Plot loss curves
plt.subplot(3, 1, 1)

plt.plot(epochs, training_loss, 'b.-', label='Training Loss')
plt.plot(epochs, testing_loss, 'r.-', label='Testing Loss')
plt.title('Plot Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3, 1, 3)
plt.ylim(top=1.2)
plt.plot(epochs, training_accuracy, 'b.-', label='Training accuracy')
plt.plot(epochs, testing_accuracy, 'r.-', label='Testing accuracy')
plt.title('Plot Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Save the figure
plt.savefig(os.path.join(model_save_path, 'losscurves.png'))

# Show the plot
plt.show()
from optparse import OptionParser
import os
import traceback


parser = OptionParser()
parser.add_option("-m", dest="model_name") #architecture to choose 
parser.add_option("-s", dest="session") #new session to create directories and save model 

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
    # y_train = tf.keras.utils.to_categorical(y_train, 100)
    # y_test = tf.keras.utils.to_categorical(y_test, 100)
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
model = network.build(10)

model.summary()

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test)
# datagen = ImageDataGenerator(
#     horizontal_flip=True,
#     # zoom_range=0.2
# )

# Create augmented data generator
# train_generator = datagen.flow(x_train, y_train, batch_size=512)


model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.003, momentum=0.9), metrics='accuracy')

model.fit(x_train, y_train , epochs=10, validation_data=(x_test, y_test), batch_size=512)
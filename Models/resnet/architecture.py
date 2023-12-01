import tensorflow as tf
from tensorflow.keras.applications import ResNet50 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, BatchNormalization, Dropout, UpSampling2D
from keras.models import Model

class Network():
    
    def build(self, num_unfreeze_layers=10):
        # Define the input layer with the shape of your data
        input_layer = tf.keras.Input(shape=(32, 32, 3))

        # Apply upsampling to match the expected input size of ResNet50
        x = UpSampling2D(size=(7, 7))(input_layer)

        # Initialize the ResNet50 model
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x)

        # Set the trainable status of the layers
        for layer in base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False

        if num_unfreeze_layers > 0:
            for layer in base_model.layers[-num_unfreeze_layers:]:
                layer.trainable = True

        # Apply the base model and continue building the model
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = BatchNormalization()(x)
        output = Dense(100, activation='softmax')(x)  # Adjust the number of classes as needed

        model = Model(inputs=input_layer, outputs=output)

        return model

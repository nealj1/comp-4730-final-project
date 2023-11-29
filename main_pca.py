'''
https://www.kaggle.com/code/adtysregita/pca-application-using-cifar10-dataset
https://towardsdatascience.com/pca-in-a-single-line-of-code-ed79ae42059b

'''

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from keras.datasets import cifar100
import seaborn as sns
import matplotlib.pyplot as plt

class_labels = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", 
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", 
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", 
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", 
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", 
    "house", "kangaroo", "computer_keyboard", "lamp", "lawn_mower", "leopard", 
    "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", 
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", 
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", 
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", 
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", 
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", 
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", 
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

superclass_labels = [
    "Aquatic mammals", "Fish", "Flowers", "Food containers", "Fruit and vegetables",
    "Household electrical devices", "Household furniture", "Insects", "Large carnivores",
    "Large man-made outdoor things", "Large natural outdoor scenes",
    "Large omnivores and herbivores", "Medium-sized mammals", "Non-insect invertebrates",
    "People", "Reptiles", "Small mammals", "Trees", "Vehicles 1", "Vehicles 2"
]


# Model configuration
img_width, img_height, img_num_channels = 32, 32, 3


# Load CIFAR-100 data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()


#print the shape of training, testing, and label data
print('Training Data Shape: ', x_train.shape)
print('Testing Data Shape: ', x_test.shape)

print('Label Training Data Shape: ', y_train.shape)
print('Label Testing Data Shape: ', y_test.shape)



#find out total number of labels and classes
classes = np.unique(y_train)
nClasses = len(classes)
print('Number of Outputs: ', nClasses)
print('Number of Output Classes: ', classes)


# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

# Parse numbers as floats
input_train = x_train.astype('float32')
input_test = x_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255


x_train = x_train/255.0
print(np.min(x_train), np.max(x_train))
x_train.shape

#flatten images
x_train_flat = x_train.reshape(-1,3072)
feat_cols = ['pixel' + str(i) for i in range(x_train_flat.shape[1])]
df_cifar = pd.DataFrame(x_train_flat, columns = feat_cols)
df_cifar['Label'] = y_train
print('Size of Data Frame: {}'.format(df_cifar.shape))

print(df_cifar.head())

#check max and min values of dataset
print(np.min(x_train), np.max(x_train))

#normalize pixels between 0 and 1
x_train = x_train/255.0
np.min(x_train), np.max(x_train)
x_train.shape


#create PCA method
pca_cifar = PCA(n_components = 2)
principalComponents_cifar = pca_cifar.fit_transform(df_cifar.iloc[:, :-1])

#convert principal components
principal_cifar_Df = pd.DataFrame(data = principalComponents_cifar,
                                  columns = ['Principal Component 1', 'Principal Component 2'])
principal_cifar_Df['Label'] = y_train
principal_cifar_Df.head()


#plotting dataset into 2d graph
plt.figure(figsize = (10,7))
sns.scatterplot(
    x = "Principal Component 1", y = "Principal Component 2",
    hue = "Label",
    palette = sns.color_palette("Set2", as_cmap=True),
    data = principal_cifar_Df,
    legend = "full",
    alpha = 1.0
)



#variance of principal components
print('Explained Variation per Principal Component: {}'.format(pca_cifar.explained_variance_ratio_))

# Create a 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    principal_cifar_Df['Principal Component 1'],
    principal_cifar_Df['Principal Component 2'],
    y_train.flatten(),
    c=y_train.flatten(),
    cmap="Set2",
    marker='o',
    alpha=0.5
)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Class Label')
ax.set_title('3D Scatter Plot of CIFAR-100 with PCA')

plt.show()

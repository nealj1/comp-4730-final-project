import random
import matplotlib.pyplot as plt
from keras.datasets import cifar100
from collections import Counter


def get_class_distribution(labels, class_labels):
    # Ensure that labels are integers
    labels = labels.flatten().astype(int)
    
    # Map integer labels to class labels
    class_labels_mapped = [class_labels[label] for label in labels]

    return Counter(class_labels_mapped)

def plot_class_distribution(class_distribution, title, color):
    plt.figure(figsize=(12, 6))
    plt.bar(class_distribution.keys(), class_distribution.values(), color=color)
    plt.title(title)
    plt.xlabel('Class Label')
    plt.ylabel('Number of Samples')
    plt.show()

def plot_superclass_distribution(superclass_distribution, title, color):
    plt.figure(figsize=(12, 6))
    plt.bar(superclass_distribution.keys(), superclass_distribution.values(), color=color)
    plt.title(title)
    plt.xlabel('Superclass Label')
    plt.ylabel('Number of Samples')
    plt.show()

def get_superclass_distribution(labels, class_labels):
    # Ensure that labels are integers
    labels = labels.flatten().astype(int)
    
    # Map integer labels to superclass labels
    superclass_labels_mapped = [class_labels[label] for label in labels]

    return Counter(superclass_labels_mapped)


# Function to display images
def display_images(images, labels, class_labels, num_images=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.title(class_labels[labels[i][0]])
        plt.axis('off')
    plt.show()


# Function to display images for a specific class in a single plot
def display_images_for_classes(images, labels, class_labels, target_classes, num_images_per_class=5):
    plt.figure(figsize=(15, 15))
    subplot_count = 1

    for target_class in target_classes:
        indices = [i for i in range(len(labels)) if class_labels[labels[i][0]] == target_class]
        random_indices = random.sample(indices, min(num_images_per_class, len(indices)))

        for i, idx in enumerate(random_indices):
            plt.subplot(5, 5, subplot_count)
            plt.imshow(images[idx])
            plt.title(class_labels[labels[idx][0]])
            plt.axis('off')
            subplot_count += 1

    plt.tight_layout()
    plt.show()



(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# Class labels for reference
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


# Display the first 5 images
#display_images(x_test, y_test, class_labels, num_images=25)

# Display basic information about the dataset
print("Number of training samples:", len(x_train))
print("Number of testing samples:", len(x_test))
print("Image dimensions:", x_train[0].shape)
print("Number of classes:", len(class_labels))

# Display the first 5 images for a specific class (e.g., "dog")
random_classes = random.sample(class_labels, 5)
display_images_for_classes(x_test, y_test, class_labels, target_classes=random_classes, num_images_per_class=5)

# Display unique class labels
unique_labels = set(class_labels)
print("Unique class labels:", unique_labels)

# Display the number of unique classes
num_unique_classes = len(unique_labels)
print("Number of unique classes:", num_unique_classes)

'''
# Plot class distribution in the training set
class_distribution_train = get_class_distribution(y_train, class_labels)
plot_class_distribution(class_distribution_train, 'Class Distribution in the Training Set', 'blue')

# Plot class distribution in the testing set
class_distribution_test = get_class_distribution(y_test, class_labels)
plot_class_distribution(class_distribution_test, 'Class Distribution in the Testing Set', 'orange')


# Plot superclass distribution in the training set
superclass_distribution_train = get_superclass_distribution(y_train, superclass_labels)
plot_superclass_distribution(superclass_distribution_train, 'Superclass Distribution in the Training Set', 'green')

# Plot superclass distribution in the testing set
superclass_distribution_test = get_superclass_distribution(y_test, superclass_labels)
plot_superclass_distribution(superclass_distribution_test, 'Superclass Distribution in the Testing Set', 'red')
'''

# Print unique superclass values
unique_superclasses = set(superclass_labels)
num_unique_classes = len(unique_superclasses)
print("Unique Superclasses:",num_unique_classes, unique_superclasses)


from keras.datasets import cifar100

# Load CIFAR-100 dataset
(_, y_train_fine), (_, _) = cifar100.load_data(label_mode='fine')  # Fine-grained labels
(_, y_train_coarse), (_, _) = cifar100.load_data(label_mode='coarse')  # Coarse-grained labels

# Print the first 10 examples
print(y_train_coarse[:1])
'''
# Print examples with both fine-grained and coarse-grained labels
for i in range(len(y_train_fine)):
    fine_grained_label = y_train_fine[i][0]
    coarse_grained_label = y_train_coarse[i][0]
    
    fine_grained_class_name = class_labels[fine_grained_label]
    coarse_grained_superclass_name = superclass_labels[coarse_grained_label]
    print(f"Example {i + 1}: Fine-grained class: {fine_grained_class_name.ljust(20)} Superclass: {coarse_grained_superclass_name}")
'''



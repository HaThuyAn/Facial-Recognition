import os
from glob import glob
import random
from build_siamese_triplet_loss import *
from util import preprocess_triplets

IMG_WIDTH = 200
IMG_HEIGHT = 200

anchor_images_path = []
positive_images_path = []

# Dictionary to store images by class (person)
class_images = {}

# Loop through each class (person's folder)
for class_ in os.listdir("C:/Users/PC/Downloads/classification_data/train_data_lite"):
    # Get all images of that person
    class_imgs_path = glob(f"C:/Users/PC/Downloads/classification_data/train_data_lite/{class_}/*.jpg")

    # If the number of images is odd, remove the first image to make it even
    if len(class_imgs_path) % 2:
        class_imgs_path = class_imgs_path[1:]

    # Store images by class
    class_images[class_] = sorted(class_imgs_path)

    # Split the images into two halves
    n = len(class_imgs_path) // 2
    # First half goes to anchor list
    anchor_images_path.extend(class_imgs_path[:n])
    # Second half goes to positive list
    positive_images_path.extend(class_imgs_path[n:])


anchor_images = sorted(anchor_images_path)
positive_images = sorted(positive_images_path)

image_count = len(anchor_images)

# Create anchor and positive datasets
anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)

# Generate negative images ensuring they belong to a different class
negative_images = []

# Get all classes (persons) to select from
all_classes = list(class_images.keys())

for anchor, positive in zip(anchor_images, positive_images):
    # Extract the current person's class from the image path
    current_class = next((cls for cls in all_classes if cls in anchor), None)

    # Exclude the current class when picking a negative image
    other_classes = [cls for cls in all_classes if cls != current_class]

    # Randomly select a different class
    negative_class = random.choice(other_classes)
    negative_image = random.choice(class_images[negative_class])

    negative_images.append(negative_image)

# Convert the negative images list to a dataset
negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)

# Combine anchor, positive, and negative datasets
dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)

# Split into training and validation sets
train_dataset = dataset.take(round(image_count * 0.8))
val_dataset = dataset.skip(round(image_count * 0.8))

train_dataset = train_dataset.batch(32, drop_remainder=False).prefetch(8)
val_dataset = val_dataset.batch(32, drop_remainder=False).prefetch(8)

# steps_per_epoch = len(train_dataset)
# half_epoch_steps = steps_per_epoch // 32
siamese_model = build_siamese()
siamese_model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# siamese_model.build(input_shape=[
#     (None, IMG_WIDTH, IMG_HEIGHT, 3),  # Anchor input
#     (None, IMG_WIDTH, IMG_HEIGHT, 3),  # Positive input
#     (None, IMG_WIDTH, IMG_HEIGHT, 3)   # Negative input
# ])

dummy = tf.random.normal((1, IMG_WIDTH, IMG_HEIGHT, 3))
siamese_model([dummy, dummy, dummy])

siamese_model.save("C:/Users/PC/Downloads/Applied ML/FacialProject/metric_learning_triplet_loss_resnet50.keras")



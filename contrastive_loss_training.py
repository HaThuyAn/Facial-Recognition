import os
from glob import glob
import random
from build_siamese_contrastive_loss import *
from util import preprocess_pairs

IMG_WIDTH = 200
IMG_HEIGHT = 200

# Organize images by class (person)
class_images = {}

# Define the directory containing the images
base_dir = "C:/Users/PC/Downloads/classification_data/train_data_lite"

# Load all images grouped by class
for class_ in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_)
    # Get all images from the current class
    class_imgs_path = glob(os.path.join(class_path, "*.jpg"))
    if class_imgs_path:
        class_images[class_] = sorted(class_imgs_path) # Sorting to ensure consistent order

# Get a list of all class names
all_classes = list(class_images.keys())

similar_pairs = []
dissimilar_pairs = []

# Generate Similar Pairs (Positive Pairs)
for cls, imgs in class_images.items():
    num_images = len(imgs)
    similar_pairs.extend([(imgs[i], imgs[j], 1) for i in range(num_images) for j in range(i + 1, num_images)])

# Generate Dissimilar Pairs (Negative Pairs)
for cls, imgs in class_images.items():
    other_classes = [c for c in all_classes if c != cls]
    for img in imgs:
        # Randomly select a different class and its image
        negative_class = random.choice(other_classes)
        negative_img = random.choice(class_images[negative_class])
        dissimilar_pairs.append((img, negative_img, 0))

# Balance the number of similar and dissimilar pairs
similar_pairs = similar_pairs[:len(dissimilar_pairs)]

# Combine similar and dissimilar pairs and shuffle once
random.shuffle(similar_pairs)
random.shuffle(dissimilar_pairs)

half = 10000 // 2

similar_sample = similar_pairs[:half]
dissimilar_sample = dissimilar_pairs[:half]

all_pairs = similar_sample + dissimilar_sample
random.shuffle(all_pairs)

# Train-Validation Split
split_ratio = 0.8
split_index = int(len(all_pairs) * split_ratio)

# Split pairs
train_pairs = all_pairs[:split_index]
val_pairs = all_pairs[split_index:]

# Data Generator Function
def pair_generator(pairs):
    """Yields pairs of image paths and labels."""
    for img1, img2, label in pairs:
        yield img1, img2, label

# Create Dataset from Generator
def create_dataset(pairs):
    """Create a tf.data.Dataset from a generator."""
    dataset = tf.data.Dataset.from_generator(
        lambda: pair_generator(pairs),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),  # Image1 path
            tf.TensorSpec(shape=(), dtype=tf.string),  # Image2 path
            tf.TensorSpec(shape=(), dtype=tf.float32), # Label
        )
    )
    return dataset

# Create Training and Validation Datasets
train_dataset = create_dataset(train_pairs)
val_dataset = create_dataset(val_pairs)

train_dataset = (
    train_dataset
    .repeat()
    .map(preprocess_pairs, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    val_dataset
    .repeat()
    .map(preprocess_pairs, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

print(f"Total similar pairs: {len(similar_pairs)}")
print(f"Total dissimilar pairs: {len(dissimilar_pairs)}")
print(f"Total training pairs: {len(train_pairs)}")
print(f"Total validation pairs: {len(val_pairs)}")

steps_per_epoch = len(train_pairs) // 40
validation_steps = len(val_pairs) // 10
siamese_model = build_siamese()
siamese_model.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=10, validation_data=val_dataset, validation_steps=validation_steps)

# siamese_model.build(input_shape=[
#     (None, IMG_WIDTH, IMG_HEIGHT, 3),  # Anchor input
#     (None, IMG_WIDTH, IMG_HEIGHT, 3),  # Positive input
#     (None, IMG_WIDTH, IMG_HEIGHT, 3)   # Negative input
# ])

dummy = tf.random.normal((1, IMG_WIDTH, IMG_HEIGHT, 3))
siamese_model([dummy, dummy])

siamese_model.save("C:/Users/PC/Downloads/Applied ML/FacialProject/metric_learning_contrastive_loss_resnet50.keras")

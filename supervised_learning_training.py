import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

train_dataset_path = "C:/Users/PC/Downloads/classification_data/train_data_lite"
val_dataset_path = "C:/Users/PC/Downloads/classification_data/val_data_lite"
test_dataset_path = "C:/Users/PC/Downloads/classification_data/test_data_lite"

train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input) # Rescale pixel values to [-1, 1]
train_generator = train_datagen.flow_from_directory(train_dataset_path,
                                                   target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                   batch_size=BATCH_SIZE,
                                                   class_mode="categorical", # One‑hot labels
                                                   shuffle=True) # Shuffle so that the model will not be able to remember the order of the class

val_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input)
val_generator = val_datagen.flow_from_directory(val_dataset_path,
                                                 target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                 batch_size=BATCH_SIZE,
                                                 class_mode="categorical",
                                                 shuffle=True)

labels = {value: key for key, value in train_generator.class_indices.items()}

for key, value in labels.items():
    print(f"{key} : {value}")

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal"),
  tf.keras.layers.RandomRotation(0.2),
])

# Create the base model from the pre-trained model ResNet50 V2
base_model = tf.keras.applications.ResNet50V2(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

inputs = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.Flatten()(x) # Flatten the output of the base model
x = tf.keras.layers.Dense(512, activation='relu')(x) # Fully connected layer
x = tf.keras.layers.Dropout(0.5)(x) # Dropout to prevent overfitting
x = tf.keras.layers.Dense(256, activation='relu')(x) # Additional dense layer
x = tf.keras.layers.Dropout(0.3)(x) # Dropout layer
outputs = tf.keras.layers.Dense(200, activation='softmax')(x) # Output layer
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history_fine = model.fit(train_generator, epochs=10, validation_data=val_generator,
                        verbose=1)

model.save('supervised_learning_resnet50.keras')

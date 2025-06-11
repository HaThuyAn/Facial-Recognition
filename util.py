import os
import pickle

import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import resnet
from tensorflow.keras.models import load_model
from build_siamese_triplet_loss import SiameseModel, DistanceLayer
# from build_siamese_contrastive_loss import SiameseModel, DistanceLayer
from tensorflow.keras.applications import resnet_v2

IMG_WIDTH = 200
IMG_HEIGHT = 200


def get_button(window, text, color, command, fg='white'):
    button = tk.Button(
                        window,
                        text=text,
                        activebackground="black",
                        activeforeground="white",
                        fg=fg,
                        bg=color,
                        command=command,
                        height=2,
                        width=20,
                        font=('Helvetica bold', 20)
                    )

    return button


def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label


def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label


def get_entry_text(window):
    inputtxt = tk.Text(window,
                       height=2,
                       width=15, font=("Arial", 32))
    return inputtxt


def msg_box(title, description):
    messagebox.showinfo(title, description)


def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )


def preprocess_pairs(image1_path, image2_path, label):
    """Preprocess function for the dataset."""
    image1 = preprocess_image(image1_path)
    image2 = preprocess_image(image2_path)
    label = tf.cast(label, tf.float32)
    return (image1, image2), label


def euclidean_distance(embedding1, embedding2):
    euclidean_distance_score = np.linalg.norm(embedding1 - embedding2) # Lower is better
    euclidean_similarity_score = 1 / (1 + euclidean_distance_score) # Convert distance to similarity - Closer to 1 is better
    return euclidean_similarity_score


def recognize(img, db_path):
    siamese_model = load_model("C:/Users/PC/Downloads/Applied ML/FacialProject/metric_learning_triplet_loss_resnet50.keras",
                               custom_objects={
                                   "SiameseModel": SiameseModel,
                                   "DistanceLayer": DistanceLayer
                               },
                               compile=False)
    functional_model = siamese_model.get_layer("functional")
    embedding_model = functional_model.get_layer("Embedding")
    embedded_img = embedding_model(tf.expand_dims(resnet.preprocess_input(preprocess_image(img)), axis=0))

    # model = load_model('supervised_learning_resnet50.keras')
    # embedding_model = tf.keras.Model(
    #     inputs=model.input,
    #     outputs=model.get_layer(index=-2).output  # -2 to access the layer before softmax
    # )
    # image = tf.io.read_file(img)
    # image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.resize(image, (224, 224))
    # image = resnet_v2.preprocess_input(image)
    # image = tf.expand_dims(image, axis=0)
    # embedded_img = embedding_model.predict(image)[0]

    max_similarity = 0
    best_match = "Not found"

    for filename in os.listdir(db_path):
        file_path = os.path.join(db_path, filename)

        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Preprocess the database image
            embedded_db_img = embedding_model(tf.expand_dims(resnet.preprocess_input(preprocess_image(file_path)), axis=0))

            # db_image = tf.io.read_file(file_path)
            # db_image = tf.image.decode_jpeg(db_image, channels=3)
            # db_image = tf.image.resize(db_image, (224, 224))
            # db_image = resnet_v2.preprocess_input(db_image)
            # db_image = tf.expand_dims(db_image, axis=0)
            # embedded_db_img = embedding_model.predict(db_image)[0]

            # Calculate similarity
            similarity = euclidean_distance(embedded_img, embedded_db_img)

            # Update best match if similarity is higher
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = filename

            print(similarity)

    # Return "Not found" if the highest similarity is below the threshold
    if max_similarity < 0.26:
        return "Not found"
    return best_match


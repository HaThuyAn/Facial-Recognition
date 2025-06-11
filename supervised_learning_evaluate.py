import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import resnet_v2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, classification_report
from util import euclidean_distance

IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

test_dataset_path = "C:/Users/PC/Downloads/classification_data/test_data_lite"
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input)
test_generator = test_datagen.flow_from_directory(test_dataset_path,
                                                target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                batch_size=BATCH_SIZE,
                                                class_mode="categorical",
                                                shuffle=False)

model = load_model('supervised_learning_resnet50.keras')
embedding_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer(index=-2).output # -2 to access the layer before softmax
)

y_true = test_generator.classes
predictions_fine = model.predict(test_generator)
y_pred_fine = np.argmax(predictions_fine, axis=1)
accuracy = accuracy_score(y_true, y_pred_fine)

print(f"Test Accuracy: {accuracy:.4f}") # 0.5425
print(f1_score(y_true, y_pred_fine, average="macro")) # 0.5061
print(precision_score(y_true, y_pred_fine, average="macro")) # 0.5328
print(recall_score(y_true, y_pred_fine, average="macro")) # 0.5425


def load_verification_data(file_path):
    pairs = []
    with open(file_path, 'r') as file:
        for line in file:
            img1, img2, label = line.strip().split()
            img1 = os.path.join(base_path, img1.split('/')[-1])
            img2 = os.path.join(base_path, img2.split('/')[-1])
            label = int(label)
            pairs.append((img1, img2, label))
    return pairs


base_path = "C:/Users/PC/Downloads/verification_data"

verification_pairs = load_verification_data("C:/Users/PC/Downloads/verification_pairs_val.txt")

for pair in verification_pairs:
    print(pair)


def preprocess_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = resnet_v2.preprocess_input(image)
    return image


def embedding(image_path):
    img = preprocess_image(image_path)
    img = tf.expand_dims(img, axis=0)
    embedding = embedding_model.predict(img)
    return embedding[0]


y_true = []  # True labels (1 for same person, 0 for different)
y_scores_distance = []  # Distance scores

stop_point = len(verification_pairs) // 4

for img1, img2, label in verification_pairs[0:stop_point]:
    embedding1 = embedding(img1)
    embedding2 = embedding(img2)

    euclidean_dist = euclidean_distance(embedding1, embedding2)
    print(f"Euclidean: {euclidean_dist}")

    # Store results
    y_true.append(label)  # Actual match (1) or mismatch (0)
    y_scores_distance.append(euclidean_dist)

# Calculate ROC curve and AUC for euclidean distance
fpr_euclidean, tpr_euclidean, thresholds_euclidean = roc_curve(y_true, y_scores_distance)
roc_auc_euclidean = auc(fpr_euclidean, tpr_euclidean)

# Choose the best threshold
youden_j = tpr_euclidean - fpr_euclidean
optimal_idx = np.argmax(youden_j)
optimal_threshold = thresholds_euclidean[optimal_idx]
print(f'Optimal threshold: {optimal_threshold}') # 0.04435793570041878

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr_euclidean, tpr_euclidean, color='blue', label=f'ROC Curve (AUC = {roc_auc_euclidean:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig("roc_curve_supervised_learning.png", dpi=300, bbox_inches='tight')
plt.show()


import matplotlib.pyplot as plt
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import resnet
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model
from util import preprocess_image, euclidean_distance
from build_siamese_contrastive_loss import SiameseModel, DistanceLayer

siamese_model = load_model("C:/Users/PC/Downloads/Applied ML/FacialProject/metric_learning_contrastive_loss_resnet50.keras",
                           custom_objects={
                               "SiameseModel": SiameseModel,
                               "DistanceLayer": DistanceLayer
                           },
                           compile=False)
functional_model = siamese_model.get_layer("functional")
embedding_model = functional_model.get_layer("Embedding")


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

y_true = []  # True labels (1 for same person, 0 for different)
y_scores_distance = []  # Distance scores

midpoint = len(verification_pairs) // 4

for img1, img2, label in verification_pairs[0:midpoint]:
    embedding1 = embedding_model(tf.expand_dims(resnet.preprocess_input(preprocess_image(img1)), axis=0))
    embedding2 = embedding_model(tf.expand_dims(resnet.preprocess_input(preprocess_image(img2)), axis=0))

    euclidean_dist = euclidean_distance(embedding1, embedding2)
    print(f"Euclidean: {euclidean_dist}")

    # Store results
    y_true.append(label)
    y_scores_distance.append(euclidean_dist)

# Calculate ROC curve and AUC for euclidean distance
fpr_euclidean, tpr_euclidean, thresholds_euclidean = roc_curve(y_true, y_scores_distance)
roc_auc_euclidean = auc(fpr_euclidean, tpr_euclidean)

# Choose the best threshold
youden_j = tpr_euclidean - fpr_euclidean
optimal_idx = np.argmax(youden_j)
optimal_threshold = thresholds_euclidean[optimal_idx]
# Optimal threshold (triplet loss): 0.26618034207089164
# Optimal threshold (contrastive loss): 0.5699760706219016
print(f'Optimal threshold: {optimal_threshold}')

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr_euclidean, tpr_euclidean, color='blue', label=f'ROC Curve (AUC = {roc_auc_euclidean:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig("roc_curve_contrastive_loss.png", dpi=300, bbox_inches='tight')
plt.show()

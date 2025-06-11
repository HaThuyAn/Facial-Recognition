import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# Load data from the text file
data_file = "C:/Users/PC/Downloads/Applied ML/FacialProject/emotion_data.txt"
data = np.loadtxt(data_file)

# Split data into features (X) and labels (y)
X = data[:, :-1]  # Features are all columns except the last one
y = data[:, -1]   # Labels are the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=y)

# # Initialize the XGB Classifier
# xgb_model = XGBClassifier(
#     n_estimators=300,
#     max_depth=15,
#     learning_rate=0.1,
#     random_state=42
# )
#
# # Train the classifier on the training data
# xgb_model.fit(X_train, y_train)

# # Initialize the Random Forest Classifier
# rf_classifier = RandomForestClassifier(
#     n_estimators=500,
#     max_depth=20,
#     min_samples_split=15,
#     min_samples_leaf=5,
#     max_features='sqrt',
#     random_state=42,
#     verbose=1
# )
#
# # Train the classifier on the training data
# rf_classifier.fit(X_train, y_train)

mlp_model = MLPClassifier(hidden_layer_sizes=(256, 128),
                          activation='relu',
                          max_iter=500,
                          alpha=1e-4,
                          random_state=42)
mlp_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = mlp_model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(confusion_matrix(y_test, y_pred))

with open('./emotion_model', 'wb') as f:
    pickle.dump(mlp_model, f)


import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: Load and Prepare the Dataset

# Load the dataset (assuming TF-IDF features and labels are pre-saved in CSV files)
X_tfidf = pd.read_csv('C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 4/tfidf_features.csv')  # TF-IDF features
labels = pd.read_csv('C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 4/processed_sms_data.csv')['label']  # Labels

# Preprocess the labels: Map 'ham' -> 0, 'spam' -> 1
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# STEP 2: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# STEP 3: Build the Improved Neural Network Model
model = keras.Sequential([
    layers.InputLayer(input_shape=(X_train.shape[1],)),  # Input layer

    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),

    layers.Dense(1, activation='sigmoid')  # Output layer (binary classification)
])

# Compile the model with a lower learning rate for better convergence
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# STEP 4: Train the Model
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# STEP 5: Evaluate the Model on Test Data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# STEP 6: Make Predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# STEP 7: Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display Results
print('--- Improved Neural Network Results ---')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

# STEP 8: Plot Training History
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('Model_Accuracy.png', dpi=300)
plt.clf()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('Model_Loss.png', dpi=300)
plt.clf()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.title('Confusion Matrix - Improved Neural Network')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('Confusion_Matrix.png', dpi=300)
plt.show()

# Save Model
model.save("improved_sms_spam_classifier.keras")
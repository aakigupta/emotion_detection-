
import os  
from IPython.display import Audio, display
# Replace with the correct path to a valid audio file
file_path = "C:\\Users\\KIIT\\mini_project\\archive\Actor_01\\03-01-01-01-01-01-01.wav"
if os.path.isfile(file_path):
    print("File exists and is accessible.")
else:
    print("File does not exist or is not accessible.")
    
import sounddevice as sd
import librosa

file_path = "C:\\Users\\KIIT\\mini_project\\archive\Actor_01\\03-01-01-01-01-01-01.wav"
audio_path = file_path
audio_data, sample_rate = librosa.load(audio_path)

sd.play(audio_data, sample_rate)
sd.wait()  # Wait until playback is finished

# Display the audio file
display(Audio(filename=file_path))


import numpy as np
import librosa

#Extracts MFCC features from an audio file.    
def extract_mfcc(file_path, sr=22050, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=sr, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Replace with the path to your dataset
dataset_path = "C:\\Users\\KIIT\\mini_project\\archive"

mfcc_features = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.wav'):
            try:
                file_path = os.path.join(root, file)
                mfccs = extract_mfcc(file_path)
                file_class = int(file[7:8]) - 1  # Extracting class from the file name
                mfcc_features.append((mfccs, file_class))
            except ValueError as err:
                print(f"Error processing file {file}: {err}")
                continue

# Check if the extraction worked
print(f"Extracted {len(mfcc_features)} files")

import joblib

# Unzipping the list and converting to NumPy arrays
x, y = zip(*mfcc_features)
x, y = np.asarray(x), np.asarray(y)


# Define the directory path to save the arrays
SAVE_DIR_PATH = "C:\\Users\\KIIT\\mini_project\\array"
# Check if the directory exists; if not, create it
if not os.path.isdir(SAVE_DIR_PATH):
    os.makedirs(SAVE_DIR_PATH)

# Save the arrays to files using joblib
joblib.dump(x, os.path.join(SAVE_DIR_PATH, 'x.joblib'))
joblib.dump(y, os.path.join(SAVE_DIR_PATH, 'y.joblib'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Define the model
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # Define layers
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(32 * 98, 64)  # Adjusted for input shape (100 - 3 + 1 = 98)
        self.dense2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1d(x))
        x = self.dropout(x)
        x = self.flatten(x)
        x = torch.relu(self.dense1(x))
        x = self.dense2(x)
        return x

# Instantiate the model
model = CustomModel()

# Verify model parameters
print("Model Parameters:")
for name, param in model.named_parameters():
    print(name, param.shape)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Optimizer initialized successfully.")


# Input shape: (batch_size, sequence_length) -> Reshape to (batch_size, in_channels, sequence_length)
sequence_length = 100
x_train = torch.randn(1000, sequence_length)  # 1000 samples, each with 100 time steps
y_train = torch.randint(0, 10, (1000,))      # 1000 labels (0–9 for classification)
x_test = torch.randn(200, sequence_length)   # 200 test samples
y_test = torch.randint(0, 10, (200,))        # 200 test labels

# Reshape data to match Conv1d input requirements: (batch_size, in_channels, sequence_length)
x_train = x_train.unsqueeze(1)  # Shape: (1000, 1, 100)
x_test = x_test.unsqueeze(1)    # Shape: (200, 1, 100)

# Create datasets and dataloaders
train_dataset = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(x_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop
num_epochs = 10
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_dataloader)
    train_acc = 100 * correct / total
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)\
            
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(test_dataloader)
    val_acc = 100 * val_correct / val_total
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

# Plot the metrics
plt.plot(train_acc_history, label='Training Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 100])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()

plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

x_train = x_train.unsqueeze(1)  # Shape: (1000, 1, 100)
x_test = x_test.unsqueeze(1)    # Shape: (200, 1, 100)

from joblib import load
# Check if the directory exists; if not, create it
if not os.path.isdir(SAVE_DIR_PATH):
    os.makedirs(SAVE_DIR_PATH)

# Define the model path
model_path = os.path.join(SAVE_DIR_PATH, 'random_forest_model.joblib')

# Load the pre-trained Random Forest model
try:
    clf = load(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Training a new model...")
    
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay

# Load the saved features and labels
x = joblib.load(os.path.join(SAVE_DIR_PATH, 'x.joblib'))
y = joblib.load(os.path.join(SAVE_DIR_PATH, 'y.joblib'))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

# Evaluate the model
y_pred = clf.predict(x_test)

# Print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

from joblib import dump
# Save the trained model
dump(clf, model_path)
print(f"Model saved to {model_path}")


# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

import os
import numpy as np
from scipy.stats import zscore
import librosa
from sklearn.ensemble import RandomForestClassifier

from IPython.display import Audio, display
import matplotlib.pyplot as plt
from joblib import load

# RAVDESS Database Emotion Mapping
label_dict_ravdess = {
    '02': 'NEU',  # Neutral=1
    '03': 'HAP',  # Happy=2
    '04': 'SAD',  # Sad=3
    '05': 'ANG',  # Angry=4
    '06': 'FEA',  # Fearful=5
    '07': 'DIS',  # Disgust=6
    '08': 'SUR'   # Surprised=7
}

def detect_emotion(audio_file_path, model):
    """
    Detects the emotion of an audio file using a pre-trained model.
    """
    # Step 1: Extract MFCC features
    features = extract_mfcc(audio_file_path)
    
    # Reshape features for compatibility with the model
    features = features.reshape(1, -1)  # Shape: (1, feature_length)
    
    # Step 2: Predict the emotion
    prediction = model.predict(features)[0]  # Get the predicted class
    
    # Map the prediction to an emotion label
    emotion_labels = list(label_dict_ravdess.values())
    predicted_emotion = emotion_labels[prediction]
    
    return predicted_emotion


# Path to the audio file
audio_file_path = "C:\\Users\\KIIT\\mini_project\\archive\\Actor_01\\03-01-01-01-02-02-01.wav"

# Detect the emotion
predicted_emotion = detect_emotion(audio_file_path, clf)

# Load and normalize the audio signal
y, sample_rate = librosa.load(audio_file_path, sr=16000)
y = zscore(y)

# Plot the waveform of the audio file
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(y)) / float(sample_rate), y)
plt.xlim(0, len(y) / sample_rate)
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Amplitude (dB)', fontsize=16)
plt.title(f"Signal wave of file with detected emotion: {predicted_emotion}")
plt.show()

# Play the audio file
print(f"Playing audio file with detected emotion: {predicted_emotion}")

sd.play(audio_data, sample_rate)
sd.wait()
display(Audio(y, rate=sample_rate))

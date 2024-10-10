import os
import numpy as np
import librosa
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load Your Dataset
def load_data(data_folder):
    features = []
    labels = []

    # Iterate through each sub-folder in the dataset
    for label in os.listdir(data_folder):
        label_folder = os.path.join(data_folder, label)
        
        # Ensure it's a directory
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                if filename.endswith('.wav'):  # Change if your audio files have a different extension
                    file_path = os.path.join(label_folder, filename)
                    y, sr = librosa.load(file_path, sr=None)
                    
                    # Step 2: Extract Features (MFCCs in this case)
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    mfccs_mean = np.mean(mfccs, axis=1)  # Average over time
                    features.append(mfccs_mean)
                    labels.append(label)  # Assuming folder name is the label

    return np.array(features), np.array(labels)

# Step 3: Prepare Your Dataset
data_folder = r'D:\RISS\Tone Analysis\TESS Toronto emotional speech set data'  # Change this to your dataset path
X, y = load_data(data_folder)

# Step 4: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Model
model = svm.SVC(kernel='linear')  # You can choose a different kernel if needed
model.fit(X_train, y_train)

# Step 6: Make Predictions and Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Save the trained model
joblib.dump(model, 'tone_analysis_model.joblib')
print('Model saved as tone_analysis_model.joblib')

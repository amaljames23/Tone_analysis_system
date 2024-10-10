import numpy as np
import librosa
import joblib

# Step 1: Load the trained model
model = joblib.load('tone_analysis_model.joblib')

# Step 2: Define a function to extract features from a new audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCCs and compute the mean
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    return mfccs_mean.reshape(1, -1)  # Reshape for prediction

# Step 3: Load and process the new audio file for prediction
audio_file_path = r'D:\RISS\Tone Analysis\sad\OAF_back_sad.wav'  # Change this to your audio file path
features = extract_features(audio_file_path)

# Step 4: Make predictions
predicted_tone = model.predict(features)
print(f'The predicted tone is: {predicted_tone[0]}')

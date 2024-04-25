import os
import numpy as np
import pickle
import librosa
from keras.models import load_model
from Preprocess.data_preprocessing import remove_silence, extract_features_from_audio


model_type = load_model('Models/drums_vs_harmonic_model.h5')
model_drums = load_model('Models/model_drums.h5')
model_harmonic = load_model('Models/model_harmonic.h5')

instrument_type = ["drums", "harmonic"]
harmonic_instruments = [
    "Cello",
    "Clarinet",
    "Flute",
    "Acoustic Guitar",
    "Electric Guitar",
    "Organ",
    "Piano",
    "Saxophone",
    "Trumpet",
    "Violin",
    "Voice"]
drum_instruments = ["Cymbals", "Hi Hats", "Kicks", "Snares"]


def predict_instrument(file_path):
    file_path = remove_silence(file_path)
    audio_data, sr = librosa.load(file_path, sr=44100)

    if len(audio_data) == 0:
        os.remove(file_path)
        return "Empty"  # Zwraca "Empty" w przypadku pustego pliku

    if (predict_type(file_path) == 'harmonic'):
        dataset = harmonic_instruments
        model = model_harmonic
        with open('Scalers/scaler_harmonic_finall.pkl', 'rb') as file:
            scaler = pickle.load(file)
    else:
        dataset = drum_instruments
        model = model_drums
        with open('Scalers/scaler_drums_final.pkl', 'rb') as file:
            scaler = pickle.load(file)

    audio_data, sr = librosa.load(file_path, sr=44100)
    audio_data /= np.max(np.abs(audio_data))  # Normalize audio data

    desired_duration = 3

    if len(audio_data) > sr * desired_duration:
        segments = len(audio_data) // (sr * desired_duration)
        chunks = np.array_split(audio_data, segments)
    else:
        chunks = [audio_data]

    predicted_instruments = []

    for chunk in chunks:
        if len(chunk) > sr * desired_duration:
            chunk = chunk[:sr * desired_duration]
        else:
            chunk = np.pad(
                chunk, (0, sr * desired_duration - len(chunk)), 'constant')

        features = extract_features_from_audio(chunk, sr)
        features = features.flatten()
        features = features.reshape(1, -1)

        scaled_features = scaler.transform(features)
        predictions = model.predict(scaled_features)
        instrument_index = np.argmax(predictions)
        predicted_instrument = dataset[instrument_index]
        predicted_instruments.append(predicted_instrument)

    predicted_instrument = max(
        set(predicted_instruments),
        key=predicted_instruments.count)
    os.remove(file_path)
    return predicted_instrument


def predict_type(file_path):
    audio_data, sr = librosa.load(file_path, sr=44100)

    if len(audio_data) == 0:
        print("Empty audio file:", file_path)
        return None
    audio_data /= np.max(np.abs(audio_data))  # Normalize audio data

    desired_duration = 3
    with open('Scalers/scaler_all.pkl', 'rb') as file:
        scaler = pickle.load(file)

    if len(audio_data) > sr * desired_duration:
        segments = len(audio_data) // (sr * desired_duration)
        chunks = np.array_split(audio_data, segments)
    else:
        chunks = [audio_data]

    predicted_types = []

    for chunk in chunks:
        if len(chunk) > sr * desired_duration:
            chunk = chunk[:sr * desired_duration]
        else:
            chunk = np.pad(
                chunk, (0, sr * desired_duration - len(chunk)), 'constant')

        features = extract_features_from_audio(chunk, sr)
        features = features.flatten()
        features = features.reshape(1, -1)

        scaled_features = scaler.transform(features)
        predictions = model_type.predict(scaled_features)
        instrument_index = np.argmax(predictions)
        predicted_type = instrument_type[instrument_index]
        predicted_types.append(predicted_type)

    predicted_type = max(set(predicted_types), key=predicted_types.count)
    return predicted_type

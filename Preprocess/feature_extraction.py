import os
import librosa
from librosa import feature
import numpy as np
import pandas as pd


def extract_features_from_audio(audio_data, sr, target_duration=3):

    audio_duration = librosa.get_duration(y=audio_data, sr=sr)


    if audio_duration < target_duration:

        samples_needed = int((target_duration - audio_duration) * sr)
        audio_data = np.pad(audio_data, (0, samples_needed), mode='constant')


    elif audio_duration > target_duration:

        samples_to_trim = int((audio_duration - target_duration) * sr)
        audio_data = audio_data[:len(audio_data) - samples_to_trim]

    mfcc = feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    chroma = feature.chroma_stft(y=audio_data, sr=sr)
    centroid = feature.spectral_centroid(y=audio_data, sr=sr)
    bandwidth = feature.spectral_bandwidth(y=audio_data, sr=sr)
    contrast = feature.spectral_contrast(y=audio_data, sr=sr)
    rms = feature.rms(y=audio_data)
    zcr = feature.zero_crossing_rate(y=audio_data)


    all_features = np.vstack(
        [mfcc, chroma, centroid, bandwidth, contrast, rms, zcr])
    all_features_reshaped = all_features.reshape(
        all_features.shape[0], all_features.shape[1], 1)
    return all_features_reshaped


def create_feature_table(data_parent_path, instruments):
    records = []

    for instrument in instruments:
        data_path = os.path.join(data_parent_path, instrument)
        instrument_records = []

        for file in os.listdir(data_path):
            if file.endswith(".wav"):
                file_path = os.path.join(data_path, file)
                try:
                    audio_data, sr = librosa.load(file_path, sr=44100)
                    audio_data /= audio_data.max()  # normalizacja

                    # Extract features
                    instrument_records.append((audio_data, sr, instrument))
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")


        instrument_df = pd.DataFrame(
            data=instrument_records,
            columns=[
                "audio_data",
                "sampling_frequency",
                "instrument"])
        records.append(instrument_df)

    full_dataset = pd.concat(records, ignore_index=True)  # polaczenie
    return full_dataset


def features_to_array(features_df):
    features_list = []
    for index, row in features_df.iterrows():
        mfccs = row['mfcc']
        mfccs_flat = mfccs.flatten()
        features_list.append(mfccs_flat.tolist())
    return np.array(features_list)

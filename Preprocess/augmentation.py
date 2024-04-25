import numpy as np
import librosa
import pandas as pd


def augment_audio(audio_data, sr, augmentation_factor=2):

    augmented_data = []
    for _ in range(augmentation_factor):

        speed_change = np.random.uniform(0.8, 1.2)
        augmented_audio = librosa.effects.time_stretch(
            audio_data, rate=speed_change)
        pitch_change = np.random.randint(-3, 4)
        augmented_audio = librosa.effects.pitch_shift(
            augmented_audio, sr=sr, n_steps=pitch_change)

        augmented_data.append(augmented_audio)

    return augmented_data


def create_augmented_dataset(original_dataset, augmentation_factor=2):

    augmented_dataset = []
    for index, row in original_dataset.iterrows():
        audio_data = row["audio_data"]
        instrument = row["instrument"]

        augmented_audio_data = augment_audio(
            audio_data, 44100, augmentation_factor)

        for augmented_audio in augmented_audio_data:

            augmented_dataset.append(
                {"audio_data": augmented_audio, "instrument": instrument})

    augmented_df = pd.DataFrame(augmented_dataset)
    return augmented_df

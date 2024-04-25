from .feature_extraction import extract_features_from_audio
from pydub import AudioSegment


def remove_silence(audio_file_path):
    sound = AudioSegment.from_file(audio_file_path)
    non_silent = sound.strip_silence(silence_len=400, silence_thresh=-30)
    temp_file_path = "audio_without_silence.wav"
    non_silent.export(temp_file_path, format="wav")
    return temp_file_path


def preprocess_data(features_dataset, augmented_dataset, sr=44100):
    mfccs = []
    mfccs_augmented = []

    for data in features_dataset["audio_data"]:
        mfcc = extract_features_from_audio(data, sr)
        mfccs.append(mfcc)

    for data in augmented_dataset["audio_data"]:
        mfcc_augmented = extract_features_from_audio(data, sr)
        mfccs_augmented.append(mfcc_augmented)

    features_dataset["mfcc"] = mfccs
    augmented_dataset["mfcc"] = mfccs_augmented

    features_dataset['mfcc'] = [mfcc.flatten()
                                for mfcc in features_dataset['mfcc']]
    augmented_dataset['mfcc'] = [mfcc.flatten()
                                 for mfcc in augmented_dataset['mfcc']]

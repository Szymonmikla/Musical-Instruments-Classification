from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from Utils.build_model import create_model
import numpy as np


def preprocess_data(features_dataset, augmented_dataset):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(
        features_dataset["instrument"])

    x = np.array(features_dataset["mfcc"].to_list())
    x_train_augmented = np.array(augmented_dataset["mfcc"].to_list())
    y_train_augmented = label_encoder.fit_transform(
        augmented_dataset["instrument"])

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, encoded_labels, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=42)

    X_train_combined = np.vstack((x_train, x_train_augmented))
    y_train_combined = np.concatenate((y_train, y_train_augmented))

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train_combined)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    y_train = to_categorical(y_train_combined)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)

    return x_train_scaled, x_val_scaled, x_test_scaled, y_train, y_val, y_test


def build_and_train_model(x_train, y_train, x_val, y_val,instruments):

    model = create_model(input_shape=(x_train.shape[1],), output_shape=len(instruments))


    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    history = model.fit(
        x_train,
        y_train,
        epochs=30,
        batch_size=64,
        validation_data=(
            x_val,
            y_val),
        callbacks=[early_stopping])

    return model, history



from Interface.predict import predict_instrument
import os
import shutil


def sort_files(file_paths, destination_folder):
    sorted_files = []  # To store names of sorted files
    for file_path in file_paths:
        predicted_instrument = predict_instrument(file_path)
        if predicted_instrument == "Empty":
            # result_label.config(text="One or more files are empty.")
            return

        instrument_folder = os.path.join(
            destination_folder, predicted_instrument)
        os.makedirs(instrument_folder, exist_ok=True)

        filename = os.path.basename(file_path)
        destination_path = os.path.join(instrument_folder, filename)
        shutil.move(file_path, destination_path)

        # Add filename to the list of sorted files
        sorted_files.append(filename)

    # Display sorted filenames in the label

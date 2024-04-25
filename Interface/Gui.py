import tkinter as tk
from tkinter import filedialog
from Interface.sort_files import sort_files


def root():
    root_window = tk.Tk()
    root_window.title("Audio Stems Sort")


    root_window.grid_rowconfigure(0, weight=1)
    root_window.grid_rowconfigure(1, weight=1)
    root_window.grid_columnconfigure(0, weight=1)
    root_window.grid_columnconfigure(1, weight=1)

    browse_files_button = tk.Button(
        root_window,
        text="Browse Files",
        command=browse_files)
    browse_files_button.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    browse_destination_button = tk.Button(
        root_window,
        text="Select Destination Folder",
        command=browse_destination_folder)
    browse_destination_button.grid(
        row=0, column=1, padx=10, pady=10, sticky="nsew")

    window_width = 400
    window_height = 100


    center_window(root_window, window_width, window_height)

    return root_window


def browse_destination_folder():
    destination_folder = filedialog.askdirectory()
    return destination_folder


def browse_files():
    file_paths = filedialog.askopenfilenames(
        filetypes=[("Audio files", "*.wav")])
    if file_paths:
        destination_folder = browse_destination_folder()
        if destination_folder:
            sort_files(file_paths, destination_folder)


def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x_coordinate = (screen_width - width) // 2
    y_coordinate = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x_coordinate}+{y_coordinate}")

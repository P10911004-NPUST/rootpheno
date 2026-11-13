import sys
import time

import tkinter as tk
import numpy as np
import pandas as pd
import multiprocessing as mp

from pathlib import Path

from utils import *


use_cores = max(1, mp.cpu_count() - 1)


def get_folder_path() -> None:
    input_path = tk.filedialog.askdirectory()
    input_path = Path(input_path).expanduser().resolve()
    choose_folder_path.set(input_path)
    print(input_path)


def get_img_list():
    pass



if __name__ == "__main__":
    mp.freeze_support()
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # App's window
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    app = tk.Tk()
    app.iconbitmap(resource_path("icon.ico"))
    app.title("rootpheno")
    app.geometry("600x600")

    title_label = tk.Label(
        master=app, 
        text="Root Phenotyping", 
        font="Calibri 21 bold"
    ).pack(pady=5)

    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Folder path
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    choose_folder_path = tk.StringVar()
    choose_folder_frame = tk.Frame(master=app).pack(padx=20)
    choose_folder_button = tk.Button(
        master=choose_folder_frame,
        text="Select folder",
        font="Calibri 16",
        relief="ridge",
        command=get_folder_path
    ).pack(padx=20, pady=10, fill='x')

    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Images list
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    app.mainloop()
from tkinter import filedialog
import sys
import time

import tkinter as tk
import numpy as np
import pandas as pd
import multiprocessing as mp

from pathlib import Path


def resource_path(relative_path):
    ''' Get absolute path to resource (for PyInstaller)

        Used to get the icon path during runtime
    
        try:
           base_path = sys._MEIPASS
        except Exception:
           base_path = os.path.abspath(".")
           return os.path.join(base_path, relative_path)
    '''
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    return Path(__file__).parent / relative_path



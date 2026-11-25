import re
import sys

import numpy as np
import pandas as pd

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




# lst0 = ['a-1-1.tif', 'a-1-2.czi', 'b-1-1.czi', 'b-1-2.png', 'b-1-3.bmp', 'c-1-1.tiff']
# guess_grouping(lst0)
# csv = pd.read_csv("./tests/img/RO-index/czi/img_grouping.csv")
# csv['grouping'].isna().any()

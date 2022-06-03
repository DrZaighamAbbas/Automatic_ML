import pandas as pd
import numpy as np
import glob

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

def load_data(directory_path = None):
    all_files = glob.glob(directory_path + "/*.csv")

    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame

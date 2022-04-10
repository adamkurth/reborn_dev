import pandas as pd
import numpy as np
from reborn.viewers import pandaviews

d = {'col1': [1, 2]*10000, 'col2': [3, 4]*10000}

df = pd.DataFrame(data=d)

pandaviews.view_pandas_dataframe(df)

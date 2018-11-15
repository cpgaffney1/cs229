import pandas as pd
import numpy as np


def get_country_mapping():
    # returns mapping of country code to name
    table = pd.read_csv('data/country_codes.csv')
    codes_to_names = {}
    for i in range(len(table)):
        name = table['StateNme'].iloc[i]
        code = table['CCode'].iloc[i]
        codes_to_names[code] = name
    return codes_to_names

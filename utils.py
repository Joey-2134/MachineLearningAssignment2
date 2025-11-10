# This file will contain methods to evaluating both algorithms the same way (and some other utilities)
import pandas as pd

def load_data():
    df = pd.read_csv('steel.csv')
    print(f"Loaded {len(df)} rows from CSV")
    return df
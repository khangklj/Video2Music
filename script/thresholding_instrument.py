import pandas as pd
import os

source_path = r'dataset\vevo_instrument\origin'
target_path = r'dataset\vevo_instrument\thresholding'

for file in os.listdir(source_path):
    file_path = os.path.join(source_path, file)
    data = pd.read_csv(file_path)
    print(data)
    break
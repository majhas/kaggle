import numpy as np
import pandas as pd

file = 'data/train.csv'

# data = []
# with open(file, 'r') as f:
#     next(f)
#     for line in f:
#         line_data = line.strip().split(',')
#         data.append(line_data)
#
# data = np.array(data)
# print(data.shape)
# print(data[0])
data = pd.read_csv(file)
print(data)

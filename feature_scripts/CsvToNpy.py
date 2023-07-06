import pandas as pd
import numpy as np

train_data = pd.read_excel("Test.xlsx")['seq'].values
label = pd.read_excel("Test.xlsx")['y'].values
pos_data = train_data[np.where(label == 1)]
neg_data = train_data[np.where(label == 0)]
print(pos_data.shape)
print(neg_data.shape)
np.save("test_pos_data", pos_data)
np.save("test_neg_data", neg_data)
f1 = np.load("pos_data.npy")
print(f1.shape)

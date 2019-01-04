


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the data
test = pd.read_csv("../test.csv").values.reshape(-1,28,28,1)
results = np.ones([test.shape[0]],dtype=np.int64)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
print("complited")


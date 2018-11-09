
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

data = np.ones([5,5], dtype=np.float32) #生成随机数据,5行5列,最大值1,最小值-1
data[0,0] = 0
data[0,1] = 0
im = plt.imshow(data, cmap=plt.cm.gray)

plt.show()
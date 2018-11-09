
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

data = np.ones([5,5], dtype=np.float32) #生成随机数据,5行5列,最大值1,最小值-1
data[0,0] = 0
plt.imshow(data, cmap=plt.cm.gray)
plt.xlabel("test x label")
plt.title("test title")
plt.show()

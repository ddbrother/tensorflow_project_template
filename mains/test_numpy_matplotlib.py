
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

data = np.ones([5,5], dtype=np.float32) #生成随机数据,5行5列,最大值1,最小值-1
data[0,0] = 0
plt.figure("figure name-a")
plt.imshow(data, cmap=plt.cm.gray)
plt.xlabel("test x label")
plt.title("test title")

# 测试绘制多子图 1
ROWS = 2
COLS = 3

fig = plt.figure("test subplot -1")
ax = []
for row in range(ROWS):
    for col in range(COLS):
        ax_index = row*COLS+col
        ax.append(fig.add_subplot(ROWS, COLS, ax_index+1))
        img = data.copy()
        img[row+1][col+1] = 0.5
        ax[ax_index].imshow(img, cmap=plt.cm.gray)
        plt.title("subplot %d" %(ax_index+1))

# 测试绘制多子图 1
ROWS = 2
COLS = 3

plt.figure("test subplot -2")
for row in range(ROWS):
    for col in range(COLS):
        ax_index = row*COLS+col+1
        plt.subplot(ROWS, COLS, ax_index)
        plt.imshow(img, cmap=plt.cm.gray)
        plt.title("subplot %d" %(ax_index))
        plt.xlabel("x")
        plt.ylabel("y")
    plt.grid()

plt.show(block=True)

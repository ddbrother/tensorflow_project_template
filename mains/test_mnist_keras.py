import sys
print("sys.version =", sys.version)

print("Import dependencies. sys.version =", sys.version, sys.version_info)
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten,  MaxPooling2D, Conv2D
from keras.callbacks import TensorBoard

# mnist.npz is in /home/work/.keras/datasets/mnist.npz
print("Load and process the MNIST data...")
(X_train,y_train), (X_test, y_test) = mnist.load_data(path="mnist.npz")
print("loading mnist data completed. train data size =", X_train.size, "test data size =", X_test.size)

X_train = X_train.reshape(60000,28,28,1).astype('float32')
X_test = X_test.reshape(10000,28,28,1).astype('float32')
print("reshaped. train data size =", X_train.size, "test data size =", X_test.size)

X_train /= 255
X_test /= 255

n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test  = keras.utils.to_categorical(y_test,  n_classes)

print("Create the LeNet-5 neural network architecture")
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)) )
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())          
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

print("Compile the model")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Set log data to feed to TensorBoard for visual analysis")
tensor_board = TensorBoard('./logs/LeNet-MNIST-1')

print("Train the model")
model.fit(X_train, y_train, batch_size=1024, epochs=3, verbose=1,
          validation_data=(X_test,y_test), callbacks=[tensor_board])
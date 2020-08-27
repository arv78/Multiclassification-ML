import struct as st
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
from sklearn.preprocessing import OneHotEncoder
import time

def to_classlabel(z):
    return z.argmax(axis=1)

def T(y, K):
  """ one hot encoding """
  one_hot = np.zeros((len(y), K))
  one_hot[np.arange(len(y)), y] = 1
  return one_hot

def softmax(z):
    return np.transpose(np.transpose(np.exp(z))/(np.sum(np.exp(z), axis=1)))

def model(x_train_flat, theta):

    result = np.dot(x_train_flat, theta) #NOTE: (60000 * 784) * (784 * 10) = (60000 * 10)
    # print("1", np.shape(x_train_flat))
    # print("2", np.shape(theta))
    # print("3", np.shape(result))
    return result

def main():

    # filename = {'images' : 'train-images.idx3-ubyte' ,'labels' : 'train-labels.idx1-ubyte'} # dictionary
    # train_imagesfile = open(filename['images'],'rb')

    # train_imagesfile.seek(0)
    # magic = st.unpack('>4B',train_imagesfile.read(4))
    # size = st.unpack('>I',train_imagesfile.read(4))[0] #num of images
    # nR = st.unpack('>I',train_imagesfile.read(4))[0] #num of rows
    # nC = st.unpack('>I',train_imagesfile.read(4))[0] #num of column

    # print(magic)
    # print(size)
    # print(nC)
    # print(nR)

    # images_array = np.zeros((size,nR,nC))
    # nBytesTotal = size*nR*nC*1 # since each pixel data is 1 byte
    # images_array = 255 - np.asarray(st.unpack('>'+'B'*nBytesTotal,train_imagesfile.read(nBytesTotal))).reshape((size,nR,nC))

    # plt.imshow(images_array[0,:,:], cmap='gray')
    # plt.show()

    x_train = idx2numpy.convert_from_file("train-images.idx3-ubyte")
    y_train = idx2numpy.convert_from_file("train-labels.idx1-ubyte")

    x_test = idx2numpy.convert_from_file("t10k-images.idx3-ubyte")
    y_test = idx2numpy.convert_from_file("t10k-labels.idx1-ubyte")

    # because of numpy (,) != (,1)
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    print(np.shape(x_train))
    print(np.shape(y_train))
    print(np.shape(x_test))
    print(np.shape(y_test))

    # plt.imshow(x_test[20], cmap='gray')
    # plt.show()

    lr = 0.0001
    num_iter = 5
    theta = np.zeros((784, 10))

    x_train_flat = x_train.reshape(-1,28*28)

    ohr = OneHotEncoder(sparse=False)
    y_train_ohr = ohr.fit_transform(y_train) #NOTE: (60000 * 10)

    # print(np.shape(y_train_ohr))

    start_time = time.time()
    for i in range(num_iter):
        # print("i: ",i)
        # gradient = np.zeros((784, 10))

        for j in range(100):
            # print("j: ",j)
            X_train_batch = x_train_flat[((j)*600+0):(j+1)*600,] #NOTE: (600 * 784)
            y_train_batch = y_train_ohr[((j)*600+0):(j+1)*600,] #NOTE: (600 * 10)

            z = model(X_train_batch, theta)
            h = softmax(z)
            # print("4", np.shape(h)) #NOTE: (600 * 10)

            gradient = np.dot(np.transpose(X_train_batch), (y_train_batch - h)) #NOTE: (784 * 600) * ((600 * 10) - (600 * 10)) = (784 * 10)
            gradient = gradient / y_train_batch.size
            # np.set_printoptions(threshold=np.inf)
            # print(y_train_batch[1])

            theta = theta + np.dot(lr, gradient)
    
    # accuracy train data
    z_0 = np.dot(x_train_flat, theta)
    y_p_0 = softmax(z_0)
    predicted_labels = to_classlabel(y_p_0)

    count = 0
    for i in range(60000):
        if (predicted_labels[i] == y_train[i]):
            count += 1
    print("accuracy_train: ",count/60000)
    
    x_test_flat = x_test.reshape(-1,28*28)
    z = np.dot(x_test_flat, theta) #NOTE: (10000 * 784) * (784 , 10) = (10000 * 10)
    # print(np.shape(z))
    # print(z[0])

    y_p = softmax(z)
    # print(np.shape(y_p))

    predicted_labels = to_classlabel(y_p)
    end_time = time.time()

    print(predicted_labels)
    print(y_test)

    count = 0
    for i in range(10000):
        if (predicted_labels[i] == y_test[i]):
            count += 1
    print("accuracy_test: ",count/10000)
    print("execution time: ",end_time - start_time)

main()  
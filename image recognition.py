import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from utils import load_dataset


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#index = 25
#plt.imshow(train_set_x_orig[index])
#plt.show()
#print(str(train_set_y[:,index]) + "it's a "+classes[np.squeeze(train_set_y[:,index])].decode("utf-8") )

print(train_set_x_orig.shape) #(209, 64, 64, 3)
print(train_set_y.shape) #(1, 209)
print(test_set_x_orig.shape)#(50, 64, 64, 3)
print(test_set_y.shape)#(1, 50)
print(classes.shape)#(2,)

print(train_set_x_orig.shape[0])

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

#dimensions and shapes of problem
#reshape
#standardize.

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s


def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b


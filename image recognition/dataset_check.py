import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from utils import load_dataset
import random

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

print (train_set_x_orig.shape)#(209, 64, 64, 3)
index = random.randrange(0,209)
print(train_set_y[0][index])
plt.imshow(train_set_x_orig[index])
plt.show()
#print (train_set_y)

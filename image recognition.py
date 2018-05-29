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

#m_train=
#m_test=
#num_px=

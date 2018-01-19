import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image #Python Image Library,暂时不支持python3
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

#在python执行中不需要这条语句
#%matplotlib inline
np.random.seed(1)

# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Example of a picture
index = 6
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

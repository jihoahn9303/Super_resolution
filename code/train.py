import cv2, os, glob
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import SVG

from keras.layers import Conv2D, Input, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils.vis_utils import model_to_dot

from skimage.transform import pyramid_expand
from Subpixel import Subpixel
from DataGenerator import DataGenerator

base_path = 'C:/Users/user/Desktop/OpenSource/project/dataset/celeba-dataset/processed'
batch_size = 16

x_train_list = sorted(glob.glob(base_path + '/' + 'x_train' + '/' + '*'))
x_val_list = sorted(glob.glob(base_path + '/'  + 'x_val' + '/' + '*'))

# Custom Generator
train_gen = DataGenerator(list_IDs=x_train_list, labels=None, batch_size=batch_size, dim=(44,44), n_channels=3, n_classes=None, shuffle=True)
val_gen = DataGenerator(list_IDs=x_val_list, labels=None, batch_size=batch_size, dim=(44,44), n_channels=3, n_classes=None, shuffle=False)

upscale_factor = 4
inputs = Input(shape=(44, 44, 3))

# make convolutional nerual network model scheme 
net = Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
net = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = Conv2D(filters=upscale_factor**2, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = Subpixel(filters=3, kernel_size=3, r=upscale_factor, padding='same')(net)
outputs = Activation('relu')(net)

model = Model(inputs=inputs, outputs=outputs)   # make model object
model.compile(optimizer='adam', loss='mse')     # set model training&test method

# train,validate data by using model, and save model weight per epochs
history = model.fit_generator(train_gen, validation_data=val_gen, epochs=10, verbose=1, workers = 5, callbacks=[
    ModelCheckpoint('C:/Users/user/Desktop/OpenSource/project/dataset/hentai_decensor_model/model.h5', monitor='val_loss', verbose=1, save_best_only=True)
])

x_test_list = sorted(glob.glob(base_path + '/' + 'x_test' + '/' + '*'))
y_test_list = sorted(glob.glob(base_path + '/' + 'y_test' + '/' + '*'))

# testing
test_idx = 21      # pick up the 21th test image

x1_test = np.load(x_test_list[test_idx])        # downsampled image
x1_test_resized = pyramid_expand(x1_test, 4, multichannel = True)   # upsampled image
y1_test = np.load(y_test_list[test_idx])        # original cropped image
y_pred = model.predict(x1_test.reshape((1, 44, 44, 3)))     # predictied image from downsample image

# scaling pixel value by 255 for converting gray-scale image to color image
x1_test = (x1_test * 255).astype(np.uint8)
x1_test_resized = (x1_test_resized * 255).astype(np.uint8)
y1_test = (y1_test * 255).astype(np.uint8)
y_pred = np.clip(y_pred.reshape((176, 176, 3)), 0, 1)

# gray-scale -> RGB color
x1_test = cv2.cvtColor(x1_test, cv2.COLOR_BGR2RGB)
x1_test_resized = cv2.cvtColor(x1_test_resized, cv2.COLOR_BGR2RGB)
y1_test = cv2.cvtColor(y1_test, cv2.COLOR_BGR2RGB)
y_pred = cv2.cvtColor(y_pred, cv2.COLOR_BGR2RGB)

# plotting
fig, ax = plt.subplots(2,2)
fig.set_size_inches(15,15)
title_list = ['input', 'resized', 'output', 'groundtruth']
image_list = [x1_test, x1_test_resized, y_pred, y1_test]
for idx in range(4):
    i_row = math.floor(idx / 2.0)
    i_col = idx % 2
    ax[i_row][i_col].imshow(image_list[idx])
    ax[i_row][i_col].set_title(title_list[idx])
plt.suptitle('Sequence of Images', fontsize=18, fontweight='bold')
plt.show()

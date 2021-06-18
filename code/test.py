import cv2 
import numpy as np
# image = cv2.imread('C:/Users/user/Desktop/multimedia/jihoAhn.jpg')
image = cv2.imread('C:/Users/user/Desktop/jihoAhn1.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# pixel = np.array(image_rgb)
# pixel = np.array(image_rgb)
# print(pixel)
r, g, b = cv2.split(image_rgb)








# Custom Generator
train_gen = DataGenerator(list_IDs=x_train_list, labels=None, batch_size=batch_size, dim=(44,44), n_channels=3, n_classes=None, shuffle=True)
val_gen = DataGenerator(list_IDs=x_val_list, labels=None, batch_size=batch_size, dim=(44,44), n_channels=3, n_classes=None, shuffle=False)

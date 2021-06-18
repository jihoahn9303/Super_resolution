import os, cv2, glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import pyramid_reduce 
plt.style.use('dark_background')

base_path = 'C:/Users/user/Desktop/OpenSource/project/dataset/celeba-dataset'
img_base_path = base_path + '/' + 'img_align_celeba' + '/' + 'img_align_celeba'
target_img_path = base_path + '/' + 'processed'

# load csv file(file name, label(0 : train, 1: val, 2 : test))
eval_list = np.loadtxt(base_path + '/' + 'list_eval_partition.csv', dtype=str, delimiter=',', skiprows=1)

# # load sample original image
# img_sample = cv2.imread(os.path.join(img_base_path, eval_list[0][0]))       
# h, w, _ = img_sample.shape

# crop_sample = img_sample[int((h-w)/2):int(-(h-w)/2), :]     # crop
# resized_sample = pyramid_reduce(crop_sample, downscale=4, multichannel = True)  # down sampling
# pad = int((crop_sample.shape[0] - resized_sample.shape[0]) / 2) 
# padded_sample = cv2.copyMakeBorder(resized_sample, top=pad, bottom=pad, left=pad, right=pad, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

# print(crop_sample.shape, padded_sample.shape)

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 4, 1)
# plt.imshow(img_sample)
# plt.subplot(1, 4, 2)
# plt.imshow(crop_sample)
# plt.subplot(1, 4, 3)
# plt.imshow(resized_sample)
# plt.subplot(1, 4, 4)
# plt.imshow(padded_sample)

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 4, 1)
# plt.imshow(img_sample)
# plt.subplot(1, 4, 2)
# plt.imshow(crop_sample)
# plt.subplot(1, 4, 3)
# plt.imshow(resized_sample)
# plt.subplot(1, 4, 4)
# plt.imshow(padded_sample)

downscale = 4
# n_train = 162770
n_train = 3000
# n_val = 19867
n_val = 3000
# n_test = 19962
n_test = 3000

# 20만장의 사진 중 2000장의 사진이 동일한 확률로 선택 됨. -> 각각 학습용, 검증용, 테스트용 사진으로 분배
for i, e in enumerate(eval_list[np.random.choice(200000, 2000, p = [1/200000 for i in range(200000)])]):
    filename, ext = os.path.splitext(e[0])
    img_path = img_base_path + '/' +  e[0]
    
    img = cv2.imread(img_path)      # load image
    h, w, _ = img.shape
    
    crop = img[int((h-w)/2):int(-(h-w)/2), :]       # crop original image
    crop = cv2.resize(crop, dsize=(176, 176))       # 176 x 176 size로 조정
    resized = pyramid_reduce(crop, downscale=downscale, multichannel = True)       # down sampling -> 해상도 저하
    norm = cv2.normalize(crop.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)     # 학습 속도 향상을 위해, original image 정보를 정규화
    
    if int(e[1]) == 0:      # train
        np.save(target_img_path + '/' + 'x_train' + '/' + filename + '.npy', resized)
        np.save(target_img_path + '/' + 'y_train' + '/' + filename + '.npy', norm)
    elif int(e[1]) == 1:    # validation
        np.save(target_img_path + '/' 'x_val' + '/' + filename + '.npy', resized)
        np.save(target_img_path + '/' 'y_val' + '/' + filename + '.npy', norm)
    elif int(e[1]) == 2:    # test
        np.save(target_img_path + '/' + 'x_test' + '/' + filename + '.npy', resized)
        np.save(target_img_path + '/' + 'y_test' + '/' + filename + '.npy', norm)
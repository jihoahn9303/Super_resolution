import numpy as np
import keras
import cv2, os

"""
<Class Motivation>
대용량 데이터셋 처리에 많은 메모리를 소비해서 Trick을 통해 알아서 매끄럽게 처리되는 상황을 원함.
실시간으로 멀티 코어에서 처리가능한 데이터셋을 생성하고 트레이닝하는 방법
"""

# 커스텀 데이터 제너레이터 코드를 작성할 시, keras.utils.Sequence 클래스를 상속받아야 함.
# Sequence는 __getitem__, __len__, on_epoch_end, __iter__를 sub method로서 가지고 있음 -> 데이터에 맞게 변형
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs        # npy 파일들의 절대 경로 시퀀스(list)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # 각 call request는 배치 index : 0 ~ 총 batch 크기 만큼 될 수 있다.
        # 이부분이 __len__을 통해 제어 됨.
        length = int(np.floor(len(self.list_IDs) / self.batch_size))
        return length

    def __getitem__(self, index):
        'Generate one batch of data'
        # batch 프로세싱이 주어진 index(by __len__() function)에 따라 호출 될 때 generator는 __getitem__을 호출함
        # 결국 batch size만큼의 entry를 계산해서 리턴
        # 예) batch size가 2이고 index가 10이라면 아래 코드에 의해 indexes에 10, 11이 리턴되고 이에 상응하는 list_IDs[10], list_IDs[11]이 list_IDs_temp에 리턴됨
        # 이를 통해 __data_generation(list_IDs_temp)를 통해 알맞은 X, y가 구해짐
        
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # on_epoch_end 메소드는 각 epoch의 맨처음과 맨 끝에 실행됨
        # shuffle 파라미터가 True이면 각 epoch마다 새로운 order를 만들어냄
        # 단순 index를 shuffle하는 것임
        # shuffle을 통해 각 batch마다 identical한 데이터셋을 학습시키는 것을 방지하여 모델을 좀더 robust하게 만듦
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # generation process에서 데이터의 batch를 생성함
        # data generation동안에 이 코드는 ID.npy에 상응하는 example를 NumPy 배열로 만들어냄
        
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels)) # X(x_train) : (n_samples, *dim, n_channels)
        y = np.empty((self.batch_size, self.dim[0] * 4, self.dim[1] * 4, self.n_channels))  # upscaling image(suffling the multiple output images), upscale rate = 4

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i] = np.load(ID)

            # x_train -> y_train
            # 정답(original image) 이미지에 해당하는 numpy 파일 데이터를 numpy 객체로 역직렬화하는 과정
            splited = ID.split('/')
            splited[-1] = 'y' + splited[-1][1:] 
            y_path = ''
            for idx in range(len(splited)):
                if idx != 0:
                    y_path += ('/' + splited[idx])
                else:
                    y_path += splited[idx]

            # Store class
            y[i] = np.load(y_path)

        return X, y
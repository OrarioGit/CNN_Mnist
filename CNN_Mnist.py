############
# Step1
# 辨識資料處裡
############

# 匯入模組
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
np.random.seed(10)

# 獲取mnist資料
(x_Train, y_Train), (x_Test, y_Test) = mnist.load_data()

# 將影像特徵值轉換成4維
x_Train4D = x_Train.reshape(x_Train.shape[0], 28, 28, 1).astype('float32')
x_Test4D = x_Test.reshape(x_Test.shape[0], 28, 28, 1).astype('float32')

# 將影像特徵值做標準化
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255

# 將數字的真實數值做One-hot encoding轉換
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)

###########
# Step2
# 建立模型
###########

# 匯入所需要之模組
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# 建立Sequential模型
model = Sequential()

# 建立卷積層
#
# filters 建立濾鏡個數
# kernel_size 設定濾鏡大小
# activation 設定激活函數
model.add(Conv2D(filters = 16
				,kernel_size = (5,5)
				,padding = 'same'
				,input_shape = (28, 28, 1)
				,activation = 'relu'))

# 建立池化層
model.add(MaxPooling2D(pool_size = (2,2)))

# 建立卷積層2
model.add(Conv2D(filters = 36
				,kernel_size = (5,5)
				,padding = 'same'
				,activation = 'relu'))

# 建立池化層2
model.add(MaxPooling2D(pool_size = (2,2)))

# 加入Dropout避免overfitting
model.add(Dropout(0.25))

# 建立平坦層
model.add(Flatten())

# 建立隱藏層
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))

# 建立輸出層
model.add(Dense(10, activation = 'softmax'))

##############
# Step3
# 開始訓練
##############

# 定義訓練模式
# loss設定損失函數
# optimizer設定優化方法
# metrics設定評估模型方式
model.compile(loss = 'categorical_crossentropy',
			  optimizer = 'adam',
			  metrics = ['accuracy'])

# 訓練
# validation_split設定驗證資料之比率
# epochs 設定週期次數
# batch_size 設定每一批次多少筆資料
train_history = model.fit(x = x_Train4D_normalize,
						  y = y_TrainOneHot,
						  validation_split = 0.2,
						  epochs = 10,
						  batch_size = 300,
						  verbose = 2)

# 建立函數來顯示訓練過程
import matplotlib.pylot as plt
def show_train_history(train_history, train, validation):
	plt.plot(train_history, history[train])
	plt.plot(train_history, history[validation])
	plt.title('Train History')
	plt.ylabel(train)
	plt.xlabel('Epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()

# 繪製accuracy準確率執行結果
show_train_history(train_history, 'acc', 'val_acc')

# 繪製loss誤差執行結果
show_train_history(train_history, 'loss', 'val_loss')
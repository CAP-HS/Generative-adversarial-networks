
#Loading the MNIST dataset in Keras
import keras
from keras.callbacks import History 
history = History()
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#import the MNIST dataset from Keras

batch_size = 128

num_classes = 10

epochs = 25



# input image dimensions

img_rows, img_cols = 28, 28





#MNIST dataset using the Keras



from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()



#normalize inputs from 0-255 to 0-1



if K.image_data_format() == 'channels_first':

    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)

    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

    input_shape = (1, img_rows, img_cols)

else:

    x_train = x_train.reshape(x_train.shape[0],img_rows, img_cols, 1)

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')



#convert the dependent variable in the form of integers to a binary class matrix



y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)



#Design a Model



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



#Compile and Train Model

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])

model.save("first_test")



#Test with Handwritten Digits

import imageio
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.image as mpimg
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
import datetime as t
from skimage.color import rgb2gray
import cv2
import os
import glob
img_dir = ('.') # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
images = []
for f1 in files:
    img = cv2.imread(f1)
    images.append(img)
    images2 = np.expand_dims(img,axis=0)
    images2 = np.expand_dims(images2,axis=3)
    #images2 = np.array(img)
    gray = rgb2gray(images2)
    gray = gray.reshape(1,img_rows, img_cols,1)
    gray /= 255
    gray = np.dot(images2[...,:3], [0.299, 0.587, 0.114])
    plt.imshow(img, cmap = plt.get_cmap('gray'))
    plt.show()
    model = load_model("first_test")
    # predict digit
    prediction = model.predict(gray)
    print(prediction.argmax())


'''
path = "C:\\Users\\fitsu\\OneDrive\\Documents\\EEGR 491\\mnistasjpg\\testSet\\mytestset\\testSet"
for file in glob.glob(path):
    print(file)
    a= cv2.imread(file)
    im = imageio.imread(a)
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    gray = mpimg.imread(a)
    img = mpimg.imread(a)
    gray = rgb2gray(img)
    plt.imshow(gray, cmap = plt.get_cmap('gray'))
    gray = gray.reshape(1, img_rows, img_cols, 1)
    gray /= 255
    from keras.models import load_model
    model = load_model("first_test")
    prediction = model.predict(gray)
    print(prediction.argmax())
'''



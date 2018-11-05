import numpy as np
data = {}
try:
    data = np.load('position.npz')
except Exception as e:
    import test_position_make_data

import keras
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.models import Sequential, load_model
from PIL import Image
from random import randint

test_name = 'tester-2'
count = 20000
epoch = 5
rand = randint(0, 50000 - count)
count = rand + count

(x_train, y_train), (x_test, y_test) = (data['data'][:50000], data['label'][:50000]), (data['data'][50000:], data['label'][50000:])
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], 28*2, 28*2, 1)
x_test = x_test.reshape(x_test.shape[0], 28*2, 28*2, 1)

window_size = 4
test_size = 12
w2x = 2
input_shape = (int(28 * w2x), int(28 * w2x), 1)
# window_shape = (28*window_size, 28*window_size, 1)
window_shape = input_shape
# 数据处理

try:
    model = load_model('%s.hdf5' % test_name)
except Exception as e:
    model = Sequential()
    model.add(Conv2D(5, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=window_shape,
                     # input_shape=(28, 28, 1),
                     name='C1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='S1'))
    model.add(Conv2D(7, (5, 5), activation='relu', name='C2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='S2'))
    model.add(Flatten())
    model.add(Dense(360, activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


model.fit(x_train[rand:count], y_train[rand:count], epochs=epoch)
'''
from tqdm import *

for e in range(epoch):
    print('epoch {0}/{1}'.format(e, epoch))
    t = tqdm(range(rand, count))
    for i in t:
        noi = get_noice(window_shape)
        x = x_train[i].reshape(28, 28)
        img = plus(noi, x * 255).reshape(1, window_shape[0], window_shape[1], 1)
        y = y_train[i].reshape(1, 1)
        r = model.train_on_batch(img, y)
        # r = model.train_on_batch(x.reshape(1, 28, 28, 1), y)
        t.set_description('loss: ' + str(r[0]) + ' acc: ' + str(r[1]))

        im = Image.fromarray(img.reshape(window_shape[0], window_shape[1]))
        ids = ids + 1
        im.save(img_path + str(ids) + '.jpg')
        # print(y_train[i])
        with open(img_path + str(ids) + '.dat', 'w') as f:
            f.write(str(y_train[i]))
        with open(img_path + 'ids.txt', 'w') as f:
            f.write(str(ids))
'''
model.save('%s.hdf5' % test_name)

print(model.evaluate(x_test, y_test))
'''
t = tqdm(range(1000))
for i in t:
    noi = get_noice(window_shape)
    x = x_test[i]
    img = plus(noi, x).reshape(1, window_shape[0], window_shape[1], 1)
    y = y_test[i].reshape(1, 1)
    r = model.test_on_batch(img, y)
    # r = model.test_on_batch(x.reshape(1, 28, 28, 1), y)
    t.set_description('loss: ' + str(r[0]) + ' acc: ' + str(r[1]))
'''
print(model.metrics_names)


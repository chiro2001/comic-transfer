import keras
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.models import Sequential, load_model
from random import randint
# from keras.utils import plot_model
from PIL import Image
from random import randint

# 获取噪声
noice = Image.open('noice.jpg').convert('L')


# noice = np.array(noice)

def get_noice(size):
    randx = randint(0, 1600 - size[0])
    randy = randint(0, 900 - size[1])
    pos = noice.crop((randx, randy, randx + size[0], randy + size[1]))
    return np.array(pos)


def plus(src, img, pos=None):
    # print(img)
    if pos is None:
        pos = (randint(0, src.shape[0] - img.shape[0]), randint(0, src.shape[1] - img.shape[1]))
    # res = np.zeros(src.shape, dtype=np.uint8)
    res = copy.deepcopy(src)
    if pos[0] > src.shape[0] - img.shape[0] or pos[1] > src.shape[1] - img.shape[1]:
        return res
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[i + pos[0]][j + pos[1]] = img[i][j]
    return res.reshape(res.shape[0], res.shape[1], 1)

def get_image():
    pass


test_name = 'tester-origin-1'
count = 6000
epoch = 5
rand = randint(0, 60000 - count)
count = rand + count

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

window_size = 4
test_size = 12
w2x = 2
input_shape = (28 * w2x, 28 * w2x, 1)
# window_shape = (28*window_size, 28*window_size, 1)
window_shape = input_shape
# 数据处理

# exit(0)

try:
    model = load_model('%s.hdf5' % test_name)
except Exception as e:
    model = Sequential()
    model.add(Conv2D(3, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=(28, 28, 1),
                     name='C1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='S1'))
    model.add(Conv2D(3, (3, 3), activation='relu', name='C2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='S2'))
    model.add(Flatten())
    model.add(Dense(360, activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# model.fit(x_train[rand:count], y_train[rand:count], epochs=epoch)

from tqdm import *

img_path = 'data\\'

with open(img_path + 'ids.txt', 'r') as f:
    ids = int(f.read())

# model.fit(x_train, y_train, epochs=1)

for e in range(epoch):
    print('epoch {0}/{1}'.format(e, epoch))
    t = tqdm(range(rand, count // 500))
    for i in t:
        # noi = get_noice(window_shape)
        x = x_train[i:i+500]
        # img = plus(noi, x).reshape(1, window_shape[0], window_shape[1], 1)
        y = y_train[i:i+500].reshape(1, 1)
        # r = model.train_on_batch(img, y)
        r = model.train_on_batch(x.reshape(500, 28, 28, 1), y)
        t.set_description('loss: ' + str(r[0]) + ' acc: ' + str(r[1]))

        im = Image.fromarray(np.array((x[0]*255),dtype=np.uint8).reshape(28, 28))
        ids = ids + 1
        im.save(img_path + str(ids) + '.jpg')
        # print(y_train[i])
        with open(img_path + str(ids) + '.dat', 'w') as f:
            f.write(str(y_train[i]))
        with open(img_path + 'ids.txt', 'w') as f:
            f.write(str(ids))

model.save('%s.hdf5' % test_name)

# print(model.evaluate(x_test, y_test))
acc = 0
loss = 0
t = tqdm(range(1000))
for i in t:
    # noi = get_noice(window_shape)
    x = x_test[i]
    # img = plus(noi, x).reshape(1, window_shape[0], window_shape[1], 1)
    y = y_test[i].reshape(1, 1)
    # r = model.test_on_batch(img, y)
    r = model.test_on_batch(x.reshape(1, 28, 28, 1), y)
    # t.set_description('loss: ' + str(r[0]) + ' acc: ' + str(r[1]))
    loss = loss + r[0]
    acc = acc + r[1]

print(str(loss/1000)[:4], acc/1000)
    
print(model.metrics_names)

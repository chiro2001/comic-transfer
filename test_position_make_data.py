import keras
from tqdm import *
import copy
from random import randint
from PIL import Image
import numpy as np

# 获取噪声
noise = Image.open('noise.jpg').convert('L')

npz_name = 'position.npz'
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


def get_noise(size):
    randx = randint(0, noise.size[0] - size[0])
    randy = randint(0, noise.size[1] - size[1])
    pos = noise.crop((randx, randy, randx + size[0], randy + size[1]))
    return np.array(pos)


def plus(src, img, pos=None):
    if pos is None:
        pos = (randint(0, src.shape[0] - img.shape[0]), randint(0, src.shape[1] - img.shape[1]))
    res = copy.deepcopy(src)
    if pos[0] > src.shape[0] - img.shape[0] or pos[1] > src.shape[1] - img.shape[1]:
        return res
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[i + pos[0]][j + pos[1]] = img[i][j]
    return res.reshape(res.shape[0], res.shape[1], 1)


def get_image(i):
    src = get_noise((28*2, 28*2))
    img = np.array(x_train[i].reshape(28, 28) * 255, dtype=np.uint8)
    res = np.array(plus(src, img), dtype=np.uint8)
    return res


data = []
def save_image(img):
    data.append(img)


t = tqdm(range(60000))
for i in t:
    img = get_image(i)
    save_image(img)

for i in range(len(y_train)):
    if y_train[i] > 0:
        y_train[i] = 1

np.savez(npz_name, data=np.array(data), label=y_train)


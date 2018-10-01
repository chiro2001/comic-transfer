import cv2
from time import sleep
from PIL import Image
from random import randint
import numpy as np

# 获取噪声
noice = Image.open('noice.jpg').convert('L')
# noice = cv2.imread('noice.jpg', cv2.IMREAD_GRAYSCALE)
# noice = np.array(noice)

def get_noice(size):
    randx = randint(0, 1600 - size[0])
    randy = randint(0, 900 - size[1])
    pos = noice.crop((randx, randy, randx + size[0], randy + size[1]))
    return np.array(pos, dtype=np.uint8)


def plus(src, img, pos):
    res = np.zeros(src.shape, dtype=np.uint8)
    if pos[0] > src.shape[0] - img.shape[0] or pos[1] > src.shape[1] - img.shape[1]:
        return res
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if pos[0] <= i < img.shape[0] + pos[0] and pos[1] <= j < img.shape[1] + pos[1]:
                res[i][j] = img[i-pos[0]][j-pos[1]]
            else:
                res[i][j] = src[i][j]
    return res


while True:
    x1 = get_noice((280, 280))
    x2 = get_noice((28, 28))
    x3 = plus(x1, x2, (randint(0, 280-28), randint(0, 280-28)))

    cv2.imshow('noice', x3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

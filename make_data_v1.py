import os
import numpy as np
from random import randint
from PIL import Image
os.chdir('dataset-v1')

input_shape = (56, 56)


def get_fixed_img(src):
    target = [0, 0]
    if src.size[0] < src.size[1]:
        target[1] = input_shape[0]
        target[0] = int(src.size[0] / src.size[1] * input_shape[1])
    else:
        target[0] = input_shape[1]
        target[1] = int(src.size[1] / src.size[0] * input_shape[0])

    img = src.resize(target)
    res = Image.new("RGB", input_shape, (255, 255, 255))
    res.paste(img, ((input_shape[0]-target[0])//2, (input_shape[1]-target[1])//2))
    return res


def sign(x):
    if x > 0:
        return x
    return 0


create_new = False


li = os.listdir()

if create_new:
    for i in li:
        if '.jpg' not in i or 'n' in i:
            continue
        get_fixed_img(Image.open(i)).save(i)

    for i in li:
        if 'n' not in i or 'f' in i:
            continue
        # 每张图选择15个区域
        im = Image.open(i)
        pos = [0, 0, 0, 0]
        size = im.size
        for j in range(15):
            pos[0] = randint(0, sign(size[0] - input_shape[0] * 3))
            pos[1] = randint(0, sign(size[1] - input_shape[1] * 3))
            pos[2] = min((pos[0] + input_shape[0] * 3, size[0]))
            pos[3] = min((pos[1] + input_shape[1] * 3, size[1]))
            res = im.crop(pos).resize(input_shape)
            res.save(i.split('.jpg')[0] + str(j) + 'f' + '.jpg')

data = []
label = []
li = os.listdir()
for i in li:
    if 'n' in i and 'f' not in i:
        continue
    im = Image.open(i).convert('L')
    data.append(np.array(im))
    if 'n' in i:
        label.append(0)
    else:
        label.append(1)

np.savez('position2.npz', data=data, label=label)


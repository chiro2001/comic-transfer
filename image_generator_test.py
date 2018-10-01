from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os

os.chdir('dataset-v1')

Datagen = ImageDataGenerator(rotation_range=10,
                             shear_range=0.1,
                             horizontal_flip=True,
                             vertical_flip=True,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.3,
                             fill_mode='nearest')

x_img = []

for i in os.listdir():
    if 'hand' in i:
        continue
    if 'n' in i and 'f' not in i:
        continue
    img = load_img(i)
    x_img.append(img_to_array(img))

x_img = np.array(x_img)
x_img = x_img.reshape(x_img.shape[0], 56, 56, 3)

i = 0
for img_batch in Datagen.flow(x_img,
                              batch_size=1,
                              save_to_dir='.',
                              save_prefix='hand',
                              save_format='jpg'):
    i += 1
    if i > 1200:
        break
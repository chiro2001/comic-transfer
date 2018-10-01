import numpy as np
from PIL import Image

data = np.load('position.npz')

print(data['label'])

im = Image.fromarray(data['data'][1203].reshape(28*2, 28*2))
im.save('sample.jpg')
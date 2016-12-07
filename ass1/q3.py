from pylab import *
from numpy import *
from mnsit import *
images, labels = load_mnist('testing', digits=[5])
imshow(images.mean(axis=0), cmap=cm.gray)
show()
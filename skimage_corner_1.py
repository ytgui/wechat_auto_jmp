import time
import numpy as np
import skimage.io, skimage.feature, skimage.transform, skimage.draw, skimage.morphology, skimage.color
import matplotlib.pyplot as plt


image = skimage.io.imread('/home/bosskwei/Pictures/singlemarkersoriginal.png')
gray = skimage.color.rgb2gray(image)
edge = skimage.feature.canny(gray, sigma=1.2)

coords = skimage.feature.corner_peaks(skimage.feature.corner_harris(edge), min_distance=8)
coords_subpix = skimage.feature.corner_subpix(edge, coords, window_size=20)

fig, ax = plt.subplots()
ax.imshow(edge, interpolation='nearest', cmap='gray')
ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=3)
ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
plt.show()
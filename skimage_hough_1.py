import time
import numpy as np
import skimage.io, skimage.feature, skimage.transform, skimage.draw, skimage.morphology, skimage.color
import matplotlib.pyplot as plt


frame = skimage.io.imread('1.png')
gray = skimage.color.rgb2gray(frame)
edge = skimage.feature.canny(gray, sigma=2)
# edge = skimage.morphology.dilation(edge, skimage.morphology.square(1))
skimage.io.imshow(edge)
skimage.io.show()

plt.xlim(gray.shape[1])
plt.ylim(gray.shape[0])
plt.imshow(frame)

h, theta, d = skimage.transform.hough_line(edge)
for _, angle, dist in zip(*skimage.transform.hough_line_peaks(h, theta, d, min_distance=1)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - edge.shape[1] * np.cos(angle)) / np.sin(angle)
    plt.plot((0, edge.shape[1]), (y0, y1), '-r')

plt.show()

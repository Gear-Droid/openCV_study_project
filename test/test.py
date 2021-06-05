import imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import cv2


def GetMostPopulatedRegion(bins, counts):
    imax = np.argmax(counts)
    binWidth = np.gradient(bins)[0]
    lo = bins[imax] - binWidth/2.0
    hi = bins[imax] + binWidth/2.0
    return lo, hi


def open_image(filename):
    img = cv2.imread(filename)
    return img


filename = "cropped_resistor_1.jpg"
bgr_resistor_img = open_image(filename)
rgb_resistor_img = cv2.cvtColor(bgr_resistor_img, cv2.COLOR_BGR2RGB)

# красный, зеленый и синий цвета
red = rgb_resistor_img[:,:,0]
green = rgb_resistor_img[:,:,1]
blue = rgb_resistor_img[:,:,2]

# среднее в поперечном направлении
red_mean = red.mean(axis = 0)
green_mean = green.mean(axis = 0)
blue_mean = blue.mean(axis = 0)

grayscale = red_mean * 0.2989 + green_mean * 0.5870 + blue_mean * 0.1140

grad = np.abs(np.gradient(grayscale))
length = np.linspace(0, 1, len(grayscale))

plt.clf()
counts, bins, bars = plt.hist(grayscale, bins=7)
print(bins, counts, bars)
plt.show()

lo1, hi1 = GetMostPopulatedRegion(bins, counts)

counts2, bins2, bars2 = plt.hist(grad, bins=5)
lo2, hi2 = GetMostPopulatedRegion(bins2, counts2)

ix = ((grayscale < lo1) | (grayscale > hi1)) & (grad < max(grad)*0.2)

plt.clf()
plt.subplot(2, 1, 1)
plt.scatter(length[ix], grayscale[ix])
plt.subplot(2, 1, 2)
plt.imshow(rgb_resistor_img)
plt.show()

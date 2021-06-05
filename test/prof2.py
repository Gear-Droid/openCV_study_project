import imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import cv2


def open_image(filename):
    img = cv2.imread(filename)
    return img

def GetMostPopulatedRegion(bins, counts):
    imax = np.argmax(counts)
    binWidth = np.gradient(bins)[0]
    lo = bins[imax] - binWidth/2.0
    hi = bins[imax] + binWidth/2.0
    return lo, hi

stripCount = 5

img = imageio.imread('example_fine.png')

# красный, зеленый и синий цвета
red = img[:,:,0]
green = img[:,:,1]
blue = img[:,:,2]

# среднее в поперечном направлении
red_mean = red.mean(axis = 0)
green_mean = green.mean(axis = 0)
blue_mean = blue.mean(axis = 0)

grayscale = red_mean * 0.2989 + green_mean * 0.5870 + blue_mean * 0.1140

grad = np.abs(np.gradient(grayscale))
length = np.linspace(0, 1, len(grayscale))

counts, bins, bars = plt.hist(grayscale, bins=6)
lo1, hi1 = GetMostPopulatedRegion(bins, counts)

counts2, bins2, bars2 = plt.hist(grad, bins=5)
lo2, hi2 = GetMostPopulatedRegion(bins2, counts2)

ix = ((grayscale < lo1) | (grayscale > hi1)) & (grad < max(grad)*0.2)

plt.clf()
plt.subplot(2, 1, 1)
plt.scatter(length[ix], grayscale[ix])
plt.subplot(2, 1, 2)
plt.imshow(img)
plt.show()

### --- ###

from sklearn.cluster import KMeans

length_ix = length[ix].reshape(-1, 1)
GRAY_ix = grayscale[ix].reshape(-1, 1)
RED_ix = red_mean[ix].reshape(-1, 1)
GREEN_ix = green_mean[ix].reshape(-1, 1)
BLUE_ix = blue_mean[ix].reshape(-1, 1)

kmeans = KMeans(n_clusters=stripCount, random_state=0).fit(length_ix)
print()
print(kmeans.labels_)
print()
stripColors = np.empty((stripCount, 3))

for i in range(stripCount):
	ixx = kmeans.labels_ == i
	stripColors[i,0] = np.mean(RED_ix[ixx])
	stripColors[i,1] = np.mean(GREEN_ix[ixx])
	stripColors[i,2] = np.mean(BLUE_ix[ixx])


farben = ['brown', 'black', 'blue', 'yellow', 'orange']

y = [0, 1, 2, 3, 4]

print(kmeans.cluster_centers_)

order = np.argsort(kmeans.cluster_centers_[:,0])

farben_ordered = [farben[i] for i, m in enumerate(order)]
print(farben_ordered)

from sklearn import svm

clf = svm.SVC()
clf.fit(stripColors, y)

print(clf.predict(stripColors))

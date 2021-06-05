import numpy as np
from sklearn import svm
import cv2


def viewImage(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1000, 1000)
    cv2.imshow(name, image)

    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def GetMostPopulatedRegion(bins, counts):
    imax = np.argmax(counts)
    binWidth = np.gradient(bins)[0]
    lo = bins[imax] - binWidth/2.
    hi = bins[imax+1] + binWidth/2.
    return lo, hi


def open_image(filename):
    img = cv2.imread(filename)
    return img


learning_images_filenames = (
    '1_brown', '2_black', '3_yellow', '4_white', '5_orange',
    '6_silver', '7_violet', '8_red', '9_green',
)

y = [0, 1, 2, 3, 4, 5, 6, 7, 8]
farben = {
    0: 'brown',
    1: 'black',
    2: 'yellow',
    3: 'white',
    4: 'orange',
    5: 'silver',
    6: 'violet',
    7: 'red',
    8: 'green'
}

clf = svm.SVC()

n = len(learning_images_filenames)
colors_vector = np.empty((n, 3))

for i, fname in zip(range(n), learning_images_filenames):
    bgr_img = open_image('learning/{}.jpg'.format(fname))
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h_mean, s_mean, v_mean = hsv_img[:,:,0].mean(), hsv_img[:,:,1].mean(), hsv_img[:,:,2].mean()
    colors_vector[i,0], colors_vector[i,1], colors_vector[i,2] = h_mean, s_mean, v_mean

clf.fit(colors_vector, y)





prediction = clf.predict(colors_vector[1:3])
print(list(map(lambda x: farben[x], prediction)))

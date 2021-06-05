import imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.cluster import KMeans
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

from sklearn.neighbors import KNeighborsClassifier
KNN_model = KNeighborsClassifier(n_neighbors=9)

n = len(learning_images_filenames)

colors_vector = np.empty((n, 3))
for i, fname in zip(range(n), learning_images_filenames):
    bgr_img = open_image('learning/{}.jpg'.format(fname))
    hsv_img = bgr_img
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h_mean, s_mean, v_mean = hsv_img[:,:,0].mean(), hsv_img[:,:,1].mean(), hsv_img[:,:,2].mean()
    colors_vector[i,0], colors_vector[i,1], colors_vector[i,2] = h_mean, s_mean, v_mean
    clf.fit(colors_vector, y)
    KNN_model.fit(colors_vector, y)





for i in range(n):
    print(farben[i], colors_vector[i])
print()

stripped_resistors = []
for i in range(1, 7):
    filename = "cropped_resistor_{}.jpg".format(i)
    bgr_resistor_img = open_image(filename)
    rgb_resistor_img = cv2.cvtColor(bgr_resistor_img, cv2.COLOR_BGR2RGB)
    grayscale = cv2.cvtColor(bgr_resistor_img, cv2.COLOR_BGR2GRAY)

    ret, threshhold = cv2.threshold(
        grayscale, 225, 255, cv2.THRESH_BINARY
    )
    kernel = np.ones((4,4),np.uint8)
    dilation = cv2.dilate(threshhold, kernel, iterations = 1)
    mask = dilation
    inpainted_img = cv2.inpaint(bgr_resistor_img, mask, 128, cv2.INPAINT_NS)
    rgb_inpainted_img = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB)
    hsv_resistor_img = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2HSV)
    hsv_resistor_img = inpainted_img

    # Усредненный оттенок в поперечном направлении сечения резистора
    hue_mean_line = hsv_resistor_img[:,:,0].mean(axis = 0)
    saturation_mean_line = hsv_resistor_img[:,:,1].mean(axis = 0)
    visibility_mean_line = hsv_resistor_img[:,:,2].mean(axis = 0)

    plt.clf()
    counts, bins, bars = plt.hist(
        hue_mean_line, bins=6
    )
    lo1, hi1 = GetMostPopulatedRegion(bins, counts)
    # plt.show()

    grad = np.abs(np.gradient(hue_mean_line))
    length = np.linspace(0, len(hue_mean_line)-1, len(hue_mean_line))

    # 1
    # ix = ((hue_mean_line > lo1) & (hue_mean_line < hi1)) | (grad > max(grad)*0.05)
    # 2
    ix = ((hue_mean_line < lo1) | (hue_mean_line > hi1)) & (grad < max(grad)*0.13)

    # KMeans
    stripCount = 5
    length_ix = length[ix].reshape(-1, 1)
    kmeans = KMeans(n_clusters=stripCount, random_state=0).fit(length_ix)
    resistor_stripes = []
    for ctr_float in kmeans.cluster_centers_:
        ctr_int = int(ctr_float)
        H_mean = round(hue_mean_line[ctr_int-1:ctr_int+2].sum()/3., 2)
        S_mean = round(saturation_mean_line[ctr_int-1:ctr_int+2].sum()/3., 2)
        V_mean = round(visibility_mean_line[ctr_int-1:ctr_int+2].sum()/3., 2)
        resistor_stripes.append((ctr_int, [H_mean, S_mean, V_mean]))

    resistor_stripes = sorted(resistor_stripes, key=lambda index: index[0])
    stripped_resistors.append(resistor_stripes)


for resistor in stripped_resistors:
    n = len(resistor)
    resistor_colors_vector = np.empty((n, 3))
    for i in range(n):
        h, s, v = resistor[i][1][0], resistor[i][1][1], resistor[i][1][2]
        resistor_colors_vector[i,0], resistor_colors_vector[i,1], resistor_colors_vector[i,2] = h, s, v
    print(resistor_colors_vector)
    print()

    prediction = clf.predict(resistor_colors_vector)
    print('SVC', list(map(lambda x: farben[x], prediction)))

    prediction = KNN_model.predict(resistor_colors_vector)
    # print('KNN', list(map(lambda x: farben[x], prediction)))


    plt.clf()

    # 1 - оригинал изображения
    ax1 = plt.subplot(4, 1, 1)
    ax1.margins(x=0, y=0)
    plt.imshow(rgb_resistor_img)

    # 2 - закрашивание бликов
    ax2 = plt.subplot(4, 1, 2)
    ax2.margins(x=0, y=0)
    plt.imshow(rgb_inpainted_img)

    # 3 - полный график усредненных оттенков в разрезе
    ax3 = plt.subplot(4, 1, 3)
    ax3.margins(x=0, y=0)
    plt.scatter(length, hue_mean_line)

    # 4 - результат фильтрации оттенков
    ax4 = plt.subplot(4, 1, 4)
    ax4.margins(x=0, y=0)
    plt.scatter(length[ix], hue_mean_line[ix])

    plt.show()

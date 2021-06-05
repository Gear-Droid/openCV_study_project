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


for i in stripped_resistors:
    print(i)
    print()


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

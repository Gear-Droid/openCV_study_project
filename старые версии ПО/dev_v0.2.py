import cv2
import numpy as np
# import pprint
from sklearn.decomposition import PCA
import math
import copy


def open_image(filename):
    img = cv2.imread(filename)
    return img


def viewImage(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1000, 1000)
    cv2.imshow(name, image)

    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def BGR2HSV(BGR_image):
    HSV_image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2HSV)
    return HSV_image


def get_range(manual=False):
    green_low, green_high = None, None
    if manual:
        min = input("Введите минимальный порог: ")
        max = input("Введите максимальный порог: ")
        _min = min.split(" ")
        _max = max.split(" ")
        green_low = np.array(_min, dtype="uint8")
        green_high = np.array(_max, dtype="uint8")
    else:
        green_low = np.array([41, 40, 34], dtype="uint8")
        green_low = np.array([42, 40, 35], dtype="uint8")
        green_high = np.array([85, 155, 90], dtype="uint8")

    return (green_low, green_high)


def findGreatesContour(contours):
    largest_area = 0
    largest_contour_index = -1
    i = 0
    total_contours = len(contours)

    while (i < total_contours):
        area = cv2.contourArea(contours[i])

        if (area > largest_area):
            largest_area = area
            largest_contour_index = i

        i += 1

    return largest_area, largest_contour_index


def get_square_matrix(matrix, dtype="uint8"):
    # Дополняем до квадратного изображения
    if matrix.shape[0] != matrix.shape[1]:
        z = 0
        if matrix.shape[0] > matrix.shape[1]:
            z = np.zeros((matrix.shape[0], matrix.shape[0]-matrix.shape[1]),
                         dtype)
            matrix = np.hstack((matrix, z))
        else:
            z = np.zeros((matrix.shape[1]-matrix.shape[0], matrix.shape[1]),
                         dtype)
            matrix = np.vstack((matrix, z))

    return matrix


class SchemeFinder():

    def __init__(self, filename, image=None, rotation=True):
        self.rotation = rotation
        self.original_img = None
        if image is None:
            self.filename = filename
            img = open_image(self.filename)
        else:
            img = image
            self.original_img = copy.deepcopy(img)
        # Высчитываем погрешность DELTA для последующего использования
        self.DELTA = (img.shape[0] + img.shape[1])/2. * 0.005
        self.image = img
        self.contour_img = None
        self.contours = None
        self.largest_contour_index = None
        self.cX = None
        self.cY = None
        self.w = None
        self.h = None

    def show_image(self, screen_name):
        cv2.namedWindow(screen_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(screen_name, 1000, 1000)
        cv2.imshow(screen_name, self.image)

        while(True):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def filter(self):
        # Приминение фильтров удаляющих артефакты камеры
        self.image = cv2.medianBlur(self.image, 3)
        self.image = cv2.GaussianBlur(self.image, (3, 3), 0)

    def to_HSV(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

    def to_gray(self):
        RGB_again = cv2.cvtColor(self.image, cv2.COLOR_HSV2RGB)
        self.image = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)

    def print_point(self, x, y):
        print(self.image[[y], [x]])

    def clear_white(self):
        # mask1 - фильтрует засветы, затемняя их
        mask1 = cv2.inRange(
            self.image, np.array([0, 0, 250]), np.array([180, 10, 256])
        )
        self.image[mask1 > 0] = ([0, 0, 0])

    def paint_scheme(self, manual=False):
        green_low, green_high = get_range(manual)
        mask2 = cv2.inRange(self.image, green_low, green_high)
        self.image[mask2 > 0] = ([179, 0, 255])

    def treshold(self):
        ret, threshold = cv2.threshold(self.image, 254, 255, cv2.THRESH_BINARY)
        self.image = threshold

    def closing(self):
        # Заполнение пробелов
        kernel = np.ones((int(2.5*self.DELTA), int(2.5*self.DELTA)), np.uint8)
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)

    def opening(self):
        # Устранение артефактов
        kernel = np.ones((int(self.DELTA), int(self.DELTA)), np.uint8)
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)

    def contouring(self):
        self.contours, hierarchy = cv2.findContours(
            self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if self.original_img is None:
            self.contour_img = open_image(self.filename)
        else:
            self.contour_img = copy.deepcopy(self.original_img)
        cv2.drawContours(self.contour_img, self.contours, -1, (0, 0, 255), -1)
        largest_area, self.largest_contour_index = findGreatesContour(
            self.contours
        )
        """
        pprint.pprint("Макс площадь: {}".format(largest_area))
        pprint.pprint("Индекс контура: {}".format(self.largest_contour_index))
        pprint.pprint("Кол-во контуров: {}".format(len(self.contours)))
        """
        if largest_area > 0.95 * self.image.shape[0] * self.image.shape[1]:
            self.DELTA = 0

    def find_center(self):
        # Найдем центр наибольшего контура
        cnt = self.contours[self.largest_contour_index]
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        """
        pprint.pprint("x = {}".format(cX))
        pprint.pprint("y = {}".format(cY))
        """
        return (cX, cY)

    def show_contour_image(self, screen_name):
        cv2.namedWindow(screen_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(screen_name, 1000, 1000)
        cv2.imshow(screen_name, self.contour_img)

        while(True):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def get_rectangle(self):
        x, y, w, h = cv2.boundingRect(
            self.contours[self.largest_contour_index]
        )
        cv2.rectangle(
            self.image,
            (x-int(self.DELTA*w/h)//2, y-int(self.DELTA*h/w)//2),
            (x+w+int(self.DELTA*w/h)//2, y+h+int(self.DELTA*h/w)//2),
            (0, 255, 0),
            2
        )
        return x, y, w, h

    def get_rotation_angle(self, img):
        sqr_image = get_square_matrix(img, dtype="uint8")
        sqr_image = cv2.resize(sqr_image, (256, 256))
        # Устранение артефактов
        kernel = np.ones((2, 2), np.uint8)
        sqr_image = cv2.morphologyEx(sqr_image, cv2.MORPH_OPEN, kernel)
        viewImage('Квадратная матрица для метода главных компонент', sqr_image)

        x = []
        y = []
        for j in np.arange(0, sqr_image.shape[1]):
            for i in np.arange(0, sqr_image.shape[0]):
                if sqr_image[[i], [j]] >= 50:
                    x.append(j)
                    y.append(i)
        X = np.vstack((x, y))

        pca = PCA(n_components=1)
        pca.fit_transform(np.transpose(X))
        # print('Mean vector: ', pca.mean_)
        # print('Projection: ', pca.components_)
        vec = np.zeros((2))
        sign = 1.
        if pca.components_[[0], [0]] < 0:
            sign = -1.
        vec[0] = sign * pca.components_[[0], [0]]
        vec[1] = sign * pca.components_[[0], [1]]
        print(vec)

        rotation_angle = math.acos(vec.dot(np.array([1., 0])))
        if vec[1] >= 0:
            rotation_angle = -1. * rotation_angle
        rotation_angle = rotation_angle * 180. / math.pi
        return rotation_angle

    def rotate(self):
        # Поворот изображения в направлении вектора максимальной дисперсии
        rotation_angle = self.get_rotation_angle(self.image)
        rows, cols = self.image.shape
        # cols-1 and rows-1 are the coordinate limits
        M = cv2.getRotationMatrix2D(((cols-1)/2., (rows-1)/2.),
                                    rotation_angle, 1)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))
        # self.show_image('Повернутая матрица после метода главных компонент')

        return M

    def draw_rect(self, M):
        self.cX, self.cY, self.w, self.h = self.get_rectangle()
        if self.original_img is None:
            self.image = open_image(self.filename)
        else:
            self.image = copy.deepcopy(self.original_img)

        if M is not None:
            rows, cols, _ = self.image.shape
            self.image = cv2.warpAffine(self.image, M, (cols, rows))

        x, y, w, h = self.cX, self.cY, self.w, self.h
        cropped_scheme = sf.image[
            y-int(self.DELTA*h/w)//2: y+h+int(self.DELTA*h/w)//2+1,
            x-int(self.DELTA*w/h)//2: x+w+int(self.DELTA*w/h)//2+1,
            :
        ]
        cv2.rectangle(
            sf.image,
            (x-int(self.DELTA*w/h)//2, y-int(self.DELTA*h/w)//2),
            (x+w+int(self.DELTA*w/h)//2, y+h+int(self.DELTA*h/w)//2),
            (0, 255, 0),
            2
        )
        return cropped_scheme

    def crop_scheme(self):
        sf.show_image("Исходник")
        sf.filter()

        # Переводим изображение в HSV
        sf.to_HSV()
        # sf.show_image("RGB в HSV")  # 1
        # Принтим значение каналов в точке (x, y)
        # sf.print_point(2195, 2400)
        sf.clear_white()

        sf.paint_scheme(manual=False)
        # sf.show_image("Наложение маски на плату")  # 2
        sf.to_gray()
        # sf.show_image("HSV в RGB и в серые оттенки")  # 3
        sf.treshold()
        # sf.show_image("Пороговая обработка")  # 4
        sf.closing()
        # sf.show_image("Заполнение пробелов")  # 5
        sf.opening()
        # sf.show_image("Устранение артефактов")  # 6
        M = None
        if self.rotation:
            M = sf.rotate()
        sf.contouring()
        # sf.show_contour_image("Контурирование")  # 7
        sf.find_center()

        cropped_scheme = sf.draw_rect(M)
        sf.show_image("Результат")  # 8
        return cropped_scheme


def on_update(val):
    edges = cv2.Canny(cropped_scheme, val, 200)
    viewImage('Edges', edges)


if __name__ == "__main__":
    filename = 'test7.jpg'

    sf = SchemeFinder(filename, rotation=True)
    cropped_scheme = sf.crop_scheme()

    viewImage('Схемка', cropped_scheme)

    # Поиск резисторов на схемке
    # edges = cv2.Canny(cropped_scheme, 10, 200)

    # cv2.namedWindow("Canny")
    # cv2.createTrackbar('First', 'Canny', 10, 1000, on_update)

    # viewImage('Canny', edges)
    """
    hsv_scheme = cv2.cvtColor(cropped_scheme, cv2.COLOR_BGR2HSV)

    print(hsv_scheme[[914], [103]])
    print(hsv_scheme[[726], [149]])
    print(hsv_scheme[[574], [540]])

    # mask1 - фильтрует засветы затемняя их
    mask1 = cv2.inRange(
        hsv_scheme, np.array([0, 0, 250]), np.array([180, 2, 255])
    )
    hsv_scheme[mask1 > 0] = ([253, 1, 253])

    resistor_low = np.array([78, 30, 60], dtype="uint8")
    resistor_high = np.array([110, 170, 200], dtype="uint8")
    mask2 = cv2.inRange(hsv_scheme, resistor_low, resistor_high)
    hsv_scheme[mask2 > 0] = ([255, 0, 255])
    # viewImage("Наложение маски на плату", hsv_scheme)  # 10

    RGB_again = cv2.cvtColor(hsv_scheme, cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)
    viewImage("HSV в RGB и в серые оттенки", gray)  # 11

    ret, threshold = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
    viewImage("Пороговая обработка", threshold)  # 12

    # Высчитывание относительной погрешности DELTA
    DELTA = (threshold.shape[0] + threshold.shape[1])/2. * 0.005

    # Заполнение пробелов
    kernel = np.ones((int(3.5*DELTA), int(3.5*DELTA)), np.uint8)
    closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    viewImage("Заполнение пробелов", closing)  # 13

    # Устранение артефактов
    kernel = np.ones((int(2*DELTA), int(2*DELTA)), np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    viewImage("Устранение артефактов", opening)  # 14
    """

    cv2.destroyAllWindows()

import cv2
from sklearn.decomposition import PCA
import copy
import math
import numpy as np


class SchemeFinderError(ValueError):
    """Ошибка обработки изображения"""
    pass


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
        # green_low = np.array([42, 40, 35], dtype="uint8")
        # green_high = np.array([85, 155, 90], dtype="uint8")
        green_low = np.array([70, 120, 20], dtype="uint8")
        green_high = np.array([90, 255, 180], dtype="uint8")

    return (green_low, green_high)


def find_greatest_contour(contours):
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


def find_contours_by_area(contours, min_area, max_area):
    cnt_indexes = []
    i = 0
    total_contours = len(contours)

    while (i < total_contours):
        area = cv2.contourArea(contours[i])

        if (area >= min_area and area <= max_area):
            cnt_indexes.append(i)

        i += 1

    return cnt_indexes


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


def clear_white(img):
    # mask1 - фильтрует засветы, затемняя их
    mask1 = cv2.inRange(
        img, np.array([0, 0, 250]), np.array([180, 10, 256])
    )
    img[mask1 > 0] = ([0, 0, 0])
    return img


def get_rectangle(contours, contour_index):
    x, y, w, h = cv2.boundingRect(
        contours[contour_index]
    )
    return x, y, w, h


def draw_resistors(image, centers, width, height):
    w = width
    h = height
    shift = 0

    for x, y in centers:
        cv2.rectangle(
            image,
            (int(x-w//2-int(shift*w/h)), int(y-h//2-int(shift*h/w))),
            (int(x+w//2+int(shift*w/h)), int(y+h//2+int(shift*h/w))),
            (0, 255, 0),
            2
        )

    return image


class SchemeFinder():

    def __init__(self, filename, image=None, rotation=True,
                 k_DELTA=0.005, filtration=True):
        self.rotation = rotation
        self.original_img = None
        self.filtration = filtration

        if image is None:
            self.filename = filename
            img = open_image(self.filename)
            self.original_img = copy.deepcopy(img)
        else:
            img = image
            self.original_img = copy.deepcopy(img)

        if img is None:
            raise SchemeFinderError(
                """При создании обработчика отсутствует изображение схемы. """
                """Не работает камера или путь до изображения неверный""")

        # Высчитываем погрешность DELTA для последующего использования
        self.DELTA = (img.shape[0] + img.shape[1])/2. * k_DELTA
        self.image = img
        self.contour_img = None
        self.contours = None
        self.largest_contour_index = None

    def filter(self, image):
        if self.filtration:
            # Приминение фильтров удаляющих артефакты камеры
            image = cv2.medianBlur(self.image, 3)
            image = cv2.GaussianBlur(self.image, (3, 3), 0)

        return image

    def show_image(self, screen_name):
        cv2.namedWindow(screen_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(screen_name, 1000, 1000)
        cv2.imshow(screen_name, self.image)

        while(True):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def to_HSV(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        return image

    def to_gray(self, image):
        RGB_again = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        image = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)

        return image

    def print_point(self, x, y):
        print(self.image[[y], [x]])

    def paint_scheme(self, image, manual=False,
                     green_low=None, green_high=None):
        if green_low is None and green_high is None:
            green_low, green_high = get_range(manual)

        mask2 = cv2.inRange(image, green_low, green_high)
        image[mask2 > 0] = ([179, 0, 255])

        return image

    def threshold(self, image):
        ret, threshold = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)

        return threshold

    def closing(self, image, ky=2.5, kx=2.5):
        # Заполнение пробелов
        kernel = np.ones((int(ky*self.DELTA), int(kx*self.DELTA)), np.uint8)

        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def opening(self, image, ky=1., kx=1.):
        # Устранение артефактов
        kernel = np.ones((int(ky*self.DELTA), int(kx*self.DELTA)), np.uint8)

        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def contouring(self, image):
        self.contours, hierarchy = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if self.original_img is None:
            self.contour_img = open_image(self.filename)
        else:
            self.contour_img = copy.deepcopy(self.original_img)
        if self.M is not None:
            self.contour_img = cv2.warpAffine(
                self.contour_img, self.M, (self.cols, self.rows)
            )
        contour_image = cv2.drawContours(
            self.contour_img, self.contours, -1, (0, 0, 255), -1)
        largest_area, self.largest_contour_index = find_greatest_contour(
            self.contours
        )
        """
        pprint.pprint("Макс площадь: {}".format(largest_area))
        pprint.pprint("Индекс контура: {}".format(self.largest_contour_index))
        pprint.pprint("Кол-во контуров: {}".format(len(self.contours)))
        """
        if largest_area > 0.95 * self.image.shape[0] * self.image.shape[1]:
            self.DELTA = 0

        return contour_image

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

    def get_rotation_angle(self, img):
        sqr_image = get_square_matrix(img, dtype="uint8")
        sqr_image = cv2.resize(sqr_image, (256, 256))
        # Устранение артефактов
        kernel = np.ones((2, 2), np.uint8)
        sqr_image = cv2.morphologyEx(sqr_image, cv2.MORPH_OPEN, kernel)
        """viewImage(
            'Квадратная матрица для метода главных компонент', sqr_image
        )"""

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
        # print(vec)

        rotation_angle = -math.acos(vec.dot(np.array([1., 0])))
        if vec[1] >= 0:
            rotation_angle = -1. * rotation_angle
        rotation_angle = rotation_angle * 180. / math.pi
        # print(rotation_angle)
        return rotation_angle

    def rotate(self, image):
        # Поворот изображения в направлении вектора максимальной дисперсии
        rotation_angle = self.get_rotation_angle(image)
        rows, cols = image.shape
        # cols-1 and rows-1 are the coordinate limits
        M = cv2.getRotationMatrix2D(((cols-1)/2., (rows-1)/2.),
                                    rotation_angle, 1)
        image = cv2.warpAffine(image, M, (cols, rows))
        # self.show_image('Повернутая матрица после метода главных компонент')

        return M, image

    def draw_rect(self, M):
        image = copy.deepcopy(self.original_img)
        x, y, w, h = get_rectangle(
            self.contours, self.largest_contour_index
        )

        if M is not None:
            self.M = M
            self.rows, self.cols, _ = image.shape
            image = cv2.warpAffine(image, M, (self.cols, self.rows))

        cropped_scheme = image[
            y-int(self.DELTA*h/w)//2: y+h+int(self.DELTA*h/w)//2+1,
            x-int(self.DELTA*w/h)//2: x+w+int(self.DELTA*w/h)//2+1,
            :
        ]
        detected_scheme = cv2.rectangle(
            image,
            (x-int(self.DELTA*w/h)//2, y-int(self.DELTA*h/w)//2),
            (x+w+int(self.DELTA*w/h)//2, y+h+int(self.DELTA*h/w)//2),
            (0, 255, 0),
            2
        )

        return cropped_scheme, detected_scheme

    def crop_scheme(self):
        # self.show_image("Исходник")
        if self.filtration:
            self.original_img = self.filter(self.original_img)

        # Переводим изображение в HSV
        self.to_HSV(self.image)
        # self.show_image("RGB в HSV")  # 1
        # Принтим значение каналов в точке (x, y)
        # self.print_point(687, 235)

        self.image = clear_white(self.image)
        self.paint_scheme(manual=False)
        # self.show_image("Наложение маски на плату")  # 2
        self.to_gray()
        # self.show_image("HSV в RGB и в серые оттенки")  # 3
        self.treshold()
        # self.show_image("Пороговая обработка")  # 4
        self.closing()
        # self.show_image("Заполнение пробелов")  # 5
        self.opening()
        # self.show_image("Устранение артефактов")  # 6

        M = None
        if self.rotation:
            M = self.rotate()

        if M is not None:
            self.M = M
            self.rows, self.cols = self.image.shape

        self.contouring()
        # self.show_contour_image("Контурирование")  # 7
        self.find_center()

        cropped_scheme = self.draw_rect(M)
        # self.show_image("Результат")  # 8
        return cropped_scheme

    def get_original_image(self):
        if self.filtration:
            self.original_img = self.filter(self.original_img)

        return self.original_img

    def get_HSV_image(self, image):
        return self.to_HSV(image)


class ResistorsFinder():

    def __init__(self, cropped_scheme):
        self.scheme_img = cropped_scheme
        # Высчитывание относительной погрешности DELTA
        self.DELTA = (cropped_scheme.shape[0] + cropped_scheme.shape[1])/2.  \
            * 0.005
        self.contour_img = None
        self.contours = None
        self.largest_contour_index = None

    def to_HSV(self, image):
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return hsv_img

    def paint_resistors(self, hsv_scheme, resistor_low, resistor_high):
        mask2 = cv2.inRange(hsv_scheme, resistor_low, resistor_high)
        res_img = copy.deepcopy(hsv_scheme)
        res_img[mask2 > 0] = ([179, 0, 255])
        # viewImage("Наложение маски на плату", hsv_scheme)  # 10

        return res_img

    def to_gray(self, hsv_image):
        RGB_again = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        gray_img = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)
        # viewImage("HSV в RGB и в серые оттенки", self.gray_img)  # 11

        return gray_img

    def threshold(self, gray_img):
        ret, threshold = cv2.threshold(
            gray_img, 254, 255, cv2.THRESH_BINARY
        )
        # viewImage("Пороговая обработка", threshold)  # 12

        return threshold

    def closing(self, image, kx=3.5, ky=3.5):
        # Заполнение пробелов
        kernel = np.ones((int(ky*self.DELTA), int(kx*self.DELTA)), np.uint8)
        res_img = cv2.morphologyEx(
            image, cv2.MORPH_CLOSE, kernel
        )
        # viewImage("Заполнение пробелов", res_img)  # 13

        return res_img

    def opening(self, image, kx=2., ky=2.):
        # Устранение артефактов
        kernel = np.ones((int(ky*self.DELTA), int(kx*self.DELTA)), np.uint8)
        res_img = cv2.morphologyEx(
            image, cv2.MORPH_OPEN, kernel
        )
        # viewImage("Устранение артефактов", res_img)  # 14

        return res_img

    def contouring(self, threshold_img, min_area, max_area):
        # Заполнение пробелов
        kernel = np.ones((int(4*self.DELTA), int(6.25*self.DELTA)), np.uint8)
        new_threshold_img = cv2.morphologyEx(
            threshold_img, cv2.MORPH_CLOSE, kernel
        )

        self.contours, hierarchy = cv2.findContours(
            new_threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour_img = copy.deepcopy(self.scheme_img)
        cv2.drawContours(
            contour_img, self.contours, -1, (0, 0, 255), -1
        )
        res = find_contours_by_area(
            self.contours, min_area, max_area
        )

        return res

    def draw_rects(self, contours):
        image = copy.deepcopy(self.scheme_img)
        DELTA = self.DELTA

        for idx in contours:
            x, y, w, h = get_rectangle(self.contours, idx)
            cv2.rectangle(
                image,
                (x-int(DELTA*w/h)//2, y-int(DELTA*h/w)//2),
                (x+w+int(DELTA*w/h)//2, y+h+int(DELTA*h/w)//2),
                (0, 255, 0),
                2
            )

        return image

    def find_centers(self, contours, idxs):
        cnt = contours
        centers = []

        for i in idxs:
            M = cv2.moments(cnt[i])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))

        return centers

    def find_resistors(self, image):
        self.hsv_scheme = self.to_HSV(image)
        hsv_scheme = clear_white(self.hsv_scheme)

        self.paint_resistors(hsv_scheme)
        self.to_gray()
        self.threshold()
        self.closing()
        self.opening()
        img = None
        contours_idxs = self.contouring(img, min_area, max_area)

        # Найдем центры резисторов
        detected_resistors_centers = self.find_centers(
            self.contours, contours_idxs
        )
        resistors_img = self.draw_rects(contours_idxs)

        return detected_resistors_centers, resistors_img

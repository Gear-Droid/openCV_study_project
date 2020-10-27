import cv2
import numpy as np
import pprint


def open_image(filename):
    img = cv2.imread(filename)
    return img


def open_imageBGR2HSV(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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


if __name__ == "__main__":
    filename = 'rotated.jpg'

    while(True):
        original_img = open_image(filename)
        # viewImage("Исходник", original_img)  # original

        # Высчитываем погрешность DELTA для использования в
        # заполнении пробелов и кропе схемки
        # print((original_img.shape[0] + original_img.shape[1])/2. * 0.005)
        DELTA = (original_img.shape[0] + original_img.shape[1])/2. * 0.005

        # Узнаём значение зеленого в HSV
        green_bgr = np.uint8([[[0, 255, 0]]])
        green_hsv = BGR2HSV(green_bgr)
        # В HSV green = (60, 255, 255)

        img = cv2.imread(filename)

        # фильтры удаляющие артефакты камеры
        img = cv2.medianBlur(img, 3)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # Переводим изображение в HSV
        hsv_img = open_imageBGR2HSV(img)
        # viewImage("RGB в HSV", hsv_img)  # 1

        # Принтим значение каналов в [[y],[x]]
        # print(hsv_img[[1320], [2064]])

        # mask1 - фильтрует засветы затемняя их
        mask1 = cv2.inRange(
            hsv_img, np.array([0, 0, 250]), np.array([180, 2, 255])
        )
        hsv_img[mask1 > 0] = ([253, 1, 253])

        manual = False
        green_low, green_high = get_range(manual)
        mask2 = cv2.inRange(hsv_img, green_low, green_high)
        hsv_img[mask2 > 0] = ([255, 0, 255])
        # viewImage("Наложение маски на плату", hsv_img)  # 2

        RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)
        # viewImage("HSV в RGB и в серые оттенки", gray)  # 3

        ret, threshold = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
        # viewImage("Пороговая обработка", threshold)  # 4

        # Заполнение пробелов
        kernel = np.ones((int(2.5*DELTA), int(2.5*DELTA)), np.uint8)
        closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        # viewImage("Заполнение пробелов", closing)  # 5

        # Устранение артефактов
        kernel = np.ones((int(DELTA), int(DELTA)), np.uint8)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        # viewImage("Устранение артефактов", opening)  # 6

        contours, hierarchy = cv2.findContours(
            opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = open_image(filename)
        cv2.drawContours(contour_img, contours, -1, (0, 0, 255), -1)
        # viewImage("Контурирование", contour_img)  # 7

        largest_area, largest_contour_index = findGreatesContour(contours)

        pprint.pprint("Макс площадь: {}".format(largest_area))
        pprint.pprint("Индекс контура: {}".format(largest_contour_index))
        pprint.pprint("Кол-во контуров: {}".format(len(contours)))

        if largest_area > 0.95 * img.shape[0] * img.shape[1]:
            DELTA = 0

        # Далее найдем центр наибольшего контура
        cnt = contours[largest_contour_index]
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        pprint.pprint("x = {}".format(cX))
        pprint.pprint("y = {}".format(cY))

        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(
            img,
            (x-int(DELTA*w/h)//2, y-int(DELTA*h/w)//2),
            (x+w+int(DELTA*w/h)//2, y+h+int(DELTA*h/w)//2),
            (0, 255, 0),
            2
        )

        # viewImage('Результат', img)  # 8

        cropped_scheme = img[
            y-int(DELTA*h/w)//2: y+h+int(DELTA*h/w)//2+1,
            x-int(DELTA*w/h)//2: x+w+int(DELTA*w/h)//2+1,
            :
        ]

        viewImage("Кропнутая схемка", cropped_scheme)  # 9

        def on_update(val):
            edges = cv2.Canny(cropped_scheme, 10, 200)
            viewImage('Canny', edges)

        # Поиск резисторов на схемке
        hsv_scheme = open_imageBGR2HSV(cropped_scheme)

        """
        print(hsv_scheme[[914], [103]])
        print(hsv_scheme[[726], [149]])
        print(hsv_scheme[[574], [540]])
        """
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

        cv2.destroyAllWindows()
        break

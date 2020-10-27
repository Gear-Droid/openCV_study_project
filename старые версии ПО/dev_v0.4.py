import cv2
# import pprint
from Finders.finders import SchemeFinder, ResistorsFinder
from Finders.finders import clear_white, draw_resistors
import copy
import numpy as np


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


if __name__ == "__main__":
    filename = 'waka.jpg'

    k_DELTA = 0.005
    sf = SchemeFinder(filename, rotation=True,
                      k_DELTA=k_DELTA, filtration=True)

    # 1 Экран исходника
    orig_scheme_img = sf.get_original_image()

    # 2 HSV
    hsv_scheme_img = sf.get_HSV_image(orig_scheme_img)
    clear_scheme_image = clear_white(hsv_scheme_img)

    green_low = np.array([70, 120, 20], dtype="uint8")
    green_high = np.array([90, 255, 180], dtype="uint8")

    # 3 Закрашенная схемка
    painted_scheme = sf.paint_scheme(
        clear_scheme_image,
        green_low=green_low, green_high=green_high,
        manual=False)

    gray_scheme = sf.to_gray(image=painted_scheme)
    threshold_scheme = sf.threshold(image=gray_scheme)

    # 4_1 Закрашивание
    closing_scheme = sf.closing(image=threshold_scheme, kx=2.5, ky=2.5)
    # 4_2 Устранение шумов
    opening_scheme = sf.opening(image=closing_scheme, kx=1., ky=1.)

    M = None
    rotated_scheme = None

    if sf.rotation:
        M, rotated_scheme = sf.rotate(opening_scheme)
    else:
        rotated_scheme = copy.deepcopy(opening_scheme)

    if M is not None:
        sf.M = M
        sf.rows, sf.cols = opening_scheme.shape

    # 5 Контурирование
    contoured_scheme = sf.contouring(rotated_scheme)

    # 6 Результат
    cropped_scheme, detected_scheme = sf.draw_rect(M)
    viewImage('Cropped', cropped_scheme)

    # Поиск резисторов на схемке
    rf = ResistorsFinder(cropped_scheme)

    hsv_resistors_img = rf.to_HSV(cropped_scheme)

    print(hsv_resistors_img[138][168])
    resistors_clear_img = clear_white(hsv_resistors_img)

    resistor_low = np.array([83, 21, 170], dtype="uint8")
    resistor_high = np.array([105, 85, 240], dtype="uint8")
    painted_resistors_img = rf.paint_resistors(
        hsv_resistors_img,
        resistor_low=resistor_low, resistor_high=resistor_high
    )

    gray_resistors_scheme = rf.to_gray(painted_resistors_img)
    threshold_resistors_scheme = rf.threshold(gray_resistors_scheme)
    closing_resistors_scheme = rf.closing(
        threshold_resistors_scheme, kx=3.5, ky=3.5)
    opening_resistors_scheme = rf.opening(
        closing_resistors_scheme, kx=2., ky=2.)
    viewImage('Обнаруженные резисторы v2', opening_resistors_scheme)
    viewImage('Cropped', cropped_scheme)

    # Ширина резистора
    w = 10
    # Высота резистора
    h = 10

    # Перерисовка резисторов
    """res_img = draw_resistors(
        cropped_scheme, detected_resistors_centers, w, h
    )
    viewImage('Обнаруженные резисторы v2', res_img)
    """
    cv2.destroyAllWindows()

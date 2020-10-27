import cv2
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage

# import pprint
from Finders.finders import SchemeFinder, ResistorsFinder
from Finders.finders import clear_white, draw_resistors
import copy
import numpy as np


def open_image(filename):
    img = cv2.imread(filename)
    return img


def threads_are_running(*args):
    running_threads = []

    for th in args:
        if th.isRunning():
            running_threads.append(th)

    return running_threads


class LogicThread(QThread):
    def __init__(self, main_window):
        # self.instance = self
        super().__init__()
        self.main_window = main_window

        self.th1 = MainVideoThread(self)
        self.th1.changePixmap.connect(main_window.setImage)
        self.th1.changePixmap1.connect(main_window.setImage1)

    def run(self):
        self.th1.start()

        while True:
            upper_index = self.main_window.tabWidget.currentIndex()
            lower_index = self.main_window.tabWidget_2.currentIndex()

            if upper_index == 0:
                # Раздел "Обнаружение схемы"

                if lower_index == 0:
                    # Подраздел "Исходник"
                    self.th1.current = '1_original'

                if lower_index == 1:
                    # Подраздел "HSV"
                    self.th1.current = '1_HSV'

                if lower_index == 2:
                    # Подраздел "Закрашивание"
                    self.th1.current = '1_painted'

                if lower_index == 3:
                    # Подраздел "Пороговая обработка"
                    self.th1.current = '1_closing_opening'

                if lower_index == 4:
                    # Подраздел "Контурирование"
                    self.th1.current = '1_contouring'

                if lower_index == 5:
                    # Подраздел "Результат"
                    self.th1.current = '1_result'

            self.msleep(100)


class MainVideoThread(QThread):
    changePixmap = pyqtSignal(QImage)
    changePixmap1 = pyqtSignal(QImage)

    def __init__(self, current):
        super().__init__()
        self.images_dict = {
            '1_original': 1,
            '1_HSV': 2,
            '1_painted': 3,
            '1_closing_opening': 4,
            '1_contouring': 5,
            '1_result': 6,
        }
        self.current = '1_original'

    def run(self):
        while True:
            filename = 'scheme_photos/waka.jpg'

            number_of_image_to_view = self.images_dict[self.current]

            k_DELTA = 0.005
            sf = SchemeFinder(filename, rotation=True,
                              k_DELTA=k_DELTA, filtration=True)

            # 1 Экран исходника
            orig_scheme_img = sf.get_original_image()
            image_to_view = orig_scheme_img

            closing_scheme = orig_scheme_img
            opening_scheme = orig_scheme_img

            if number_of_image_to_view > 1:
                # 2 HSV
                hsv_scheme_img = sf.get_HSV_image(orig_scheme_img)
                clear_scheme_image = clear_white(hsv_scheme_img)
                image_to_view = clear_scheme_image

                if number_of_image_to_view > 2:
                    green_low = np.array([70, 120, 20], dtype="uint8")
                    green_high = np.array([90, 255, 180], dtype="uint8")

                    # 3 Закрашенная схемка
                    painted_scheme = sf.paint_scheme(
                        clear_scheme_image,
                        green_low=green_low, green_high=green_high,
                        manual=False)

                    gray_scheme = sf.to_gray(image=painted_scheme)
                    threshold_scheme = sf.threshold(image=gray_scheme)
                    image_to_view = threshold_scheme

                    if number_of_image_to_view > 3:
                        # 4_1 Закрашивание
                        closing_scheme = sf.closing(
                            image=threshold_scheme, kx=2.5, ky=2.5)
                        # 4_2 Устранение шумов
                        opening_scheme = sf.opening(
                            image=closing_scheme, kx=1., ky=1.)

                        M = None
                        rotated_scheme = None

                        if sf.rotation:
                            M, rotated_scheme = sf.rotate(opening_scheme)
                        else:
                            rotated_scheme = copy.deepcopy(opening_scheme)

                        if M is not None:
                            sf.M = M
                            sf.rows, sf.cols = opening_scheme.shape

                        image_to_view = opening_scheme

                        if number_of_image_to_view > 4:
                            # 5 Контурирование
                            contoured_scheme = sf.contouring(rotated_scheme)

                            image_to_view = contoured_scheme

                            if number_of_image_to_view > 5:
                                # 6 Результат
                                cropped_scheme, detected_scheme = sf.draw_rect(M)
                                # viewImage('Cropped', cropped_scheme)
                                image_to_view = cropped_scheme

                                # Поиск резисторов на схемке
                                rf = ResistorsFinder(cropped_scheme)

                                hsv_resistors_img = rf.to_HSV(cropped_scheme)

                                # print(hsv_resistors_img[138][168])
                                resistors_clear_img = clear_white(
                                    hsv_resistors_img)

                                resistor_low = np.array(
                                    [83, 21, 170], dtype="uint8")
                                resistor_high = np.array(
                                    [105, 85, 240], dtype="uint8")
                                painted_resistors_img = rf.paint_resistors(
                                    hsv_resistors_img,
                                    resistor_low=resistor_low,
                                    resistor_high=resistor_high
                                )

                                gray_resistors_scheme = rf.to_gray(
                                    painted_resistors_img)
                                threshold_resistors_scheme = rf.threshold(
                                    gray_resistors_scheme)
                                closing_resistors_scheme = rf.closing(
                                    threshold_resistors_scheme, kx=3.5, ky=3.5)
                                opening_resistors_scheme = rf.opening(
                                    closing_resistors_scheme, kx=2., ky=2.)

                                # Ширина резистора
                                w = 10
                                # Высота резистора
                                h = 10

                                # Перерисовка резисторов
                                """res_img = draw_resistors(
                                    cropped_scheme, detected_resistors_centers, w, h
                                )
                                # viewImage('Обнаруженные резисторы v2', res_img)
                                """

            if number_of_image_to_view != 4:
                rgbImage = cv2.cvtColor(image_to_view, cv2.COLOR_BGR2RGB)

                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(
                    rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

            else:
                rgbImage = cv2.cvtColor(
                    closing_scheme, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(
                    rgbImage.data, w, h,
                    bytesPerLine, QImage.Format_RGB888)
                p1 = convertToQtFormat.scaled(320, 240, Qt.KeepAspectRatio)
                self.changePixmap.emit(p1)

                rgbImage = cv2.cvtColor(
                    opening_scheme, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(
                    rgbImage.data, w, h,
                    bytesPerLine, QImage.Format_RGB888)
                p2 = convertToQtFormat.scaled(320, 240, Qt.KeepAspectRatio)
                self.changePixmap1.emit(p2)

            self.msleep(100)


class SourceVideoThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while True:
            ret, frame = self.cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(
                    rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
            print('th1 is running')
            self.msleep(200)

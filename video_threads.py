import cv2
from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage

# import pprint
from Finders.finders import SchemeFinder, ResistorsFinder
from Finders.finders import clear_white, draw_resistors
import copy
import numpy as np

import json


class SettingsFileReadingError(KeyError):
    """Ошибка парсинга файла settings.json"""


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

            if upper_index == 0:
                # Раздел "Обнаружение схемы"
                lower_index = self.main_window.tabWidget_2.currentIndex()

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

            if upper_index == 1:
                # Раздел "Выделение резисторов"
                lower_index = self.main_window.tabWidget_3.currentIndex()

                if lower_index == 0:
                    # Подраздел "Выбор опорной точки"
                    self.th1.current = '2_point'

                if lower_index == 1:
                    # Подраздел "Выделение резисторов"
                    self.th1.current = '2_paint_resistors'

                if lower_index == 2:
                    # Подраздел "Результат"
                    self.th1.current = '2_result'

            if upper_index == 2:
                # Раздел "Проверка резисторов"
                pass

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
            '2_point': 10,
            '2_paint_resistors': 11,
            '2_result': 12,
        }
        self.current = '1_original'

    def run(self):
        while True:
            second_stage = False
            number_of_image_to_view = self.images_dict[self.current]

            settings_dict = {}
            with open('settings.json', 'r') as infile:
                settings_dict = json.load(infile)

            if settings_dict.get('1_original').get('source') == "Image":
                filename = settings_dict.get('1_original').get('filename')
            else:
                image = None

            filtration = bool(settings_dict.get(
                '1_original').get('image filtration')
                )
            k_DELTA = float(settings_dict.get('1_original').get('DELTA'))
            if filtration is None or k_DELTA is None:
                raise SettingsFileReadingError

            rotation = bool(settings_dict.get(
                '1_result').get('image rotation')
                )
            if rotation is None:
                raise SettingsFileReadingError

            sf = SchemeFinder(filename, rotation=rotation,
                              k_DELTA=k_DELTA, filtration=filtration)
            self.sf = sf

            # 1 Экран исходника
            orig_scheme_img = sf.get_original_image()
            image_to_view = orig_scheme_img

            closing_scheme = None
            opening_scheme = None
            cropped_scheme = None

            if number_of_image_to_view > 1:
                # 2 HSV
                hsv_scheme_img = sf.get_HSV_image(orig_scheme_img)
                clear_scheme_image = clear_white(hsv_scheme_img)
                image_to_view = clear_scheme_image

                if number_of_image_to_view > 2:
                    low_H = int(settings_dict.get('1_painted').get('min_H'))
                    low_S = int(settings_dict.get('1_painted').get('min_S'))
                    low_V = int(settings_dict.get('1_painted').get('min_V'))
                    high_H = int(settings_dict.get('1_painted').get('max_H'))
                    high_S = int(settings_dict.get('1_painted').get('max_S'))
                    high_V = int(settings_dict.get('1_painted').get('max_V'))
                    if low_H is None or low_S is None or low_V is None  \
                            or high_H is None or high_S is None or  \
                            high_V is None:
                        raise SettingsFileReadingError

                    green_low = np.array(
                        [low_H, low_S, low_V], dtype="uint8")
                    green_high = np.array(
                        [high_H, high_S, high_V], dtype="uint8")

                    # 3 Закрашенная схемка
                    painted_scheme = sf.paint_scheme(
                        clear_scheme_image,
                        green_low=green_low, green_high=green_high,
                        manual=False)

                    gray_scheme = sf.to_gray(image=painted_scheme)
                    threshold_scheme = sf.threshold(image=gray_scheme)
                    image_to_view = threshold_scheme

                    if number_of_image_to_view > 3:
                        closing_kx = float(
                            settings_dict.get(
                                '1_closing_opening').get('closing_kx'))
                        closing_ky = float(
                            settings_dict.get(
                                '1_closing_opening').get('closing_ky'))
                        opening_kx = float(
                            settings_dict.get(
                                '1_closing_opening').get('opening_kx'))
                        opening_ky = float(
                            settings_dict.get(
                                '1_closing_opening').get('opening_ky'))

                        if closing_kx is None or closing_ky is None  \
                                or opening_kx is None or opening_ky is None:
                            raise SettingsFileReadingError

                        # 4_1 Закрашивание
                        closing_scheme = sf.closing(
                            image=threshold_scheme,
                            kx=closing_kx, ky=closing_ky)
                        # 4_2 Устранение шумов
                        opening_scheme = sf.opening(
                            image=closing_scheme,
                            kx=opening_kx, ky=opening_ky)

                        M = None
                        rotated_scheme = None

                        if sf.rotation:
                            M, rotated_scheme = sf.rotate(opening_scheme)
                        else:
                            rotated_scheme = copy.deepcopy(opening_scheme)

                        if M is not None:
                            sf.M = M
                            sf.rows, sf.cols = opening_scheme.shape
                        else:
                            sf.M = None

                        image_to_view = closing_scheme

                        if number_of_image_to_view > 4:
                            # 5 Контурирование
                            contoured_scheme = sf.contouring(rotated_scheme)

                            image_to_view = contoured_scheme

                            if number_of_image_to_view > 5:
                                # 6 Результат
                                cropped_scheme, detected_scheme = sf.draw_rect(
                                    M
                                )
                                image_to_view = cropped_scheme

                                second_stage = True

            if second_stage:
                # Поиск резисторов на схемке
                rf = ResistorsFinder(cropped_scheme)
                self.rf = rf
                image_to_view = cropped_scheme

                if number_of_image_to_view > 10:
                    hsv_resistors_img = rf.to_HSV(cropped_scheme)
                    resistors_clear_img = clear_white(hsv_resistors_img)

                    low_H = int(settings_dict.get(
                        '2_paint_resistors').get('min_H'))
                    low_S = int(settings_dict.get(
                        '2_paint_resistors').get('min_S'))
                    low_V = int(settings_dict.get(
                        '2_paint_resistors').get('min_V'))
                    high_H = int(settings_dict.get(
                        '2_paint_resistors').get('max_H'))
                    high_S = int(settings_dict.get(
                        '2_paint_resistors').get('max_S'))
                    high_V = int(settings_dict.get(
                        '2_paint_resistors').get('max_V'))
                    if low_H is None or low_S is None or low_V is None  \
                            or high_H is None or high_S is None  \
                            or high_V is None:
                        raise SettingsFileReadingError

                    resistor_low = np.array(
                        [low_H, low_S, low_V], dtype="uint8")
                    resistor_high = np.array(
                        [high_H, high_S, high_V], dtype="uint8")

                    painted_resistors_img = rf.paint_resistors(
                        resistors_clear_img,
                        resistor_low=resistor_low,
                        resistor_high=resistor_high
                    )

                    gray_resistors_scheme = rf.to_gray(
                        painted_resistors_img)
                    threshold_resistors_scheme = rf.threshold(
                        gray_resistors_scheme)
                    image_to_view = threshold_resistors_scheme

                    if number_of_image_to_view > 11:

                        closing_resistors_scheme = rf.closing(
                            threshold_resistors_scheme, kx=3.5, ky=3.5)
                        opening_resistors_scheme = rf.opening(
                            closing_resistors_scheme, kx=2., ky=2.)

                        # Ширина резистора
                        w = 220
                        # Высота резистора
                        h = 96

                        # min_area, max_area = 5000, 30000
                        min_area, max_area = 3000, 15000
                        contours_idxs = rf.contouring(opening_resistors_scheme,
                                                      min_area, max_area)

                        # Найдем центры резисторов
                        detected_resistors_centers = rf.find_centers(
                            rf.contours, contours_idxs
                        )
                        resistors_img = rf.draw_rects(contours_idxs)
                        image_to_view = resistors_img
                        # Перерисовка резисторов
                        res_img = draw_resistors(
                            cropped_scheme, detected_resistors_centers, w, h
                        )
                        # viewImage('Обнаруженные резисторы v2', res_img)
                        image_to_view = res_img

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

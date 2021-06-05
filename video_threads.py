import cv2
import copy
import json
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

from sklearn import svm
from sklearn.cluster import KMeans
# import pprint
from Finders.finders import (
    SchemeFinder, ResistorsFinder,
    clear_white, draw_resistors
)


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


def parse_HSV(*args):
    try:
        if len(args) == 3:
            HSV = [None, None, None]
            ind = 0
            for arg in args:
                uint_value = int(arg)
                if uint_value is None:
                    raise ValueError
                if uint_value > 255:
                    raise ValueError

                HSV[ind] = uint_value
                ind += 1
            return HSV[0], HSV[1], HSV[2]
        else:
            return None
    except ValueError:
        return None


def GetMostPopulatedRegion(bins, counts):
    imax = np.argmax(counts)
    binWidth = np.gradient(bins)[0]
    lo = bins[imax] - binWidth/2.
    hi = bins[imax+1] + binWidth/2.
    return lo, hi


class LogicThread(QThread):
    def __init__(self, main_window):
        # self.instance = self
        super().__init__()
        self.main_window = main_window

        self.th1 = MainVideoThread(self)
        self.th1.changePixmap.connect(main_window.setImage)
        self.th1.changePixmap1.connect(main_window.setImage1)

    def run(self):
        if not self.th1.isRunning():
            self.th1.start()
        refresh_flag_1 = True
        refresh_flag_2 = True
        refresh_flag_3 = True
        refresh_flag_4 = True
        refresh_flag_5 = True


        while True:
            upper_index = self.main_window.tabWidget.currentIndex()

            if upper_index == 0:
                # Раздел "Обнаружение схемы"
                lower_index = self.main_window.tabWidget_2.currentIndex()

                if lower_index == 0:
                    # Подраздел "Исходник"
                    self.th1.current = '1_original'

                    settings_dict = {}
                    with open('settings.json', 'r') as infile:
                        settings_dict = json.load(infile)

                    if refresh_flag_5:
                        if bool(settings_dict['1_original']['image filtration']):
                            self.main_window.checkBox.setChecked(True)
                        refresh_flag_5 = False
                if lower_index != 0:
                    refresh_flag_5 = True

                if lower_index == 1:
                    # Подраздел "HSV"
                    self.th1.current = '1_HSV'

                if lower_index == 2:
                    # Подраздел "Закрашивание"
                    self.th1.current = '1_painted'

                    settings_dict = {}
                    with open('settings.json', 'r') as infile:
                        settings_dict = json.load(infile)

                    if refresh_flag_1:
                        self.main_window.lineEdit_8.clear()
                        self.main_window.lineEdit_8.insert(
                            str(settings_dict['1_painted']['min_H'])
                        )
                        self.main_window.lineEdit_9.clear()
                        self.main_window.lineEdit_9.insert(
                            str(settings_dict['1_painted']['min_S'])
                        )
                        self.main_window.lineEdit_10.clear()
                        self.main_window.lineEdit_10.insert(
                            str(settings_dict['1_painted']['min_V'])
                        )
                        self.main_window.lineEdit_11.clear()
                        self.main_window.lineEdit_11.insert(
                            str(settings_dict['1_painted']['max_H'])
                        )
                        self.main_window.lineEdit_12.clear()
                        self.main_window.lineEdit_12.insert(
                            str(settings_dict['1_painted']['max_S'])
                        )
                        self.main_window.lineEdit_13.clear()
                        self.main_window.lineEdit_13.insert(
                            str(settings_dict['1_painted']['max_V'])
                        )
                        refresh_flag_1 = False
                if lower_index != 2:
                    refresh_flag_1 = True

                if lower_index == 3:
                    # Подраздел "Пороговая обработка"
                    self.th1.current = '1_closing_opening'

                    settings_dict = {}
                    with open('settings.json', 'r') as infile:
                        settings_dict = json.load(infile)

                    if refresh_flag_3:
                        self.main_window.lineEdit_5.clear()
                        self.main_window.lineEdit_5.insert(
                            str(settings_dict['1_closing_opening']['closing_kx'])
                        )
                        self.main_window.lineEdit_6.clear()
                        self.main_window.lineEdit_6.insert(
                            str(settings_dict['1_closing_opening']['closing_ky'])
                        )
                        self.main_window.lineEdit_7.clear()
                        self.main_window.lineEdit_7.insert(
                            str(settings_dict['1_closing_opening']['opening_kx'])
                        )
                        self.main_window.lineEdit_14.clear()
                        self.main_window.lineEdit_14.insert(
                            str(settings_dict['1_closing_opening']['opening_ky'])
                        )
                        refresh_flag_3 = False
                if lower_index != 3:
                    refresh_flag_3 = True

                if lower_index == 4:
                    # Подраздел "Контурирование"
                    self.th1.current = '1_contouring'

                if lower_index == 5:
                    # Подраздел "Результат"
                    self.th1.current = '1_result'

                    settings_dict = {}
                    with open('settings.json', 'r') as infile:
                        settings_dict = json.load(infile)

                    if refresh_flag_4:
                        if bool(settings_dict['1_result']['image rotation']):
                            self.main_window.checkBox_2.setChecked(True)
                        refresh_flag_4 = False
                if lower_index != 5:
                    refresh_flag_4 = True

            if upper_index != 0:
                refresh_flag_1 = True
                refresh_flag_3 = True
                refresh_flag_4 = True
                refresh_flag_5 = True

            if upper_index == 1:
                # Раздел "Выделение резисторов"
                lower_index = self.main_window.tabWidget_3.currentIndex()

                if lower_index == 0:
                    # Подраздел "Выбор опорной точки"
                    self.th1.current = '2_point'

                if lower_index == 1:
                    # Подраздел "Выделение резисторов"
                    self.th1.current = '2_paint_resistors'

                    settings_dict = {}
                    with open('settings.json', 'r') as infile:
                        settings_dict = json.load(infile)

                    if refresh_flag_2:
                        self.main_window.lineEdit_15.clear()
                        self.main_window.lineEdit_15.insert(
                            str(settings_dict['2_paint_resistors']['min_H'])
                        )
                        self.main_window.lineEdit_16.clear()
                        self.main_window.lineEdit_16.insert(
                            str(settings_dict['2_paint_resistors']['min_S'])
                        )
                        self.main_window.lineEdit_17.clear()
                        self.main_window.lineEdit_17.insert(
                            str(settings_dict['2_paint_resistors']['min_V'])
                        )
                        self.main_window.lineEdit_18.clear()
                        self.main_window.lineEdit_18.insert(
                            str(settings_dict['2_paint_resistors']['max_H'])
                        )
                        self.main_window.lineEdit_19.clear()
                        self.main_window.lineEdit_19.insert(
                            str(settings_dict['2_paint_resistors']['max_S'])
                        )
                        self.main_window.lineEdit_20.clear()
                        self.main_window.lineEdit_20.insert(
                            str(settings_dict['2_paint_resistors']['max_V'])
                        )
                        refresh_flag_2 = False
                if lower_index != 1:
                    refresh_flag_2 = True

                if lower_index == 2:
                    # Подраздел "Результат"
                    self.th1.current = '2_result'
            if upper_index != 1:
                refresh_flag_2 = True

            if upper_index == 2:
                # Раздел "Проверка резисторов"
                lower_index = self.main_window.tabWidget_4.currentIndex()
                if lower_index == 0:
                    # Подраздел "Обучение модели"
                    self.th1.current = '3_cropped_resistors'

                if lower_index == 1:
                    # Подраздел "Обнаружение цветовых полосок"
                    self.th1.current = '3_strip_detection'

                if lower_index == 2:
                    # Подраздел "Результат"
                    self.th1.current = '3_result'

                    settings_dict = {}
                    with open('settings.json', 'r') as infile:
                        settings_dict = json.load(infile)

                    self.main_window.label_34.setText(settings_dict['result_text'])
            if upper_index != 2:
                pass

            self.msleep(100)


def paint_finded_stripes(res, pixels):
    for pix in pixels:
        res[:,int(pix)] = np.array((255.,255.,255.))
    return res


def calculate_resistance(stripes):
    r_value = 0
    st_value = {
        'Черный': 0,
        'Коричневый': 1,
        'Красный': 2,
        'Оранжевый': 3,
        'Желтый': 4,
        'Зеленый': 5,
        'Синий': 6,
        'Фиолетовый': 7,
        'Серый': 8,
        'Белый': 9,
    }
    st_koef = {
        'Черный': 1,
        'Коричневый': 10,
        'Красный': 100,
        'Оранжевый': 1000,
        'Желтый': 10000,
        'Зеленый': 100000,
        'Синий': 1000000,
        'Фиолетовый': 10000000,
        'Серый': 100000000,
        'Белый': 1000000000,
        'Золотой': 0.1,
        'Серебристый': 0.01,
    }
    for i in range(4):
        st = stripes[i]
        if i == 0:
            if st_value[st] == 0:
                return 0
        if i < 3:
            r_value += st_value[st] * 10**(2-i)
        if i == 3:
            try:
                r_value = r_value * st_koef[st]
            except KeyError:
                return 0
    return r_value


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
            '3_cropped_resistors': 21,
            '3_strip_detection': 22,
            '3_result': 23,
        }
        self.current = '1_original'

    def run(self):
        learning_flag = True
        tmp = True

        while True:
            second_stage = False
            third_stage = False

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

            stripCount = 5
            cropped_resistors = []
            kmeans = None

            text_output = str()

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

            detected_resistors_centers= []
            if second_stage:
                final_scheme_img = copy.deepcopy(cropped_scheme)
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
                        # w = 220
                        w = 220
                        # Высота резистора
                        # h = 96
                        h = 86

                        # min_area, max_area = 5000, 30000
                        min_area, max_area = 7000, 25000
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
                        image_to_view = cropped_scheme
                        green_border_width = 2
                        res_h, res_w = cropped_scheme.shape[0], cropped_scheme.shape[1]
                        for ctr in detected_resistors_centers:
                            resistor_crop = cropped_scheme[
                                ctr[1]-h//4+green_border_width:ctr[1]+h//4-green_border_width,
                                ctr[0]-w//2+green_border_width:ctr[0]+w//2-green_border_width,
                                :
                            ]
                            cropped_resistors.append(resistor_crop)
                        if number_of_image_to_view > 12:
                            third_stage = True

            if third_stage:
                # Обучение модели классификатора
                if learning_flag:
                    learning_images_filenames = (
                        '1_brown', '2_black', '3_yellow', '4_white', '5_orange',
                        '6_gray', '7_violet', '8_red', '9_green',
                    )
                    y = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                    farben = {
                        0: 'Коричневый',
                        1: 'Черный',
                        2: 'Желтый',
                        3: 'Белый',
                        4: 'Оранжевый',
                        5: 'Серый',
                        6: 'Фиолетовый',
                        7: 'Красный',
                        8: 'Зеленый'
                    }
                    clf = svm.SVC()
                    n = len(learning_images_filenames)
                    colors_vector = np.empty((n, 3))
                    k_h = 2.
                    k_s = 1.4
                    k_v = 1.
                    for i, fname in zip(range(n), learning_images_filenames):
                        bgr_img = open_image('learning/{}.jpg'.format(fname))
                        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
                        h_mean, s_mean, v_mean = round(hsv_img[:,:,0].mean(), 3), round(hsv_img[:,:,1].mean(), 3), round(hsv_img[:,:,2].mean(), 3)
                        colors_vector[i,0], colors_vector[i,1], colors_vector[i,2] = h_mean*k_h, s_mean*k_s, v_mean*k_v
                        clf.fit(colors_vector, y)
                    learning_flag = False

                resistors_stacked_img = tuple([res for res in cropped_resistors])
                stacked_img = copy.deepcopy(np.vstack(resistors_stacked_img))
                image_to_view = stacked_img

                if number_of_image_to_view > 21:
                    # Распознавание номинала резисторов
                    stripped_resistors = []
                    detected_pixels_list = []
                    whitestipe_imgs_list = []

                    for resistor_img in cropped_resistors:
                        bgr_resistor_img = resistor_img
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

                        counts, bins, bars = plt.hist(
                            hue_mean_line, bins=6
                        )
                        lo1, hi1 = GetMostPopulatedRegion(bins, counts)

                        grad = np.abs(np.gradient(hue_mean_line))
                        length = np.linspace(0, len(hue_mean_line)-1, len(hue_mean_line))

                        ix = ((hue_mean_line < lo1) | (hue_mean_line > hi1)) & (grad < max(grad)*0.13)

                        # KMeans
                        stripCount = 5
                        detected_pixels = length[ix]
                        length_ix = detected_pixels.reshape(-1, 1)
                        kmeans = KMeans(n_clusters=stripCount, random_state=0).fit(length_ix)

                        detected_pixels_list.append(detected_pixels)
                        whitestipe_imgs_list = tuple(
                            [paint_finded_stripes(res, pixels) for res, pixels in zip(cropped_resistors, detected_pixels_list)]
                        )
                        # Определение ориентации резистора по крупной полосе с краю
                        reverse_stripes = False
                        # reverse stripes detector
                        start_pix_value = length[ix][0]
                        start_pix_count = 0
                        current_expected = start_pix_value
                        for pixel_coord in length[ix]:
                            if pixel_coord == current_expected:
                                start_pix_count+=1
                                current_expected+=1
                            else:
                                break
                        finish_pix_value = length[ix][-1]
                        finish_pix_count = 0
                        current_expected = finish_pix_value
                        for pixel_coord in reversed(length[ix]):
                            if pixel_coord == current_expected:
                                finish_pix_count+=1
                                current_expected-=1
                            else:
                                break
                        if start_pix_count <= finish_pix_count:
                            reverse_stripes = True

                        resistor_stripes = []
                        for ctr_float in kmeans.cluster_centers_:
                            ctr_int = int(ctr_float)
                            H_mean = round(hue_mean_line[ctr_int-2:ctr_int+3].sum()/5., 2)
                            S_mean = round(saturation_mean_line[ctr_int-2:ctr_int+3].sum()/5., 2)
                            V_mean = round(visibility_mean_line[ctr_int-2:ctr_int+3].sum()/5., 2)
                            resistor_stripes.append((ctr_int, [H_mean, S_mean, V_mean]))

                        resistor_stripes = sorted(resistor_stripes, key=lambda index: index[0], reverse=reverse_stripes)
                        stripped_resistors.append(resistor_stripes)

                    for img, resistor in zip(whitestipe_imgs_list, stripped_resistors):
                        for strip in resistor:
                            strip_ctr_pixel = strip[0]
                            img[img.shape[0]//2-2:img.shape[0]//2+2,strip_ctr_pixel-2:strip_ctr_pixel+2] = np.array((0.,0.,255.))
                    image_to_view = np.vstack(whitestipe_imgs_list)
                    for_conf_matrix = []
                    y_true = [
                        [farben[0], farben[1], farben[6], farben[2], farben[0]],
                        [farben[0], farben[4], farben[0], farben[3], farben[0]],
                        [farben[0], farben[4], farben[4], farben[4], farben[0]],
                        [farben[0], farben[4], farben[6], farben[5], farben[2]],
                        [farben[0], farben[4], farben[4], farben[7], farben[8]],
                        [farben[0], farben[1], farben[6], farben[2], farben[0]],
                    ]
                    painting_resistors = []
                    resistors_farben_lists = []
                    for res_num in range(len(stripped_resistors)):
                        stripColors = np.empty((stripCount, 3))
                        for i in range(stripCount):
                            stripColors[i,0] = stripped_resistors[res_num][i][1][0]*k_h
                            stripColors[i,1] = stripped_resistors[res_num][i][1][1]*k_s
                            stripColors[i,2] = stripped_resistors[res_num][i][1][2]*k_v
                        prediction = clf.predict(stripColors)
                        farben_list = list(map(lambda x: farben[x], prediction))
                        resistors_farben_lists.append(farben_list)
                        for_conf_matrix.extend(farben_list)

                        text_output += 'Резистор #{}\n{}\n'.format(res_num, farben_list)
                        if y_true[res_num] == farben_list:
                            text_output += 'Верно!\n\n'
                            painting_resistors.append(True)
                        else:
                            text_output += 'Неверно!\n\n'
                            painting_resistors.append(False)

                    if number_of_image_to_view > 22:
                        image_to_view = stacked_img

                        settings_dict = {}
                        with open('settings.json', 'r') as infile:
                            settings_dict = json.load(infile)
                        settings_dict['result_text'] = text_output
                        with open('settings.json', 'w') as outfile:
                            json.dump(settings_dict, outfile)

                        y_true = [
                            farben[0], farben[1], farben[6], farben[2], farben[0],
                            farben[0], farben[4], farben[0], farben[3], farben[0],
                            farben[0], farben[4], farben[4], farben[4], farben[0],
                            farben[0], farben[4], farben[6], farben[5], farben[2],
                            farben[0], farben[4], farben[4], farben[7], farben[8],
                            farben[0], farben[1], farben[6], farben[2], farben[0],
                        ]
                        y_pred = for_conf_matrix

                        from sklearn.metrics import confusion_matrix
                        print("Правильные ответы:")
                        for i in range(0, 30, 5):
                            print("{}".format(y_true[i:i+5]))
                        print()
                        print("Ответы полученные алгоритмом:")
                        for i in range(0, 30, 5):
                            print("{}".format(y_pred[i:i+5]))
                        print()
                        print("Матрица ошибок (confusion matrix):")
                        print(confusion_matrix(y_true, y_pred))
                        for flag, ctr, farbens in zip(painting_resistors, detected_resistors_centers, resistors_farben_lists):
                            r_value = calculate_resistance(farbens)
                            if r_value//1000 == 0:
                                r_value = "{} Om".format(r_value)
                            if r_value//1000000 > 0:
                                r_value = "{} MOm".format(r_value/1000000)
                            elif r_value//1000 > 0:
                                r_value = "{} kOm".format(r_value/1000)
                            print(flag, '-', r_value)
                            # Center coordinates
                            center_coordinates = ctr
                            # Radius of circle
                            radius = 20
                            if flag:
                                # Green color in BGR
                                color = (0, 255, 0)
                            else:
                                # Red color in BGR
                                color = (0, 0, 255)
                            # Line thickness of -1 px
                            thickness = -1
                            # Using cv2.circle() method
                            # Draw a circle of red color of thickness -1 px
                            image_to_view = cv2.circle(final_scheme_img, center_coordinates, radius, color, thickness)

                            # font
                            font = cv2.FONT_HERSHEY_SIMPLEX                            
                            # fontScale
                            fontScale = 1
                            # Line thickness of 2 px
                            thickness = 2
                            # Using cv2.putText() method
                            coord = (center_coordinates[0]-20, center_coordinates[1]-40)
                            image = cv2.putText(final_scheme_img, r_value, coord, font, 
                                            fontScale, color, thickness, cv2.LINE_AA)

            if number_of_image_to_view != 4:
                if image_to_view.shape[0] != 0 and image_to_view.shape[1] != 0:
                    rgbImage = cv2.cvtColor(image_to_view, cv2.COLOR_BGR2RGB)

                    h, w, ch = rgbImage.shape
                    bytesPerLine = ch * w
                    convertToQtFormat = QImage(
                        rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                    p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                    self.changePixmap.emit(p)

                else:
                    pixmap = QPixmap('images for Qt5/no_image.png')
                    self.changePixmap.emit(pixmap)

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

import sys  # sys нужен для передачи argv в QApplication
from PyQt5 import QtWidgets
import os
import re

from Формы.output import design_main  # Это наш конвертированный файл дизайна

from PyQt5.QtWidgets import QLabel, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSlot

from video_threads import LogicThread

import json


class MainApp(QtWidgets.QMainWindow, design_main.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design_main.py
        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна

        # Создаем camera_label1, в котором будем хранить фото или видео
        pixmap = QPixmap('images for Qt5/no_image.png')

        # метки камер
        self.camera_label1 = QLabel(self)
        self.camera_label1.setPixmap(pixmap)
        self.camera_label2 = QLabel(self)
        self.camera_label2.setPixmap(pixmap)
        self.camera_label3 = QLabel(self)
        self.camera_label3.setPixmap(pixmap)
        self.camera_label4_1 = QLabel(self)
        self.camera_label4_1.setPixmap(pixmap)
        self.camera_label4_2 = QLabel(self)
        self.camera_label4_2.setPixmap(pixmap)
        self.camera_label5 = QLabel(self)
        self.camera_label5.setPixmap(pixmap)
        self.camera_label6 = QLabel(self)
        self.camera_label6.setPixmap(pixmap)
        self.camera_label7 = QLabel(self)
        self.camera_label7.setPixmap(pixmap)
        self.camera_label8 = QLabel(self)
        self.camera_label8.setPixmap(pixmap)
        self.camera_label9 = QLabel(self)
        self.camera_label9.setPixmap(pixmap)
        self.camera_label10 = QLabel(self)
        self.camera_label10.setPixmap(pixmap)
        self.camera_label11 = QLabel(self)
        self.camera_label11.setPixmap(pixmap)
        self.camera_label12 = QLabel(self)
        self.camera_label12.setPixmap(pixmap)

        # лэйауты для хранения изображений с камеры
        self.horizontalLayout_5.addWidget(self.camera_label1)
        self.horizontalLayout_18.addWidget(self.camera_label2)
        self.horizontalLayout_8.addWidget(self.camera_label3)
        self.horizontalLayout_20.addWidget(self.camera_label4_1)
        self.horizontalLayout_26.addWidget(self.camera_label4_2)
        self.horizontalLayout_14.addWidget(self.camera_label5)
        self.horizontalLayout_16.addWidget(self.camera_label6)
        self.horizontalLayout_22.addWidget(self.camera_label7)
        self.horizontalLayout_28.addWidget(self.camera_label8)
        self.horizontalLayout_25.addWidget(self.camera_label9)
        self.horizontalLayout_31.addWidget(self.camera_label10)
        self.horizontalLayout_34.addWidget(self.camera_label11)
        self.horizontalLayout_38.addWidget(self.camera_label12)

        # поток логики приложения
        self.th3 = LogicThread(main_window=self)
        self.th3.start()

        # Логика сохранения настроек в файл settings.json
        self.pushButton.clicked.connect(self.apply_1_original)
        # pushButton — имя объекта, который мы определили в Qt design_mainer
        # clicked — событие, которое мы хотим привязать
        # connect() — метод, который привязывает событие к вызову
        # переданной функции
        # self.apply_1_original — просто функция (метод), которую
        # мы описали в классе MainApp
        self.pushButton_5.clicked.connect(self.apply_1_painted)
        self.pushButton_7.clicked.connect(self.apply_1_closing_opening)
        self.pushButton_11.clicked.connect(self.apply_1_result)
        self.pushButton_17.clicked.connect(self.apply_2_paint_resistors)
        self.comboBox.activated[str].connect(self.change_source_type)
        if self.comboBox.currentText() == 'Камера':
            self.pushButton_19.setVisible(False)
        if self.comboBox.currentText() == 'Изображение':
            self.pushButton_19.setVisible(True)
        self.pushButton_19.clicked.connect(self.browse_folder)

    def change_source_type(self, text):
        # Video or Image
        if self.comboBox.currentText() == 'Камера':
            self.pushButton_19.setVisible(False)
            self.listWidget.setVisible(False)
        if self.comboBox.currentText() == 'Изображение':
            self.pushButton_19.setVisible(True)
            self.listWidget.setVisible(True)

    def closeEvent(self, event):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setWindowTitle("Выход из программы")
        msg.setText("Вы уверены, что хотите выйти?")

        cancel_button = msg.addButton('Нет', QMessageBox.RejectRole)
        ok_button = msg.addButton('Да', QMessageBox.AcceptRole)
        msg.setDefaultButton(cancel_button)

        msg.exec()
        if msg.clickedButton() == ok_button:
            # self.th1.cap.release()
            self.th3.th1.msleep(100)
            self.th3.msleep(100)
            """
            if 'cap' in self.th3.__dict__:
                self.th3.cap.release()
            self.th3.th1.terminate()
            self.th3.terminate()
            """
            event.accept()

        else:
            event.ignore()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.camera_label1.setPixmap(QPixmap.fromImage(image))
        self.camera_label2.setPixmap(QPixmap.fromImage(image))
        self.camera_label3.setPixmap(QPixmap.fromImage(image))
        self.camera_label4_1.setPixmap(QPixmap.fromImage(image))
        self.camera_label5.setPixmap(QPixmap.fromImage(image))
        self.camera_label6.setPixmap(QPixmap.fromImage(image))
        self.camera_label7.setPixmap(QPixmap.fromImage(image))
        self.camera_label8.setPixmap(QPixmap.fromImage(image))
        self.camera_label9.setPixmap(QPixmap.fromImage(image))
        self.camera_label10.setPixmap(QPixmap.fromImage(image))
        self.camera_label11.setPixmap(QPixmap.fromImage(image))
        self.camera_label12.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def setImage1(self, image):
        self.camera_label4_2.setPixmap(QPixmap.fromImage(image))

    def browse_folder(self):
        self.listWidget.clear()  # На случай, если в списке уже есть элементы
        filename = QtWidgets.QFileDialog.getOpenFileName(
            self, "Выберите файл...", '/home/vlad/Документы/OpenCV/Курсач')[0]
        # открыть диалог выбора директории и установить значение переменной
        # равной пути к выбранной директории

        if filename:  # не продолжать выполнение,
            # если пользователь не выбрал директорию
            self.listWidget.addItem(filename)   # добавить путь файла в
            # listWidget

    def parse_delta(self, text):
        try:
            result = re.match(r'^\d+\.\d*$', text)
            if result is None:
                raise ValueError
            result = float(result.group(0))
            if result > 1.0 and result < 0.0:
                raise ValueError
            DELTA = result
            return DELTA
        except ValueError:
            return None

    def successfully(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setWindowTitle("Успешно")
        msg.setText("Вы успешно изменили значение(-я)")

        ok_button = msg.addButton('Хорошо', QMessageBox.AcceptRole)
        msg.setDefaultButton(ok_button)

        msg.exec()
        if msg.clickedButton() == ok_button:
            pass

    def error(self, text):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)

        msg.setWindowTitle("Неудачно")
        msg.setText(text)

        ok_button = msg.addButton('Хорошо', QMessageBox.AcceptRole)
        msg.setDefaultButton(ok_button)

        msg.exec()
        if msg.clickedButton() == ok_button:
            pass

    def apply_1_original(self):
        settings_dict = {}
        with open('settings.json', 'r') as infile:
            settings_dict = json.load(infile)

        new_settings = {}
        DELTA = self.parse_delta(self.lineEdit.text())

        if self.comboBox.currentText() == 'Камера':
            new_settings['source'] = 'Camera'
        if self.comboBox.currentText() == 'Изображение':
            file_obj = self.listWidget.takeItem(0)
            if file_obj is None:
                self.error('Вы не указали путь к изображению!')
                return

            filename = file_obj.text()
            if not os.path.isfile(filename):
                text = "<{}>\nне является файлом, либо отсутствует!".format(
                    filename)
                self.error(text)
                return
            new_settings['source'] = 'Image'
            new_settings['filename'] = filename

        if DELTA:
            new_settings['DELTA'] = DELTA
            new_settings['image filtration'] = self.checkBox.isChecked()
            settings_dict['1_original'] = new_settings

            with open('settings.json', 'w') as outfile:
                json.dump(settings_dict, outfile)
            print('apply_1_original DONE!')
            self.successfully()
        else:
            print('apply_1_original ERROR!')

    def parse_uint(self, text):
        result = re.match(r'^\d+\.*$', text)
        if result is None:
            return None
        result = int(result.group(0))
        return result

    def parse_HSV(self, *args):
        try:
            if len(args) == 3:
                HSV = [None, None, None]
                ind = 0
                for arg in args:
                    uint_value = self.parse_uint(arg)
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

    def apply_1_painted(self):
        settings_dict = {}
        with open('settings.json', 'r') as infile:
            settings_dict = json.load(infile)

        new_settings = {}
        min_HSV = self.parse_HSV(
            self.lineEdit_8.text(),
            self.lineEdit_9.text(),
            self.lineEdit_10.text()
        )
        max_HSV = self.parse_HSV(
            self.lineEdit_11.text(),
            self.lineEdit_12.text(),
            self.lineEdit_13.text()
        )
        None_flag = True

        if min_HSV is not None and max_HSV is not None:
            None_flag = False

        if not None_flag:
            new_settings['min_H'] = min_HSV[0]
            new_settings['min_S'] = min_HSV[1]
            new_settings['min_V'] = min_HSV[2]
            new_settings['max_H'] = max_HSV[0]
            new_settings['max_S'] = max_HSV[1]
            new_settings['max_V'] = max_HSV[2]
            settings_dict['1_painted'] = new_settings

            with open('settings.json', 'w') as outfile:
                json.dump(settings_dict, outfile)
            print('apply_1_painted DONE!')
            self.successfully()
        else:
            print('apply_1_painted ERROR!')

    def parse_k(self, text):
        try:
            result = re.match(r'^\d+\.*\d*$', text)
            if result is None:
                raise ValueError
            result = float(result.group(0))
            if result > 20.0 or result < 1.0:
                raise ValueError
            k = result
            return k
        except ValueError:
            return None

    def parse_kx_ky(self, *args):
        try:
            if len(args) != 2:
                raise ValueError

            k = [None, None]
            ind = 0
            for arg in args:
                k_i = self.parse_k(arg)
                if k_i is None:
                    raise ValueError
                k[ind] = k_i
                ind += 1

            return k[0], k[1]
        except ValueError:
            return None

    def apply_1_closing_opening(self):
        settings_dict = {}
        with open('settings.json', 'r') as infile:
            settings_dict = json.load(infile)

        new_settings = {}
        closing = self.parse_kx_ky(
            self.lineEdit_5.text(),
            self.lineEdit_6.text()
        )
        opening = self.parse_kx_ky(
            self.lineEdit_7.text(),
            self.lineEdit_14.text()
        )

        if closing is not None and opening is not None:
            new_settings['closing_kx'] = closing[0]
            new_settings['closing_ky'] = closing[1]
            new_settings['opening_kx'] = opening[0]
            new_settings['opening_ky'] = opening[1]
            settings_dict['1_closing_opening'] = new_settings

            with open('settings.json', 'w') as outfile:
                json.dump(settings_dict, outfile)
            print('apply_1_closing_opening DONE!')
            self.successfully()
        else:
            print('apply_1_closing_opening ERROR!')

    def apply_1_result(self):
        settings_dict = {}
        with open('settings.json', 'r') as infile:
            settings_dict = json.load(infile)

        new_settings = {}

        new_settings['image rotation'] = self.checkBox_2.isChecked()
        settings_dict['1_result'] = new_settings

        with open('settings.json', 'w') as outfile:
            json.dump(settings_dict, outfile)
        print('apply_1_result DONE!')
        self.successfully()

    def apply_2_paint_resistors(self):
        settings_dict = {}
        with open('settings.json', 'r') as infile:
            settings_dict = json.load(infile)

        new_settings = {}
        min_HSV = self.parse_HSV(
            self.lineEdit_15.text(),
            self.lineEdit_16.text(),
            self.lineEdit_17.text()
        )
        max_HSV = self.parse_HSV(
            self.lineEdit_18.text(),
            self.lineEdit_19.text(),
            self.lineEdit_20.text()
        )
        None_flag = True

        if min_HSV is not None and max_HSV is not None:
            None_flag = False

        if not None_flag:
            new_settings['min_H'] = min_HSV[0]
            new_settings['min_S'] = min_HSV[1]
            new_settings['min_V'] = min_HSV[2]
            new_settings['max_H'] = max_HSV[0]
            new_settings['max_S'] = max_HSV[1]
            new_settings['max_V'] = max_HSV[2]
            settings_dict['2_paint_resistors'] = new_settings

            with open('settings.json', 'w') as outfile:
                json.dump(settings_dict, outfile)
            print('apply_2_paint_resistors DONE!')
            self.successfully()
        else:
            print('apply_2_paint_resistors ERROR!')


def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = MainApp()  # Создаём объект класса MainApp
    window.show()  # Показываем окно

    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()

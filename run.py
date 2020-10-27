import sys  # sys нужен для передачи argv в QApplication
from PyQt5 import QtWidgets
import os

from Формы.output import design_main  # Это наш конвертированный файл дизайна

from PyQt5.QtWidgets import QLabel, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSlot

from video_threads import LogicThread, SourceVideoThread, \
    MainVideoThread


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

        # лэйауты для хранения изображений с камеры
        self.horizontalLayout_5.addWidget(self.camera_label1)
        self.horizontalLayout_18.addWidget(self.camera_label2)
        self.horizontalLayout.addWidget(self.camera_label3)
        self.horizontalLayout_20.addWidget(self.camera_label4_1)
        self.horizontalLayout_21.addWidget(self.camera_label4_2)
        self.horizontalLayout_14.addWidget(self.camera_label5)
        self.horizontalLayout_16.addWidget(self.camera_label6)

        # поток логики приложения
        self.th3 = LogicThread(main_window=self)
        self.th3.start()

        # self.btnBrowse.clicked.connect(self.browse_folder)
        # btnBrowse — имя объекта, который мы определили в Qt design_mainer
        # clicked — событие, которое мы хотим привязать
        # connect() — метод, который привязывает событие к вызову
        # переданной функции
        # self.browse_folder — просто функция (метод), которую
        # мы описали в классе MainApp

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

    @pyqtSlot(QImage)
    def setImage1(self, image):
        self.camera_label4_2.setPixmap(QPixmap.fromImage(image))

    def browse_folder(self):
        self.listWidget.clear()  # На случай, если в списке уже есть элементы
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Выберите папку")
        # открыть диалог выбора директории и установить значение переменной
        # равной пути к выбранной директории

        if directory:  # не продолжать выполнение,
            # если пользователь не выбрал директорию
            for file_name in os.listdir(directory):  # для каждого файла
                # в директории
                self.listWidget.addItem(file_name)   # добавить файл в
                # listWidget


def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = MainApp()  # Создаём объект класса MainApp
    window.show()  # Показываем окно

    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()

import cv2
import sys
#import dlib
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import *

from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import tensorflow as tf
import pygame
from tflite_runtime.interpreter import Interpreter
import imageio


class dds(object):
    face_cascade_name = './haarcascades/haarcascade_frontalface_default.xml'
    eyes_cascade_name = './haarcascades/haarcascade_eye_tree_eyeglasses.xml'
    face_cascade = cv2.CascadeClassifier()
    eyes_cascade = cv2.CascadeClassifier()
    # 안면과 눈 영역 검출을 위한 파일 로드
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)
    SZ = 24
    status = 'Awake'
    number_closed = 0
    closed_limit = 7  # number_closed 7 초과 시, 알람
    max_count = closed_limit + 3  # 최대10까지만 증가하도록 설정
    show_frame = None
    # global sign
    sign = None
    color = None
    frame_width = 368
    frame_height = 272
    frame_resolution = [frame_width, frame_height]
    frame_rate = 32
    is_running = False

    # 라즈베리파이 카메라 초기화 및 설정
    camera = PiCamera()
    camera.rotation = 90
    camera.hflip = True
    camera.resolution = frame_resolution
    camera.framerate = frame_rate
    rawCapture = PiRGBArray(camera, size=(frame_resolution))

    blank_array = imageio.imread('janie.jpg')
    blank_image = QtGui.QImage(blank_array.data,
                               frame_width,
                               frame_height,
                               blank_array.strides[0],
                               QtGui.QImage.Format_RGB888)

    def __init__(self):
        pass

    def set_input_tensor(interpreter, image):
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    # tflite모델을 이용해 눈 개폐 여부 판별하는 함수
    def classify_image(interpreter, image, top_k=1):
        """Returns a sorted array of classification results."""
        dds.set_input_tensor(interpreter, image)
        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]
        output = np.squeeze(interpreter.get_tensor(output_details['index']))
        print('output1 :', interpreter.get_tensor(output_details['index']))
        if output_details['dtype'] == np.uint8:
            scale, zero_point = output_details['quantization']
            output = scale * (output - zero_point)

        print('output :', output)
        ordered = np.argpartition(-output, top_k)
        return [(i, output[i]) for i in ordered[:top_k]]


class ShowVideo(QtCore.QObject):
    VideoSignal1 = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, interpreter, sign_label, parent=None):
        super(ShowVideo, self).__init__(parent)

    @QtCore.pyqtSlot()
    def startVideo(self):
        minimum_brightness = 0.7
        #  이미 실행중이라면 동작하지 않음
        if dds.is_running:
            return
        else:
            dds.is_running = True
        # 카메라로 입력받은 프레임마다 수행
        for frame in dds.camera.capture_continuous(dds.rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            dds.show_frame = image

            # 졸음판단시, 흑백 화면 송출
            height, width = image.shape[:2]
            frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.equalizeHist(frame_gray)
            cols, rows = frame_gray.shape
            brightness = np.sum(frame_gray) / (255 * cols * rows)
            ratio = brightness / minimum_brightness
            if ratio < 1:
                frame_gray = cv2.convertScaleAbs(frame_gray, alpha=1 / ratio, beta=0)
            faces = dds.face_cascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=2)  # 얼굴영역 검출

            for (x, y, w, h) in faces:
                dds.show_frame = cv2.rectangle(dds.show_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                faceROI = frame_gray[y:y + h, x:x + w]

                eyes = dds.eyes_cascade.detectMultiScale(faceROI)  # 검출된 얼굴영역 중, 눈 영역 검출
                results = []

                for (x2, y2, w2, h2) in eyes:
                    eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
                    radius = int(round((w2 + h2) * 0.25))
                    dds.show_frame = cv2.circle(dds.show_frame, eye_center, radius, (0, 255, 255), 2)
                    eye = faceROI[y2:y2 + h2, x2:x2 + w2]
                    eye = cv2.resize(eye, (dds.SZ, dds.SZ))
                    eye = eye / 255
                    eye = eye.reshape(dds.SZ, dds.SZ, -1)
                    eye = np.expand_dims(eye, axis=0)
                    prediction = dds.classify_image(interpreter, eye)
                    result, confidence = prediction[0]
                    results.append(result)

                if np.mean(results) == 1:
                    dds.color = (0, 255, 0)
                    dds.status = 'Awake'
                    dds.number_closed = dds.number_closed - 1
                    if dds.number_closed < 0:
                        dds.number_closed = 0
                else:
                    dds.color = (0, 0, 255)
                    dds.status = 'Sleep'
                    dds.number_closed = dds.number_closed + 1
                    if dds.number_closed > dds.max_count:
                        dds.number_closed = dds.max_count

                dds.sign = dds.status + ', Sleep count : ' + str(dds.number_closed) + ' / ' + str(dds.closed_limit)
                if dds.number_closed > dds.closed_limit:
                    dds.show_frame = frame_gray  # 흑백화면 송출
                    sign_label.setStyleSheet("Color : red")

                    # get_busy() 반환값이 False일 때까지, 계속 알람 실행
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()
                else:
                    sign_label.setStyleSheet("Color : black")

            # 프레임 출력
            sign_label.setText(dds.sign)
            dds.show_frame = cv2.cvtColor(dds.show_frame, cv2.COLOR_BGR2RGB)
            qt_image1 = QtGui.QImage(dds.show_frame.data,
                                     width,
                                     height,
                                     dds.show_frame.strides[0],
                                     QtGui.QImage.Format_RGB888)
            self.VideoSignal1.emit(qt_image1)
            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(25, loop.quit)  # 25 ms
            loop.exec_()
            sign_label.setText(dds.sign)
            # 다음 프레임을 위해 스트림 초기화
            dds.rawCapture.truncate(0)
            if not dds.is_running:
                image_viewer1.setImage(dds.blank_image)
                break

    def stopVideo(self):
        dds.is_running = False


class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

    def initUI(self):
        self.setWindowTitle('Test')

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()


class ChangeAlarmDialog(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle('Change Alarm')
        self.setGeometry(150, 150, 300, 200)

        self.textLabel = QLabel('Alarm', self)
        self.textLabel.move(35, 20)

        self.qb = QComboBox(self)
        self.qb.addItems(['alarm 1', 'alarm 2', 'alarm 3', 'alarm 4'])
        self.qb.move(35, 80)

        self.btn1 = QtWidgets.QPushButton('Confirm', self)
        self.btn1.move(35, 150)
        self.btn1.clicked.connect(self.changeAlarm)
        self.btn1.clicked.connect(self.close)

        self.btn2 = QtWidgets.QPushButton('Cancel', self)
        self.btn2.move(165, 150)
        self.btn2.clicked.connect(self.close)

    def changeAlarm(self):
        idx = self.qb.currentIndex()
        if idx is 0:
            pygame.mixer.music.load('alarm.wav')
        elif idx is 1:
            pygame.mixer.music.load('alarm2.mp3')
        elif idx is 2:
            pygame.mixer.music.load('alarm3.mp3')
        elif idx is 3:
            pygame.mixer.music.load('alarm4.mp3')


class ChangeSleepCountDialog(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle('Change Sleep Count')
        self.setGeometry(150, 150, 300, 200)

        self.textLabel = QLabel('Sleep Count', self)
        self.textLabel.move(35, 20)

        self.qb = QComboBox(self)
        self.qb.addItems(['5', '7', '9', '11'])
        self.qb.move(35, 80)

        self.btn1 = QtWidgets.QPushButton('Confirm', self)
        self.btn1.move(35, 150)
        self.btn1.clicked.connect(self.changeSleepCount)
        self.btn1.clicked.connect(self.close)

        self.btn2 = QtWidgets.QPushButton('Cancel', self)
        self.btn2.move(165, 150)
        self.btn2.clicked.connect(self.close)

    def changeSleepCount(self):
        dds.closed_limit = int(self.qb.currentText())
        dds.max_count = dds.closed_limit + 3


def setting_button():
    push_button1.clicked.disconnect()
    push_button2.clicked.disconnect()
    push_button3.clicked.disconnect()

    push_button1.setText('Change Alarm')
    push_button2.setText('Change Sleep Count')
    push_button3.setText('Back To Menu')

    push_button1.clicked.connect(change_alarm_dialog.show)
    push_button2.clicked.connect(change_sleep_count_dialog.show)
    push_button3.clicked.connect(menu_button)


def menu_button():
    push_button1.clicked.disconnect()
    push_button2.clicked.disconnect()
    push_button3.clicked.disconnect()

    push_button1.setText('Start')
    push_button2.setText('Stop')
    push_button3.setText('Setting')

    push_button1.clicked.connect(vid.startVideo)
    push_button2.clicked.connect(vid.stopVideo)
    push_button3.clicked.connect(setting_button)


if __name__ == '__main__':
    pygame.mixer.init()
    pygame.mixer.music.load('alarm.wav')  # 졸음판단시, alarm 파일 실행
    # tflite모델 로드
    interpreter = Interpreter('drowsinessDetection.tflite')
    interpreter.allocate_tensors()

    app = QtWidgets.QApplication(sys.argv)
    change_alarm_dialog = ChangeAlarmDialog()
    change_sleep_count_dialog = ChangeSleepCountDialog()

    thread = QtCore.QThread()
    thread.start()
    sign_label = QtWidgets.QLabel()
    vid = ShowVideo(interpreter, sign_label)
    vid.moveToThread(thread)

    image_viewer1 = ImageViewer()
    image_viewer1.setImage(dds.blank_image)
    vid.VideoSignal1.connect(image_viewer1.setImage)

    sign_label = QtWidgets.QLabel()
    sign_label.setFont(QtGui.QFont("consolas", 20, weight=QtGui.QFont.Bold))
    push_button1 = QtWidgets.QPushButton('Start')
    push_button2 = QtWidgets.QPushButton('Stop')
    push_button3 = QtWidgets.QPushButton('Setting')
    vertical_layout = QtWidgets.QVBoxLayout()
    push_button1.clicked.connect(vid.startVideo)
    push_button2.clicked.connect(vid.stopVideo)
    push_button3.clicked.connect(setting_button)

    vertical_layout = QtWidgets.QVBoxLayout()
    horizontal_layout = QtWidgets.QHBoxLayout()
    horizontal_layout.addWidget(image_viewer1)
    vertical_layout.addLayout(horizontal_layout)
    vertical_layout.addWidget(sign_label)
    vertical_layout.addWidget(push_button1)
    vertical_layout.addWidget(push_button2)
    vertical_layout.addWidget(push_button3)

    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(vertical_layout)

    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.show()
    sys.exit(app.exec_())
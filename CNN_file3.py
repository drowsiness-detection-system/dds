from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
import tensorflow as tf
import pygame
from tflite_runtime.interpreter import Interpreter


# 안면과 눈 영역 검출을 위한 파일 로드 
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

#tflite모델 로드
interpreter = Interpreter('drowsinessDetection.tflite')
interpreter.allocate_tensors()

SZ = 24
status = 'Awake'
number_closed = 0
closed_limit = 7 # number_closed 7 초과 시, 알람 
max_count = 10 # 최대10까지만 증가하도록 설정
show_frame = None
sign = None
color = None

frame_width = 320
frame_height = 240
frame_resolution = [frame_width, frame_height]
frame_rate = 32


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

#tflite모델을 이용해 눈 개폐 여부 판별하는 함수 
def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]


# 라즈베리파이 카메라 초기화 및 설정
camera = PiCamera()
camera.rotation = 90
camera.hflip = True
camera.resolution = frame_resolution
camera.framerate = frame_rate
rawCapture = PiRGBArray(camera, size=(frame_resolution))
# 카메라로 입력받은 프레임마다 수행
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    show_frame = image
    
    # 졸음판단시, 흑백 화면 송출 
    height, width = image.shape[:2]
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    
    faces = face_cascade.detectMultiScale(frame_gray) # 얼굴영역 검출
    for (x, y, w, h) in faces:
        show_frame = cv2.rectangle(show_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        faceROI = frame_gray[y:y + h, x:x + w]
       
        eyes = eyes_cascade.detectMultiScale(faceROI) #검출된 얼굴영역 중, 눈 영역 검출
        results = []

        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            show_frame = cv2.circle(show_frame, eye_center, radius, (0, 255, 255), 2)
            eye = faceROI[y2:y2 + h2, x2:x2 + w2]
            eye = cv2.resize(eye, (SZ, SZ))
            eye = eye / 255
            eye = eye.reshape(SZ, SZ, -1)
            eye = np.expand_dims(eye, axis=0)
            prediction = classify_image(interpreter, eye)
            result, confidence = prediction[0]
            results.append(result)
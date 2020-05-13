# https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/raspberry_pi/classify_picamera.py
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
import tensorflow as tf
import time
import pygame
from tflite_runtime.interpreter import Interpreter

pygame.mixer.init()
pygame.mixer.music.load('alarm.wav')

face_cascade_name = './haarcascades/haarcascade_frontalface_alt.xml'
eyes_cascade_name = './haarcascades/haarcascade_eye_tree_eyeglasses.xml'

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

#-- 1. Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

#-- 2. Load TF model
interpreter = Interpreter('drowsinessDetection.tflite')
interpreter.allocate_tensors()

SZ = 24
status = 'Awake'
number_closed = 0
closed_limit = 7
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


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.rotation = 90
camera.hflip = True
camera.resolution = frame_resolution
camera.framerate = frame_rate
rawCapture = PiRGBArray(camera, size=(frame_resolution))
# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    start_time = time.time()
    # grab the raw NumPy array representing the image
    image = frame.array
    show_frame = image
    # transform gray image
    height,width = image.shape[:2]
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        show_frame = cv2.rectangle(show_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        results = []

        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            show_frame = cv2.circle(show_frame, eye_center, radius, (0, 255, 255 ), 2)
            eye = faceROI[y2:y2+h2,x2:x2+w2]
            eye = cv2.resize(eye,(SZ,SZ))
            eye= eye/255
            eye=  eye.reshape(SZ,SZ,-1)
            eye = np.expand_dims(eye,axis=0)
            #result = model.predict_classes(eye)
            prediction = classify_image(interpreter, eye)
            result, confidence = prediction[0]
            results.append(result)
        #print(result, results, np.mean(results))
        if( np.mean(results)==1 ):
            color = (0, 255, 0)
            status = 'Awake'
            number_closed = number_closed - 1
            if( number_closed<0 ):
                number_closed = 0
        else:
            color = (0, 0, 255)
            status = 'Sleep'
            number_closed = number_closed + 1
            
        sign = status + ', Sleep count : ' + str(number_closed) + ' / ' + str(closed_limit)
        if( number_closed > closed_limit ):
            show_frame = frame_gray
            # play SOUND
            if(pygame.mixer.music.get_busy()==False):
                pygame.mixer.music.play()
            #while pygame.mixer.music.get_busy() == True:
            #    continue
        
    # show the frame
    cv2.putText(show_frame, sign , (10,height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow("Drowsiness Detection", show_frame)
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

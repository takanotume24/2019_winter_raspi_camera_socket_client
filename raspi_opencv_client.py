import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import picamera
import numpy
import time

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.0.32', 8485))
connection = client_socket.makefile('wb')

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
cam = cv2.VideoCapture(0)

cam.set(3, 320)
cam.set(4, 240)

while True:
    ret, frame = cam.read()
    result, frame = cv2.imencode('.jpg', frame, encode_param)
    data = pickle.dumps(frame, 0)
    size = len(data)

    print("{}: {}".format(img_counter, size))
    client_socket.sendall(struct.pack(">L", size) + data)
    img_counter += 1
    data = client_socket.recv(1024)

cam.release()

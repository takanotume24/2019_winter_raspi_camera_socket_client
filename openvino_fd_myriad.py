import cv2 as cv
import picamera
import picamera.array
import socket

import threading

xmin = 0
ymin = 0
xmax = 0
ymax = 0


def fun_thread_socket():
    while 1:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # サーバを指定
            s.connect(('192.168.22.103', 1234))
            # サーバにメッセージを送る
            string = str(xmin)
            s.sendall(string.encode())
            print(string, " send.")


thread_1 = threading.Thread(target=fun_thread_socket)
thread_1.start()

with picamera.PiCamera() as camera:
    # カメラの画像をリアルタイムで取得するための処理
    with picamera.array.PiRGBArray(camera) as stream:
        # 解像度の設定
        camera.resolution = (512, 384)
        net = cv.dnn.readNet('face-detection-adas-0001.xml',
                             'face-detection-adas-0001.bin')
        net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

        while True:
            # カメラから映像を取得する（OpenCVへ渡すために、各ピクセルの色の並びをBGRの順番にする）
            camera.capture(stream, 'bgr', use_video_port=True)
            # Load the model.

            # Specify target device.
            # Read an image.
            frame = stream.array
            if frame is None:
                raise Exception('Image not found!')
            # Prepare input blob and perform an inference.
            blob = cv.dnn.blobFromImage(
                frame, size=(672, 384), ddepth=cv.CV_8U)
            net.setInput(blob)
            out = net.forward()
            # Draw detected faces on the frame.
            for detection in out.reshape(-1, 7):
                confidence = float(detection[2])
                xmin = int(detection[3] * frame.shape[1])
                ymin = int(detection[4] * frame.shape[0])
                xmax = int(detection[5] * frame.shape[1])
                ymax = int(detection[6] * frame.shape[0])
                if confidence > 0.5:
                    cv.rectangle(frame, (xmin, ymin),
                                 (xmax, ymax), color=(0, 255, 0))
                    print(xmin, xmax,  ymin, ymax)

            # Save the frame to an image file.
                    # 結果の画像を表示する
            cv.imshow('camera', stream.array)

            # カメラから読み込んだ映像を破棄する
            stream.seek(0)
            stream.truncate()

            # 何かキーが押されたかどうかを検出する（検出のため、1ミリ秒待つ）
            if cv.waitKey(1) > 0:
                break

        cv.destroyAllWindows()

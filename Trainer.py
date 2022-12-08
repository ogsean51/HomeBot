# program to capture single image from webcam in python
import asyncio

# importing OpenCV library
import cv2


# initialize the camera
# If you have multiple camera connected with
# current device, assign a value in cam_port
# variable according to that
num = 0
cam_port = 0
cam = cv2.VideoCapture(cam_port)

while num < 100:
    result, image = cam.read()
    if result:
        cv2.imwrite("Dataset/" + str(num) + ".png", image)
        num += 1

# If captured image is corrupted, moving to else part
else:
    print("No image detected. Please! try again")
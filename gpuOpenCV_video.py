from datetime import datetime
import torch
import cv2
import numpy as np
import cupy as cp
import torchvision
import torchvision.transforms as transforms
from mapClassesGPU import vectorizedMap
import csv

# Initialize global variables
res = 3163
frameCount = 0
header_written = False
startTime = datetime.now()
loopTime = datetime.now()
speed = 15
angle = 0
radius = 200
center = (res/2, res/2)

while((datetime.now() - loopTime).total_seconds() < 30.0):
    # Open the CSV file in write mode
    with open('OpenCV_GPU_Video_Canny.csv', mode='a', newline='') as file:
        # Create a csv.writer object
        writer = csv.writer(file)
        if not header_written:
            writer.writerow(['FPS_1x10_7'])
            header_written = True
        
        # Data preprocessing
        img = cv2.imread('largeImage.jpg', 1)
        h, w = img.shape[1:]
        type_map = cv2.CV_8UC3
        gpu_mat = cv2.cuda_GpuMat(h, w, type_map)
        gpu_mat.upload(img)
        resized_img = cv2.cuda_GpuMat(res, res, type_map)
        cv2.cuda.resize(gpu_mat, (res, res), resized_img)
        
        # Calculate the new position of the image
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))

        # Perform Canny edge detection on the frame
        M = cv2.getRotationMatrix2D((x, y), angle, 1) # Create a rotation matrix
        rotated_img = cv2.cuda.warpAffine(resized_img, M, (res, res)) # Apply the rotation to the image
        rotated_img = cv2.cuda.bilateralFilter(rotated_img, 9, 75, 75)
        gray_gpu_mat = cv2.cuda_GpuMat(res, res, cv2.CV_8UC1)
        rotated_img = cv2.cuda.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY, gray_gpu_mat)
        edges_mat = cv2.cuda_GpuMat(res, res, cv2.CV_8UC1)
        canny_detector = cv2.cuda.createCannyEdgeDetector(100,200)
        canny_detector.detect(gray_gpu_mat, edges_mat)
        
        # Download the result to a regular cv2.Mat object
        np_array = np.empty((res, res, 1), dtype=np.uint8)
        edges_mat.download(np_array)

        # Increment the counter and render the image
        frameCount += 1
        cv2.imshow('Animated Image', np_array)


        angle += np.pi / speed # Update the angle of rotation
        if angle >= 20 * np.pi:
            angle = 0

        endTime = datetime.now()
        timeElapsed = (endTime - startTime) # Stop the stopwatch and calculate the elapsed time
        print(timeElapsed)
        fps = frameCount / timeElapsed.total_seconds() #Calculate the fps and write it to a csv
        writer.writerow([fps])

        if cv2.waitKey(1) & 0xFF == ord('q'): # Enter 'q' to stop video feed
            break

        frameCount = 0 # Reset the frame counter and start time
        startTime = datetime.now()

# Close the window
cv2.destroyAllWindows()

from datetime import datetime
import cupy as cp
import numpy as np
import pyglet
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from pyglet.gl import *
#import cupy as cp
from mapClassesGPU import vectorizedMap
import csv
import cv2

#Gonna need something to convert a Gpu_Mat to cupy_ndarray or find a way to convert Gpu_Mat to bytes object
#Set global variables
res = 3163 #317, 1000, 3163
xOffset, yOffset = 10, 10
texture = pyglet.image.Texture.create(res, res)
window = pyglet.window.Window(width=res+xOffset, height=res+yOffset)
sprite = pyglet.sprite.Sprite(texture)

#Initialize fps variables and csv header row
fps_label = pyglet.text.Label("FPS: 0", font_size=12, x=10, y=window.height-15)
frameCount = 0
timeElapsed = 0
header_written = False
# Define the center and radius of the circle
center = (res/2, res/2)
radius = 200
# Define the angle of rotation
angle = 0
# Define the animation speed
speed = 15

#Define a function that creates a new image and performs similar conversions as the deploy script
def update(dt):
    global header_written, angle, radius, center, speed, res
    
    # Open the CSV file in write mode
    with open('OpenGL_GPU_Video_Canny.csv', mode='a', newline='') as file:
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

        # Create a rotation matrix
        M = cv2.getRotationMatrix2D((x, y), angle, 1)
        
        # Apply the rotation to the image
        rotated_img = cv2.cuda.warpAffine(resized_img, M, (res, res))

        #Apply Canny Edge Detection
        rotated_img = cv2.cuda.bilateralFilter(rotated_img, 9, 75, 75)

        # Convert the frame to grayscale
        gray_gpu_mat = cv2.cuda_GpuMat(res, res, cv2.CV_8UC1)
        cv2.cuda.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY, gray_gpu_mat)
        
        # Create a cv2.cuda_GpuMat() object to store the result
        edges_mat = cv2.cuda_GpuMat(res, res, cv2.CV_8UC1)
        
        #Create a CUDA-accelerated Canny edge detector
        canny_detector = cv2.cuda.createCannyEdgeDetector(100,200)
        
        #Apply the Canny edge detection on the gpu_mat
        canny_detector.detect(gray_gpu_mat, edges_mat)
        
        # Create a 3-channel cv2.cuda_GpuMat() object for rendering in pyglet
        mat_3C = cv2.cuda_GpuMat(res, res, cv2.CV_8UC3)
        cv2.cuda.cvtColor(edges_mat, cv2.COLOR_GRAY2BGR, mat_3C)
        
        # Download the result to a regular cv2.Mat object
        np_array = np.empty((res, res, 3), dtype=np.uint8)
        mat_3C.download(np_array)
        
        # Update the angle of rotation
        angle += np.pi / speed
        if angle >= 20 * np.pi:
            angle = 0
        
        #Create a bytes object on the device from arr_device and create a memoryview object to avoid creating new variables every time the image data is updated
        bytes_obj = np_array.tobytes()
        bytes_obj = memoryview(bytes_obj)

        #Convert the image frame to a pyglet image
        image_data = pyglet.image.ImageData(res, res, 'RGB', bytes_obj, -3 * res)
        
        #Update the sprite
        sprite.image = image_data
        sprite.x = xOffset // 2
        sprite.y = yOffset // 2

        # Increment the counter and the time elapsed
        global frameCount, timeElapsed
        frameCount += 1
        timeElapsed += dt
        roundedTimeElapsed = round(timeElapsed, 6)
        print(roundedTimeElapsed, "seconds")
        
        #Calculate FPS
        fps = frameCount / timeElapsed
        writer.writerow([fps])
        fps_label.text = f"FPS: {fps:.2f}"
        
        #Reset the frame counter and start time
        frameCount = 0
        timeElapsed = 0
        
        # Schedule the window to close after N seconds
        N = 30
        pyglet.clock.schedule_once(close_window, N)

# Define the close window function
def close_window(dt):
    pyglet.app.exit() 
       
#Override the on_draw function on the window object, 'window'
@window.event
def on_draw():
    global timeElapsed
    start_time = datetime.now()
    window.clear()
    sprite.draw()
    fps_label.draw()
    end_time = datetime.now()
    timeElapsed += (end_time - start_time).total_seconds()
    
#Set the update interval to 60 fps and run the event loop
pyglet.clock.schedule_interval(update, 1/60)
pyglet.app.run()




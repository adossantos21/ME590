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


#Set global variables
res = 3163 #317, 1000, 3163
xOffset, yOffset = 10, 10
texture = pyglet.image.Texture.create(res, res)
window = pyglet.window.Window(width=res+xOffset, height=res+yOffset)
sprite = pyglet.sprite.Sprite(texture)

#Initialize fps variables and csv header row
center = (res/2, res/2)
radius = 200
angle = 0
speed = 15
fps_label = pyglet.text.Label("FPS: 0", font_size=12, x=10, y=window.height-15)
frameCount = 0
timeElapsed = 0
header_written = False

#Define a function that creates a new image and performs similar conversions as the deploy script
def update(dt):
    global header_written, angle, radius, center, speed, res
    
    # Open the CSV file in write mode
    with open('OpenGL_CPU_Video_Canny.csv', mode='a', newline='') as file:
        # Create a csv.writer object
        writer = csv.writer(file)
        if not header_written:
            writer.writerow(['FPS_1x10_7'])
            header_written = True
        
        # Data preprocessing
        img = cv2.imread('largeImage.jpg', 1)
        img = cv2.resize(img, (res, res))
        
        # Calculate the new position of the image
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))

        # Create a rotation matrix
        M = cv2.getRotationMatrix2D((x, y), angle, 1)

        # Apply the rotation to the image
        rotated_img = cv2.warpAffine(img, M, img.shape[1::-1])

        #Apply Canny Edge Detection
        rotated_img = cv2.GaussianBlur(rotated_img, (7, 7), 1.41)

        # Convert the frame to grayscale
        rotated_img = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)

        # Perform Canny edge detection on the frame
        rotated_img = cv2.Canny(rotated_img, 25, 75)
        rotated_img = cv2.cvtColor(rotated_img, cv2.COLOR_GRAY2BGR)
        
        # Update the angle of rotation
        angle += np.pi / speed
        if angle >= 20 * np.pi:
            angle = 0

        #Create a bytes object and create a memoryview object to avoid creating new variables every time the image data is updated
        bytes_obj = rotated_img.tobytes()
        bytes_obj = memoryview(bytes_obj)

        #Convert the image frame to a pyglet image
        image_data = pyglet.image.ImageData(res, res, 'RGB', bytes_obj, -3 * res)

        #Update the sprite
        sprite.image = image_data
        sprite.x = xOffset // 2
        sprite.y = yOffset // 2
        sprite.draw()
        
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




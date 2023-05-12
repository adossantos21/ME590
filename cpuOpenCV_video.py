from datetime import datetime
import cv2
import numpy as np
import csv

# Define global variables
res = 3163 # Values should be 3163x3163, 1000x1000, 317x317
frameCount = 0
header_written = False
startTime = datetime.now()
loopTime = datetime.now()
angle = 0
speed = 15
center = (res/2, res/2)
radius = 200

while((datetime.now() - loopTime).total_seconds() < 30.0):
    # Open the CSV file in append mode
    with open('OpenCV_CPU_Video_Canny.csv', mode='a', newline='') as file:
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

        # Perform Canny edge detection on the frame
        M = cv2.getRotationMatrix2D((x, y), angle, 1) # Create a rotation matrix
        rotated_img = cv2.warpAffine(img, M, img.shape[1::-1]) # Apply the rotation to the image
        rotated_img = cv2.GaussianBlur(rotated_img, (7, 7), 1.41) # Blur
        rotated_img = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
        rotated_img = cv2.Canny(rotated_img, 25, 75)

        frameCount += 1 # Increment the counter and render the image
        cv2.imshow('Animated Image', rotated_img)

        angle += np.pi / speed # Update the angle of rotation
        if angle >= 20 * np.pi:
            angle = 0

        endTime = datetime.now()
        timeElapsed = (endTime - startTime) # Stop the stopwatch and calculate the elapsed time
        fps = frameCount / timeElapsed.total_seconds() #Calculate the fps and write it to a csv
        writer.writerow([fps])

        if cv2.waitKey(1) & 0xFF == ord('q'): # Enter 'q' to stop video feed
            break

        frameCount = 0 # Reset the frame counter and restart time
        startTime = datetime.now()

# Close the window
cv2.destroyAllWindows()

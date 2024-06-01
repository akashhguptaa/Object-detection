#car detection model using opencv

import cv2
import os

# Define the paths to the XML files
car_classifier = 'cars.xml'
body_classifier = 'fullbody.xml'

# Check if the car classifier file exists
if not os.path.exists(car_classifier):
    raise FileNotFoundError(f"Car classifier file '{car_classifier}' not found. Please download it and place it in the current directory.")

# Check if the pedestrian classifier file exists
if not os.path.exists(body_classifier):
    raise FileNotFoundError(f"Pedestrian classifier file '{body_classifier}' not found. Please download it from https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_fullbody.xml and place it in the current directory.")

# Load the classifiers
car_classifier_file = cv2.CascadeClassifier(car_classifier)
body_classifier_file = cv2.CascadeClassifier(body_classifier)

# Load the image
image_file = 'car.jpg'
if not os.path.exists(image_file):
    raise FileNotFoundError(f"Image file '{image_file}' not found.")

img = cv2.imread(image_file)

# Convert the image to grayscale
bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect cars and pedestrians in the image
cars_coordinates = car_classifier_file.detectMultiScale(bw_img)
pedestrain_coordinates = body_classifier_file.detectMultiScale(bw_img)

# Draw rectangles around the detected cars
for (x, y, w, h) in cars_coordinates:
    cv2.rectangle(img, (x+1, y+1), (x+w, y+h), (255, 0, 0), 5)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)

# Draw rectangles around the detected pedestrians
for (x, y, w, h) in pedestrain_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 5)

# Display the image with detections
cv2.imshow('Car and Pedestrian Detector', img)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()

import cv2 as cv
import random
import os
import json

# Program for generating trainig images (volleyball on random background) with labels and saving them to "data" catalog

# Paths to balls and background images
balls = os.listdir("balls/")
backgounds = os.listdir("backgrounds/")

# How many images to generate
IMAGES_NUM = input("Podaj liczbe zdjec do wygenerowania: ")

# Delete all old images and labels (bounding box cordinates)
old_images = os.listdir("data/images/")
old_labels = os.listdir("data/labels/")

for file in old_images: os.remove("data/images/"+file)
for file in old_labels: os.remove("data/labels/"+file)

print("Generowanie danych...")
for i in range(int(IMAGES_NUM)):

    # Get a random image and scale it to 1280x720 (HD) resolution
    bg_id = random.randint(0, len(backgounds) - 1)
    bg = cv.imread("backgrounds/" + backgounds[bg_id])
    bg = cv.resize(bg, [1280, 720])

    # Get a random ball image
    ball_id = random.randint(0, len(balls) - 1) 
    ball = cv.imread("balls/" + balls[ball_id], -1)

    # Scale ball image by a random number from 0.1 to 0.2
    scale = random.randint(1, 2)/10.0
    ball_size = int(ball.shape[0]*scale)
    ball_r = int(ball.shape[0]*scale/2)
    ball = cv.resize(ball, [ball_size, ball_size])

    # Rotate ball image by a random angle (0-360 degree)
    rotation_angle = random.randint(0, 360)
    M = cv.getRotationMatrix2D((ball_size/2, ball_size/2), rotation_angle, 1.0)
    ball = cv.warpAffine(ball, M, (ball_size, ball_size))

    # Get a random valid position to insert ball image
    x_pos = random.randint(0, bg.shape[1] - ball_size)
    y_pos = random.randint(0, bg.shape[0] - ball_size)

    image = bg.copy()

    # Insert ball image into background (with a transparency)
    alpha_s = ball[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        image[y_pos:y_pos+ball_size, x_pos:x_pos+ball_size, c] = (alpha_s * ball[:, :, c] +
                              alpha_l * image[y_pos:y_pos+ball_size, x_pos:x_pos+ball_size, c])

    # Write image to catalog
    cv.imwrite("data/images/" + str(i)+ ".png", image)

    # Save bounding box cordinates in json file
    bounding_box = [x_pos, y_pos, x_pos + ball_size, y_pos + ball_size]
    data = {"data":bounding_box}

    with open('data/labels/' + str(i) + '.json', 'w') as f:
        json.dump(data, f)

print("Wygenerowano dane")
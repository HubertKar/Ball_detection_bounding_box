import cv2 as cv
import numpy as np
import random
import os
import json

balls = os.listdir("balls/")
backgounds = os.listdir("backgrounds/")

IMAGES_NUM = input("Podaj liczbe zdjec do wygenerowania: ")

old_images = os.listdir("data/images/")
old_labels = os.listdir("data/labels/")

for file in old_images: os.remove("data/images/"+file)
for file in old_labels: os.remove("data/labels/"+file)

print("Generowanie danych...")
for i in range(int(IMAGES_NUM)):

    bg_id = random.randint(0, len(backgounds) - 1)

    bg = cv.imread("backgrounds/" + backgounds[bg_id])
    bg = cv.resize(bg, [1280, 720])

    ball_id = random.randint(0, len(balls) - 1) 
    ball = cv.imread("balls/" + balls[ball_id], -1)

    scale = random.randint(1, 2)/10.0
    ball_size = int(ball.shape[0]*scale)
    ball_r = int(ball.shape[0]*scale/2)

    ball = cv.resize(ball, [ball_size, ball_size])

    rotation_angle = random.randint(0, 360)

    M = cv.getRotationMatrix2D((ball_size/2, ball_size/2), rotation_angle, 1.0)
    ball = cv.warpAffine(ball, M, (ball_size, ball_size))

    x_pos = random.randint(0, bg.shape[1] - ball_size)
    y_pos = random.randint(0, bg.shape[0] - ball_size)

    image = bg.copy()

    alpha_s = ball[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        image[y_pos:y_pos+ball_size, x_pos:x_pos+ball_size, c] = (alpha_s * ball[:, :, c] +
                              alpha_l * image[y_pos:y_pos+ball_size, x_pos:x_pos+ball_size, c])


    cv.imwrite("data/images/" + str(i)+ ".png", image)

    bounding_box = [x_pos, y_pos, x_pos + ball_size, y_pos + ball_size]
    data = {"data":bounding_box}

    with open('data/labels/' + str(i) + '.json', 'w') as f:
        json.dump(data, f)

print("Wygenerowano dane")

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import json
import glob
from matplotlib.widgets import Button

IMAGE_W = 800
IMAGE_H = 800
LR_VALUE = 0.001

NUM_EPOCHS = 10
FINE_TUNE_EPOCHS = 30
FINE_TUNE_AT = 35
BATCH_SIZE = 8

class Next_button(object):  # Klasa do przycisku zarządzania przyciskiem w funkcji test_model
    def __init__(self, axes, label, images, labels, model):
        self.button = Button(axes, label)
        self.button.on_clicked(self.clicked)
        self.images = images
        self.labels = labels
        self.model = model

    def clicked(self, event):
        i = np.random.randint(0,len(self.images)-1) # Losowa zdjęcie i masska ze zbioru testowego
        sample_image = self.images[i] 
        sample_box = self.labels[i]

        prediction = self.model.predict(sample_image[tf.newaxis, ...])[0]
        prediction_box = [int(prediction[0]* sample_image.shape[0]), int(prediction[1]* sample_image.shape[1]), int(prediction[2]* sample_image.shape[0]), int(prediction[3]* sample_image.shape[1])]
        sample_box = [int(sample_box[0]* sample_image.shape[0]), int(sample_box[1]* sample_image.shape[1]), int(sample_box[2]* sample_image.shape[0]), int(sample_box[3]* sample_image.shape[1])]
        
        image_copy = sample_image.copy()
        image_copy /= 255.0
        image_copy2 = image_copy.copy()

        predicted_box = cv.rectangle(image_copy, (prediction_box[0], prediction_box[1]), (prediction_box[2], prediction_box[3]), (0, 0, 255), 3)
        true_box = cv.rectangle(image_copy2, (sample_box[0], sample_box[1]), (sample_box[2], sample_box[3]), (0, 0, 255), 3)

        plt.subplot(121)
        plt.imshow(true_box)
        plt.axis('off')
        plt.title('Ground true')

        plt.subplot(122)
        plt.imshow(predicted_box)
        plt.axis('off')
        plt.title('Predicted')

        plt.draw()

def test_model(model, trainig_images, trainig_labels, valid_images, valid_labels): # Funckja do testowania działania sieci 

    i = np.random.randint(0,len(trainig_images)-1) # Losowa zdjęcie i masska ze zbioru testowego
    sample_image = trainig_images[i] 
    sample_box = trainig_labels[i]

    prediction = model.predict(sample_image[tf.newaxis, ...])[0] # Przewidywanie maski, tf.newaxis powiększa wymiar macierzy o jeden, predict oczukuje zbioru danych  
    prediction_box = [int(prediction[0]* sample_image.shape[0]), int(prediction[1]* sample_image.shape[1]), int(prediction[2]* sample_image.shape[0]), int(prediction[3]* sample_image.shape[1])]
    sample_box = [int(sample_box[0]* sample_image.shape[0]), int(sample_box[1]* sample_image.shape[1]), int(sample_box[2]* sample_image.shape[0]), int(sample_box[3]* sample_image.shape[1])]
    
    image_copy = sample_image.copy()
    image_copy /= 255.0
    image_copy2 = image_copy.copy()

    predicted_box = cv.rectangle(image_copy, (prediction_box[0], prediction_box[1]), (prediction_box[2], prediction_box[3]), (0, 0, 255), 3)
    true_box = cv.rectangle(image_copy2, (sample_box[0], sample_box[1]), (sample_box[2], sample_box[3]), (0, 0, 255), 3)

    fig = plt.figure(figsize=(10,5))

    plt.subplot(121)
    plt.imshow(true_box)
    plt.axis('off')
    plt.title('Ground true')

    plt.subplot(122)
    plt.imshow(predicted_box)
    plt.axis('off')
    plt.title('Predicted')

    axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    a2xnext = fig.add_axes([0.71, 0.05, 0.1, 0.075])
    bnext = Next_button(axnext, "NEXT(Trainig)", trainig_images, trainig_labels, model)
    b2next = Next_button(a2xnext, "NEXT(Valid)", valid_images, valid_labels, model)

    plt.show()

def create_model():
    
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model = tf.keras.applications.MobileNetV2(weights="imagenet", 
                                                        include_top=False,
                                                        input_shape=(IMAGE_W, IMAGE_H, 3))
    # layers_to_freze = 1
    # for layer in pretarined_model.layers[:-layers_to_freze]:
    #     layer.trainable=False
    # for layer in pretarined_model.layers[-layers_to_freze:]:
    #     layer.trainable=True

    base_model.trainable = False
    inputs=Input(shape=(IMAGE_W, IMAGE_H, 3))

    x = inputs 
    x = preprocess_input(x)
    x = base_model(x, training=False)
    
    x = tf.keras.layers.Conv2D(256, kernel_size=1, padding="same", kernel_regularizer=regularizers.L2(0.0001))(x)
    #x = tf.keras.layers.Conv2D(256, kernel_size=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(4, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

    opt = Adam(lr=LR_VALUE)
    model.compile(loss="binary_crossentropy", optimizer=opt)
    print(model.summary())

    return model

IMAGES_PATH = "data/images"
ANNOTS_PATH = "data/labels"

def main():
    data = []
    targets = []
    filenames = []

    with tf.device('/CPU:0'):
        print("Loading data...")
        x = 1
        for i in range(len(os.listdir(ANNOTS_PATH))):
            if x <= 300:
                f = open(ANNOTS_PATH + "/" + str(i) +".json")
                data_j = json.load(f)
                data_j = data_j[str(i)]
                startX = data_j[0]
                startY = data_j[1]
                endX = data_j[2]
                endY = data_j[3]

                imagePath = IMAGES_PATH + "/" + str(i) + ".png"
                image = cv.imread(imagePath)
                (h, w) = image.shape[:2]

                startX = float(startX) / w
                startY = float(startY) / h
                endX = float(endX) / w
                endY = float(endY) / h

                image = load_img(imagePath, target_size=(IMAGE_W, IMAGE_H))
                image = img_to_array(image)

                data.append(image)
                targets.append((startX, startY, endX, endY))
                x += 1

        data = np.array(data, dtype="float32")

        targets = np.array(targets, dtype="float32")

        split = train_test_split(data, targets, test_size=0.10,
            random_state=42)

        (trainImages, testImages) = split[:2]
        (trainTargets, testTargets) = split[2:4]

    model2 = create_model()

    choice = input("Load or create new model [0, 1]: ")

    if choice == "0":
        model2.load_weights("model_fine_tuning.keras")

    if choice == "1":
        print("Training bounding box regressor...")
        H = model2.fit(
            trainImages, trainTargets,
            validation_data=(testImages, testTargets),
            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCHS,
            verbose=1)

        model2.save("model1.keras")

        N = NUM_EPOCHS

        loss = H.history["loss"]
        val_loss = H.history["val_loss"]

        plt.figure(figsize=[6, 4.8])
        plt.plot(np.arange(0, N), loss, label="train_loss")
        plt.plot(np.arange(0, N), val_loss, label="val_loss")
        plt.title("Bounding Box Regression Loss on Training Set")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
        plt.savefig("plots")
        plt.show()

        print("Finetunning...")

        base_model = model2.layers[3]

        base_model.trainable = True

        for layer in base_model.layers[:-FINE_TUNE_AT]:
            layer.trainable=False
        for layer in base_model.layers[-FINE_TUNE_AT:]:
            layer.trainable=True
        
        loss_function=tf.keras.losses.BinaryCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1 * LR_VALUE)
        model2.compile(loss=loss_function,
                optimizer = optimizer)
         
        print(model2.summary())

        history_fine = model2.fit(
            trainImages, trainTargets,
            validation_data=(testImages, testTargets),
            batch_size=BATCH_SIZE,
            epochs=FINE_TUNE_EPOCHS,
            verbose=1)
        
        N += FINE_TUNE_EPOCHS

        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']

        plt.figure(figsize=[6, 4.8])
        plt.plot(np.arange(0, N), loss, label="train_loss")
        plt.plot(np.arange(0, N), val_loss, label="val_loss")
        plt.title("Bounding Box Regression Loss on Training Set (with fine tunning)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
        plt.savefig("plots")
        plt.show()

        model2.save("model_fine_tuning.keras")
        

    test_model(model=model2, trainig_images=trainImages, trainig_labels=trainTargets, valid_images=testImages, valid_labels=testTargets)

if __name__ == "__main__":
   main()
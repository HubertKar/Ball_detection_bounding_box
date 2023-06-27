import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import json
from matplotlib.widgets import Button

IMAGE_W = 800
IMAGE_H = 800
LR_VALUE = 0.001

NUM_EPOCHS = 69
FINE_TUNE_EPOCHS = 30
FINE_TUNE_AT = 20
BATCH_SIZE = 16

DATASET_PATH = "data"

def show_image(model, trainig_images, trainig_labels):
    i = np.random.randint(0,len(trainig_images)-1) # Losowa zdjęcie i masska ze zbioru testowego
    sample_image = cv.imread(trainig_images[i])
    sample_image = cv.cvtColor(sample_image, cv.COLOR_BGR2RGB)

    sample_box = trainig_labels[i]
    sample_box = [int(sample_box[0] / sample_image.shape[1] * IMAGE_W), int(sample_box[1] / sample_image.shape[0] * IMAGE_H), int(sample_box[2] / sample_image.shape[1] * IMAGE_W), int(sample_box[3] / sample_image.shape[0] * IMAGE_H)]
    
    sample_image = cv.resize(sample_image, (IMAGE_W, IMAGE_H))


    prediction = model.predict(sample_image[tf.newaxis, ...])[0] # Przewidywanie maski, tf.newaxis powiększa wymiar macierzy o jeden, predict oczukuje zbioru danych  
    prediction_box = [int(prediction[0]* sample_image.shape[0]), int(prediction[1]* sample_image.shape[1]), int(prediction[2]* sample_image.shape[0]), int(prediction[3]* sample_image.shape[1])]
    
    image_copy = sample_image.copy()
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

class Next_button(object):  # Klasa do przycisku zarządzania przyciskiem w funkcji test_model
    def __init__(self, axes, label, images, labels, model):
        self.button = Button(axes, label)
        self.button.on_clicked(self.clicked)
        self.images = images
        self.labels = labels
        self.model = model

    def clicked(self, event):
        show_image(self.model, self.images, self.labels)
        plt.draw()

def test_model(model, trainig_images, trainig_labels, valid_images, valid_labels): # Funckja do testowania działania sieci 

    fig = plt.figure(figsize=(10,5))
    show_image(model, trainig_images, trainig_labels)
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

    base_model.trainable = False
    inputs = tf.keras.layers.Input(shape=(IMAGE_W, IMAGE_H, 3))

    x = inputs 
    x = preprocess_input(x)
    x = base_model(x, training=False)
    
    x = tf.keras.layers.Conv2D(256, kernel_size=1, padding="same", kernel_regularizer=tf.keras.regularizers.L2(0.0001))(x)
    #x = tf.keras.layers.Conv2D(256, kernel_size=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(4, activation="sigmoid")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    opt = tf.keras.optimizers.Adam(lr=LR_VALUE)
    model.compile(loss="binary_crossentropy", optimizer=opt)
    print(model.summary())

    return model

def load_dataset(path, split=0.1, max_img_number = -1):
    
    images = []
    bboxes = []

    labels_path = os.path.join(path, "labels")

    x = 1
    for i in range(len(os.listdir(labels_path))):
        if x <= max_img_number or max_img_number == -1:
            f = open(labels_path + "/" + str(i) +".json")
            data_j = json.load(f)
            data_j = data_j[str(i)]
            startX = data_j[0]
            startY = data_j[1]
            endX = data_j[2]
            endY = data_j[3]

            filename = str(i) + ".png"

            image = os.path.join(path, "images", filename)
            bbox = [startX, startY, endX, endY]

            images.append(image)
            bboxes.append(bbox)

            x += 1

    split_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(bboxes, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y)

def read_image_bbox(path, bbox):
    """ Image """
    path = path.decode()
    image = cv.imread(path, cv.IMREAD_COLOR)
    h, w, _ = image.shape
    image = cv.resize(image, (IMAGE_W, IMAGE_H))
    #image = (image - 127.5) / 127.5  ## [-1, +1]
    image = image.astype(np.float32)

    """ Bounding box """
    x1, y1, x2, y2 = bbox

    norm_x1 = float(x1 / w)
    norm_y1 = float(y1 / h)
    norm_x2 = float(x2 / w)
    norm_y2 = float(y2 / h)
    norm_bbox = np.array([norm_x1, norm_y1, norm_x2, norm_y2], dtype=np.float32)

    return image, norm_bbox

def parse(x, y):
    x, y = tf.numpy_function(read_image_bbox, [x, y], [tf.float32, tf.float32])
    x.set_shape([IMAGE_H, IMAGE_W, 3])
    y.set_shape([4])
    return x, y

def tf_dataset(images, bboxes, batch=8):
    ds = tf.data.Dataset.from_tensor_slices((images, bboxes))
    ds = ds.map(parse).batch(batch).prefetch(10)
    return ds

def main():

    (train_x, train_y), (valid_x, valid_y) = load_dataset(DATASET_PATH)

    model = create_model()

    choice = input("Load or create new model [0, 1]: ")

    if choice == "0":
        model.load_weights("model_fine_tuning.keras")

    if choice == "1":
        print("Training bounding box regressor...")

        train_ds = tf_dataset(train_x, train_y, batch=BATCH_SIZE)
        valid_ds = tf_dataset(valid_x, valid_y, batch=BATCH_SIZE)

        # H= model2.fit(
        # train_ds,
        # epochs=NUM_EPOCHS,
        # validation_data=valid_ds,
        # )

        # model2.save("model1.keras")

        # loss = H.history["loss"]
        # val_loss = H.history["val_loss"]

        # plt.figure(figsize=[6, 4.8])
        # plt.plot(np.arange(0, NUM_EPOCHS), loss, label="train_loss")
        # plt.plot(np.arange(0, NUM_EPOCHS), val_loss, label="val_loss")
        # plt.title("Bounding Box Regression Loss on Training Set")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.legend(loc="lower left")
        # plt.savefig("plots")
        # plt.show()
        base_model = model.layers[3]

        base_model.trainable = True

        for layer in base_model.layers[:-FINE_TUNE_AT]:
            layer.trainable=False
        for layer in base_model.layers[-FINE_TUNE_AT:]:
            layer.trainable=True
        
        loss_function=tf.keras.losses.BinaryCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1 * LR_VALUE)
        model.compile(loss=loss_function,
                optimizer = optimizer)
         
        print(model.summary())

        print("Training model...")
        H= model.fit(
        train_ds,
        epochs=NUM_EPOCHS,
        validation_data=valid_ds,
        )
        
        loss = H.history['loss']
        val_loss = H.history['val_loss']

        plt.figure(figsize=[6, 4.8])
        plt.plot(np.arange(0, NUM_EPOCHS), loss, label="train_loss")
        plt.plot(np.arange(0, NUM_EPOCHS), val_loss, label="val_loss")
        plt.title("Bounding Box Regression Loss on Training Set (with fine tunning)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
        plt.savefig("plots/plot")
        plt.show()

        model.save("model/model_fine_tuning.keras")
        
        test_model(model, train_x, train_y, valid_x, valid_y)

if __name__ == "__main__":
   main()
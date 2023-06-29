import torchvision
import torch
import yaml
from PIL import Image, ImageDraw
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from tkinter import Tcl

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import utils as utils

INPUT_W, INPUT_H = 2048, 1080
IMG_W, IMG_H = 1280, 720

def show_test_image(model, dataset1, dataset2, device, parameters):
    i = np.random.randint(0, len(dataset1)-1)

    img1 = dataset1[i]
    transform = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
    img1 = transform(img1)
    img1 = img1/255.0

    img2 = dataset2[i]
    img2 = transform(img2)
    img2 = img2/255.0

    model.eval()
 
    with torch.no_grad():
        prediction1 = model([img1.to(device)])
        prediction2 = model([img2.to(device)])

    prediction_boxes1 = prediction1[0]["boxes"]
    prediction_scores1 = prediction1[0]["scores"]
    prediction_scores1 = torch.Tensor.cpu(prediction_scores1)

    prediction_boxes2 = prediction2[0]["boxes"]
    prediction_scores2 = prediction2[0]["scores"]
    prediction_scores2 = torch.Tensor.cpu(prediction_scores2)

    img1 = Image.fromarray(img1.mul(255).permute(1, 2, 0).byte().numpy())
    img2 = Image.fromarray(img2.mul(255).permute(1, 2, 0).byte().numpy())

    if len(prediction_scores1) != 0: 
        prediction_box1 = prediction_boxes1[np.argmax(prediction_scores1)] 
        img1_d = ImageDraw.Draw(img1)
        img1_d.rectangle((prediction_box1[0], prediction_box1[1], prediction_box1[2], prediction_box1[3]), outline ="blue", width=3)
        
    if len(prediction_scores2) != 0: 
        prediction_box2= prediction_boxes2[np.argmax(prediction_scores2)]
        img2_d = ImageDraw.Draw(img2)
        img2_d.rectangle((prediction_box2[0], prediction_box2[1], prediction_box2[2], prediction_box2[3]), outline ="blue", width=3)


    if len(prediction_scores1) != 0 and len(prediction_scores2) != 0:
        print("Wykryto piłke")
        center_point1 = [(prediction_box1[1] + prediction_box1[0])/2 * INPUT_W/IMG_W, (prediction_box1[3] + prediction_box1[2])/2 * INPUT_H/IMG_H]
        center_point2 = [(prediction_box2[1] + prediction_box2[0])/2 * INPUT_W/IMG_W, (prediction_box2[3] + prediction_box2[2])/2 * INPUT_H/IMG_H]

        center_point1[0], center_point1[1]  = torch.Tensor.numpy(torch.Tensor.cpu(center_point1[0])), torch.Tensor.numpy(torch.Tensor.cpu(center_point1[1]))
        center_point2[0], center_point2[1]  = torch.Tensor.numpy(torch.Tensor.cpu(center_point2[0])), torch.Tensor.numpy(torch.Tensor.cpu(center_point2[1]))

        point3D = triangulate(parameters[0], parameters[1], parameters[2], parameters[3], center_point1, center_point2)     
        print(point3D)
    else:
        print("Nie wykryto piłki")
        
    plt.subplot(121)
    plt.imshow(img1)
    plt.axis('off')
    plt.title('First camera')

    plt.subplot(122)
    plt.imshow(img2)
    plt.axis('off')
    plt.title('Second camera')

class Next_button_test(object):  # Klasa do przycisku zarządzania przyciskiem w funkcji test_model
    def __init__(self, axes, label, dataset1, dataset2, model, device, parameters):
        self.button = Button(axes, label)
        self.button.on_clicked(self.clicked)
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.model = model
        self.device = device
        self.parameters = parameters

    def clicked(self, event):
        show_test_image(self.model, self.dataset1, self.dataset2, self.device, self.parameters)
        plt.draw()

def test_model_on_random_img(model, dataset1, dataset2, device, parameters): # Funckja do testowania działania sieci 

    fig = plt.figure(figsize=(15, 8))
    show_test_image(model, dataset1, dataset2, device, parameters)
    a3xnext = fig.add_axes([0.61, 0.05, 0.1, 0.075])
    b3next = Next_button_test(a3xnext, "NEXT IMG", dataset1, dataset2,  model, device, parameters)
    plt.show()

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root

        self.imgs = list(sorted(os.listdir(root))) 
        self.imgs = Tcl().call('lsort', '-dict', self.imgs) # Correct sorting 

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((1280, 720))

        return img

    def __len__(self):
        return len(self.imgs)
    
def triangulate(mtx1, mtx2, R, T, point1, point2):
 
    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = mtx1 @ RT1 #projection matrix for C1
 
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = mtx2 @ RT2 #projection matrix for C2
 
    def DLT(P1, P2, point1, point2):
 
        A = [point1[1]*P1[2,:] - P1[1,:],
             P1[0,:] - point1[0]*P1[2,:],
             point2[1]*P2[2,:] - P2[1,:],
             P2[0,:] - point2[0]*P2[2,:]
            ]
        A = np.array(A).reshape((4,4))
 
        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices = False)

        return Vh[3,0:3]/Vh[3,3]
 
    
    point3D = DLT(P1, P2, point1, point2)

    return point3D

def create_trajectory(model, dataset1, dataset2, device, parameters):
    trajecotry = []
    trajecotry_x = []
    trajecotry_y = []
    trajecotry_z = []
    print("calcutating trajectory...")

    for img1, img2 in zip(dataset1, dataset2):
        transform = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
        img1 = transform(img1)
        img1 = img1/255.0

        img2 = transform(img2)
        img2 = img2/255.0

        model.eval()
    
        with torch.no_grad():
            prediction1 = model([img1.to(device)])
            prediction2 = model([img2.to(device)])

        prediction_boxes1 = prediction1[0]["boxes"]
        prediction_scores1 = prediction1[0]["scores"]
        prediction_scores1 = torch.Tensor.cpu(prediction_scores1)

        prediction_boxes2 = prediction2[0]["boxes"]
        prediction_scores2 = prediction2[0]["scores"]
        prediction_scores2 = torch.Tensor.cpu(prediction_scores2)

        if len(prediction_scores1) != 0 and len(prediction_scores2) != 0:
            prediction_box1 = prediction_boxes1[np.argmax(prediction_scores1)]
            prediction_box2= prediction_boxes2[np.argmax(prediction_scores2)]

            center_point1 = [(prediction_box1[1] + prediction_box1[0])/2 * INPUT_W/IMG_W, (prediction_box1[3] + prediction_box1[2])/2 * INPUT_H/IMG_H]
            center_point2 = [(prediction_box2[1] + prediction_box2[0])/2 * INPUT_W/IMG_W, (prediction_box2[3] + prediction_box2[2])/2 * INPUT_H/IMG_H]

            center_point1[0], center_point1[1]  = torch.Tensor.numpy(torch.Tensor.cpu(center_point1[0])), torch.Tensor.numpy(torch.Tensor.cpu(center_point1[1]))
            center_point2[0], center_point2[1]  = torch.Tensor.numpy(torch.Tensor.cpu(center_point2[0])), torch.Tensor.numpy(torch.Tensor.cpu(center_point2[1]))

            point3D = triangulate(parameters["mtx1"], parameters["mtx2"], parameters["R"], parameters["T"], center_point1, center_point2)    
            trajecotry.append(point3D)
            trajecotry_x.append(point3D[0])
            trajecotry_y.append(point3D[1])
            trajecotry_z.append(point3D[2])
            
            print(point3D)
        else:
            print("Nie wykryto piłki")

    print("done")
    print(trajecotry)
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(trajecotry_x, trajecotry_y, trajecotry_z, label='trajectory')
    ax.set_xlim(-2, 1)
    ax.set_ylim(0, 4)
    ax.set_zlim(5, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def trajectory_visualization(model, dataset1, dataset2, device, parameters):
    trajecotry = []
    trajecotry_x = []
    trajecotry_y = []
    trajecotry_z = []

    fig = plt.figure(figsize=(30, 10))
    for img1, img2 in zip(dataset1, dataset2):
        plt.clf()
        transform = torchvision.transforms.Compose([torchvision.transforms.PILToTensor()])
        img1 = transform(img1)
        img1 = img1/255.0

        img2 = transform(img2)
        img2 = img2/255.0

        model.eval()
    
        with torch.no_grad():
            prediction1 = model([img1.to(device)])
            prediction2 = model([img2.to(device)])

        prediction_boxes1 = prediction1[0]["boxes"]
        prediction_scores1 = prediction1[0]["scores"]
        prediction_scores1 = torch.Tensor.cpu(prediction_scores1)

        prediction_boxes2 = prediction2[0]["boxes"]
        prediction_scores2 = prediction2[0]["scores"]
        prediction_scores2 = torch.Tensor.cpu(prediction_scores2)
        
        img1 = Image.fromarray(img1.mul(255).permute(1, 2, 0).byte().numpy())
        img2 = Image.fromarray(img2.mul(255).permute(1, 2, 0).byte().numpy())

        if len(prediction_scores1) != 0: 
            prediction_box1 = prediction_boxes1[np.argmax(prediction_scores1)] 
            img1_d = ImageDraw.Draw(img1)
            img1_d.rectangle((prediction_box1[0], prediction_box1[1], prediction_box1[2], prediction_box1[3]), outline ="blue", width=3)
        
        if len(prediction_scores2) != 0: 
            prediction_box2= prediction_boxes2[np.argmax(prediction_scores2)]
            img2_d = ImageDraw.Draw(img2)
            img2_d.rectangle((prediction_box2[0], prediction_box2[1], prediction_box2[2], prediction_box2[3]), outline ="blue", width=3)

        if len(prediction_scores1) != 0 and len(prediction_scores2) != 0:
            prediction_box1 = prediction_boxes1[np.argmax(prediction_scores1)]
            prediction_box2= prediction_boxes2[np.argmax(prediction_scores2)]

            center_point1 = [(prediction_box1[1] + prediction_box1[0])/2 * INPUT_W/IMG_W, (prediction_box1[3] + prediction_box1[2])/2 * INPUT_H/IMG_H]
            center_point2 = [(prediction_box2[1] + prediction_box2[0])/2 * INPUT_W/IMG_W, (prediction_box2[3] + prediction_box2[2])/2 * INPUT_H/IMG_H]

            center_point1[0], center_point1[1]  = torch.Tensor.numpy(torch.Tensor.cpu(center_point1[0])), torch.Tensor.numpy(torch.Tensor.cpu(center_point1[1]))
            center_point2[0], center_point2[1]  = torch.Tensor.numpy(torch.Tensor.cpu(center_point2[0])), torch.Tensor.numpy(torch.Tensor.cpu(center_point2[1]))

            point3D = triangulate(parameters["mtx1"], parameters["mtx2"], parameters["R"], parameters["T"], center_point1, center_point2)    
            trajecotry.append(point3D)
            trajecotry_x.append(point3D[0])
            trajecotry_y.append(point3D[1])
            trajecotry_z.append(point3D[2])
            
            print(point3D)
        else:
            print("Nie wykryto piłki")

        plt.subplot(131)
        plt.imshow(img1)
        plt.axis('off')
        plt.title('First camera')

        plt.subplot(132)
        plt.imshow(img2)
        plt.axis('off')
        plt.title('Second camera')

        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.plot3D(trajecotry_x, trajecotry_y, trajecotry_z, label='trajectory')
        ax.set_title("Trajectory")
        ax.set_xlim(-3.5, 1)
        ax.set_ylim(-0.1, 3)
        ax.set_zlim(5, 10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.draw()
        plt.pause(0.001)

    plt.show()

def load_calibration_parameters(sequence_nr):

    with open("calibration_parameters/" + sequence_nr + "/camera_1.yaml") as f:
        loadeddict = yaml.safe_load(f)
    mtx1 = loadeddict.get('camera_matrix')

    with open("calibration_parameters/" + sequence_nr + "/camera_2.yaml") as f:
        loadeddict = yaml.safe_load(f)
    mtx2 = loadeddict.get('camera_matrix')

    with open("calibration_parameters/" + sequence_nr + "/stereovision.yaml") as f:
        loadeddict = yaml.safe_load(f)
    R = loadeddict.get('rotation matrix')
    T = loadeddict.get('translation vector')

    parameters = { 
        "mtx1" : mtx1,
        "mtx2" : mtx2,
        "R": R,
        "T": T 
    }
    return parameters

def load_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    num_classes = 2 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    model.load_state_dict(torch.load("model/model.pt"))
    model.eval()

    return model, device

def main():
    # model = torch.jit.load('model/model_scripted.pt') # load model
    # model.eval()
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    sequnce_nr = "sequence_1"

    camera1_dataset = TestDataset("sekwencje/" + sequnce_nr + "/camera_1/frames")
    camera2_dataset = TestDataset("sekwencje/" + sequnce_nr + "/camera_2/frames")

    model, device = load_model()

    parameters = load_calibration_parameters(sequnce_nr)

    test_model_choice = input("[0] Test_mode_on_random_img, [1] create_trajectory, [2] trajectory_visualization:  ")

    if test_model_choice == "0":
        test_model_on_random_img(model, camera1_dataset, camera2_dataset, device, parameters)
    elif test_model_choice == "1":
        create_trajectory(model, camera1_dataset, camera2_dataset, device, parameters)
    elif test_model_choice == "2":
        trajectory_visualization(model, camera1_dataset, camera2_dataset, device, parameters)

if __name__ == "__main__":
    main()
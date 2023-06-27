import torchvision
import torch
import torch.utils.data
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image, ImageDraw
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import json

from engine import train_one_epoch, evaluate
import transforms as T
import utils as utils

<<<<<<< HEAD

def show_image(model, dataset, device):

=======
def show_image(model, dataset, device):

>>>>>>> 85ad28b793ddf255b99a390c876d0ff5b1f554e6
    i = np.random.randint(0, len(dataset)-1)
    img, targets = dataset[i] # Losowa zdjęcie i maska ze zbioru testowego
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    prediction_boxes = prediction[0]["boxes"]
    prediction_scores = prediction[0]["scores"]
    prediction_scores = torch.Tensor.cpu(prediction_scores)

    prediction_box = prediction_boxes[np.argmax(prediction_scores)]

    img1 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    img2 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

    sample_box = targets["boxes"]
    img1_d = ImageDraw.Draw(img1) 
    img2_d = ImageDraw.Draw(img2)

    img1_d.rectangle((sample_box[0, 0], sample_box[0, 1], sample_box[0, 2], sample_box[0, 3]), outline ="red", width=3)
    img2_d.rectangle((prediction_box[0], prediction_box[1], prediction_box[2], prediction_box[3]), outline ="blue", width=3)

    plt.subplot(121)
    plt.imshow(img1)
    plt.axis('off')
    plt.title('Ground true')

    plt.subplot(122)
    plt.imshow(img2)
    plt.axis('off')
    plt.title('Predicted')

def show_test_image(model, dataset, device):
    i = np.random.randint(0, len(dataset)-1)
    img = dataset[i]

    transform = transforms.Compose([transforms.PILToTensor()])
    img = transform(img)
    img = img/255.0

    model.eval()
 

    with torch.no_grad():
        prediction = model([img.to(device)])

    print(prediction)

    prediction_boxes = prediction[0]["boxes"]
    prediction_scores = prediction[0]["scores"]
    prediction_scores = torch.Tensor.cpu(prediction_scores)

    if len(prediction_scores) != 0: 
        prediction_box = prediction_boxes[np.argmax(prediction_scores)]
        print("Wykryto piłke")
    else:
        prediction_box = [0,0,0,0]
        print("Nie wykryto piłki")

    img1 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    img2 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

    img2_d = ImageDraw.Draw(img2)

    img2_d.rectangle((prediction_box[0], prediction_box[1], prediction_box[2], prediction_box[3]), outline ="blue", width=3)

    plt.subplot(121)
    plt.imshow(img1)
    plt.axis('off')
    plt.title('Image')

    plt.subplot(122)
    plt.imshow(img2)
    plt.axis('off')
    plt.title('Predicted box')

class Next_button(object):  # Klasa do przycisku zarządzania przyciskiem w funkcji test_model
    def __init__(self, axes, label, dataset, model, device):
        self.button = Button(axes, label)
        self.button.on_clicked(self.clicked)
        self.dataset = dataset
        self.model = model
        self.device = device

    def clicked(self, event):
        show_image(self.model, self.dataset, self.device)
        plt.draw()

class Next_button_test(object):  # Klasa do przycisku zarządzania przyciskiem w funkcji test_model
    def __init__(self, axes, label, dataset, model, device):
        self.button = Button(axes, label)
        self.button.on_clicked(self.clicked)
        self.dataset = dataset
        self.model = model
        self.device = device

    def clicked(self, event):
        show_test_image(self.model, self.dataset, self.device)
        plt.draw()

def test_model(model, trainig_dataset, valid_dataset, test_dataset, device): # Funckja do testowania działania sieci 

    fig = plt.figure(figsize=(10,5))
    show_image(model, trainig_dataset, device)
    axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    a2xnext = fig.add_axes([0.71, 0.05, 0.1, 0.075])
    a3xnext = fig.add_axes([0.61, 0.05, 0.1, 0.075])
    bnext = Next_button(axnext, "NEXT(Trainig)", trainig_dataset, model, device)
    b2next = Next_button(a2xnext, "NEXT(Valid)", valid_dataset, model, device)
    b3next = Next_button_test(a3xnext, "NEXT(TEST)", test_dataset, model, device)

    
    plt.show()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels"))))     

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, "images", self.imgs[idx])
        labels_path = os.path.join(self.root, "labels", self.labels[idx])
        img = Image.open(img_path).convert("RGB")

        num_objs = 1

        boxes = []
        f = open(labels_path)
        data_j = json.load(f)
        data_j = data_j["data"]
        startX = data_j[0]
        startY = data_j[1]
        endX = data_j[2]
        endY = data_j[3]

        boxes.append([startX, startY, endX, endY])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root, "images")))) 

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((1280, 720))

        return img

    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

NUM_EPOCHS = 5

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
<<<<<<< HEAD

    dataset = Dataset('data', get_transform(train=False))
    dataset_valid = Dataset('data', get_transform(train=False))
    dataset_test = TestDataset('test_data', get_transform(train=False))
=======
    # use our dataset and defined transformations
    dataset = Dataset('data', get_transform(train=False))
    dataset_valid = Dataset('data', get_transform(train=False))
    dataset_test = TestDataset('test_data1', get_transform(train=False))
>>>>>>> 85ad28b793ddf255b99a390c876d0ff5b1f554e6

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_valid = torch.utils.data.Subset(dataset_valid, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

<<<<<<< HEAD
    num_classes = 2 
=======
    num_classes = 2  
>>>>>>> 85ad28b793ddf255b99a390c876d0ff5b1f554e6
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # move model to the right device
    model.to(device)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    
    choice = input("Load or create new model [0, 1]: ")

    if choice == "0":
        model.load_state_dict(torch.load("model/model.pt"))
        model.eval()

    if choice == "1":
        for epoch in range(NUM_EPOCHS):
<<<<<<< HEAD
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
=======
            train_one_epoch(model, optimizer, data_loader, device, NUM_EPOCHS, print_freq=10)
>>>>>>> 85ad28b793ddf255b99a390c876d0ff5b1f554e6
            lr_scheduler.step()
            evaluate(model, data_loader_valid, device=device)

        torch.save(model.state_dict(), "model/model.pt")
<<<<<<< HEAD
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save('model/model_scripted.pt') # Save
=======
>>>>>>> 85ad28b793ddf255b99a390c876d0ff5b1f554e6

    test_model(model, dataset, dataset_valid, dataset_test, device)
    
if __name__ == "__main__":
   main()
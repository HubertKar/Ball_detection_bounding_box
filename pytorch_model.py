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
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import json

from engine import train_one_epoch, evaluate
import transforms as T
import utils as utils

# Program for training and testing model for boudning box detecion with pytorch

# Function for displaying two images using matplotlib (trainig and valid set)
def show_image(model, dataset, device):
    
    # Get random image from dataset and predict bounding box for it
    i = np.random.randint(0, len(dataset)-1)
    img, targets = dataset[i] # Losowa zdjęcie i maska ze zbioru testowego
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    # Get bounding box
    prediction_boxes = prediction[0]["boxes"]
    prediction_scores = prediction[0]["scores"]
    prediction_scores = torch.Tensor.cpu(prediction_scores)

    # Get one bounding box with max score
    prediction_box = prediction_boxes[np.argmax(prediction_scores)]

    # Convert to image
    img1 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    img2 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

    sample_box = targets["boxes"]
    img1_d = ImageDraw.Draw(img1) 
    img2_d = ImageDraw.Draw(img2)

    # Draw bounding box on image
    img1_d.rectangle((sample_box[0, 0], sample_box[0, 1], sample_box[0, 2], sample_box[0, 3]), outline ="red", width=3)
    img2_d.rectangle((prediction_box[0], prediction_box[1], prediction_box[2], prediction_box[3]), outline ="blue", width=3)

    # Show "Ground true" image
    plt.subplot(121)
    plt.imshow(img1)
    plt.axis('off')
    plt.title('Ground true')

    # Show "Predicted" image
    plt.subplot(122)
    plt.imshow(img2)
    plt.axis('off')
    plt.title('Predicted')

# Function for prediction result on real photos ("test" dataset, images without known boudnig box positions)
def show_test_image(model, dataset, device):

    # Get random image from dataset and proces it 
    i = np.random.randint(0, len(dataset)-1)
    img = dataset[i]

    transform = transforms.Compose([transforms.PILToTensor()])
    img = transform(img)
    img = img/255.0

    model.eval()

    with torch.no_grad():
        prediction = model([img.to(device)])

    print(prediction)

    # Get prediction boxes
    prediction_boxes = prediction[0]["boxes"]
    prediction_scores = prediction[0]["scores"]
    prediction_scores = torch.Tensor.cpu(prediction_scores)

    # Check if any ball was detected
    if len(prediction_scores) != 0: 
        prediction_box = prediction_boxes[np.argmax(prediction_scores)]
        print("ball detected")
    else:
        prediction_box = [0,0,0,0]
        print("ball NOT detected")

    img1 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    img2 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

    # Draw bounding box on image
    img2_d = ImageDraw.Draw(img2)
    img2_d.rectangle((prediction_box[0], prediction_box[1], prediction_box[2], prediction_box[3]), outline ="blue", width=3)

    # Show photo
    plt.subplot(121)
    plt.imshow(img1)
    plt.axis('off')
    plt.title('Image')

    # Show photo wit predicted bounding box
    plt.subplot(122)
    plt.imshow(img2)
    plt.axis('off')
    plt.title('Predicted box')

# Button class for test_model function
class Next_button(object): 
    def __init__(self, axes, label, dataset, model, device):
        self.button = Button(axes, label)
        self.button.on_clicked(self.clicked)
        self.dataset = dataset
        self.model = model
        self.device = device

    def clicked(self, event):
        show_image(self.model, self.dataset, self.device)
        plt.draw()

# Button class for test_model function
class Next_button_test(object):  
    def __init__(self, axes, label, dataset, model, device):
        self.button = Button(axes, label)
        self.button.on_clicked(self.clicked)
        self.dataset = dataset
        self.model = model
        self.device = device

    def clicked(self, event):
        show_test_image(self.model, self.dataset, self.device)
        plt.draw()

# Function for testing and viusalization 
def test_model(model, trainig_dataset, valid_dataset, test_dataset, device): 

    fig = plt.figure(figsize=(10,5))
    show_image(model, trainig_dataset, device)
    axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    a2xnext = fig.add_axes([0.71, 0.05, 0.1, 0.075])
    a3xnext = fig.add_axes([0.61, 0.05, 0.1, 0.075])
    bnext = Next_button(axnext, "NEXT(Trainig)", trainig_dataset, model, device)
    b2next = Next_button(a2xnext, "NEXT(Valid)", valid_dataset, model, device)
    b3next = Next_button_test(a3xnext, "NEXT(TEST)", test_dataset, model, device)

    plt.show()

# Dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels"))))     

    def __getitem__(self, idx):
        
        # Get image
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        labels_path = os.path.join(self.root, "labels", self.labels[idx])
        img = Image.open(img_path).convert("RGB")

        num_objs = 1

        # Read bounding box position
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

# Test Dataset class (for data without info about bounding box, unlabeled data)
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

# Number of trainig epochs
NUM_EPOCHS = 5

def main():
    # Get device, gpu with cuda is default
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Get trainig, valid and "test"* dataset
    # *test dataset contains real frames from cameras, without info about bounding box, only for visualization
    dataset = Dataset('data', get_transform(train=False))
    dataset_valid = Dataset('data', get_transform(train=False))
    dataset_test = TestDataset('test_data', get_transform(train=False))

    # Split the dataset in train and valid set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_valid = torch.utils.data.Subset(dataset_valid, indices[-50:])

    # Define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # Load pretrained model resnet50
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    num_classes = 2 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Move model to the right device
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    
    # Load trained model or train a new one
    choice = input("Load or create new model [0, 1]: ")

    if choice == "0": 
        model.load_state_dict(torch.load("model/model.pt")) # Load model from "model" catalog
        model.eval()

    if choice == "1":
        for epoch in range(NUM_EPOCHS): # Traing loop
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            train_one_epoch(model, optimizer, data_loader, device, NUM_EPOCHS, print_freq=10)
            lr_scheduler.step()
            evaluate(model, data_loader_valid, device=device)

        torch.save(model.state_dict(), "model/model.pt") # Save model
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save('model/model_scripted.pt') # Save (this version doesn't work for some reason)

    # Test model
    test_model(model, dataset, dataset_valid, dataset_test, device)
    
if __name__ == "__main__":
   main()

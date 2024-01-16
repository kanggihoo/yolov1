import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss
torch.manual_seed(42)

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 10
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"

class Compose():
    def __init__(self):
        self.transforms=transforms.Compose([
        transforms.Resize(size=(448,448)),
        transforms.ToTensor()
        ])

    def __call__(self , img , bboxes):
        img  = self.transforms(img)
        return img , bboxes
    
data_transforms = Compose()

def train_fn(train_loader , model , optimizer , loss_fn , device):
    mean_loss= []
    for batch_idx , (x,y) in tqdm(enumerate(train_loader)):
        print(x)
        x , y = x.to(device) , y.to(device)
        out = model(x)
        loss = loss_fn(out , y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss.append(loss.item())
    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def main():
    model = Yolov1(split_size = 7 , num_boxes=2 , num_classes=20).to(DEVICE)
    loss_fn = YoloLoss()
    optimizer = optim.Adam(model.parameters() , lr = LEARNING_RATE , weight_decay=WEIGHT_DECAY)
    
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
    
    train_dataset = VOCDataset("./data/train.csv" , transforms=data_transforms)
    test_dataset = VOCDataset("./data/test.csv" , transforms=data_transforms)
    
    train_loader = DataLoader(train_dataset , batch_size = BATCH_SIZE , shuffle=True , pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset , batch_size = BATCH_SIZE , shuffle=True , pin_memory=PIN_MEMORY)
    
    
    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4,device=DEVICE
        )
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")
        
        train_fn(train_loader, model, optimizer, loss_fn,DEVICE)
if __name__ == "__main__":
    main()

    
    
    
    
    
    
    

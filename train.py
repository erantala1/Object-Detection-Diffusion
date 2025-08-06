
import torch
import torch.nn as nn
import os
import numpy as np
from model import *
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader



def collate(batch):
    imgs = []
    box_tensors = []
    label_tensors = []
    for img, anns in batch:
        imgs.append(img)
        boxes = []
        labels = []
        for obj in anns:
            x, y, w, h = obj["bbox"]
            cx, cy = x + w/2, y + h/2
            boxes.append([cx, cy, w, h])
            labels.append(obj["category_id"])

        if not boxes:
            boxes.append([0., 0., 0., 0.])
            labels.append(-1)

        box_tensors.append(torch.tensor(boxes, dtype=torch.float32))
        label_tensors.append(torch.tensor(labels,dtype=torch.long))

    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)
    B = len(imgs)
    padded_imgs = torch.zeros(len(imgs), 3, max_h, max_w)
    for i, img in enumerate(imgs):
        c, h, w = img.shape
        padded_imgs[i, :, :h, :w] = img

    max_n = max(b.size(0) for b in box_tensors)
    padded_boxes  = torch.zeros(B, max_n, 4, dtype=torch.float32)
    padded_labels = torch.full((B, max_n), -1, dtype=torch.long)   # -1 pad
    for i, (b, l) in enumerate(zip(box_tensors, label_tensors)):
        n = b.size(0)
        padded_boxes[i, :n]  = b
        padded_labels[i, :n] = l

    return padded_imgs, padded_boxes, padded_labels


def set_prediction_loss(prediction, gt_boxes, loss_fn):
    gt = pad_boxes(prediction.shape[1],gt_boxes)
    return loss_fn(prediction,gt)


#add random boxes to ground truth images so that each image has N boxes
def pad_boxes(N, gt_boxes):
    B, M, _ = gt_boxes.shape
    if M > N:
        return gt_boxes[:,:N,:]
    elif M == N:
        return gt_boxes
    num = N - M
    extra_center = torch.randn(B, num, 2, device=gt_boxes.device)
    extra_wh  = torch.randn(B, num, 2, device=gt_boxes.device).abs() * 0.2
    extra = torch.cat([extra_center, extra_wh], dim=-1).clamp(-1.5, 1.5)
    #create random centers between (0,1)
    #with width and heigh between (0,0.2)
    padded = torch.cat([gt_boxes, extra], dim=1)
    return padded

def train_loss(images, gt_boxes, gt_labels):
    B = images.size(0)
    feats = encode(images)
    pb = pad_boxes(N, gt_boxes)
    pb = (pb * 2 - 1) * scale
    t = torch.randint(0, T, (B,), device=device)
    pb_crpt = diffusion.forward_process(pb,t)
    pb_pred, logits = decode(pb_crpt, feats, t)
    loss = criterion(pb_pred, logits, gt_boxes, gt_labels) 
    return loss

if __name__=="__main__":
    T = 1000
    device = 'cpu' #change to cude if available
    num_classes = 91
    roi_size = 7
    hidden_dim = 256
    diffusion = Diffusion(T,device).to(device)
    encode = ImageEncoder(device, trainable = False, weights = True, norm_layer = nn.BatchNorm2d).to(device)
    decode = DetectionDecoder(num_classes, device, roi_size, hidden_dim).to(device)
    criterion = HungarianSetCriterion(num_classes=81,lambda_cls=1.0,lambda_l1=5.0,lambda_giou=2.0,k_best=3).to(device)
    learning_rate = 1e-5
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(decode.parameters(), lr = learning_rate)
    scale = 2.0 #adjust
    N = 300 #proposal boxes
    epochs = 50


    ROOT = "/Users/evanrantala/Downloads/COCO/coco2017"
    TRAIN_IMG_DIR = os.path.join(ROOT, "train2017")
    VAL_IMG_DIR   = os.path.join(ROOT, "val2017")
    TRAIN_ANN = os.path.join(ROOT, "annotations", "instances_train2017.json")
    VAL_ANN   = os.path.join(ROOT, "annotations", "instances_val2017.json")

    to_tensor = transforms.ToTensor()
    train_ds = CocoDetection(TRAIN_IMG_DIR, TRAIN_ANN, transform=to_tensor)
    val_ds   = CocoDetection(VAL_IMG_DIR,   VAL_ANN,   transform=to_tensor)

    train_loader = DataLoader(
    train_ds, batch_size=16, shuffle=True,
    num_workers=4, collate_fn=collate, pin_memory=True)

    val_loader = DataLoader(
    val_ds, batch_size=16, shuffle=False,
    num_workers=4, collate_fn=collate, pin_memory=True)


    for ep in range(epochs):
        running_loss = 0.0
        for images, gt_boxes, gt_labels in train_loader:
            images = images.to(device)
            gt_boxes = gt_boxes.to(device)
            optimizer.zero_grad()
            loss = train_loss(images, gt_boxes, gt_labels)
            print(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().item()
        net_loss = (running_loss/(len(train_loader)))
        print(f"epoch:{ep}, running_loss = {running_loss}")
        print(f"epoch:{ep}, net_loss = {net_loss}")
    torch.save(decode.state_dict(), "decoder.pth")
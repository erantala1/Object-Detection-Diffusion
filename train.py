
import torch
import torch.nn as nn
import os
import numpy as np
from model import *
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb



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
            # Ensure boxes are valid (positive dimensions)
            if w > 0 and h > 0:
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

def train_loss(images, gt_boxes, gt_labels, diffusion, encode, decode, criterion, N, scale, T, device):
    B, _, H_img, W_img = images.shape
    feats = encode(images)
    gt_norm = gt_boxes.clone()
    gt_norm[..., 0] /= W_img   # cx
    gt_norm[..., 2] /= W_img   # w
    gt_norm[..., 1] /= H_img   # cy
    gt_norm[..., 3] /= H_img   # h
    pb = pad_boxes(N, gt_norm)
    pb_scaled = (pb * 2 - 1) * scale

    t = torch.randint(0, T, (B,), device=device)
    pb_crpt = diffusion.forward_process(pb_scaled, t)
    pb_pred_scaled, logits = decode(pb_crpt, feats, t)

    pb_pred_norm = (pb_pred_scaled / scale + 1) / 2

    pb_pred_px = pb_pred_norm.clone()
    pb_pred_px[..., 0] *= W_img
    pb_pred_px[..., 2] *= W_img
    pb_pred_px[..., 1] *= H_img
    pb_pred_px[..., 3] *= H_img

    loss, loss_dict = criterion(pb_pred_px, logits, gt_boxes, gt_labels)
    return loss, loss_dict

if __name__=="__main__":
    T = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'  # Use mixed precision only on CUDA
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    num_classes = 91
    roi_size = 7
    hidden_dim = 256
    diffusion = Diffusion(T,device).to(device)
    encode = ImageEncoder(device, trainable = False, weights = True, norm_layer = nn.BatchNorm2d).to(device)
    decode = DetectionDecoder(num_classes, device, roi_size, hidden_dim).to(device)
    
    # Set models to appropriate modes
    encode.eval()  # Encoder should always be in eval mode since it's frozen
    diffusion.eval()  # Diffusion model should be in eval mode during training
    decode.train()  # Only decoder should be in train mode
    
    criterion = HungarianSetCriterion(num_classes,lambda_cls=1.0,lambda_l1=5.0,lambda_giou=2.0,k_best=3).to(device)
    learning_rate = 2.5 * 1e-5
    optimizer = torch.optim.AdamW(decode.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Add warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    scale = 2.0 #adjust
    N = 300 #proposal boxes
    epochs = 50
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="evan-rantala-university-of-california",
        # Set the wandb project where this run will be logged.
        project="Diffusion Object Detection",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 2.5 * 1e-5,
            "architecture": "DiffusionDet",
            "dataset": "COCO2017",
            "N_train": 300,
            "Hidden Dimension": 256,
            "epochs": 50,
        },
    )
    wandb.define_metric("global_step")
    wandb.define_metric("batch/*", step_metric="global_step")
    wandb.define_metric("epoch")
    wandb.define_metric("epoch/*", step_metric="epoch")

    ROOT = "/Users/evanrantala/Downloads/COCO/coco2017"
    TRAIN_IMG_DIR = os.path.join(ROOT, "train2017")
    VAL_IMG_DIR   = os.path.join(ROOT, "val2017")
    TRAIN_ANN = os.path.join(ROOT, "annotations", "instances_train2017.json")
    VAL_ANN   = os.path.join(ROOT, "annotations", "instances_val2017.json")

    to_tensor = transforms.ToTensor()
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0), ratio=(0.75, 1.33))
    ])
    val_transform = transforms.ToTensor()
    
    train_ds = CocoDetection(TRAIN_IMG_DIR, TRAIN_ANN, transform=train_transform)
    val_ds   = CocoDetection(VAL_IMG_DIR,   VAL_ANN,   transform=val_transform)

    train_loader = DataLoader(
    train_ds, batch_size=16, shuffle=True,
    num_workers=4, collate_fn=collate, pin_memory=True)

    val_loader = DataLoader(
    val_ds, batch_size=16, shuffle=False,
    num_workers=4, collate_fn=collate, pin_memory=True)

    ckpt_dir = ".checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    '''
    ckpt_path = os.path.join(ckpt_dir,"last.pth")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path,map_location=device)
        decode.load_state_dict(ckpt["decoder"])
        encode.load_state_dict(ckpt["encoder"])
        diffusion.load_state_dict(ckpt["diffusion"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("warmup_scheduler"):
            warmup_scheduler.load_state_dict(ckpt["warmup_scheduler"])
        if scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch",-1) + 1
        best_loss = ckpt.get("epoch_loss", float("inf"))
        global_step = ckpt.get("global_step")
    '''
    start_epoch = 0
    global_step = 0
    for ep in range(start_epoch,epochs):
        print(f"Starting epoch {ep+1}/{epochs}")
        running_loss = 0.0
        num_batches = 0
        for images, gt_boxes, gt_labels in train_loader:
            images = images.to(device)
            gt_boxes = gt_boxes.to(device)
            gt_labels = gt_labels.to(device)
            
            # Validate data
            if torch.isnan(images).any() or torch.isnan(gt_boxes).any():
                print("Warning: NaN detected in input data, skipping batch")
                continue
                
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss, loss_dict = train_loss(images, gt_boxes, gt_labels, diffusion, encode, decode, criterion, N, scale, T, device)
            else:
                loss, loss_dict = train_loss(images, gt_boxes, gt_labels, diffusion, encode, decode, criterion, N, scale, T, device)
                
            print(loss)
            
            if use_amp:
                scaler.scale(loss).backward()
                # Add gradient clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(decode.parameters(), max_norm=1.0)
                
                # Log gradient norms for debugging
                total_norm = 0
                for p in decode.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # Add gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(decode.parameters(), max_norm=1.0)
                
                # Log gradient norms for debugging
                total_norm = 0
                for p in decode.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                optimizer.step()
            run.log({
                "global_step": global_step,
                "batch/loss_total": loss.item(),
                "batch/loss_cls":   loss_dict["cls"],
                "batch/loss_l1":    loss_dict["l1"],
                "batch/loss_giou":  loss_dict["giou"],
                "batch/learning_rate": optimizer.param_groups[0]['lr'],
                "batch/gradient_norm": total_norm,
            })

            global_step += 1
            running_loss += loss.item()
            num_batches += 1
            
            # Apply warmup scheduler for first 5 epochs
            if ep < 5:
                warmup_scheduler.step()
        epoch_loss = (running_loss/num_batches)
        print(f"epoch:{ep}, running_loss = {running_loss}")
        print(f"epoch:{ep}, epoch_loss = {epoch_loss}")
        run.log({
        "epoch": ep,
        "epoch/loss": epoch_loss,
        })
        # Save checkpoint every 5 epochs or if it's the best so far
        if (ep + 1) % 5 == 0 or epoch_loss < best_loss:
            ckpt = {
            "epoch": ep,
            "decoder": decode.state_dict(),
            "encoder": encode.state_dict(),
            "diffusion": diffusion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "warmup_scheduler": warmup_scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "epoch_loss": epoch_loss,
            "global_step": global_step
            }
            torch.save(ckpt, os.path.join(ckpt_dir, "last.pth"))
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(ckpt, os.path.join(ckpt_dir, "best.pth"))
                patience_counter = 0
            else:
                patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
        
        # Validation loop
        decode.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, gt_boxes, gt_labels in val_loader:
                images = images.to(device)
                gt_boxes = gt_boxes.to(device)
                gt_labels = gt_labels.to(device)
                
                # Validate data
                if torch.isnan(images).any() or torch.isnan(gt_boxes).any():
                    print("Warning: NaN detected in validation data, skipping batch")
                    continue
                    
                loss, loss_dict = train_loss(images, gt_boxes, gt_labels, diffusion, encode, decode, criterion, N, scale, T, device)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        print(f"epoch:{ep}, val_loss = {val_loss}")
        run.log({
            "epoch": ep,
            "epoch/loss": epoch_loss,
            "epoch/val_loss": val_loss,
            "epoch/learning_rate": scheduler.get_last_lr()[0],
        })
        decode.train()
        
        # Update learning rate
        scheduler.step()
        
    print("Training Done")
    run.finish()
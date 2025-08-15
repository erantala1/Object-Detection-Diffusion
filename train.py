
import torch
import torch.nn as nn
import os
import numpy as np
from model import *
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb


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

def train_loss(images, gt_boxes, gt_labels, sizes, diffusion, encode, decode, criterion, N, scale, T, device):
    B, _, H_pad, W_pad = images.shape
    feats = encode(images)
    targets = []
    gt_norm_list = []
    for b in range(B):
        H_i, W_i = sizes[b].tolist()
        l  = gt_labels[b]
        bx = gt_boxes[b]
        valid = (l != -1) & (bx.abs().sum(dim=-1) > 0)
        l  = l[valid]
        bx = bx[valid]

        g_norm = bx.clone()
        g_norm[:, [0,2]] /= W_i
        g_norm[:, [1,3]] /= H_i
        gt_norm_list.append(g_norm)

        g_xyxy = cxcywh_to_xyxy(bx)
        g_xyxy = clamp_xyxy(g_xyxy, H_i, W_i)

        targets.append({
            "labels": l.long(),
            "boxes": g_norm,
            "boxes_xyxy": g_xyxy,
            "image_size_xyxy": torch.tensor([W_i, H_i, W_i, H_i], device=device, dtype=torch.float32),
            "image_size_xyxy_tgt": torch.tensor([W_i, H_i, W_i, H_i], device=device, dtype=torch.float32),
        })

    pb_list = []
    for b in range(B):
        gn  = gt_norm_list[b]
        gn  = gn.unsqueeze(0)
        pb_b = pad_boxes(N, gn).squeeze(0)
        pb_list.append(pb_b)
    pb = torch.stack(pb_list, dim=0)

    pb_scaled = (pb * 2 - 1) * scale

    t = torch.randint(0, T, (B,), device=device)
    pb_crpt = diffusion.forward_process(pb_scaled, t)
    pb_pred_scaled, logits = decode(pb_crpt, feats, t)

    pb_pred_norm = (pb_pred_scaled / scale + 1) / 2
    pb_pred_norm = torch.cat(
        [pb_pred_norm[..., :2],
        torch.clamp(pb_pred_norm[..., 2:], 1e-6, 1.0)],
        dim=-1
    )

    pred_xyxy = cxcywh_to_xyxy(pb_pred_norm)  # [B,N,4] (normalized)
    pred_xyxy_scaled = []
    for b in range(B):
        H_i, W_i = sizes[b].tolist()
        xy = pred_xyxy[b]  # [N,4]
        # scale to pixels without modifying in place
        scale_vec = xy.new_tensor([W_i, H_i, W_i, H_i])[None, :]
        xy = xy * scale_vec
        xy = clamp_xyxy(xy, H_i, W_i)  # returns a new tensor
        pred_xyxy_scaled.append(xy)

    pred_xyxy = torch.stack(pred_xyxy_scaled, dim=0)  # [B,N,4]
    outputs = {"pred_logits": logits, "pred_boxes": pred_xyxy}

    loss_dict = criterion(outputs, targets)
    total = (criterion.weight_dict["loss_ce"]   * loss_dict["loss_ce"] +
             criterion.weight_dict["loss_bbox"] * loss_dict["loss_bbox"] +
             criterion.weight_dict["loss_giou"] * loss_dict["loss_giou"])
    return total, loss_dict



if __name__=="__main__":
    ROOT = "/Users/evanrantala/Downloads/COCO/coco2017"
    #ROOT = "./data"
    TRAIN_IMG_DIR = os.path.join(ROOT, "train2017")
    VAL_IMG_DIR   = os.path.join(ROOT, "val2017")
    TRAIN_ANN = os.path.join(ROOT, "annotations", "instances_train2017.json")
    VAL_ANN   = os.path.join(ROOT, "annotations", "instances_val2017.json")

    to_tensor = transforms.ToTensor()
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
    ])
    val_transform = transforms.ToTensor()
    
    train_ds = CocoDetection(TRAIN_IMG_DIR, TRAIN_ANN, transform=train_transform)
    val_ds   = CocoDetection(VAL_IMG_DIR,   VAL_ANN,   transform=val_transform)

    cat_ids = sorted(train_ds.coco.getCatIds())
    cat_id_to_contig = {cid: i for i, cid in enumerate(cat_ids)}
    num_classes = len(cat_ids)

    def collate(batch):
        imgs, box_tensors, label_tensors, sizes = [], [], [], []
        for img, anns in batch:
            C, H, W = img.shape
            sizes.append((H, W))
            imgs.append(img)
            boxes, labels = [], []
            for obj in anns:
                x, y, w, h = obj["bbox"]
                if w > 0 and h > 0:
                    cx, cy = x + w/2, y + h/2
                    boxes.append([cx, cy, w, h])
                    labels.append(cat_id_to_contig.get(obj["category_id"], -1))

            if not boxes:
                boxes.append([0.,0.,0.,0.])
                labels.append(-1)

            box_tensors.append(torch.tensor(boxes, dtype=torch.float32))
            label_tensors.append(torch.tensor(labels, dtype=torch.long))

        max_h = max(h for h, w in sizes)
        max_w = max(w for h, w in sizes)
        B = len(imgs)
        padded_imgs = torch.zeros(B, 3, max_h, max_w)
        for i, img in enumerate(imgs):
            c, h, w = img.shape
            padded_imgs[i, :, :h, :w] = img

        max_n = max(b.size(0) for b in box_tensors)
        padded_boxes  = torch.zeros(B, max_n, 4, dtype=torch.float32)
        padded_labels = torch.full((B, max_n), -1, dtype=torch.long)
        for i, (b, l) in enumerate(zip(box_tensors, label_tensors)):
            n = b.size(0)
            padded_boxes[i, :n]  = b
            padded_labels[i, :n] = l

        sizes_tensor = torch.tensor(sizes, dtype=torch.long)
        return padded_imgs, padded_boxes, padded_labels, sizes_tensor


    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0, collate_fn=collate, pin_memory=True)

    T = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    num_classes = 80
    roi_size = 7
    hidden_dim = 256
    diffusion = Diffusion(T,device).to(device)
    encode = ImageEncoder(device, trainable = False, weights = True, norm_layer = nn.BatchNorm2d).to(device)
    decode = DetectionDecoder(num_classes, device, roi_size, hidden_dim).to(device)
    matcher = MatcherDynamicK(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0, use_focal=True, ota_k=10).to(device)
    criterion = SetCriterionDynamicKLite(num_classes=num_classes, matcher=matcher, weight_dict={"loss_ce":1.0,"loss_bbox":5.0,"loss_giou":2.0}, use_focal=True).to(device)
    
    encode.eval()
    diffusion.eval()
    decode.train()
    learning_rate = 2.5 * 1e-5
    optimizer = torch.optim.AdamW(decode.parameters(), lr = learning_rate)

    steps_per_epoch = len(train_loader)
    warmup_iters = 5* steps_per_epoch

    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
        schedulers = [torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=0.1,total_iters=warmup_iters),
        torch.optim.lr_scheduler.StepLR(optimizer,step_size=10*steps_per_epoch,gamma=0.5)], milestones=[warmup_iters])

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
            "N_train": 500,
            "Hidden Dimension": 256,
            "epochs": 50,
        },
    )
    wandb.define_metric("global_step")
    wandb.define_metric("batch/*", step_metric="global_step")
    wandb.define_metric("epoch")
    wandb.define_metric("epoch/*", step_metric="epoch")

    ckpt_dir = ".checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    ckpt_path = os.path.join(ckpt_dir,"last.pth")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path,map_location=device)
        dec_state = ckpt["decoder"]
        dec_state = {k: v for k, v in dec_state.items() if not k.startswith("fc_cls.")}
        missing, unexpected = decode.load_state_dict(dec_state, strict=False)
        print("decoder loaded - missing:", missing, "unexpected:", unexpected)
        decode.fc_cls.reset_parameters()
        encode.load_state_dict(ckpt["encoder"])
        diffusion.load_state_dict(ckpt["diffusion"])
        '''
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        '''
        start_epoch = ckpt.get("epoch",-1) + 1
        best_loss = ckpt.get("epoch_loss", float("inf"))
        global_step = ckpt.get("global_step")
    
    
    start_epoch = 0
    global_step = 0
    torch.autograd.set_detect_anomaly(True)
    for ep in range(start_epoch,epochs):
        print(f"Starting epoch {ep+1}/{epochs}")
        running_loss = 0.0
        num_batches = 0
        for images, gt_boxes, gt_labels, sizes in train_loader:
            images, gt_boxes, gt_labels, sizes = images.to(device), gt_boxes.to(device), gt_labels.to(device), sizes.to(device)

            if torch.isnan(images).any() or torch.isnan(gt_boxes).any():
                print("Warning: NaN detected in input data, skipping batch")
                continue
                
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    loss, loss_dict = train_loss(images, gt_boxes, gt_labels, sizes, diffusion, encode, decode, criterion, N, scale, T, device)
            else:
                loss, loss_dict = train_loss(images, gt_boxes, gt_labels, sizes, diffusion, encode, decode, criterion, N, scale, T, device)
                
            print(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decode.parameters(), max_norm=1.0)
            
            optimizer.step()
            run.log({
                "global_step": global_step,
                "batch/loss_total": loss.item(),
                "batch/loss_ce":    loss_dict["loss_ce"].item(),
                "batch/loss_bbox":  loss_dict["loss_bbox"].item(),
                "batch/loss_giou":  loss_dict["loss_giou"].item(),
                "batch/learning_rate": optimizer.param_groups[0]['lr'],
            })

            global_step += 1
            running_loss += loss.item()
            num_batches += 1

            scheduler.step()
        epoch_loss = (running_loss/num_batches)
        print(f"epoch:{ep}, running_loss = {running_loss}")
        print(f"epoch:{ep}, epoch_loss = {epoch_loss}")
        run.log({
        "epoch": ep,
        "epoch/loss": epoch_loss,
        })

        if (ep + 1) % 5 == 0 or epoch_loss < best_loss:
            ckpt = {
            "epoch": ep,
            "decoder": decode.state_dict(),
            "encoder": encode.state_dict(),
            "diffusion": diffusion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
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

        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break

        decode.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, gt_boxes, gt_labels, sizes in val_loader:
                images = images.to(device)
                gt_boxes = gt_boxes.to(device)
                gt_labels = gt_labels.to(device)
                sizes = sizes.to(device)

                if torch.isnan(images).any() or torch.isnan(gt_boxes).any():
                    print("Warning: NaN detected in validation data, skipping batch")
                    continue
                    
                loss, loss_dict = train_loss(images, gt_boxes, gt_labels, sizes, diffusion, encode, decode, criterion, N, scale, T, device)
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

        scheduler.step()
        
    print("Training Done")
    run.finish()
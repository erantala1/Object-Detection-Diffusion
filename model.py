import torch
import torch.nn as nn
import numpy as np
import math
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch.nn.functional as F
from torchvision.ops import box_convert, generalized_box_iou
from scipy.optimize import linear_sum_assignment

class Diffusion(nn.Module):
    def __init__(self,T,device):
        super().__init__()
        self.device = device
        self.T = T
        self.steps  = torch.arange(T+1, dtype=torch.float32).to(device)
        self.alphas_c = (torch.cos((self.steps / T + 0.008) / 1.008 * math.pi / 2) ** 2).to(device)
        self.alphas_c /= (self.alphas_c[0]).clone().to(device)
        self.betas = (1 - (self.alphas_c[1:] / self.alphas_c[:-1])).to(device)
        self.betas = self.betas.clamp(1e-8, 0.999).to(device) 

        self.alpha = (1-self.betas).to(device)
        self.alpha_cumprod = torch.cumprod(self.alpha,dim=0).to(device)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_cumprod).to(device)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1-self.alpha_cumprod).to(device)

    def forward_process(self, x0, t): 
        c1 = self.sqrt_alpha_hat[t].view(-1, 1, 1)
        c2 = self.sqrt_one_minus_alpha_hat[t].view(-1,1,1)
        eps = torch.randn_like(x0)
        return c1 * x0 + c2 * eps



class ImageEncoder(nn.Module):
    #encodes images and returns features 
    def __init__(self, device, trainable = False, weights = True, norm_layer = nn.BatchNorm2d):
        super().__init__()
        self.device = device
        self.backbone = resnet_fpn_backbone(
            backbone_name = "resnet50",
            weights     = weights,
            norm_layer     = norm_layer,
            trainable_layers = 5 if trainable else 0
        )
        #Optionally freeze (or partially fine-tune) parameters
        if not trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor):
        return self.backbone(images)
   
class DetectionDecoder(nn.Module):
    #returns coordinates of cleaned boxes based on encoded features and input image with noisy boxes at specified time step t
    #inputs: ->
    #boxes_t : (B, N, 4)
    #feats : dict()
    #t : time step

    #outputs: ->
    #boxes_cleaned : (B, N, 4)
    #logits : (B, N, C)

    def __init__(self,num_classes,device, roi_size, hidden_dim):
        super().__init__()
        self.device = device
        self.roi_align = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0','1','2','3'], output_size=roi_size, sampling_ratio=2)
        in_dim = 256 * roi_size * roi_size
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_reg  = nn.Linear(hidden_dim, 4)
        self.fc_cls  = nn.Linear(hidden_dim, num_classes)

        self.t_dim = 128
        self.t_embed_proj = nn.Sequential(
            nn.Linear(self.t_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def time_embedding(self,t, dim, device):
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=device) / half
        )
        t = t.float().unsqueeze(-1)
        ang = t * freqs
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb        
    
    def center_to_corner(self,boxes):
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5
        return torch.stack((x1,y1,x2,y2),dim=-1)

    def forward(self, boxes_t, feats, t):
        B, N, _ = boxes_t.shape
        boxes_xyxy = self.center_to_corner(boxes_t)
        boxes_list = [boxes_xyxy[i] for i in range(B)]
        _, _, H_pad, W_pad = next(iter(feats.values())).shape
        H_img, W_img = H_pad * 4, W_pad * 4
        crops = self.roi_align(feats, boxes_list, [(H_img, W_img)] * B)

        feats_flat = crops.flatten(1)
        h = self.act(self.fc1(feats_flat)) 

        sin_emb = self.time_embedding(t.float(), self.t_dim, boxes_t.device)
        t_embed = self.t_embed_proj(sin_emb)
        t_embed = t_embed[:, None, :].expand(B, N, -1).reshape(B*N, -1)
        h = h + t_embed

        h = self.act(self.fc2(h))
        delta  = self.fc_reg(h).reshape(B, N, 4)
        logits = self.fc_cls(h).reshape(B, N, -1)
        boxes_cleaned = boxes_t + delta
        return boxes_cleaned, logits
    


class HungarianSetCriterion(torch.nn.Module):
    def __init__(self, num_classes,lambda_cls=1.0,lambda_l1=5.0,lambda_giou=2.0, k_best=3):
        super().__init__()
        self.num_classes = num_classes
        self.l_cls  = lambda_cls
        self.l_l1   = lambda_l1
        self.l_giou = lambda_giou
        self.k_best = k_best

    def forward(self, pred_boxes, pred_logits, gt_boxes, gt_labels):
        """
        pred_boxes : [B, N, 4]  (cx cy w h) in **same scale** as GT
        pred_logits: [B, N, C]  (before softmax)
        gt_boxes   : list[Tensor(M_i, 4)]  OR padded tensor [B, M_max, 4]
        gt_labels  : list[Tensor(M_i)]     OR padded tensor [B, M_max]
        returns    : scalar loss
        """
        if torch.is_tensor(gt_boxes):
            gt_boxes  = [b[torch.any(b!=0, dim=-1)] for b in gt_boxes]
            gt_labels = [l[l!=-1]       for l in gt_labels]

        B, N, _ = pred_boxes.shape
        total_cls = total_l1 = total_giou = 0.0
        total_gt = 0

        for b in range(B):
            pb = pred_boxes[b]
            pl = pred_logits[b]
            gb = gt_boxes[b].to(pb)
            gl = gt_labels[b].long().to(pb.device)

            M = gb.size(0)
            if M == 0:
                total_cls += F.cross_entropy(pl, torch.zeros(N, dtype=torch.long, device=pb.device))
                continue

            cost_cls  = -pl[:, gl]
            cost_l1   = torch.cdist(pb, gb, p=1)
            giou = generalized_box_iou(
                     box_convert(pb, in_fmt='cxcywh', out_fmt='xyxy'),
                     box_convert(gb, in_fmt='cxcywh', out_fmt='xyxy'))
            cost_giou = 1 - giou
            C = (self.l_cls * cost_cls +
                 self.l_l1  * cost_l1  +
                 self.l_giou* cost_giou).cpu().detach().numpy()

            idx_pred, idx_gt = linear_sum_assignment(C)
            matched = [[] for _ in range(M)]
            for p,g in zip(idx_pred, idx_gt):
                matched[g].append(p)
            idx_pred_final, idx_gt_final = [], []
            for g, plist in enumerate(matched):
                for p in plist[: self.k_best]:
                    idx_pred_final.append(p)
                    idx_gt_final.append(g)
            if len(idx_pred_final) == 0:
                continue
            idx_pred_final = torch.tensor(idx_pred_final, device=pb.device)
            idx_gt_final   = torch.tensor(idx_gt_final,   device=pb.device)

            pb_m  = pb[idx_pred_final]
            gb_m  = gb[idx_gt_final]
            loss_l1   = F.l1_loss(pb_m, gb_m, reduction='sum')

            giou_m = generalized_box_iou(
                        box_convert(pb_m, in_fmt='cxcywh', out_fmt='xyxy'),
                        box_convert(gb_m, in_fmt='cxcywh', out_fmt='xyxy'))
            loss_giou = (1 - giou_m.diag()).sum()

            cls_tgt = torch.zeros_like(pl)
            cls_tgt[range(N), 0] = 1.0
            cls_tgt[idx_pred_final, 0] = 0.0
            cls_tgt[idx_pred_final, gl[idx_gt_final]] = 1.0
            loss_cls = F.cross_entropy(pl, cls_tgt.argmax(dim=-1), reduction='sum')

            total_cls  += loss_cls
            total_l1   += loss_l1
            total_giou += loss_giou
            total_gt   += M

        norm = max(total_gt, 1)
        total = (self.l_cls*total_cls + self.l_l1*total_l1 + self.l_giou*total_giou) / norm
        return total, {"cls": total_cls / norm, "l1":  total_l1  / norm, "giou":total_giou/ norm}

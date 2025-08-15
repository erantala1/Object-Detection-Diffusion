import torch
import torch.nn as nn
import numpy as np
import math
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch.nn.functional as F
from torchvision.ops import box_convert, generalized_box_iou
from scipy.optimize import linear_sum_assignment
from torchvision.ops import sigmoid_focal_loss
import torchvision.ops as ops

def logits_to_probs(logits, use_focal=True):
    # DiffusionDet uses sigmoid multi-label form (no background column)
    return torch.sigmoid(logits) if use_focal else torch.softmax(logits, dim=-1)

def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def clamp_xyxy(xyxy, H, W, eps=1e-6):
    x1, y1, x2, y2 = xyxy.unbind(-1)
    x1 = x1.clamp(0, W-1); y1 = y1.clamp(0, H-1)
    x2 = x2.clamp(0, W-1); y2 = y2.clamp(0, H-1)
    x1 = torch.minimum(x1, x2 - eps); y1 = torch.minimum(y1, y2 - eps)
    x2 = torch.maximum(x2, x1 + eps); y2 = torch.maximum(y2, y1 + eps)
    return torch.stack([x1,y1,x2,y2], dim=-1)


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
        self.act = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_reg  = nn.Linear(hidden_dim, 4)
        self.fc_cls  = nn.Linear(hidden_dim, num_classes)

        self.t_dim = 128
        self.t_embed_proj = nn.Sequential(
            nn.Linear(self.t_dim, hidden_dim),
            nn.ReLU(inplace=False)
        )
        self.apply(self._init_weights)
        self.to(device)
        self.signal_scale = 2.0

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
        boxes_norm = (boxes_t / self.signal_scale + 1.0) / 2.0  # in [0,1]
        _, _, H_pad, W_pad = next(iter(feats.values())).shape
        H_img, W_img = H_pad * 4, W_pad * 4
        boxes_xyxy = self.center_to_corner(boxes_norm)
        boxes_xyxy[..., [0, 2]] *= W_img
        boxes_xyxy[..., [1, 3]] *= H_img

        eps = 1e-6
        x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
        x1 = x1.clamp(0, W_img-1); y1 = y1.clamp(0, H_img-1)
        x2 = x2.clamp(0, W_img-1); y2 = y2.clamp(0, H_img-1)
        x1 = torch.minimum(x1, x2 - eps); y1 = torch.minimum(y1, y2 - eps)
        x2 = torch.maximum(x2, x1 + eps); y2 = torch.maximum(y2, y1 + eps)
        boxes_xyxy = torch.stack((x1,y1,x2,y2), dim=-1)

        boxes_list = [boxes_xyxy[i] for i in range(B)]
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

        cx, cy, w_, h_ = boxes_t.unbind(-1)
        dx, dy, dlogw, dlogh = delta.unbind(-1)

        cx_new = cx + dx
        cy_new = cy + dy
        w_new  = (w_.clamp_min(1e-6)) * torch.exp(dlogw.clamp(-4, 4))
        h_new  = (h_.clamp_min(1e-6)) * torch.exp(dlogh.clamp(-4, 4))
        boxes_cleaned = torch.stack([cx_new, cy_new, w_new, h_new], dim=-1)

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

class MatcherDynamicK(torch.nn.Module):
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0, use_focal=True, ota_k=10):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox  = cost_bbox
        self.cost_giou  = cost_giou
        self.use_focal  = use_focal
        self.ota_k      = ota_k

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs: {'pred_logits':[B,N,C], 'pred_boxes':[B,N,4] (xyxy px)}
        targets: list of dicts:
          {'labels':[Mi], 'boxes':[Mi,4] cxcywh norm, 'boxes_xyxy':[Mi,4] xyxy px,
           'image_size_xyxy':[4], 'image_size_xyxy_tgt':[4]}
        returns: indices: list of tuples (selected_query_mask [N], gt_indices [K])
        """
        B, N, C = outputs["pred_logits"].shape
        probs   = logits_to_probs(outputs["pred_logits"], use_focal=self.use_focal)
        pred_xyxy = outputs["pred_boxes"]

        indices = []
        for b in range(B):
            p_xyxy = pred_xyxy[b]
            p_prob = probs[b]
            tgt    = targets[b]
            gt_ids = tgt["labels"]
            M = gt_ids.numel()
            if M == 0:
                sel = torch.zeros(N, dtype=torch.bool, device=p_prob.device)
                indices.append((sel, torch.arange(0, device=p_prob.device)))
                continue

            gt_xyxy = tgt["boxes_xyxy"]

            if self.use_focal:
                alpha, gamma = 0.25, 2.0
                pos = p_prob[:, gt_ids]
                neg = 1.0 - pos
                cost_class = ( alpha * (neg**gamma) * (-(pos+1e-8).log())
                               - (1-alpha) * (pos**gamma) * (-(neg+1e-8).log()) )
            else:
                cost_class = -p_prob[:, gt_ids]

            img_wh = tgt["image_size_xyxy"]
            p_xyxy_n = p_xyxy / img_wh
            g_xyxy_n = gt_xyxy / tgt["image_size_xyxy_tgt"]
            cost_bbox = torch.cdist(p_xyxy_n, g_xyxy_n, p=1)

            giou = ops.generalized_box_iou(p_xyxy, gt_xyxy)
            cost_giou = 1.0 - giou

            cost = self.cost_class*cost_class + self.cost_bbox*cost_bbox + self.cost_giou*cost_giou

            with torch.no_grad():
                pair_ious = ops.box_iou(p_xyxy, gt_xyxy)
                topk = min(self.ota_k, max(1, N))
                topk_ious, _ = torch.topk(pair_ious, k=topk, dim=0)
                dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

            matching_matrix = torch.zeros_like(cost)
            for j in range(M):
                _, pos_idx = torch.topk(cost[:, j], k=dynamic_ks[j].item(), largest=False)
                matching_matrix[pos_idx, j] = 1.0

            anchor_match_gt = matching_matrix.sum(1)
            if (anchor_match_gt > 1).any():
                multi = anchor_match_gt > 1
                min_cost_idx = cost[multi].argmin(dim=1)
                matching_matrix[multi] *= 0
                matching_matrix[multi, min_cost_idx] = 1

            while (matching_matrix.sum(0) == 0).any():
                un = (matching_matrix.sum(0) == 0).nonzero(as_tuple=False).squeeze(1)
                for j in un:
                    p = torch.argmin(cost[:, j])
                    matching_matrix[:, j] *= 0
                    matching_matrix[p, j] = 1

            selected_query = matching_matrix.sum(1) > 0
            gt_indices     = matching_matrix[selected_query].argmax(dim=1)
            indices.append((selected_query, gt_indices))
        return indices, None


class SetCriterionDynamicKLite(torch.nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, use_focal=True):
        super().__init__()
        self.num_classes = num_classes
        self.matcher     = matcher
        self.weight_dict = weight_dict
        self.use_focal   = use_focal

    def forward(self, outputs, targets):
        """
        outputs: {'pred_logits':[B,N,C], 'pred_boxes':[B,N,4] xyxy px}
        targets: list of dicts (see above)
        returns: dict of losses
        """
        B, N, C = outputs["pred_logits"].shape
        indices, _ = self.matcher(outputs, targets)

        matched_pos = 0
        total_queries = 0

        logits = outputs["pred_logits"]
        probs  = logits_to_probs(logits, use_focal=self.use_focal)

        loss_ce_sum = torch.tensor(0., device=logits.device)
        num_pos = 0
        for b in range(B):
            sel, gt_idx = indices[b]
            matched_pos += sel.sum().item()
            total_queries += sel.numel()
            if gt_idx.numel() == 0:
                continue
            tgt_cls = targets[b]["labels"][gt_idx]
            pred_k  = logits[b][sel]

            onehot = torch.zeros_like(pred_k)
            onehot[torch.arange(tgt_cls.numel(), device=logits.device), tgt_cls] = 1.0
            loss_ce = sigmoid_focal_loss(pred_k, onehot, reduction="sum")
            loss_ce_sum += loss_ce
            num_pos += tgt_cls.numel()
        loss_ce = loss_ce_sum / max(num_pos, 1)

        loss_l1_sum = torch.tensor(0., device=logits.device)
        loss_gi_sum = torch.tensor(0., device=logits.device)
        for b in range(B):
            sel, gt_idx = indices[b]
            if gt_idx.numel() == 0:
                continue
            p_xyxy = outputs["pred_boxes"][b][sel]
            g_xyxy = targets[b]["boxes_xyxy"][gt_idx]

            wh_out = targets[b]["image_size_xyxy"] 
            wh_tgt = targets[b]["image_size_xyxy_tgt"]
            p_n = p_xyxy / wh_out
            g_n = g_xyxy / wh_tgt
            loss_l1_sum += F.l1_loss(p_n, g_n, reduction='sum')

            giou = ops.generalized_box_iou(p_xyxy, g_xyxy)
            loss_gi_sum += (1.0 - giou.diag()).sum()

        num_pos = max(num_pos, 1)
        matched_frac = float(matched_pos) / max(total_queries, 1)
        losses = {
            "loss_ce":   loss_ce,
            "loss_bbox": loss_l1_sum / num_pos,
            "loss_giou": loss_gi_sum / num_pos,
            "matched_frac": torch.as_tensor(matched_frac, device=logits.device),
        }
        return losses

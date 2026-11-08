"""Training-only text-conditioned set head and Hungarian loss."""

import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def box_area(boxes):
    return (boxes[..., 2:] - boxes[..., :2]).clamp(min=0).prod(-1)


def generalized_iou(a, b):
    inter = (
        (
            torch.minimum(a[:, None, 2:], b[None, :, 2:])
            - torch.maximum(a[:, None, :2], b[None, :, :2])
        )
        .clamp(min=0)
        .prod(-1)
    )
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    cover = (
        (
            torch.maximum(a[:, None, 2:], b[None, :, 2:])
            - torch.minimum(a[:, None, :2], b[None, :, :2])
        )
        .clamp(min=0)
        .prod(-1)
    )
    return inter / union.clamp(min=1e-08) - (cover - union) / cover.clamp(min=1e-08)


def set_loss(
    boxes,
    logits,
    targets,
    l1_weight=5.0,
    giou_weight=2.0,
    object_weight=1.0,
    no_object_weight=0.1,
):
    boxes, logits = (boxes.float(), logits.float())
    l1, giou, obj = ([], [], [])
    for pred, score, gt in zip(boxes, logits, targets):
        gt = gt.to(pred.device, torch.float32).reshape(-1, 4)
        if len(gt) > len(pred):
            raise ValueError(f"{len(gt)} targets exceed {len(pred)} queries")
        labels = torch.zeros_like(score)
        if len(gt):
            with torch.no_grad():
                cost = (
                    l1_weight * torch.cdist(pred, gt, p=1)
                    - giou_weight * generalized_iou(pred, gt)
                    - object_weight * score.sigmoid()[:, None]
                )
                pi, ti = linear_sum_assignment(cost.cpu().numpy())
                pi = torch.as_tensor(pi, device=pred.device)
                ti = torch.as_tensor(ti, device=pred.device)
            labels[pi] = 1
            l1.append((pred[pi] - gt[ti]).abs().sum(-1).mean())
            giou.append((1 - generalized_iou(pred[pi], gt[ti]).diagonal()).mean())
        else:
            l1.append(pred.sum() * 0)
            giou.append(pred.sum() * 0)
        weights = torch.where(labels.bool(), 1.0, no_object_weight)
        obj.append(
            (
                F.binary_cross_entropy_with_logits(score, labels, reduction="none")
                * weights
            ).sum()
            / weights.sum()
        )
    l1, giou, obj = (torch.stack(v).mean() for v in (l1, giou, obj))
    total = l1_weight * l1 + giou_weight * giou + object_weight * obj
    counts = torch.tensor([len(t) for t in targets], device=logits.device)
    return (
        total,
        {
            "box_l1": l1.detach(),
            "box_giou": giou.detach(),
            "object_bce": obj.detach(),
            "count_mae_at_0.5": ((logits.sigmoid() >= 0.5).sum(-1) - counts)
            .abs()
            .float()
            .mean()
            .detach(),
        },
    )


class DetectionHead(nn.Module):

    def __init__(self, hidden_size, num_queries=128, head_dim=256, head_layers=2):
        super().__init__()
        self.memory = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, head_dim)
        )
        self.queries = nn.Embedding(num_queries, head_dim)
        layer = nn.TransformerDecoderLayer(
            head_dim,
            8,
            head_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, head_layers, nn.LayerNorm(head_dim))
        self.box = nn.Sequential(
            nn.Linear(head_dim, head_dim), nn.GELU(), nn.Linear(head_dim, 4)
        )
        self.objectness = nn.Linear(head_dim, 1)
        nn.init.constant_(self.objectness.bias, -2.0)

    def forward(self, hidden, attention_mask):
        memory = self.memory(hidden)
        last = attention_mask.long().sum(-1) - 1
        condition = memory[torch.arange(len(memory), device=memory.device), last]
        query = (
            self.queries.weight[None].expand(len(memory), -1, -1) + condition[:, None]
        )
        out = self.decoder(
            query, memory, memory_key_padding_mask=~attention_mask.bool()
        )
        raw = self.box(out).sigmoid()
        center = raw[..., :2]
        half = raw[..., 2:] * torch.minimum(center, 1 - center)
        return (
            torch.cat((center - half, center + half), -1),
            self.objectness(out).squeeze(-1),
        )

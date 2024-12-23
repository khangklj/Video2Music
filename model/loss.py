import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from utilities.device import get_device, use_cuda

# Borrowed from https://github.com/jason9693/MusicTransformer-pytorch/blob/5f183374833ff6b7e17f3a24e3594dedd93a5fe5/custom/criterion.py#L28
class SmoothCrossEntropyLoss(_Loss):
    """
    https://arxiv.org/abs/1512.00567
    """
    __constants__ = ['label_smoothing', 'vocab_size', 'ignore_index', 'reduction']

    def __init__(self, label_smoothing, vocab_size, ignore_index=-100, reduction='mean', is_logits=True):
        assert 0.0 <= label_smoothing <= 1.0
        super().__init__(reduction=reduction)

        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.input_is_logits = is_logits

    def forward(self, input, target):
        """
        Args:
            input: [B * T, V]
            target: [B * T]
        Returns:
            cross entropy: [1]
        """
        mask = (target == self.ignore_index).unsqueeze(-1)
        q = F.one_hot(target.long(), self.vocab_size).type(torch.float32)
        u = 1.0 / self.vocab_size
        q_prime = (1.0 - self.label_smoothing) * q + self.label_smoothing * u
        q_prime = q_prime.masked_fill(mask, 0)

        ce = self.cross_entropy_with_logits(q_prime, input)
        if self.reduction == 'mean':
            lengths = torch.sum(target != self.ignore_index)
            return ce.sum() / lengths
        elif self.reduction == 'sum':
            return ce.sum()
        else:
            raise NotImplementedError

    def cross_entropy_with_logits(self, p, q):
        return -torch.sum(p * (q - q.logsumexp(dim=-1, keepdim=True)), dim=-1)

class FocalLoss(_Loss):
    def __init__(self, weight=0.1, alpha=0.25, gamma=2.0, vocab_size=100, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        mask = (target == self.ignore_index).unsqueeze(-1).to(get_device())
        target_one_hot = F.one_hot(target, num_classes=self.vocab_size).float().to(get_device())
        target_one_hot = target_one_hot.masked_fill(mask, 0)

        if target_one_hot.shape != input.shape:
            input = input.reshape(target_one_hot.shape)

        log_prob = F.log_softmax(input, dim=-1)
        prob = log_prob.exp()

        focal_factor = (1 - prob) ** self.gamma

        loss = -focal_factor * log_prob * target_one_hot
        loss = loss.sum(dim=-1)
        # print(loss.shape, '\n', loss)

        if self.reduction == 'mean':
            length = torch.sum(target != self.ignore_index)
            return loss.sum() / length * self.weight
        elif self.reduction == 'sum':
            return loss.sum() * self.weight
        else:
            raise NotImplementedError

class TopKAuxiliaryLoss(_Loss):
    def __init__(self, k=3, weight=0.1, vocab_size=100, ignore_index=-100, reduction='mean'):
        super().__init__(reduction=reduction)
        self.k = k
        self.weight = weight
        self.ignore_index = ignore_index
        self.vocab_size = vocab_size

    def forward(self, input, target):
        mask = (target == self.ignore_index).unsqueeze(-1).to(get_device())
        target_one_hot = F.one_hot(target.long(), self.vocab_size).type(torch.float32).to(get_device())
        target_one_hot = target_one_hot.masked_fill(mask, 0)

        if target_one_hot.shape != input.shape:
            input = input.reshape(target_one_hot.shape)
        pred = F.softmax(input, dim=-1)

        loss = self.loss_with_logits(target_one_hot, pred, self.k)
        loss = loss.masked_fill(mask.squeeze(), 0)
        if self.reduction == 'mean':
            length = torch.sum(target != self.ignore_index)
            return loss.sum() / length * self.weight
        elif self.reduction == 'sum':
            return loss.sum() * self.weight
        else:
            raise NotImplementedError
        
    def loss_with_logits(self, truth, pred, k):
        topk_scores, topk_indices = torch.topk(pred, k=k, dim=-1)
        true_scores = torch.sum(pred * truth, dim=-1)

        if topk_scores.ndim == 2:
            topk_scores = topk_scores.unsqueeze(0)
        # lowest_topk_scores = topk_scores[:, :, -1].float()
        mean_topk_scores = topk_scores.sum(dim=-1) / k
        return F.relu(mean_topk_scores - true_scores)

class CombinedLoss(_Loss):
    def __init__(self, lossFunctionList=[], type='sum'):
        super().__init__()
        self.lossFunctionList = nn.ModuleList(lossFunctionList)
        self.type_ = type

    def forward(self, input, target):
        loss = torch.tensor(0.0).to(get_device())

        count = 0
        for lossFunction in self.lossFunctionList:
            l = lossFunction(input, target)
            if l > 1e-10:
                count += 1
            loss += l

        if self.type_ == 'sum':
            return loss
      
        return loss / count

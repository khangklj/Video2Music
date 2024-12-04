import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

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

class TopKAuxiliaryLoss(_Loss):
    def __init__(self, k=3, weight=0.1, vocab_size=100, ignore_index=-100, reduction='mean'):
        super().__init__(reduction=reduction)
        self.k = k
        self.weight = weight
        self.ignore_index = ignore_index
        self.vocab_size = vocab_size

    def forward(self, input, target):
        mask = (target == self.ignore_index).unsqueeze(-1)
        q = F.one_hot(target.long(), self.vocab_size).type(torch.float32)
        q = q.masked_fill(mask, 0)

        loss = self.loss_with_logits(q, input, self.k)
        if self.reduction == 'mean':
            lengths = torch.sum(target != self.ignore_index)
            return loss.sum() / lengths * self.weight
        elif self.reduction == 'sum':
            return loss.sum() * self.weight
        else:
            raise NotImplementedError
        
    def loss_with_logits(self, truth, pred, k):
        print(truth.shape, pred.shape)
        topk_scores, topk_indices = torch.topk(pred, k=k, dim=1)

        true_scores = pred.gather(1, truth.unsqueeze(1)).squeeze(1)

        lowest_top3_scores = topk_scores[:, -1]

        return F.relu(lowest_top3_scores - true_scores)

class CombinedLoss(_Loss):
    def __init__(self, lossFunctionList=[]):
        super().__init__()
        self.lossFunctionList = nn.ModuleList(lossFunctionList)

    def forward(self, input, target):
        loss = torch.Tensor(0)
        for lossFunction in self.lossFunctionList:
            loss += lossFunction(input, target)

        return loss
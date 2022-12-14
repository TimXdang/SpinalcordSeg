# own defined metrics

import torch
from torch import nn


class JaccardIndex(nn.Module):
    __name__ = "iou_score"

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def _binarize_predictions(self, y, n_classes):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.

        :param y: predictions
        :param n_classes: number of channels / classes
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = y > 0.5
            return result.long()

        _, max_index = torch.max(y, dim=0, keepdim=True)
        return torch.zeros_like(y, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, pred, y):
        """
        Computes IoU for a given target and prediction tensors.

        :param pred: prediction
        :param y: target
        """
        return torch.sum(pred & y).float() / torch.clamp(torch.sum(pred | y).float(), min=1e-8)

    def forward(self, pred, y):
        n_classes = pred.size()[1]
        assert pred.size() == y.size()

        per_batch_iou = []
        for _input, _target in zip(pred, y):
            binary_prediction = self._binarize_predictions(_input, n_classes)

            # convert to uint8
            binary_prediction = binary_prediction.byte()
            _target = _target.byte()

            per_channel_iou = []
            for c in range(n_classes):
                per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

            assert per_channel_iou, "All channels were ignored from the computation"
            mean_iou = torch.mean(torch.tensor(per_channel_iou))
            per_batch_iou.append(mean_iou)

        return torch.mean(torch.tensor(per_batch_iou))

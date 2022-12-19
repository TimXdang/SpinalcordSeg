# own defined loss functions
import torch
from torch import nn


def flatten(tensor: torch.Tensor) -> torch.Tensor:
    """
    Flattens tensor such that channel axis is first.
    Shapes are transformed as follows: (N, C, D, H, W) -> (C, N * D * H * W)

    :param tensor: tensor to be flattened
    :return: flattened tensor
    """
    # number of channels
    C = tensor.size(1)
    # new order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class VolDiceLoss(nn.Module):
    """
    Unnormalized probabilities (logits) as outputs assumed which are to be normalized. Dice (or soft Dice in this case)
    is usually used for binary data. Thus, normalizing the channels with Sigmoid is the default choice.
    To get the proper probability distribution from the output set normalization=softmax.

    :ivar normalization: sets whether binary data or probability distribution is to be obtained
    """
    __name__ = 'dice_loss'

    def __init__(self, normalization: str = 'sigmoid'):
        super(VolDiceLoss, self).__init__()
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, y: torch.Tensor, pred: torch.Tensor, epsilon: float = 1e-6) -> float:
        """
        Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 .
        Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

        :param y: NxCxSpatial input tensor
        :param pred: NxCxSpatial target tensor
        :param epsilon: float that prevents division by zero
        :return: dice coefficient
        """

        # input and target shapes must match
        assert y.size() == pred.size(), "Input and target must have the same shape!"

        y = flatten(y)
        pred = flatten(pred)
        y = y.float()

        intersect = (y * pred).sum(-1)
        denominator = (y * y).sum(-1) + (pred * pred).sum(-1)
        return 2 * (intersect / denominator.clamp(min=epsilon))

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> float:
        """
        Forward pass.

        :param pred: predictions
        :param y: targets
        :return: dice loss
        """
        # get probabilities from logits
        pred = self.normalization(pred)

        # compute Dice coefficient
        dice = self.dice(y, pred)

        # average Dice score
        return 1. - torch.mean(dice)


class CrossEntropyDice(nn.Module):
    """
    Linear combination of Binary Cross Entropy and Dice Loss as proposed in https://arxiv.org/pdf/1809.10486.pdf

    :ivar alpha: factor for BCE loss
    :ivar beta: factor for DICE loss
    :ivar bce: BCE loss
    :ivar dice: DICE loss
    """
    __name__ = 'ce_dice_loss'

    def __init__(self, alpha: float, beta: float):
        super(CrossEntropyDice, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = VolDiceLoss()

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> float:
        return self.alpha * self.bce(pred, y.float()) + self.beta * self.dice(pred, y)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss()

    def forward(self, pred, y):
        # flatten label and prediction tensors
        pred = pred.view(-1)
        y = y.view(-1)

        bce = self.bce(pred, y.float())
        bce_exp = torch.exp(-bce)
        focal_loss = self.alpha * (1-bce_exp)**self.gamma * bce

        return focal_loss


class FocalDiceLoss(nn.Module):
    """
    https://www.sciencedirect.com/science/article/pii/S1361841521003042
    """
    __name__ = 'focal_dice'
    def __init__(self, beta=3):
        super().__init__()
        self.beta = beta
        self.dice = VolDiceLoss()

    def forward(self, pred, target):
        focal_dice = 0.05 * self.dice(pred, target) ** (1 / self.beta) + 0.95 * (1 - self.dice(pred, target)) ** \
                        (1 / self.beta)
        return focal_dice

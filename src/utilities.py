import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import math

def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]

def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)

    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]

def to_attributes(var_in):
    """
    convert dictionary to object with keys as attributes
    should implement with a class inheriting dict but would cause leak in 2.7
    ignores non dict objects
    """

    class Container:
        def __str__(self):
            return str(self.__dict__)

        def update(self, up_dict):
            self.__dict__.update(up_dict.__dict__)

    if type(var_in) == dict:
        container = Container()
        container.__dict__ = {k: to_attributes(v) for k, v in var_in.items()}
        return container
    else:
        return var_in


class WNConv2d(nn.Module):
    """Weight-normalized 2d convolution.
    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels in the output.
        kernel_size (int): Side length of each convolutional kernel.
        padding (int): Padding to add on edges of input.
        bias (bool): Use bias in the convolution operation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super(WNConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias))

    def forward(self, x):
        x = self.conv(x)

        return x


def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(
        kernel_size
    ), "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)

    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def compute_loss(nll, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses


def compute_loss_y(nll, y_logits, y_weight, y, multi_class, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    if multi_class:
        y_logits = torch.sigmoid(y_logits)
        loss_classes = F.binary_cross_entropy_with_logits(
            y_logits, y, reduction=reduction
        )
    else:
        loss_classes = F.cross_entropy(
            y_logits, torch.argmax(y, dim=1), reduction=reduction
        )

    losses["loss_classes"] = loss_classes
    losses["total_loss"] = losses["nll"] + y_weight * loss_classes

    return losses


def loss(self, x, y, level):
    if self.y_condition:
        z, nll, y_logits = self.model(x, y, level) if level is not None else self.model(x, y)
        losses = compute_loss_y(nll, self.y_logits, self.y_weight, y, False)
    else:
        z, nll, y_logits = self.model(x, level) if level is not None else self.model(x)
        losses = compute_loss(nll)
    return losses


def edge_bias(x, k_size):
    """
    injects a mask into a tensor for 2d convs to indicate padding locations
    """

    pad_size = k_size // 2

    # manually pad data for conv
    x_padded = F.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])

    # generate mask to indicated padded pixels
    # x_mask_inner = tf.zeros(shape=tf.shape(x))
    x_mask_inner = torch.zeros(x)
    x_mask_inner = x_mask_inner[:, 0:1, :, :]
    x_mask = F.pad(x_mask_inner, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], value=1.0)

    # combine into 1 tensor
    x_augmented = torch.cat([x_padded, x_mask], axis=1)

    return x_augmented


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """

    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x
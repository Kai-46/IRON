import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage

import warnings
import kornia

from icecream import ic


class PyramidL2Loss(nn.Module):
    def __init__(self, use_cuda=True):
        super().__init__()

        dirac = np.zeros((7, 7), dtype=np.float32)
        dirac[3, 3] = 1.0
        f = np.zeros([3, 3, 7, 7], dtype=np.float32)
        gf = scipy.ndimage.filters.gaussian_filter(dirac, 1.0)
        f[0, 0, :, :] = gf
        f[1, 1, :, :] = gf
        f[2, 2, :, :] = gf
        self.f = torch.from_numpy(f)
        if use_cuda:
            self.f = self.f.cuda()
        self.m = torch.nn.AvgPool2d(2)

    def forward(self, pred_img, trgt_img):
        """
        pred_img, trgt_img: [B, C, H, W]
        """
        diff_0 = pred_img - trgt_img

        h, w = pred_img.shape[-2:]
        # Convolve then downsample
        diff_1 = self.m(torch.nn.functional.conv2d(diff_0, self.f, padding=3))
        diff_2 = self.m(torch.nn.functional.conv2d(diff_1, self.f, padding=3))
        diff_3 = self.m(torch.nn.functional.conv2d(diff_2, self.f, padding=3))
        diff_4 = self.m(torch.nn.functional.conv2d(diff_3, self.f, padding=3))
        loss = (
            diff_0.pow(2).sum() / (h * w)
            + diff_1.pow(2).sum() / ((h / 2.0) * (w / 2.0))
            + diff_2.pow(2).sum() / ((h / 4.0) * (w / 4.0))
            + diff_3.pow(2).sum() / ((h / 8.0) * (w / 8.0))
            + diff_4.pow(2).sum() / ((h / 16.0) * (w / 16.0))
        )
        return loss


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r"""Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def ssim_loss_fn(X, Y, mask=None, data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03)):
    r"""Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images of shape [b, c, h, w]
        Y (torch.Tensor): images of shape [b, c, h, w]
        mask (torch.Tensor): [b, 1, h, w]
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
    Returns:
        torch.Tensor: per pixel ssim results (same size as input images X, Y)
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if len(X.shape) != 4:
        raise ValueError(f"Input images should be 4-d tensors, but got {X.shape}")

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    win = _fspecial_gauss_1d(win_size, win_sigma)
    win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    ssim_map = ssim_map.mean(dim=1, keepdim=True)

    if mask is not None:
        ### pad ssim_map to original size
        ssim_map = F.pad(
            ssim_map, (win_size // 2, win_size // 2, win_size // 2, win_size // 2), mode="constant", value=1.0
        )

        mask = kornia.morphology.erosion(mask.float(), torch.ones(win_size, win_size).float().to(mask.device)) > 0.5
        # ic(ssim_map.shape, mask.shape)
        ssim_map = ssim_map[mask]

    return 1.0 - ssim_map.mean()


if __name__ == "__main__":
    pred_im = torch.rand(1, 3, 256, 256).cuda()
    # gt_im = torch.rand(1, 3, 256, 256).cuda()
    gt_im = pred_im.clone()
    mask = torch.ones(1, 1, 256, 256).bool().cuda()

    ssim_loss = ssim_loss_fn(pred_im, gt_im, mask)
    ic(ssim_loss)

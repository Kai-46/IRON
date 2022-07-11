import numpy as np
import imageio
import os
from skimage.metrics import structural_similarity
import lpips
import torch
import glob


def skimage_ssim(pred_im, trgt_im):
    ssim = 0.
    for ch in range(3):
        ssim += structural_similarity(trgt_im[:, :, ch], pred_im[:, :, ch], 
                                      data_range=1.0, win_size=11, sigma=1.5,
                                      use_sample_covariance=False, k1=0.01, k2=0.03)
    ssim /= 3.
    return ssim

def read_image(fpath):
    return imageio.imread(fpath).astype(np.float32) / 255.

mse2psnr = lambda x: -10. * np.log(x+1e-10) / np.log(10.)

import sys
folder = sys.argv[1]

all_psnr = []
all_ssim = []
all_lpips = []

loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

with open(os.path.join(folder, '../metrics.txt'), 'w') as fp:
    fp.write('img_name\tpsnr\tssim\tlpips\n')
    for _, fpath in enumerate(glob.glob(os.path.join(folder, '*_truth.png'))):
        name = os.path.basename(fpath)
        idx = name.find('_')
        idx = int(name[:idx])

        pred_im = read_image(os.path.join(folder, '{}_prediction.png'.format(idx)))
        trgt_im = read_image(os.path.join(folder, '{}_truth.png'.format(idx)))

        psnr = mse2psnr(np.mean((pred_im - trgt_im) ** 2))

        ssim = skimage_ssim(trgt_im, pred_im)

        pred_im = torch.from_numpy(pred_im).permute(2, 0, 1).unsqueeze(0) * 2. - 1.
        trgt_im = torch.from_numpy(trgt_im).permute(2, 0, 1).unsqueeze(0) * 2. - 1.
        d = loss_fn_alex(trgt_im.cuda(), pred_im.cuda()).item()

        fp.write('{}_prediction.png\t{:.3f}\t{:.3f}\t{:.4f}\n'.format(idx, psnr, ssim, d))

        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_lpips.append(d)
    fp.write('\nAverage\t{:.3f}\t{:.3f}\t{:.4f}\n'.format(np.mean(all_psnr), np.mean(all_ssim), np.mean(all_lpips)))


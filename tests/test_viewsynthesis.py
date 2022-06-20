import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import trimesh
import json
import imageio
from torch.utils.tensorboard import SummaryWriter
import configargparse
from icecream import ic
import glob

import sys

sys.path.append("../")

from models.fields import SDFNetwork, RenderingNetwork
from models.raytracer import RayTracer, Camera, render_camera
from models.renderer_ggx import GGXColocatedRenderer
from models.image_losses import PyramidL2Loss, ssim_loss_fn


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None, help="input data directory")
    parser.add_argument("--out_dir", type=str, default=None, help="output directory")
    # parser.add_argument("--neus_ckpt_fpath", type=str, default=None, help="checkpoint to load")
    parser.add_argument("--num_iters", type=int, default=100001, help="number of iterations")
    # parser.add_argument("--white_specular_albedo", action='store_true', help='force specular albedo to be white')
    parser.add_argument("--eik_weight", type=float, default=0.1, help="weight for eikonal loss")
    parser.add_argument("--ssim_weight", type=float, default=1.0, help="weight for ssim loss")
    parser.add_argument("--roughrange_weight", type=float, default=0.1, help="weight for roughness range loss")

    parser.add_argument("--plot_image_name", type=str, default=None, help="image to plot during training")
    parser.add_argument("--no_edgesample", action="store_true", help="whether to disable edge sampling")

    return parser


parser = config_parser()
args = parser.parse_args()
ic(args)


def to8b(x):
    return np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)


ggx_renderer = GGXColocatedRenderer(use_cuda=True)
pyramidl2_loss_fn = PyramidL2Loss(use_cuda=True)


def render_fn(interior_mask, color_network_dict, ray_o, ray_d, points, normals, features):
    dots_sh = list(interior_mask.shape)
    color = torch.zeros(
        dots_sh
        + [
            3,
        ],
        dtype=torch.float32,
        device=interior_mask.device,
    )
    normals_pad = color.clone()
    if interior_mask.any():
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-10)
        interior_color = color_network_dict["color_network"](points, normals, ray_d, features)

        color[interior_mask] = interior_color
        normals_pad[interior_mask] = normals

    return {
        "color": color,
        "normal": normals_pad,
    }


sdf_network = SDFNetwork(
    d_in=3,
    d_out=257,
    d_hidden=256,
    n_layers=8,
    skip_in=[
        4,
    ],
    multires=6,
    bias=0.5,
    scale=1.0,
    geometric_init=True,
    weight_norm=True,
).cuda()
raytracer = RayTracer()


color_network_dict = {
    "color_network": RenderingNetwork(
        d_in=9,
        d_out=3,
        d_feature=256,
        d_hidden=256,
        n_layers=8,
        multires=10,
        multires_view=4,
        mode="idr",
        squeeze_out=True,
        skip_in=(4,),
    ).cuda()
}

sdf_optimizer = torch.optim.Adam(sdf_network.parameters(), lr=1e-5)
color_optimizer_dict = {"color_network": torch.optim.Adam(color_network_dict["color_network"].parameters(), lr=1e-4)}


def load_datadir(datadir):
    cam_dict = json.load(open(os.path.join(datadir, "cam_dict_norm.json")))
    imgnames = list(cam_dict.keys())
    try:
        imgnames = sorted(imgnames, key=lambda x: int(x[:-4]))
    except:
        imgnames = sorted(imgnames)

    image_fpaths = []
    gt_images = []
    Ks = []
    W2Cs = []
    for x in imgnames:
        fpath = os.path.join(datadir, "image", x)
        assert fpath[-4:] in [".jpg", ".png"], "must use ldr images as inputs"
        im = imageio.imread(fpath).astype(np.float32) / 255.0
        K = np.array(cam_dict[x]["K"]).reshape((4, 4)).astype(np.float32)
        W2C = np.array(cam_dict[x]["W2C"]).reshape((4, 4)).astype(np.float32)

        image_fpaths.append(fpath)
        gt_images.append(torch.from_numpy(im))
        Ks.append(torch.from_numpy(K))
        W2Cs.append(torch.from_numpy(W2C))
    gt_images = torch.stack(gt_images, dim=0)
    Ks = torch.stack(Ks, dim=0)
    W2Cs = torch.stack(W2Cs, dim=0)
    return image_fpaths, gt_images, Ks, W2Cs


image_fpaths, gt_images, Ks, W2Cs = load_datadir(args.data_dir)
cameras = [
    Camera(W=gt_images[i].shape[1], H=gt_images[i].shape[0], K=Ks[i].cuda(), W2C=W2Cs[i].cuda())
    for i in range(gt_images.shape[0])
]
ic(len(image_fpaths), gt_images.shape, Ks.shape, W2Cs.shape, len(cameras))

#### load pretrained checkpoints
start_step = -1
ckpt_fpaths = glob.glob(os.path.join(args.out_dir, "ckpt_*.pth"))
if len(ckpt_fpaths) > 0:
    path2step = lambda x: int(os.path.basename(x)[len("ckpt_") : -4])
    ckpt_fpaths = sorted(ckpt_fpaths, key=path2step)
    ckpt_fpath = ckpt_fpaths[-1]
    start_step = path2step(ckpt_fpath)
    ic("Reloading from checkpoint: ", ckpt_fpath)
    ckpt = torch.load(ckpt_fpath, map_location=torch.device("cuda"))
    sdf_network.load_state_dict(ckpt["sdf_network"])
    for x in list(color_network_dict.keys()):
        color_network_dict[x].load_state_dict(ckpt[x])
    # logim_names = [os.path.basename(x) for x in glob.glob(os.path.join(args.out_dir, "logim_*.png"))]
    # start_step = sorted([int(x[len("logim_") : -4]) for x in logim_names])[-1]

ic(start_step)

fill_holes = False
handle_edges = not args.no_edgesample
is_training = True
inv_gamma_gt = False
if inv_gamma_gt:
    ic("linearizing ground-truth images using inverse gamma correction")
    gt_images = torch.pow(gt_images, 2.2)

ic(fill_holes, handle_edges, is_training, inv_gamma_gt)
os.makedirs(args.out_dir, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs"))


for global_step in tqdm.tqdm(range(start_step + 1, args.num_iters)):
    sdf_optimizer.zero_grad()
    for x in color_optimizer_dict.keys():
        color_optimizer_dict[x].zero_grad()

    idx = np.random.randint(0, gt_images.shape[0])
    camera_crop, gt_color_crop = cameras[idx].crop_region(trgt_W=128, trgt_H=128, image=gt_images[idx])

    results = render_camera(
        camera_crop,
        sdf_network,
        raytracer,
        color_network_dict,
        render_fn,
        fill_holes=fill_holes,
        handle_edges=handle_edges,
        is_training=is_training,
    )

    mask = results["convergent_mask"]
    if handle_edges:
        mask = mask | results["edge_mask"]

    img_loss = torch.Tensor([0.0]).cuda()
    img_l2_loss = torch.Tensor([0.0]).cuda()
    img_ssim_loss = torch.Tensor([0.0]).cuda()

    eik_points = torch.empty(camera_crop.H * camera_crop.W // 2, 3).cuda().float().uniform_(-1.0, 1.0)
    eik_grad = sdf_network.gradient(eik_points).view(-1, 3)
    eik_cnt = eik_grad.shape[0]
    eik_loss = ((eik_grad.norm(dim=-1) - 1) ** 2).sum()
    if mask.any():
        pred_img = results["color"].permute(2, 0, 1).unsqueeze(0)
        gt_img = gt_color_crop.permute(2, 0, 1).unsqueeze(0).to(pred_img.device)
        img_l2_loss = pyramidl2_loss_fn(pred_img, gt_img)
        img_ssim_loss = args.ssim_weight * ssim_loss_fn(pred_img, gt_img, mask.unsqueeze(0).unsqueeze(0))
        img_loss = img_l2_loss + img_ssim_loss

        eik_grad = results["normal"][mask]
        eik_cnt += eik_grad.shape[0]
        eik_loss = eik_loss + ((eik_grad.norm(dim=-1) - 1) ** 2).sum()
        if "edge_pos_neg_normal" in results:
            eik_grad = results["edge_pos_neg_normal"]
            eik_cnt += eik_grad.shape[0]
            eik_loss = eik_loss + ((eik_grad.norm(dim=-1) - 1) ** 2).sum()

    eik_loss = eik_loss / eik_cnt * args.eik_weight

    loss = img_loss + eik_loss
    loss.backward()
    sdf_optimizer.step()
    for x in color_optimizer_dict.keys():
        color_optimizer_dict[x].step()

    if global_step % 50 == 0:
        writer.add_scalar("loss/loss", loss, global_step)
        writer.add_scalar("loss/img_loss", img_loss, global_step)
        writer.add_scalar("loss/img_l2_loss", img_l2_loss, global_step)
        writer.add_scalar("loss/img_ssim_loss", img_ssim_loss, global_step)
        writer.add_scalar("loss/eik_loss", eik_loss, global_step)

    if global_step % 1000 == 0:
        torch.save(
            dict(
                [
                    ("sdf_network", sdf_network.state_dict()),
                ]
                + [(x, color_network_dict[x].state_dict()) for x in color_network_dict.keys()]
            ),
            os.path.join(args.out_dir, f"ckpt_{global_step}.pth"),
        )

    if global_step % 500 == 0:
        ic(
            args.out_dir,
            global_step,
            loss.item(),
            img_loss.item(),
            img_l2_loss.item(),
            img_ssim_loss.item(),
            eik_loss.item(),
        )

        for x in list(results.keys()):
            del results[x]

        idx = 0
        if args.plot_image_name is not None:
            while idx < len(image_fpaths):
                if args.plot_image_name in image_fpaths[idx]:
                    break
                idx += 1

        camera_resize, gt_color_resize = cameras[idx].resize(factor=0.25, image=gt_images[idx])
        results = render_camera(
            camera_resize,
            sdf_network,
            raytracer,
            color_network_dict,
            render_fn,
            fill_holes=fill_holes,
            handle_edges=handle_edges,
            is_training=False,
        )
        for x in list(results.keys()):
            results[x] = results[x].detach().cpu().numpy()

        gt_color_im = gt_color_resize.detach().cpu().numpy()
        color_im = results["color"]
        normal = results["normal"]
        normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-10)
        normal_im = (normal + 1.0) / 2.0
        edge_mask_im = np.tile(results["edge_mask"][:, :, np.newaxis], (1, 1, 3))
        if inv_gamma_gt:
            gt_color_im = np.power(gt_color_im + 1e-6, 1.0 / 2.2)
            color_im = np.power(color_im + 1e-6, 1.0 / 2.2)

        im = np.concatenate([gt_color_im, color_im, normal_im, edge_mask_im], axis=1)
        imageio.imwrite(os.path.join(args.out_dir, f"logim_{global_step}.png"), to8b(im))

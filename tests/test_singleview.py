import os
import time
from unittest import result
import numpy as np
import torch
import trimesh
import json
import imageio

imageio.plugins.freeimage.download()

from icecream import ic
import sys

sys.path.append("../")

from models.fields import SDFNetwork
import models.raytracer

models.raytracer.VERBOSE_MODE = False
from models.raytracer import RayTracer, Camera, render_camera


def to8b(x):
    return np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)


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

color_network_dict = {}


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
    normals_pad = torch.zeros(
        dots_sh
        + [
            3,
        ],
        dtype=torch.float32,
        device=interior_mask.device,
    )
    if interior_mask.any():
        interior_color = (
            torch.ones_like(points.view(-1, 3))
            * torch.Tensor([[237.0 / 255.0, 61.0 / 255.0, 100.0 / 255.0]]).float().cuda()
        )
        interior_color = interior_color.view(list(points.shape))
        color[interior_mask] = interior_color
        normals_pad[interior_mask] = normals

    return {"color": color, "normal": normals_pad}


gt_color = imageio.imread("./data_singleview/12.png").astype(np.float32) / 255.0
gt_color = torch.from_numpy(gt_color).cuda()

cam_dict = json.load(open("./data_singleview/cam_dict_norm.json"))
K = torch.from_numpy(np.array(cam_dict["12.png"]["K"]).reshape((4, 4)).astype(np.float32)).cuda()
W2C = torch.from_numpy(np.array(cam_dict["12.png"]["W2C"]).reshape((4, 4)).astype(np.float32)).cuda()
W, H = cam_dict["12.png"]["img_size"]

camera = Camera(W=W, H=H, K=K, W2C=W2C)

fill_holes = False
handle_edges = True
is_training = True
out_dir = f"./debug_singleview_{fill_holes}_{handle_edges}_{is_training}"
ic(out_dir)
os.makedirs(out_dir, exist_ok=True)

sdf_optimizer = torch.optim.Adam(sdf_network.parameters(), lr=1e-4)

for global_step in range(15000):
    sdf_optimizer.zero_grad()

    camera_crop, gt_color_crop = camera.crop_region(trgt_W=128, trgt_H=128, image=gt_color)

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
        # mask = mask | results["edge_mask"]
        mask = results["edge_mask"]

    img_loss = torch.Tensor(
        [
            0.0,
        ]
    ).cuda()
    rand_eik_points = torch.empty(camera_crop.H * camera_crop.W // 2, 3).cuda().float().uniform_(-1.0, 1.0)
    eik_grad = sdf_network.gradient(rand_eik_points).view(-1, 3)

    if mask.any():
        img_loss = ((results["color"][mask] - gt_color_crop[mask]) ** 2).mean()
        interior_normals = results["normal"][mask | results["convergent_mask"]]
        eik_grad = torch.cat([eik_grad, interior_normals], dim=0)
        if "edge_pos_neg_normal" in results:
            eik_grad = torch.cat([eik_grad, results["edge_pos_neg_normal"]], dim=0)
    eik_loss = ((eik_grad.norm(dim=-1) - 1) ** 2).mean()

    loss = img_loss + 0.1 * eik_loss
    loss.backward()
    sdf_optimizer.step()

    if global_step % 200 == 0:
        ic(global_step, loss.item(), img_loss.item(), eik_loss.item())
        for x in list(results.keys()):
            del results[x]

        camera_resize, gt_color_resize = camera.resize(factor=0.25, image=gt_color)
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
        im = np.concatenate([gt_color_im, color_im, normal_im, edge_mask_im], axis=1)
        imageio.imwrite(os.path.join(out_dir, f"logim_{global_step}.png"), to8b(im))

        torch.save(sdf_network.state_dict(), os.path.join(out_dir, "ckpt.pth"))

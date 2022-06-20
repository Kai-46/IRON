import os
import time
import numpy as np
import torch
import trimesh
import imageio

imageio.plugins.freeimage.download()

from icecream import ic
import sys

sys.path.append("../")

from models.fields import SDFNetwork, RenderingNetwork
import models.raytracer

models.raytracer.VERBOSE_MODE = True
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
color_network = RenderingNetwork(
    d_in=9,
    d_out=3,
    d_feature=256,
    d_hidden=256,
    n_layers=4,
    multires_view=4,
    mode="idr",
    squeeze_out=True,
).cuda()
raytracer = RayTracer()

scene = "dtu_scan69"
ckpt_fpath = f"../exp/{scene}/womask_sphere/checkpoints/ckpt_300000.pth"

ckpt = torch.load(ckpt_fpath, map_location=torch.device("cuda"))
sdf_network.load_state_dict(ckpt["sdf_network_fine"])
color_network.load_state_dict(ckpt["color_network_fine"])

color_network_dict = {"color_network": color_network}


def render_fn(interior_mask, color_network_dict, ray_o, ray_d, points, normals, features):
    interior_color = color_network_dict["color_network"](points, normals, ray_d, features)  # [..., [2, 0, 1]]

    dots_sh = list(interior_mask.shape)
    color = torch.zeros(
        dots_sh
        + [
            3,
        ],
        dtype=torch.float32,
        device=interior_mask.device,
    )
    color[interior_mask] = interior_color

    normals_pad = torch.zeros(
        dots_sh
        + [
            3,
        ],
        dtype=torch.float32,
        device=interior_mask.device,
    )
    normals_pad[interior_mask] = normals
    return {"color": color, "normal": normals_pad}


def load_datadir(data_dir):
    from glob import glob
    from models.dataset import load_K_Rt_from_P

    camera_dict = np.load(os.path.join(data_dir, "cameras_sphere.npz"))
    images_lis = sorted(glob(os.path.join(data_dir, "image/*.png")))
    n_images = len(images_lis)
    images = np.stack([imageio.imread(im_name) for im_name in images_lis]) / 255.0
    images = torch.from_numpy(images).float()
    # world_mat is a projection matrix from world to image
    world_mats_np = [camera_dict["world_mat_%d" % idx].astype(np.float32) for idx in range(n_images)]
    # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
    scale_mats_np = [camera_dict["scale_mat_%d" % idx].astype(np.float32) for idx in range(n_images)]
    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())
    intrinsics_all = torch.stack(intrinsics_all, dim=0)
    pose_all = torch.stack(pose_all, dim=0)  # C2W
    pose_all = torch.inverse(pose_all)

    ic(images.shape, intrinsics_all.shape, pose_all.shape)
    return images, intrinsics_all, pose_all


gt_images, Ks, W2Cs = load_datadir(f"../public_data/{scene}")

img_idx = 10
gt_color = gt_images[img_idx]
camera = Camera(W=gt_color.shape[1], H=gt_color.shape[0], K=Ks[img_idx].cuda(), W2C=W2Cs[img_idx].cuda())

fill_holes = False
handle_edges = True
is_training = False
out_dir = f"./debug_raytracer_{scene}_{fill_holes}_{handle_edges}_{is_training}"
ic(out_dir)
os.makedirs(out_dir, exist_ok=True)

if is_training:
    camera, gt_color = camera.crop_region(trgt_W=256, trgt_H=256, center_crop=True, image=gt_color)
ic(gt_color.shape, camera.H, camera.W)

results = render_camera(
    camera,
    sdf_network,
    raytracer,
    color_network_dict,
    render_fn,
    fill_holes=fill_holes,
    handle_edges=handle_edges,
    is_training=is_training,
)

for x in list(results.keys()):
    results[x] = results[x].detach().cpu().numpy()


def append_allones(x):
    return np.concatenate((x, np.ones_like(x[..., 0:1])), axis=-1)


imageio.imwrite(os.path.join(out_dir, "convergent_mask.png"), to8b(results["convergent_mask"]))
imageio.imwrite(os.path.join(out_dir, "distance.exr"), results["distance"])
imageio.imwrite(os.path.join(out_dir, "depth.exr"), results["depth"])
imageio.imwrite(os.path.join(out_dir, "sdf.exr"), results["sdf"])
imageio.imwrite(os.path.join(out_dir, "points.exr"), results["points"])
imageio.imwrite(os.path.join(out_dir, "normal.png"), to8b((results["normal"] + 1.0) / 2.0))
imageio.imwrite(os.path.join(out_dir, "normal.exr"), results["normal"])
imageio.imwrite(os.path.join(out_dir, "color.png"), to8b(results["color"])[..., ::-1])
imageio.imwrite(os.path.join(out_dir, "color_gt.png"), to8b(gt_color.detach().cpu().numpy()))
imageio.imwrite(os.path.join(out_dir, "uv.exr"), append_allones(results["uv"]))

imageio.imwrite(os.path.join(out_dir, "depth_grad_norm.exr"), results["depth_grad_norm"])
imageio.imwrite(os.path.join(out_dir, "depth_edge_mask.png"), to8b(results["depth_edge_mask"]))
imageio.imwrite(
    os.path.join(out_dir, "walk_edge_found_mask.png"),
    to8b(results["walk_edge_found_mask"]),
)
trimesh.PointCloud(results["edge_points"].reshape((-1, 3))).export(os.path.join(out_dir, "edge_points.ply"))
imageio.imwrite(os.path.join(out_dir, "edge_mask.png"), to8b(results["edge_mask"]))
imageio.imwrite(os.path.join(out_dir, "edge_pos_side_weight.exr"), results["edge_pos_side_weight"])
imageio.imwrite(os.path.join(out_dir, "edge_angles.exr"), results["edge_angles"])
imageio.imwrite(os.path.join(out_dir, "edge_sdf.exr"), results["edge_sdf"])
imageio.imwrite(os.path.join(out_dir, "edge_pos_side_depth.exr"), results["edge_pos_side_depth"])
imageio.imwrite(os.path.join(out_dir, "edge_neg_side_depth.exr"), results["edge_neg_side_depth"])
imageio.imwrite(os.path.join(out_dir, "edge_pos_side_color.png"), to8b(results["edge_pos_side_color"])[..., ::-1])
imageio.imwrite(os.path.join(out_dir, "edge_neg_side_color.png"), to8b(results["edge_neg_side_color"])[..., ::-1])

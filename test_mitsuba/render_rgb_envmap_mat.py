import os
import numpy as np
import json
import imageio

imageio.plugins.freeimage.download()

import sys

asset_dir = sys.argv[1]
cam_dict_fpath = sys.argv[2]
envmap_fpath = sys.argv[3]
out_dir = sys.argv[4]


d_albedo = os.path.join(asset_dir, "diffuse_albedo.exr")
s_albedo = os.path.join(asset_dir, "specular_albedo.exr")
s_roughness = os.path.join(asset_dir, "roughness.exr")
mesh_fpath = os.path.join(asset_dir, "mesh.obj")

os.makedirs(out_dir, exist_ok=True)


envmap_fpath = os.path.join(asset_dir, "../envmap.exr")

cam_dict = json.load(open(cam_dict_fpath))

use_docker = True

for img_name in list(cam_dict.keys()):
    out_fpath = os.path.join(out_dir, img_name[:-4] + ".exr")
    K = np.array(cam_dict[img_name]["K"]).reshape((4, 4))
    focal = K[0, 0]
    width, height = cam_dict[img_name]["img_size"]
    fov = np.rad2deg(np.arctan(width / 2.0 / focal) * 2.0)
    w2c = np.array(cam_dict[img_name]["W2C"]).reshape((4, 4))

    c2w = np.linalg.inv(w2c)
    c2w[:3, :2] *= -1  # mitsuba camera coordinate system: x-->left, y-->up, z-->scene
    origin = c2w[:3, 3]
    c2w = " ".join([str(x) for x in c2w.flatten().tolist()])

    cmd = (
        'mitsuba -b 10 rgb_envmap_hdr_mat.xml -D fov={} -D width={} -D height={} -D c2w="{}" '
        "-D mesh={} -D d_albedo={} -D s_albedo={}  -D s_roughness={} "
        "-D envmap={}  "
        "-o {} ".format(fov, width, height, c2w, mesh_fpath, d_albedo, s_albedo, s_roughness, envmap_fpath, out_fpath)
    )

    if use_docker:
        docker_prefix = "docker run -w `pwd` --rm -v `pwd`:`pwd` -v /phoenix:/phoenix ninjaben/mitsuba-rgb "
        cmd = docker_prefix + cmd

    os.system(cmd)
    os.system("rm mitsuba.*.log")

    to8b = lambda x: np.uint8(np.clip(x * 255.0, 0.0, 255.0))
    im = imageio.imread(out_fpath).astype(np.float32)
    imageio.imwrite(out_fpath[:-4] + ".png", to8b(np.power(im, 1.0 / 2.2)))

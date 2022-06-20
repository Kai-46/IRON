import numpy as np
import imageio
import igl
import trimesh
import os
import shutil
import torch


to8b = lambda x: np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)


def sample_surface(vertices, face_vertices, texturecoords, face_texturecoords, n_samples):
    """
    Samples point cloud on the surface of the model defined as vectices and
    faces. This function uses vectorized operations so fast at the cost of some
    memory.
    """
    vec_cross = np.cross(
        vertices[face_vertices[:, 0], :] - vertices[face_vertices[:, 2], :],
        vertices[face_vertices[:, 1], :] - vertices[face_vertices[:, 2], :],
    )
    face_areas = np.sqrt(np.sum(vec_cross**2, 1))
    face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    # Error fix by Yangyan (yangyan.lee@gmail.com) 2017-Aug-7
    n_samples_per_face = np.ceil(n_samples * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc : acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)

    A = vertices[face_vertices[sample_face_idx, 0], :]
    B = vertices[face_vertices[sample_face_idx, 1], :]
    C = vertices[face_vertices[sample_face_idx, 2], :]
    P = (1 - np.sqrt(r[:, 0:1])) * A + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + np.sqrt(r[:, 0:1]) * r[:, 1:] * C

    A = texturecoords[face_texturecoords[sample_face_idx, 0], :]
    B = texturecoords[face_texturecoords[sample_face_idx, 1], :]
    C = texturecoords[face_texturecoords[sample_face_idx, 2], :]
    P_uv = (1 - np.sqrt(r[:, 0:1])) * A + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + np.sqrt(r[:, 0:1]) * r[:, 1:] * C

    return P.astype(np.float32), P_uv.astype(np.float32)


class Groupby(object):
    def __init__(self, keys):
        """note keys are assumed to by integer"""
        super().__init__()

        self.unique_keys, self.keys_as_int = np.unique(keys, return_inverse=True)
        self.n_keys = len(self.unique_keys)
        self.indices = [[] for i in range(self.n_keys)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]

    def apply(self, function, vector):
        assert len(vector.shape) <= 2
        if len(vector.shape) == 2:
            result = np.zeros((self.n_keys, vector.shape[-1]))
        else:
            result = np.zeros((self.n_keys,))

        for k, idx in enumerate(self.indices):
            result[k] = function(vector[idx], axis=0)

        return result


def accumulate_splat_material(xyz_image, material_image, weight_image, pcd, uv, material):
    H, W = material_image.shape[:2]

    xyz_image = xyz_image.reshape((H * W, -1))
    material_image = material_image.reshape((H * W, -1))
    weight_image = weight_image.reshape((H * W,))

    ### label each 3d point with their splat pixel index
    uv[:, 0] = uv[:, 0] * W
    uv[:, 1] = H - uv[:, 1] * H

    ### repeat to a neighborhood
    pcd = np.tile(pcd, (5, 1))
    material = np.tile(material, (5, 1))
    uv_up = np.copy(uv)
    uv_up[:, 1] -= 1
    uv_right = np.copy(uv)
    uv_right[:, 0] += 1
    uv_down = np.copy(uv)
    uv_down[:, 1] += 1
    uv_left = np.copy(uv)
    uv_left[:, 0] -= 1
    uv = np.concatenate((uv, uv_up, uv_right, uv_down, uv_left), axis=0)

    ### compute pixel coordinates
    pixel_col = np.floor(uv[:, 0])
    pixel_row = np.floor(uv[:, 1])
    label = (pixel_row * W + pixel_col).astype(int)

    ### filter out-of-range points
    mask = np.logical_and(label >= 0, label < H * W)
    label = label[mask]
    uv = uv[mask]
    material = material[mask]
    pcd = pcd[mask]
    pixel_col = pixel_col[mask]
    pixel_row = pixel_row[mask]

    # compute gaussian weight
    sigma = 1.0
    weight = np.exp(-((uv[:, 0] - pixel_col - 0.5) ** 2 + (uv[:, 1] - pixel_row - 0.5) ** 2) / (2 * sigma * sigma))
    # weight = np.ones_like(uv[:, 0])

    groupby_obj = Groupby(label)
    delta_xyz = groupby_obj.apply(np.sum, weight[:, np.newaxis] * pcd)
    delta_material = groupby_obj.apply(np.sum, weight[:, np.newaxis] * material)
    delta_weight = groupby_obj.apply(np.sum, weight)

    xyz_image[groupby_obj.unique_keys] += delta_xyz
    material_image[groupby_obj.unique_keys] += delta_material
    weight_image[groupby_obj.unique_keys] += delta_weight

    xyz_image = xyz_image.reshape((H, W, -1))
    material_image = material_image.reshape((H, W, -1))
    weight_image = weight_image.reshape((H, W))

    return xyz_image, material_image, weight_image


def loadmesh_and_checkuv(obj_fpath, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    vertices, texturecoords, _, face_vertices, face_texturecoords, _ = igl.read_obj(obj_fpath, dtype="float32")

    def make_rgba_color(float_rgb):
        float_rgba = np.concatenate((float_rgb, np.ones_like(float_rgb[:, 0:1])), axis=-1)
        return np.uint8(np.clip(float_rgba * 255.0, 0.0, 255.0))

    #### create debug plot
    pcd, pcd_uv = sample_surface(vertices, face_vertices, texturecoords, face_texturecoords, n_samples=10**6)

    uv_color = np.concatenate((pcd_uv, np.zeros_like(pcd_uv[:, 0:1])), axis=-1)
    trimesh.PointCloud(vertices=pcd, colors=make_rgba_color(uv_color)).export(os.path.join(out_dir, "check_uvmap.ply"))
    W, H = 512, 512
    grid_w, grid_h = np.meshgrid(np.linspace(0.0, 1.0, W), np.linspace(1, 0.0, H))
    grid_color = np.stack((grid_w, grid_h, np.zeros_like(grid_w)), axis=2)
    imageio.imwrite(os.path.join(out_dir, "check_uvmap.png"), to8b(grid_color))

    return vertices, face_vertices, texturecoords, face_texturecoords


def export_materials(mesh_fpath, material_predictor, out_dir, max_num_pts=320000, texture_H=2048, texture_W=2048):
    """output material parameters"""
    os.makedirs(out_dir, exist_ok=True)
    vertices, face_vertices, texturecoords, face_texturecoords = loadmesh_and_checkuv(mesh_fpath, out_dir)

    xyz_image = np.zeros((texture_H, texture_W, 3), dtype=np.float32)
    material_image = np.zeros((texture_H, texture_W, 7), dtype=np.float32)
    weight_image = np.zeros((texture_H, texture_W), dtype=np.float32)

    for i in range(5):
        points, points_uv = sample_surface(
            vertices, face_vertices, texturecoords, face_texturecoords, n_samples=5 * 10**6
        )

        points = torch.from_numpy(points).cuda()
        merge_materials = []
        for points_split in torch.split(points, max_num_pts, dim=0):
            with torch.set_grad_enabled(False):
                diffuse_albedo, specular_albedo, specular_roughness = material_predictor(points_split)
                merge_materials.append(
                    torch.cat((diffuse_albedo, specular_albedo, specular_roughness), dim=-1).detach().cpu()
                )
        merge_materials = torch.cat(merge_materials, dim=0).numpy()
        points = points.detach().cpu().numpy()

        accumulate_splat_material(xyz_image, material_image, weight_image, points, points_uv, merge_materials)

    final_xyz_image = xyz_image / (weight_image[:, :, np.newaxis] + 1e-10)
    final_material_image = material_image / (weight_image[:, :, np.newaxis] + 1e-10)

    imageio.imwrite(os.path.join(out_dir, "xyz.exr"), final_xyz_image)
    imageio.imwrite(os.path.join(out_dir, "diffuse_albedo.exr"), final_material_image[:, :, :3])
    imageio.imwrite(os.path.join(out_dir, "specular_albedo.exr"), final_material_image[:, :, 3:6])
    imageio.imwrite(os.path.join(out_dir, "roughness.exr"), final_material_image[:, :, 6])

    imageio.imwrite(os.path.join(out_dir, "xyz.png"), to8b(final_xyz_image * 0.5 + 0.5))
    imageio.imwrite(os.path.join(out_dir, "diffuse_albedo.png"), to8b(final_material_image[:, :, :3]))
    imageio.imwrite(os.path.join(out_dir, "specular_albedo.png"), to8b(final_material_image[:, :, 3:6]))
    imageio.imwrite(os.path.join(out_dir, "roughness.png"), to8b(final_material_image[:, :, 6]))

    out_mesh_fpath = mesh_fpath
    with open(out_mesh_fpath, "r") as original:
        data = original.read()
    with open(out_mesh_fpath, "w") as modified:
        modified.write("usemtl ./{}\n\n".format(os.path.basename(out_mesh_fpath)[:-4] + ".mtl") + data)

    with open(os.path.join(out_dir, os.path.basename(out_mesh_fpath)[:-4] + ".mtl"), "w") as fp:
        fp.write(
            "newmtl Wood\n"
            "Ka 1.000000 1.000000 1.000000\n"
            "Kd 0.640000 0.640000 0.640000\n"
            "Ks 0.500000 0.500000 0.500000\n"
            "Ns 96.078431\n"
            "Ni 1.000000\n"
            "d 1.000000\n"
            "illum 0\n"
            "map_Kd diffuse_albedo.png\n"
        )

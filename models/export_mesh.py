import numpy as np
import torch
import trimesh
from skimage import measure


def get_grid_uniform(resolution):
    x = np.linspace(-1.0, 1.0, resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points.cuda(), "shortest_axis_length": 2.0, "xyz": [x, y, z], "shortest_axis_index": 0}


def get_grid(points, resolution, eps=0.1):
    input_min = torch.min(points, dim=0)[0].squeeze().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if shortest_axis == 0:
        x = np.linspace(input_min[shortest_axis] - eps, input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif shortest_axis == 1:
        y = np.linspace(input_min[shortest_axis] - eps, input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif shortest_axis == 2:
        z = np.linspace(input_min[shortest_axis] - eps, input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {
        "grid_points": grid_points,
        "shortest_axis_length": length,
        "xyz": [x, y, z],
        "shortest_axis_index": shortest_axis,
    }


def export_mesh(sdf, mesh_fpath, resolution=512, max_n_pts=100000):
    assert mesh_fpath.endswith(".obj"), f"must use .obj format: {mesh_fpath}"
    # get low res mesh to sample point cloud
    grid = get_grid_uniform(100)
    z = []
    points = grid["grid_points"]
    for i, pnts in enumerate(torch.split(points, max_n_pts, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0).astype(np.float32)
    verts, faces, normals, values = measure.marching_cubes(
        volume=z.reshape(grid["xyz"][1].shape[0], grid["xyz"][0].shape[0], grid["xyz"][2].shape[0]).transpose(
            [1, 0, 2]
        ),
        level=0,
        spacing=(
            grid["xyz"][0][2] - grid["xyz"][0][1],
            grid["xyz"][0][2] - grid["xyz"][0][1],
            grid["xyz"][0][2] - grid["xyz"][0][1],
        ),
    )
    verts = verts + np.array([grid["xyz"][0][0], grid["xyz"][1][0], grid["xyz"][2][0]])
    mesh_low_res = trimesh.Trimesh(verts, faces, normals)
    components = mesh_low_res.split(only_watertight=False)
    areas = np.array([c.area for c in components], dtype=np.float)
    mesh_low_res = components[areas.argmax()]
    recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
    recon_pc = torch.from_numpy(recon_pc).float().cuda()

    # Center and align the recon pc
    s_mean = recon_pc.mean(dim=0)
    s_cov = recon_pc - s_mean
    s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
    vecs = torch.eig(s_cov, True)[1].transpose(0, 1)
    if torch.det(vecs) < 0:
        vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
    helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1), (recon_pc - s_mean).unsqueeze(-1)).squeeze()

    grid_aligned = get_grid(helper.cpu(), resolution)
    grid_points = grid_aligned["grid_points"]
    g = []
    for i, pnts in enumerate(torch.split(grid_points, max_n_pts, dim=0)):
        g.append(
            (
                torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2), pnts.unsqueeze(-1)).squeeze()
                + s_mean
            )
            .detach()
            .cpu()
        )
    grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, max_n_pts, dim=0)):
        z.append(sdf(pnts.cuda()).detach().cpu().numpy())
    z = np.concatenate(z, axis=0).astype(np.float32)

    if not (np.min(z) > 0 or np.max(z) < 0):
        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(
                grid_aligned["xyz"][1].shape[0], grid_aligned["xyz"][0].shape[0], grid_aligned["xyz"][2].shape[0]
            ).transpose([1, 0, 2]),
            level=0,
            spacing=(
                grid_aligned["xyz"][0][2] - grid_aligned["xyz"][0][1],
                grid_aligned["xyz"][0][2] - grid_aligned["xyz"][0][1],
                grid_aligned["xyz"][0][2] - grid_aligned["xyz"][0][1],
            ),
        )

        verts = torch.from_numpy(verts).float()
        verts = torch.bmm(
            vecs.detach().cpu().unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2), verts.unsqueeze(-1)
        ).squeeze()
        verts = (verts + grid_points[0]).numpy()

        trimesh.Trimesh(verts, faces, normals).export(mesh_fpath)

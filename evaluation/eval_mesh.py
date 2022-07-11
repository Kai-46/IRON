import numpy as np
import os
import igl


def cal_mesh_err(va, fa, vb, fb):
	sqrD1, _, _ = igl.point_mesh_squared_distance(va, vb, fb)
	sqrD2, _, _ = igl.point_mesh_squared_distance(vb, va, fa)
	D1 = np.sqrt(sqrD1)
	D2 = np.sqrt(sqrD2)
	ret = (D1.mean() + D2.mean()) * 0.5
	return ret


def eval_obj_meshes(pred_mesh_fpath, trgt_mesh_fpath):
    v1, _, n1, f1, _, _ = igl.read_obj(pred_mesh_fpath)
    v4, _, n4, f4, _, _ = igl.read_obj(trgt_mesh_fpath)

    return cal_mesh_err(v1, f1, v4, f4)


import sys
pred_mesh_fpath = sys.argv[1]
trgt_mesh_fpath = sys.argv[2]
dist_bidirectional = eval_obj_meshes(pred_mesh_fpath, trgt_mesh_fpath)
print('\tChamfer_dist: ', dist_bidirectional)

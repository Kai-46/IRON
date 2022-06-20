# Usage: Blender --background --python export_uv.py {in_mesh_fpath} {out_mesh_fpath}

import os
import bpy
import sys


def export_uv(in_mesh_fpath, out_mesh_fpath):
    assert in_mesh_fpath.endswith(".obj"), f"must use .obj format: {in_mesh_fpath}"
    assert out_mesh_fpath.endswith(".obj"), f"must use .obj format: {out_mesh_fpath}"

    bpy.data.objects["Camera"].select_set(True)
    bpy.data.objects["Cube"].select_set(True)
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()  # delete camera, cube, light

    mesh_fname = os.path.basename(in_mesh_fpath)[:-4]
    bpy.ops.import_scene.obj(
        filepath=in_mesh_fpath,
        use_edges=True,
        use_smooth_groups=True,
        use_split_objects=True,
        use_split_groups=True,
        use_groups_as_vgroups=False,
        use_image_search=True,
        split_mode="ON",
        global_clamp_size=0,
        axis_forward="-Z",
        axis_up="Y",
    )

    obj = bpy.data.objects[mesh_fname]
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.export_scene.obj(
        filepath=out_mesh_fpath,
        axis_forward="-Z",
        axis_up="Y",
        use_selection=True,
        use_normals=True,
        use_uvs=True,
        use_materials=False,
        use_triangles=True,
    )


print(sys.argv)
export_uv(sys.argv[-2], sys.argv[-1])

# IRON
We propose a neural inverse rendering pipeline called IRON that operates on photometric images and outputs high-quality 3D content in the format of triangle meshes and material textures readily deployable in existing graphics pipelines.

## Usage

### Setup
```shell
git clone https://github.com/Kai-46/iron.git && cd iron && . ./env.sh
```

### Data
Download real-world data from [this google drive folder](https://drive.google.com/file/d/1SwQ2MX8hjAHO3V86saXlgIK9Uevlvh-i/view?usp=sharing), and unzip to this code folder

### Running
```shell
. ./train_scene.sh sai/dragon
```
Once training is done, you will see the recovered mesh and materials under the folder ```./exp_iron_stage2/sai/dragon/mesh_and_materials_50000/```. At the same time, the rendered test images are under the folder ``````./exp_iron_stage2/sai/dragon/render_test_50000/``````

## Citations
```
@inproceedings{iron-2022,
  title={IRON: Inverse Rendering by Optimizing Neural SDFs and Materials from Photometric Images},
  author={Zhang, Kai and Luan, Fujun and Li, Zhengqi and Snavely, Noah},
  booktitle={IEEE Conf. Comput. Vis. Pattern Recog.},
  year={2022}
}
```

## Example results
![example results](./readme_resources/assets_lowres.png)
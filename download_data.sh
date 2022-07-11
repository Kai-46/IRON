pip install gdown

echo "Downloading image indices for Bi et al 2020: Deep Reflectance Volumes: Relightable Reconstructions from Multi-View Photometric Images"
echo "Please ask the authors of this work for data, and then split the data using the image indices"
gdown 1BThZgEnHgsL7dgyVTQuSFYZjAkZzQozx
unzip "Bi et al 2020-image_indices.zip"

echo "Downloading real data captured by Luan et al 2021: Unified Shape and SVBRDF Recovery using Differentiable Monte Carlo Rendering"
echo "Please credit the original paper if you use this data"
gdown 1BO6XZjUm8PhHof5RZ7O0Y3C815loBlqj
unzip "Luan et al 2021.zip"

echo "Downloading synthetic assets for creating synthetic data with Mitsuba"
gdown 1EhDI06NsluXsC98ZErvB7UN_TPI1_6sn
unzip "synthetic_assets.zip"

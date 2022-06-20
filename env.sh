conda create -y -n iron python=3.8 && conda activate iron
pip install numpy scipy trimesh opencv_python scikit-image imageio imageio-ffmpeg pyhocon PyMCubes tqdm icecream configargparse
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install black  # format code nicely
pip install tensorboard
pip install kornia
conda install -c conda-forge igl

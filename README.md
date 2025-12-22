# :dart: Download the HM3D Dataset

### HM3D scene dataset
Please download the validation split of the HM3Dv2 scene data from https://github.com/matterport/habitat-matterport-3dresearch, files including:
```
hm3d-val-glb-v0.2.tar
hm3d-val-habitat-v0.2.tar
hm3d-val-semantic-annots-v0.2.tar
hm3d-val-semantic-configs-v0.2.tar
```
Then extract them as follow:
```shell
mkdir -p ./data/scene_datasets/hm3d/val
cd data
tar xvf hm3d-val-glb-v0.2.tar -C scene_datasets/hm3d/val
tar xvf hm3d-val-habitat-v0.2.tar -C scene_datasets/hm3d/val
tar xvf hm3d-val-semantic-annots-v0.2.tar -C scene_datasets/hm3d/val
tar xvf hm3d-val-semantic-configs-v0.2.tar -C scene_datasets/hm3d
ln -s ../hm3d scene_datasets/hm3d_v0.2
```
You can refer the file structure below:
```
data/scene_datasets/hm3d/val
├── 00800-TEEsavR23oF
├── 00801-HaxA7YrQdEC
├── ...
```

### HM3D Object Navigation Dataset
You can find object navigation task dataset in https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md. Please download  `objectnav_hm3d_v1.zip` and `objectnav_hm3d_v2.zip`, and extract then as follow:
```shell
cd data
mkdir -p datasets/objectnav/hm3d
unzip objectnav_hm3d_v1.zip -d datasets/objectnav/hm3d
unzip objectnav_hm3d_v2.zip -d datasets/objectnav/hm3d
cd datasets/objectnav/hm3d 
mv objectnav_hm3d_v1 v1
mv objectnav_hm3d_v2 v2
```

For evaluation of the `objectnav_hm3d_v2` navigation dataset, you should create a symbolic link `data/scene_datasets/hm3d_v0.2` pointing to `data/scene_datasets/hm3d/val`.

## :hammer_and_wrench: Configure the Environment

### CUDA Environment
We've tested CUDA 11.8 on various version of Nivida GPUs. After the installation, please execute the script below for CUDA compilation later:
```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```


### Conda Environment
Below is the recommended pipeline for cuda environment:
```bash
conda create -n nav_neural python=3.9 -y
conda activate nav_neural 
conda install habitat-sim=0.2.5 -c conda-forge -c aihabitat
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install cuml-cu11==23.10.0 --extra-index-url=https://pypi.nvidia.com  # ==23.10.0 for GPUs like 1070
pip install "git+https://github.com/naokiyokoyama/frontier_exploration.git"
pip install opencv-python==4.5.5.64
pip install scipy scikit-learn
pip install habitat-baselines==0.2.520230729 habitat-lab==0.2.520230729
```

### Third-party Dependencies
Download the checkpoint and copy it into `data`:
>https://github.com/bdaiinstitute/vlfm/raw/refs/heads/main/data/pointnav_weights.pth

## :arrow_forward: Evaluate within Habitat

### Run in Simulator
To collect data:
```bash
python -m neural.collect --episode 1000
```


## :weight_lifting: Analyze Data
To analyze the collected data, select one of the following command:
```bash
python -m neural.analyze --feature-key activations_hidden --threshold 0.2 
python -m neural.analyze --feature-key activations_rnn_output --threshold 0.2
python -m neural.analyze --feature-key activations_visual_embed --threshold 0.2
```
To visualize the result, run:
```bash
python -m neural.visualize
```

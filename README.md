# Learning to Estimate Hidden Motions with Global Motion Aggregation
This repository contains the source code for our paper:

[Learning to Estimate Hidden Motions with Global Motion Aggregation](https://arxiv.org/abs/2104.02409)<br/>
ICCV 2021 <br/>
**Shihao Jiang**, Dylan Campbell, Yao Lu, Hongdong Li, Richard Hartley<br/>
ANU, Oxford<br/>

## Environments
You will have to choose cudatoolkit version to match your compute environment. 
The code is tested on PyTorch 1.8.0 but other versions might also work. 
```Shell
conda create --name gma python==3.7
conda activate gma
conda install pytorch=1.8.0 torchvision=0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install matplotlib imageio einops scipy opencv-python
```
## Demo
```Shell
sh demo.sh
```
## Train
```Shell
sh train.sh
```
## Evaluate
```Shell
sh evaluate.sh
```
## License
WTFPL. See [LICENSE](LICENSE) file. 

## Acknowledgement
The overall code framework is adapted from [RAFT](https://github.com/princeton-vl/RAFT). We
thank the authors for the contribution. We also thank [Phil Wang](https://github.com/lucidrains)
for open-sourcing transformer implementations. 

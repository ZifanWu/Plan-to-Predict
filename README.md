# Plan-To-Predict

This code accompanies the paper "[Plan To Predict: Learning an Uncertainty-Foreseeing Model for Model-Based Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/65beb73449888fabcf601b3a3ef4b3a7-Abstract-Conference.html)".



## Installation

1. Install [MuJoCo 1.50](https://www.roboti.us/index.html) at `~/.mujoco/mjpro150` and copy your license key to `~/.mujoco/mjkey.txt`
2. Clone `P2P`

```
git clone https://github.com/ZifanWu/Plan-to-Predict.git
```

3. Create a conda environment and install Plan-to-Predict

```
cd src/Plan-to-Predict
conda env create -f environment/gpu-env.yml
conda activate p2p
pip install -e .
```



## Usage

```
python main_p2p.py --num_epoch 150
```

### Optimal parameters

The optimal parameters are contained in `./configs/` folder.

## Reference

```
@article{wu2022plan,
  title={Plan To Predict: Learning an Uncertainty-Foreseeing Model For Model-Based Reinforcement Learning},
  author={Wu, Zifan and Yu, Chao and Chen, Chen and Hao, Jianye and Zhuo, Hankz Hankui},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={15849--15861},
  year={2022}
}
```

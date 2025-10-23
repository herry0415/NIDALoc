# NIDALoc
本项目仅用于复现代码记录使用 

## Environment
- 环境同SGLoc 和DiffLoc基本一致
- 在上述基础上多了三个包 手动安装一下即可 `torchstat`、`thop`、`ptflops`（较大）


## Train  / Test  代码要修改的部分 以兼容HerCULES 和 Snail_Radar
在STCLoc的基础上有以下注意事项
- 传入参数问题：   `skip`=2 与前面保持一致 `epoch`注意100轮即可
- 数据集类构造： 多了两个新函数`grid_position` `hd_orientation` 并且也返回了新的参数
- train / test 指定GPU要在import torch之前 要不然可能会指定失败

## Run
### Hercules
注意事项同STCLoc  https://github.com/herry0415/STCLoc
- train  
```
python train_hercules_lidar.py //  python train_hercules_radar.py 

```
**其他运行训练的时候在代码里要改的**
- --gpu_id 0 对应代码里os.environ["CUDA_VISIBLE_DEVICES"] = '3'
- --全局变量 **SEQUENCE_NAME**
- --序列名 `data.hercules_lidar.py` or `data.hercules_radar.py`   里面的`sequence_name`
- `data.composition` 里面的要把引入的 `radar / lidar` 版本的 `Hercules类` 进行切换

**以下在代码里默认设置好的（参考原本代码run oxford数据集的参数）**
- --decay_step 500
- --log_dir 文件后缀记得更改`f'STCLoc_{SEQUENCE_NAME}_Lidar/'` or `f'STCLoc_{SEQUENCE_NAME}_Radar/'`
- --batch_size 80  注意更改
- --val_batch_size 80  注意更改
- --skip 2
- --dataset Hercules 
- --num_loc 10 --num_ang 10 / 8 按照原先代码设置
- --mac epoch **用100轮**



#### test  -- 1 GPU
```
python eval_hercules_lidar.py //   python eval_hercules_radar.py
```
**其他运行测试的时候在代码里要改的**
- --gpu_id 0 对应代码里`os.environ["CUDA_VISIBLE_DEVICES"] = '3'`
- --resume_model `checkpoint_epochxx.tar` 权重文件
- --全局变量 **SEQUENCE_NAME**
- --序列名`data.hercules_lidar.py` or `data.hercules_radar.py`   里面的`sequence_name`
- `data.composition` 里面的要把引入的 `radar / lidar` 版本的 `Hercules类` 进行切换


**以下在代码里默认设置好的（参考原本代码run oxford数据集的参数）**
- --log_dir `f'STCLoc_{SEQUENCE_NAME}_Lidar/'` or `f'STCLoc_{SEQUENCE_NAME}_Radar/'`
- --val_batch_size 40 **注意这个值等于1 会测试的很慢**
- --dataset `Hercules``
- --skip 2
- --num_loc 10 --num_ang 10 / 8 按照原先代码设置

## Citation

```
@ARTICLE{10296854,
  author={Yu, Shangshu and Sun, Xiaotian and Li, Wen and Wen, Chenglu and Yang, Yunuo and Si, Bailu and Hu, Guosheng and Wang, Cheng},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={NIDALoc: Neurobiologically Inspired Deep LiDAR Localization}, 
  year={2024},
  volume={25},
  number={5},
  pages={4278-4289}
}
```

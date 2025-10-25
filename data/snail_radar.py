import os
import numpy as np
import os.path as osp
from torch.utils import data
from decimal import Decimal
from copy import deepcopy
from data.augmentor import Augmentor # todo  数据增强
from sklearn.neighbors import KDTree
# import MinkowskiEngine as ME
import os.path as osp
import re
from utils.pose_util import process_poses, ds_pc
from utils.pose_util import filter_overflow_snail, interpolate_pose_snail
from utils.pose_util import poses_to_matrices_snale
from utils.pose_util import grid_position, hd_orientation  #todo 多了几个引入的函数

from data.body_t_ars548 import body_T_ars548 # todo 为从data导入

xt32_T_ars = np.array([
        [-0.0148246945939438,  0.999869704931098,  -0.00638767491813221, 0],
        [-0.967459492843472,   -0.0127297492912655,  0.252705525895304,   0],
        [ 0.252591285490489,    0.00992609917652017,  0.967522151941183,   0.07],
        [ 0,                   0,                    0,                  1]
    ])

class Snail(data.Dataset):
    def __init__(self, 
                 data_path, 
                 train=True,
                 valid=False,
                 voxel_size=0.3, # todo  x32的体素设置为0.4
                 augmentation=[],
                 num_points=4096,
                 num_loc=10,  # todo
                 num_ang=10,  # todo
                 real=False,    
                #  skip_pcs=False #todo 添加 real 和 skip_pcs 以兼容 MF 类
                 ):
        
        self.dataset_root = data_path
        sequence_name = 'if' # todo 序列名
        self.sub_sequence_name =  '20240123_3'      # if   ['20240116_eve_5', '20240116_5', '20240123_3']  # test 集
                                                    # iaf  ['20231213_3', '20240113_3', '20240116_eve_4']  # test 集
        self.sequence_name = sequence_name
        self.train = train

        self.data_dir = os.path.join(self.dataset_root, self.sequence_name)

        seqs = self._get_sequences(sequence_name, train) #todo train -> split

        self.T_body_ars548_dict = {}
        self.lidar_files = []
        self.radar_files = []
        self.samples = []
        self.T_body_xt32 = np.array([
                [0.0, -1.0,  0.0,  0.0],
                [1.0,  0.0,  0.0,  0.0],
                [0.0,  0.0,  1.0,  0.0],
                [0.0,  0.0,  0.0,  1.0]
            ], dtype=np.float32) 
        
        #todo snail数据集多的一部分参数
        # self.aug_rotation = aug_rotation
        # self.aug_translation = aug_translation
      
        # if train:
        #     seqs = getattr(self.cfgs.DATASET, sequence_name).train
        #     print(f"  Training with {seqs}  sequence \n")
        # else:
        #     seqs = getattr(self.cfgs.DATASET, sequence_name).test
        #     print(f"  Testing with {seqs}  sequence \n")
       
        ps = {}
        ts = {}
        # pcs_all = [] #todo 多了一个这个
        vo_stats = {}
        # self.pcs = [] #todo 没什么用
        
        for seq in seqs: 
            date_match = re.search(r"(\d{8})", seq)
            if date_match:
                date = int(date_match.group(1))
            else:
                    raise ValueError(f"Cannot parse date from sequence name: {seq}")
            T_body_ars548 = body_T_ars548(date)  #ars548到body的外参
            self.T_body_ars548_dict[seq] = T_body_ars548 #todo 序列不一样外参不一样
            
            seq_dir = os.path.join(self.data_dir, seq)
            lidar_ts_raw_str = np.loadtxt(seq_dir + '/xt32/times.txt',dtype=str, usecols=0)
            lidar_ts_raw = np.array([Decimal(s) for s in lidar_ts_raw_str]) # todo  只有Decimal可以保持精度
            lidar_gt_filename = os.path.join(seq_dir, 'utm50r_T_xt32.txt')
            ts[seq] = filter_overflow_snail(lidar_gt_filename, lidar_ts_raw) # todo   逐一过滤，保留那些落在 GNSS 时间范围内的时间戳
            
            lidar_pose = interpolate_pose_snail(lidar_gt_filename, ts[seq]) #[n,7]
            p = poses_to_matrices_snale(lidar_pose) #(n,4,4)
            ps[seq] = np.reshape(p[:, :3, :], (len(p), -1)) 
            
           
            radar_ts_raw_str = np.loadtxt(seq_dir+ '/ars548/points/times.txt', dtype=str, usecols=0)
            radar_ts_raw = np.array([Decimal(s) for s in radar_ts_raw_str])
            radar_gt_filename = os.path.join(seq_dir, 'utm50r_T_ars548.txt')
            radar_filter_ts = filter_overflow_snail(radar_gt_filename, radar_ts_raw)
            radar_pose = interpolate_pose_snail(radar_gt_filename, radar_filter_ts)
            radar_p = poses_to_matrices_snale(radar_pose)
            
            lidar_xy = p[:,:2, 3]
            radar_xy = radar_p[:,:2, 3]
            
            lidar_files = [os.path.join(seq_dir, 'xt32_bins', str(t)+ '.bin') for t in ts[seq]]
            radar_files = [os.path.join(seq_dir, 'ars548/multi_frame_w7', str(t)+'_multi_w7' + '.bin') for t in radar_filter_ts]
         
            tree_radar = KDTree(radar_xy)
            dists, idxs = tree_radar.query(lidar_xy, k=1)
            idxs = idxs.ravel()
            dists = dists.ravel()

            for li, ridx in enumerate(idxs):
                dist = float(dists[li])
                lf = lidar_files[li]
                rf = radar_files[ridx]
                if os.path.exists(lf) and os.path.exists(rf):
                    # self.lidar_files.append(lf)
                    # self.radar_files.append(rf)
                    self.samples.append((lf,rf, seq))
                    # pcs_all.append(rf) #! todo 加载lidar files  /  radar files
                else:
                    # 缺文件则略过并打印警告
                    if not os.path.exists(lf): print(f"Missing LIDAR file: {lf}")
                    if not os.path.exists(rf): print(f"Missing RADAR file: {rf}")
               
            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
            
  
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))

        pose_stats_filename = os.path.join(self.data_dir, self.sequence_name  + '_radar_pose_stats.txt') #todo lidar
        print(f'saving pose stats in {pose_stats_filename}')
        #todo 新增pose_max_min_filename
        pose_max_min_filename = os.path.join(self.data_dir, self.sequence_name + '_radar_pose_max_min.txt') #todo lidar files

        if self.train:
            self.mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)  # (3,)
            self.std_t = np.std(poses[:, [3, 7, 11]], axis=0)  # (3,)
            np.savetxt(pose_stats_filename, np.vstack((self.mean_t, self.std_t)), fmt='%8.7f')
        else:
            self.mean_t, self.std_t = np.loadtxt(pose_stats_filename)
            
        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))
        self.poses_max = np.empty((0, 2))
        self.poses_min = np.empty((0, 2))
        # poses_all = np.empty((0, 6))
        # rots_all = np.empty((0, 3, 3))
        # self.augmentor = Augmentor()
        for seq in seqs:
            pss, pss_max, pss_min = process_poses(poses_in=ps[seq], mean_t=self.mean_t, std_t=self.std_t,
                                                            align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                                            align_s=vo_stats[seq]['s'])
        
            self.poses = np.vstack((self.poses, pss))
            self.poses_max = np.vstack((self.poses_max, pss_max))
            self.poses_min = np.vstack((self.poses_min, pss_min))
            if train:
                self.poses_max = np.max(self.poses_max, axis=0)  # (2,)
                self.poses_min = np.min(self.poses_min, axis=0)  # (2,)
                np.savetxt(pose_max_min_filename, np.vstack((self.poses_max, self.poses_min)), fmt='%8.7f')
            else:
                self.poses_max, self.poses_min = np.loadtxt(pose_max_min_filename)

        self.augmentation = augmentation
        self.num_points = num_points
        self.num_loc = num_loc   #todo 新增参数
        self.num_ang = num_ang   #todo 新增参数


        if train:
            print("train data num:" + str(len(self.poses)))
            print("train grid num:" + str(self.num_loc * self.num_loc))#todo

        else:
            print("valid data num:" + str(len(self.poses)))
            print("valid grid num:" + str(self.num_loc * self.num_loc))#todo


    def _get_sequences(self, sequence_name, train):
        mapping = {
            'if': (
                ['20231208_4', '20231213_4', '20240115_3'],  # train 集
                [self.sub_sequence_name]
                # ['20240116_eve_5', '20240116_5', '20240123_3']  # test 集
            ),
            'iaf': (
                ['20231201_2', '20231208_5', '20231213_2'],  # train 集
                [self.sub_sequence_name]
                # ['20231213_3', '20240113_3', '20240116_eve_4']  # test 集
            )
        }
        return mapping[sequence_name][0] if train else mapping[sequence_name][1]



    def __getitem__(self, index):
        
        lidar_scan_path, radar_scan_path, seq_name = self.samples[index] #todo 每条序列外参不一致所以需要处理
        T_body_ars548 = self.T_body_ars548_dict[seq_name] # ars548到body的外参
        T_body_xt32 = self.T_body_xt32
        
        # lidar_ptcld = np.fromfile(lidar_scan_path, dtype=np.float32).reshape(-1,4)[:,:3]
        radar_ptcld = np.fromfile(radar_scan_path, dtype=np.float32).reshape(-1,9)[:,:3]
        # 统一到body
        # lidar_in_body = (T_body_xt32[:3, :3] @ lidar_ptcld.T).T + T_body_xt32[:3, 3]
        radar_in_body = (T_body_ars548[:3, :3] @ radar_ptcld.T).T + T_body_ars548[:3, 3]
        
        # #只要前向
        # HFOV = np.deg2rad(120)
        # HFOV_half = HFOV / 2
        # x, y, z = lidar_in_body[:, 0], lidar_in_body[:, 1], lidar_in_body[:, 2]
        # horiz_angle = np.arctan2(y, x)
        # mask_lidar = (x > 0) & (np.abs(horiz_angle) <= HFOV_half)
        # lidar_in_fov = lidar_in_body[mask_lidar]

        # lidar_scan = np.ascontiguousarray(lidar_in_fov)
        radar_scan = np.ascontiguousarray(radar_in_body)
        scan = radar_scan

        scan      = ds_pc(scan, self.num_points)

        for a in self.augmentation:
            scan = a.apply(scan)
        
        pose = self.poses[index]  # (6,)
        grid      = grid_position(pose, self.poses_max, self.poses_min, self.num_loc) # (2, )  #Todo 新增
        hd        = hd_orientation(pose, self.num_ang)  # (1, )  #todo 新增

        return scan, pose, grid, hd  #todo 新增
            
            
    def __len__(self):
        return len(self.poses)
   
import random
import torch

from deeplio.common.spatial import inv_SE3, rotation_matrix_to_quaternion

import math
import numpy as np

class PostProcessSiameseData(object):
    def __init__(self, seq_size=2, batch_size=1, shuffle=False, device='cpu'):
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.combinations = []
        self.shuffle = shuffle
        self.device = device

    def process(self, data):
        res_imu = []
        res_gt_local = []
        res_gt_global = []

        n_batches = len(data['imus'])
        for b in range(n_batches):
            imus = data['imus'][b]
            gts = data['gts'][b]

            combinations = [[0, x] for x in range(self.seq_size) if x > 0]
            # we do not want that the network memorizes an specific combination pattern
            if self.shuffle:
                random.shuffle(combinations)
            self.combinations.extend(combinations)

            gts_local = self.process_ground_turth(gts, combinations)
            res_gt_local.append(gts_local)
            res_gt_global.append(gts)

            imu_seq = []
            for j, combi in enumerate(combinations):
                idx_0 = combi[0]
                idx_1 = combi[1]

                # Determining IMU measurment btw. each combination
                max_idx = max(combi)
                min_idx = min(combi)
                imu_tmp = []
                for k in range(min_idx, max_idx):
                    imu_tmp.extend(imus[k])
                imu_seq.append(torch.stack(imu_tmp).to(self.device))
            res_imu.append(imu_seq)

        res_gt_global = torch.stack(res_gt_global).to(self.device, non_blocking=True)
        res_gt_local = torch.stack(res_gt_local).to(self.device, non_blocking=True)
        #res_imu = [torch.stack(imu).to(self.device, non_blocking=True) for imu_seq in res_imu for imu in imu_seq]
        return res_imu, res_gt_local, res_gt_global

    def process_ground_turth(self, gts, combinations):
        T_global = []
        v_global = []

        for gt in gts:
            t = gt[0:3]
            R = gt[3:12].reshape(3, 3)
            T = torch.eye(4)
            T[:3, 3] = t
            T[:3, :3] = R
            T_global.append(T)
            v = gt[12:]
            v_global.append(v)

        state_local = []
        for combi in combinations:
            T_i = T_global[combi[0]]
            T_i_inv = inv_SE3(T_i)
            T_ip1 = T_global[combi[1]]
            T_i_ip1 = torch.matmul(T_i_inv, T_ip1)
            dx = T_i_ip1[:3, 3].contiguous()
            dq = rotation_matrix_to_quaternion(T_i_ip1[:3, :3].contiguous())
            dv = v_global[combi[1]] - v_global[combi[0]]
            state_local.append(torch.cat([dx, dq, dv]))

        gt_local = torch.stack(state_local).to(self.device, non_blocking=True)
        return gt_local

    def __call__(self, args):
        return self.process(args)

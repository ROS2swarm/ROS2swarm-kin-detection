import os
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data

class RosSwLidDataset(data.Dataset):
    def __init__(self, config, train=True):
        self.data_root = config.DATASET_PATH
        self.arenas = config.ARENAS_TRAIN if train else config.ARENAS_VAL
        self.behaviours = config.BEHAVIOURS
        self.robots = config.ROBOTS
        self.num_robots = config.NROBOTS
        self.only_valid = config.DATA_VALID

        self.files_dict = []
        self.files_dfdict = {}
        self.files_length = []
        self.files_valid_length = []

        for arena in self.arenas:
            files = os.listdir(f'%s/%s/datasets/' % (self.data_root, arena))
            for behaviour in self.behaviours:
                for robot in self.robots:
                    for nrobot in self.num_robots:
                        ffiles = [filename for filename in files if filename.startswith(f'dataset_%s_%d_%s_%s_2023' % (arena.replace(' ', '-'), nrobot, robot, behaviour))]
                        for file in ffiles:
                            filename = f'%s/%s/datasets/%s' % (self.data_root, arena, file)
                            tmp_df = pd.read_csv(filename)
                            if self.only_valid and sum(tmp_df['Flag valid data']) == 0:
                                continue
                            self.files_dict.append(filename)
                            self.files_dfdict[filename] = tmp_df
                            self.files_length.append(len(tmp_df)) if not self.only_valid else self.files_length.append(sum(tmp_df['Flag valid data']))

        self.files_length_cum = np.array(self.files_length).cumsum()
        self.cum_sub = np.array([0] + self.files_length).cumsum()

    def __len__(self):
        return self.files_length_cum[-1]

    def __getitem__(self, item):
        search_res = np.where(self.files_length_cum - item > 0)[0][0]
        filename = self.files_dict[search_res]
        tmp_df = self.files_dfdict[filename] #pd.read_csv(filename)
        if self.only_valid:
            tmp_df = tmp_df.loc[tmp_df['Flag valid data']]
        # print(self.files_length[search_res], self.files_length_cum[search_res], item, self.files_length_cum[search_res] - item, len(tmp_df))
        # print(self.cum_sub[search_res], item - self.cum_sub[search_res])
        # print(tmp_df.iloc[0])
        inst_row = tmp_df.iloc[(item - self.cum_sub[search_res])]
        points = torch.tensor(inst_row.iloc[7:367], dtype=torch.float32)
        torch.nan_to_num_(points, posinf=3.5)
        labels = torch.tensor(inst_row.iloc[367:], dtype=torch.int64)
        # points = points.where(points != torch.inf, torch.ones_like(points) * 4.0)

        return points, labels
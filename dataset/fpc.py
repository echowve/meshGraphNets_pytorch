import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class FpcDataset(Dataset):

    def __init__(self, data_root, split):
        meta_path = os.path.join(data_root, split+'.npz')
        data_path = os.path.join(data_root, split+'.dat')
        meta_keys = ("pos", "node_type", "cells", "indices", "cindices", "all_velocity_shape")
        tmp = np.load(meta_path, allow_pickle=True)
        self.meta = {key: tmp[key] for key in meta_keys}

        shape = self.meta['all_velocity_shape']
        self.fp = np.memmap(data_path, dtype='float32', mode='r', shape=shape)

        self.tra_len = self.fp.shape[1]
        self.num_sampes_per_tra = self.tra_len -1
        tras_nums = len(self.meta['indices']) - 1 # remove the first 0 indice
        self.total_samples = tras_nums * self.num_sampes_per_tra

    def __getitem__(self, index):

        tra_index = index//self.num_sampes_per_tra
        tra_sample_index = index % (self.tra_len -1)
        tra_start_index = self.meta['indices'][tra_index]
        tra_end_index = self.meta['indices'][tra_index+1]
        ctra_start_index = self.meta['cindices'][tra_index]
        ctra_end_index = self.meta['cindices'][tra_index+1]

        tra_velocity = self.fp[tra_start_index:tra_end_index, tra_sample_index]
        tra_target = self.fp[tra_start_index:tra_end_index, tra_sample_index+1]
        pos = self.meta['pos'][tra_start_index:tra_end_index]
        node_type = self.meta['node_type'][tra_start_index:tra_end_index]
        cells = self.meta['cells'][ctra_start_index:ctra_end_index]

        x = np.concatenate([node_type, tra_velocity], axis=-1)
        x = torch.as_tensor(x.copy(), dtype=torch.float32) # .copy to writeable memory
        pos = torch.as_tensor(pos.copy(), dtype=torch.float32)
        face = torch.as_tensor(cells.T.copy(), dtype=torch.int64)
        y = torch.as_tensor(tra_target.copy(), dtype=torch.float32)
        
        graph = Data(x=x,pos=pos,face=face,y=y)

        return graph
    
    def __len__(self):
        return self.total_samples

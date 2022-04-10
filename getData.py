import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class TrainSet(Dataset):
    def __init__(self, hdf5_Idx, opt):
        
        # set the path of datasets
        train_txt_path = opt.trainset_path+"train_files.txt"
        # get the list of .h5 files
        with open(train_txt_path,'r') as f:
            h5_list = f.read()
        file_name = h5_list.split()[hdf5_Idx]

        # load the data
        print('Loading the trainset:  ' + file_name)
        h5 = h5py.File(opt.trainset_path + file_name, 'r')
        self.data = torch.Tensor(np.array(h5["data"]))
        self.lenth = self.data.size(0)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.lenth

class ValidSet(Dataset):
    def __init__(self, hdf5_Idx, opt):
        
        valid_txt_path = opt.trainset_path+"valid_files.txt"
        with open(valid_txt_path,'r') as f:
            h5_list = f.read()
        file_name = h5_list.split()[hdf5_Idx]
        h5 = h5py.File(opt.trainset_path + file_name, 'r')
        self.data = torch.Tensor(np.array(h5["data"]))
        self.lenth = self.data.size(0)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.lenth

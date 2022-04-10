import torch
import torch.nn as nn
import numpy as np
import utils

class CodedNet(nn.Module):

    def __init__(self):

        super(CodedNet, self).__init__()

        opt = utils.parse_arg()
        self.block_size = opt.blocksize
        self.channel = opt.channel

    # image compress
    def encode_CASSI(self, x, Mask):
        y = torch.mul(x, Mask)      #[48, 48, channel]
        y = y.sum(3)     #[48, 48]
        return y

    # init x0 for deep unfolding
    def init_CASSI(self, y, Mask, channel):
        y1 = y.unsqueeze(3)       #[batchsize, 48, 48, 1]
        y2 = y1.repeat([1, 1, 1, channel])    #[batchsize, 48, 48, channel]
        x0 = torch.mul(y2, Mask)
        return x0

    # random a mask∈{0, 1} for one batch
    def ran_Cu(self, batch_size):
        Cu_input = np.zeros([self.block_size, self.block_size, self.channel])
        T = np.round(np.random.rand(int(self.block_size/2), int(self.block_size/2)))
        T = np.concatenate([T,T],axis=0)
        T = np.concatenate([T,T],axis=1)
        for ch in range(self.channel):
            Cu_input[:,:,ch] = np.roll(T, shift=-ch, axis=0)      #roll the mask
        Cu_input = np.expand_dims(Cu_input, axis=0)
        Cu_input = torch.Tensor(np.tile(Cu_input, [batch_size, 1, 1, 1]))
        return Cu_input

    def forward(self, x): 

        batch_size = x.size(0)      #batchsize
        Cu_input = self.ran_Cu(batch_size).cuda()    #random mask
        y  = self.encode_CASSI(x, Cu_input)       #get measurements
        x0 = self.init_CASSI(y, Cu_input, self.channel)     #init x0

        return x0, Cu_input

def CodedModel():
    return CodedNet()